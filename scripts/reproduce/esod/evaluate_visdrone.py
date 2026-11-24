"""在 VisDrone val 上重评作者的 ESOD-YOLOv5m，并记录单图耗时。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "reproduce/esod/upstream"
LOCAL = ROOT / "reproduce/esod/local"
WEIGHT = LOCAL / "esod_yolov5m.pt"
SHA256 = "7adbe5e7fbfb968aaf7cea5dbf98ee0ed2bc5fc149c962167482a32492c29fae"
REVISION = "bde3571bd7db697e441eef0278cd425e888ea026"


def read_visdrone(image_path, annotation_path):
    """按作者规则涂灰忽略区，并从原始标注提取十类水平框。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取 {image_path}")
    boxes = []
    masked = False
    for line in annotation_path.read_text(encoding="utf-8-sig").splitlines():
        left, top, width, height, _, category, truncation, _ = map(int, line.rstrip(",").split(","))
        if category in (0, 11):
            image[top:top + height, left:left + width] = 85
            masked = True
        elif truncation < 2:
            boxes.append((category, [left, top, width, height]))
    return image, boxes, masked


def prepare_image(image_path, annotation_path):
    """COCO 重评在内存生成作者加载器读到的验证图。"""
    image, boxes, masked = read_visdrone(image_path, annotation_path)
    if masked:
        # 作者把涂灰后的图另存 JPEG 后再读入，保留这次有损编码。
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError(f"JPEG 编码失败 {image_path}")
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image, boxes


def predict(model, image, long_edge, forward_only=False):
    from utils.datasets import letterbox, norm_imgs
    from utils.general import non_max_suppression, scale_coords

    # 对齐作者 load_image + rect=True、batch=1、pad=0 的尺寸取整。
    original = image.shape[:2]
    ratio = long_edge / max(original)
    shape = tuple(math.ceil(side * long_edge / max(original) / 32) * 32 for side in original)
    resized = tuple(int(side * ratio) for side in original)
    start = perf_counter()
    if ratio != 1:
        image = cv2.resize(image, resized[::-1], interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
    padded, _, padding = letterbox(image, shape, auto=False, scaleup=False)
    tensor = torch.from_numpy(padded[:, :, ::-1].transpose(2, 0, 1).copy()).cuda().float()[None]
    tensor = norm_imgs(tensor, model)
    torch.cuda.synchronize()
    ready = perf_counter()
    (output, _), _ = model(tensor)
    torch.cuda.synchronize()
    inferred = perf_counter()
    if forward_only:
        return [], [(inferred - ready) * 1000, None]
    # 直接复用作者的多标签 NMS 与坐标还原，空预测图也计入评测和计时。
    if output is None:
        detections = []
    else:
        result = non_max_suppression(output, 0.001, 0.6, multi_label=True)[0]
        scale_coords(shape, result[:, :4], original, ((resized[0] / original[0], resized[1] / original[1]), padding))
        detections = result.cpu().tolist()
    torch.cuda.synchronize()
    done = perf_counter()
    return detections, [(inferred - ready) * 1000, (done - start) * 1000]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/visdrone")
    parser.add_argument("--head", choices=["sparse", "dense", "vanilla-timing"], default="sparse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=548)
    parser.add_argument("--long-edge", type=int, choices=(1536, 1920), default=1536)
    args = parser.parse_args()
    images = sorted((args.data / "images/val").glob("*.jpg"))
    if len(images) != 548 or not 1 <= args.limit <= len(images):
        raise ValueError(f"期望 548 张完整验证图，发现 {len(images)}，limit={args.limit}")
    images = images[:args.limit]
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if commit != REVISION:
        raise ValueError(f"作者提交与复现卡不符：{commit}")
    sys.path.insert(0, str(UPSTREAM))
    forward_only = args.head == "vanilla-timing"
    weight_info = {}
    if forward_only:
        from models.yolo import Model

        # 全图稠密网络的前向计算图与参数值无关；随机权重仅用于测速。
        torch.manual_seed(0)
        model = Model(str(UPSTREAM / "models/cfg/vanilla/visdrone_yolov5m.yaml")).cuda().float().fuse()
    else:
        with WEIGHT.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != SHA256:
            raise ValueError(f"权重 SHA-256 不匹配：{digest}")
        # 该旧格式包含模型对象，仅在来源、提交和文件哈希核对后加载。
        checkpoint = torch.load(WEIGHT, weights_only=False, map_location="cpu")
        model = (checkpoint.get("ema") or checkpoint["model"]).cuda().float().fuse()
        weight_info = {"sha256": digest, "checkpoint_epoch": checkpoint["epoch"],
                       "checkpoint_branch": "ema" if checkpoint.get("ema") else "model"}
        if args.head == "sparse":
            model.model[-1].set_sparse()
    model.eval()
    gt = COCO()
    gt.dataset = {
        "images": [], "annotations": [],
        "categories": [{"id": idx + 1, "name": name} for idx, name in enumerate(model.names)],
    }
    predictions = []
    durations = []
    for image_id, path in enumerate(tqdm(images, desc=args.head), start=1):
        image, boxes = prepare_image(path, args.data / "original_annotations/val" / (path.stem + ".txt"))
        if image_id == 1:
            for _ in range(10):
                predict(model, image, args.long_edge, forward_only)
        detections, elapsed = predict(model, image, args.long_edge, forward_only)
        durations.append(elapsed)
        if forward_only:
            continue
        gt.dataset["images"].append({"id": image_id, "file_name": path.name,
                                     "width": image.shape[1], "height": image.shape[0]})
        for category, box in boxes:
            gt.dataset["annotations"].append({"id": len(gt.dataset["annotations"]) + 1, "image_id": image_id,
                                              "category_id": category, "bbox": box,
                                              "area": box[2] * box[3], "iscrowd": 0})
        for left, top, right, bottom, score, category in detections:
            predictions.append({"image_id": image_id, "category_id": int(category) + 1,
                                "bbox": [left, top, right - left, bottom - top], "score": score})
    forward = np.mean([pair[0] for pair in durations])
    report = {
        "head": args.head, "images": len(images), "upstream": commit, **weight_info,
        "config": model.yaml_file, "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "precision": "fp32", "long_edge": args.long_edge, "batch": 1, "warmup": 10,
        "forward_ms": forward, "forward_fps": 1000 / forward,
    }
    outputs = {"durations_ms": durations, "summary": report}
    if not forward_only:
        gt.createIndex()
        if predictions:
            results = gt.loadRes(predictions)
        else:
            results = COCO()
            results.dataset = {**gt.dataset, "annotations": []}
            results.createIndex()
        evaluator = COCOeval(gt, results, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        pipeline = np.mean([pair[1] for pair in durations])
        report.update(ground_truth=len(gt.anns), detections=len(predictions),
                      AP=evaluator.stats[0] * 100, AP50=evaluator.stats[1] * 100,
                      AP_small=evaluator.stats[3] * 100, pipeline_ms=pipeline, pipeline_fps=1000 / pipeline)
        outputs["predictions"] = predictions
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        (args.output / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
