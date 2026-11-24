"""在 VisDrone val 上评测作者或本机训练的 YOLOv5m，并记录单图耗时。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from tqdm import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "reproduce/esod/upstream"
LOCAL = ROOT / "reproduce/esod/local"
WEIGHT = LOCAL / "esod_yolov5m.pt"
SHA256 = "7adbe5e7fbfb968aaf7cea5dbf98ee0ed2bc5fc149c962167482a32492c29fae"
REVISION = "bde3571bd7db697e441eef0278cd425e888ea026"
VISDRONE_NAMES = ("pedestrian", "people", "bicycle", "car", "van", "truck",
                  "tricycle", "awning-tricycle", "bus", "motor")


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
        elif not 1 <= category <= len(VISDRONE_NAMES):
            raise ValueError(f"{annotation_path} 含未知 VisDrone 类别 {category}")
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


def validate_model(model, head):
    """用内嵌配置和实际网络核对实验臂，两个配置的文件名相同。"""
    from models.common import YOLOv6Head
    from models.yolo import Detect, Model

    family = "vanilla" if head == "vanilla" else "esod"
    config = Path("models/cfg") / family / "visdrone_yolov5m.yaml"
    expected = yaml.safe_load((UPSTREAM / config).read_text(encoding="utf-8"))
    if not isinstance(model, Model) or not isinstance(model.yaml, dict):
        raise ValueError("checkpoint 必须包含作者训练保存的 Model 对象")
    if model.yaml.get("ch", 3) != 3 or any(model.yaml.get(key) != value for key, value in expected.items()):
        raise ValueError(f"--head {head} 需要与 {config.as_posix()} 一致的内嵌配置")

    # 同时核对反序列化后的层与连接，避免仅靠 YAML 给另一种检测头贴标签。
    layout = expected["backbone"] + expected["head"]
    if len(model.model) != len(layout) or any(
        type(layer).__name__ != kind.removeprefix("nn.") or layer.f != source
        for layer, (source, _, kind, _) in zip(model.model, layout)
    ):
        raise ValueError(f"checkpoint 的实际网络或检测头与 {config.as_posix()} 不一致")
    detector = model.model[-1]
    if (not isinstance(detector, Detect) or detector.nc != len(VISDRONE_NAMES)
            or detector.no != detector.nc + 5 or detector.nl != 3 or detector.na != 3
            or len(detector.m) != 3 or getattr(detector, "sparse", False)
            or not all(isinstance(branch, YOLOv6Head) and branch.nc == detector.nc
                       and branch.na == detector.na for branch in detector.m)):
        raise ValueError("需要作者训练时的三尺度十类 Detect 检测头")
    names = model.names
    if isinstance(names, dict) and set(names) == set(range(len(VISDRONE_NAMES))):
        names = [names[index] for index in range(len(VISDRONE_NAMES))]
    if not isinstance(names, (list, tuple)) or tuple(names) != VISDRONE_NAMES:
        raise ValueError(f"checkpoint 的类别名称或顺序与 VisDrone 不一致：{names}")
    return config.as_posix()


def load_trained_model(path, sha256, head, device="cuda"):
    # 旧 checkpoint 含 Python 对象；在同一文件句柄上先验摘要，再反序列化。
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != sha256.lower():
            raise ValueError(f"权重 SHA-256 不匹配：{path}，实际 {digest}")
        handle.seek(0)
        checkpoint = torch.load(handle, weights_only=False, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("期望含有 ema 或 model 的训练 checkpoint")
    branch = "ema" if checkpoint.get("ema") is not None else "model"
    model = checkpoint.get(branch)
    config = validate_model(model, head)
    detector = model.model[-1]
    # 训练验证留下的网格不全是注册缓冲区，切换设备后应由首次前向重建。
    detector.grid = [torch.empty(0) for _ in range(detector.nl)]
    if head != "vanilla":
        from models.common import HeatMapParser

        for layer in model.model:
            if isinstance(layer, HeatMapParser):
                layer.grid = layer.grid_vtx = None
    model = model.to(device).float().fuse()
    if head == "sparse":
        detector.set_sparse()
    return model.eval(), {"weights": str(path.resolve()), "sha256": digest,
                          "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_branch": branch,
                          "validated_config": config}


def predict(model, image, long_edge, forward_only=False):
    from utils.datasets import letterbox, norm_imgs
    from utils.general import non_max_suppression, scale_coords

    # 对齐作者 load_image + rect=True、batch=1、pad=0 的尺寸取整。
    original = image.shape[:2]
    ratio = long_edge / max(original)
    shape = tuple(math.ceil(side * long_edge / max(original) / 32) * 32 for side in original)
    resized = tuple(int(side * ratio) for side in original)
    device = next(model.parameters()).device
    start = perf_counter()
    if ratio != 1:
        image = cv2.resize(image, resized[::-1], interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
    padded, _, padding = letterbox(image, shape, auto=False, scaleup=False)
    tensor = torch.from_numpy(padded[:, :, ::-1].transpose(2, 0, 1).copy()).to(device).float()[None]
    tensor = norm_imgs(tensor, model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ready = perf_counter()
    # 两种 Model 都返回 ((解码预测, 各尺度原始输出), 热图)，vanilla 的热图为 None。
    (output, _), _ = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inferred = perf_counter()
    if forward_only:
        return [], [(inferred - ready) * 1000, None]
    # 直接复用作者的多标签 NMS 与坐标还原，空预测图也计入评测和计时。
    if output is None:
        detections = []
    else:
        if (not isinstance(output, torch.Tensor) or not output.is_floating_point() or output.ndim != 3
                or output.shape[0] != 1 or output.shape[2] != 5 + len(VISDRONE_NAMES)):
            raise ValueError("期望解码预测为形状 [1, N, 15] 的浮点张量")
        result = non_max_suppression(output, 0.001, 0.6, multi_label=True)[0]
        scale_coords(shape, result[:, :4], original, ((resized[0] / original[0], resized[1] / original[1]), padding))
        detections = result.cpu().tolist()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    done = perf_counter()
    return detections, [(inferred - ready) * 1000, (done - start) * 1000]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/visdrone")
    parser.add_argument("--head", choices=["sparse", "dense", "vanilla", "vanilla-timing"], default="sparse",
                        help="vanilla 评测已训练基线，vanilla-timing 仅测随机基线前向")
    parser.add_argument("--weights", type=Path, help="本机可信训练 checkpoint，与 --sha256 配套使用")
    parser.add_argument("--sha256", help="训练产物的 64 位十六进制 SHA-256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=548)
    parser.add_argument("--long-edge", type=int, choices=(640, 768, 1536, 1920), default=1536)
    args = parser.parse_args(argv)
    if (args.weights is None) != (args.sha256 is None):
        parser.error("--weights 和 --sha256 必须同时提供")
    if args.sha256 is not None and not re.fullmatch("[0-9a-fA-F]{64}", args.sha256):
        parser.error("--sha256 需要 64 位十六进制摘要")
    if args.head == "vanilla" and args.weights is None:
        parser.error("--head vanilla 需要本机训练产物 --weights 和 --sha256")
    if args.head == "vanilla-timing" and args.weights is not None:
        parser.error("评测已训练的基线请使用 --head vanilla")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
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
        model, weight_info = load_trained_model(args.weights or WEIGHT, args.sha256 or SHA256, args.head)
    model.eval()
    gt = COCO()
    gt.dataset = {
        "images": [], "annotations": [],
        "categories": [{"id": idx + 1, "name": name} for idx, name in enumerate(VISDRONE_NAMES)],
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
        "config": getattr(model, "yaml_file", "visdrone_yolov5m.yaml"), "python": platform.python_version(),
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
