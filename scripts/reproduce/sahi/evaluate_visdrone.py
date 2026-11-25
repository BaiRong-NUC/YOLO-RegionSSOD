"""在完整 VisDrone val 上配对评测 YOLO11s 整图与 SAHI 切片推理。"""

import argparse
from contextlib import redirect_stdout
from functools import partial
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from tqdm import tqdm

from scripts.reproduce.esod.evaluate_visdrone import VISDRONE_NAMES, prepare_image


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "reproduce/sahi/upstream"
REVISION = "68e61981e1f79c56299c208df1d5bac03a01811d"
IMAGE_SIZE = 640
MAX_DET = 500
CONFIDENCE = 0.001


def read_validation(data, limit, split='val', *, rgb=True):
    """复用 ESOD 的忽略区协议，预解码后在内存配对测速。"""
    records = sorted((data / 'images' / split).glob('*.jpg'))
    annotation_dir = data / 'original_annotations' / split
    expected_count = {'val': 548, 'test': 1610}[split]
    if len(records) != expected_count or {p.stem for p in records} != {p.stem for p in annotation_dir.glob('*.txt')}:
        raise ValueError(f'需要完整的 VisDrone {split} 图像和原始标注')
    gt = COCO()
    gt.dataset = {"info": {}, "images": [], "annotations": [], "categories": [
        {"id": index, "name": name} for index, name in enumerate(VISDRONE_NAMES, 1)
    ]}
    images = []
    for image_id, path in enumerate(tqdm(records[:limit], desc="准备验证图"), 1):
        image, boxes = prepare_image(path, annotation_dir / f"{path.stem}.txt")
        height, width = image.shape[:2]
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if rgb else image)
        gt.dataset["images"].append({"id": image_id, "file_name": path.name, "width": width, "height": height})
        for category, box in boxes:
            gt.dataset["annotations"].append({
                "id": len(gt.dataset["annotations"]) + 1, "image_id": image_id,
                "category_id": category, "bbox": box, "iscrowd": 0, "area": box[2] * box[3],
            })
    gt.createIndex()
    return images, gt


def evaluate(gt, detections):
    """直接按 500 框上限读取 COCO 精度张量，保留 AP50s 与 AP_small。"""
    with redirect_stdout(StringIO()):
        if detections:
            predicted = gt.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = {**gt.dataset, "annotations": []}
            predicted.createIndex()
        evaluator = COCOeval(gt, predicted, "bbox")
        evaluator.params.maxDets = [1, 10, MAX_DET]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]
    metrics = {}
    for label, area in enumerate(evaluator.params.areaRngLbl):
        suffix = "" if area == "all" else f"_{area}"
        for name, ious in (("AP", slice(None)), ("AP50", slice(0, 1))):
            values = precision[ious, :, :, label, -1]
            valid = values[values >= 0]
            metrics[name + suffix] = float(valid.mean() * 100) if valid.size else None
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/visdrone")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, choices=range(1, 549), metavar="1..548")
    args = parser.parse_args()

    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"SAHI 作者实现版本应为 {REVISION}，当前为 {revision}")
    sys.path.insert(0, str(UPSTREAM))
    from sahi.models.ultralytics import UltralyticsDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction
    import ultralytics

    # 两臂共享模型实例和输入尺寸。记录每张原图的前向次数及完整推理耗时。
    class CountedYOLO(UltralyticsDetectionModel):
        calls = 0

        def perform_inference(self, image):
            self.calls += 1
            return super().perform_inference(image)

    def predict(image, model, arm):
        if arm == "full":
            return get_prediction(image, model, verbose=0)
        return get_sliced_prediction(
            image, model, slice_height=640, slice_width=640, overlap_height_ratio=0.25,
            overlap_width_ratio=0.25, perform_standard_pred=True, postprocess_type="NMS",
            postprocess_match_metric="IOU", postprocess_match_threshold=0.5, verbose=0,
        )

    cv2.setNumThreads(1)
    images, gt = read_validation(args.data, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "revision": revision, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(), "opencv": cv2.__version__, "command": sys.argv,
        "images": len(images), "ground_truth": len(gt.anns), "precision": "fp32",
        "image_size": IMAGE_SIZE, "confidence": CONFIDENCE, "max_det": MAX_DET, "runs": [],
    }
    for weight in args.weights:
        with weight.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        model = CountedYOLO(model_path=str(weight), confidence_threshold=CONFIDENCE,
                            image_size=IMAGE_SIZE, device="cuda:0", fuse=True)
        if model.is_obb or tuple(model.category_names) != VISDRONE_NAMES:
            raise ValueError(f"权重须为 VisDrone 十类水平框模型: {weight}")
        # Ultralytics 默认按长宽比调整输入张量，这里固定两组的每次前向为 640×640。
        model.model.predict = partial(model.model.predict, max_det=MAX_DET, rect=False, iou=0.7)
        for round_id, order in enumerate((("full", "sliced"), ("sliced", "full")), 1):
            for arm in order:
                for image in images[:10]:
                    predict(image, model, arm)
                if model.model.predictor.args.rect or model.model.predictor.model.fp16:
                    raise ValueError("评测要求固定 640×640 的 FP32 输入")
                total_calls = 0
                durations, detections = [], []
                for image_id, image in enumerate(tqdm(images, desc=f"{weight.parent.parent.name}/{round_id}/{arm}"), 1):
                    model.calls = 0
                    torch.cuda.synchronize()
                    start = perf_counter()
                    result = predict(image, model, arm)
                    torch.cuda.synchronize()
                    boxes = sorted(result.object_prediction_list, key=lambda box: box.score.value, reverse=True)
                    durations.append((perf_counter() - start) * 1000)
                    total_calls += model.calls
                    for box in boxes[:MAX_DET]:
                        detections.append({"image_id": image_id, "category_id": box.category.id + 1,
                                           "bbox": box.bbox.to_xywh(), "score": box.score.value})

                report = {
                    "weight": str(weight), "sha256": digest, "round": round_id, "arm": arm,
                    "calls_per_image": total_calls / len(images), "mean_ms": float(np.mean(durations)),
                    "p95_ms": float(np.percentile(durations, 95)), "detections": len(detections),
                    **evaluate(gt, detections),
                }
                print(json.dumps(report), flush=True)
                output["runs"].append({**report, "durations_ms": durations})
                args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
