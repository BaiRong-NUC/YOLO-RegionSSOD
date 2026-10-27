from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from regionssod.data.annotations import iter_images, label_path_for_image, read_label_file
from regionssod.data.dataset import load_source_dataset
from regionssod.pseudo.generator import _prepare_output


def match_by_confidence(classes, confidences, xywhn, gt_classes, gt_xywhn, iou=0.5):
    """Class-aware, confidence-ordered greedy matching against ALL size groups."""
    gt = np.asarray(gt_xywhn, dtype=float).reshape(-1, 4)
    predictions = np.asarray(xywhn, dtype=float).reshape(-1, 4)
    matched = np.zeros(len(gt), dtype=bool)
    gt_lo, gt_hi = gt[:, :2] - gt[:, 2:] / 2, gt[:, :2] + gt[:, 2:] / 2
    for index in np.argsort(-np.asarray(confidences), kind="stable"):
        box = predictions[index]
        lo, hi = box[:2] - box[2:] / 2, box[:2] + box[2:] / 2
        intersection = np.maximum(0, np.minimum(hi, gt_hi) - np.maximum(lo, gt_lo)).prod(axis=1)
        union = box[2:].prod() + gt[:, 2:].prod(axis=1) - intersection
        overlaps = intersection / np.maximum(union, 1e-12)
        eligible = (np.asarray(gt_classes) == classes[index]) & ~matched & (overlaps >= iou)
        if eligible.any():
            best = np.where(eligible, overlaps, -1).argmax()
            matched[best] = True
    return matched


def run_size_recall(weights: Path, data: Path, output: Path, *, imgsz=960, batch=8,
                    device="0", conf=0.25, match_iou=0.5) -> dict:
    """Fixed-operating-point recall on full val; not official AP_small or AR500."""
    from ultralytics import YOLO
    import ultralytics

    if not 0 < conf <= 1 or not 0 < match_iou <= 1:
        raise ValueError("conf and matching IoU must be in (0, 1]")
    source = load_source_dataset(data)
    images = iter_images(source.images_val)
    weights, output = weights.resolve(), output.resolve()
    if not weights.is_file() or not images:
        raise ValueError("weights and nonempty full-image validation split are required")
    _prepare_output(output)
    image_list = output / "images.txt"
    image_list.write_text("\n".join(map(str, images)) + "\n", encoding="utf-8")
    counts = {s: {"gt": 0, "matched": 0} for s in ("small", "medium", "large")}
    processed = 0
    predictions = YOLO(str(weights)).predict(source=str(image_list), stream=True, imgsz=imgsz,
                                            batch=batch, device=device, conf=conf, iou=.7,
                                            max_det=500, verbose=False)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, result in enumerate(predictions):
            image = Path(result.path).resolve()
            if index >= len(images) or image != images[index].resolve():
                raise RuntimeError("validation prediction order differs from image list")
            processed += 1
            gt = read_label_file(label_path_for_image(image, source.images_val, source.labels_val))
            gt_boxes = np.array([[p.x, p.y, p.w, p.h] for p in gt]).reshape(-1, 4)
            boxes = result.boxes
            cls = boxes.cls.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            xywhn = boxes.xywhn.cpu().numpy()
            matched = match_by_confidence(cls, scores, xywhn, [p.class_id for p in gt], gt_boxes, match_iou)
            height, width = result.orig_shape
            areas = gt_boxes[:, 2] * width * gt_boxes[:, 3] * height
            for size, mask in (("small", areas < 32**2), ("medium", (areas >= 32**2) & (areas < 96**2)),
                               ("large", areas >= 96**2)):
                counts[size]["gt"] += int(mask.sum())
                counts[size]["matched"] += int((mask & matched).sum())
            handle.write(json.dumps({"image": str(image), "classes": cls.tolist(),
                                     "confidences": scores.tolist(), "xywhn": xywhn.tolist()}) + "\n")
    if processed != len(images):
        raise RuntimeError("incomplete validation predictions")
    for values in counts.values():
        values["recall"] = values["matched"] / values["gt"] if values["gt"] else None
    summary = {"weights": str(weights), "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
               "data": str(data.resolve()), "images": processed, "imgsz": imgsz, "conf": conf,
               "match_iou": match_iou, "nms_iou": .7, "max_det": 500,
               "ultralytics_version": ultralytics.__version__, "size_pixel_space": "native full-image GT",
               "matching": "confidence-descending class-aware greedy, all GT sizes jointly",
               "metrics_scope": "YOLO full-val diagnostic recall; not official AP_small or AR500",
               "per_size": counts}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
