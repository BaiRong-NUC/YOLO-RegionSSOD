from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from regionssod.pseudo.generator import _link_or_copy, _prepare_output, read_image_list


@dataclass(frozen=True)
class ScalePolicy:
    small_threshold: float = 0.30
    medium_threshold: float = 0.50
    large_threshold: float = 0.50
    conf_floor: float = 0.05
    small_area: float = 1024.0
    large_area: float = 9216.0

    def __post_init__(self) -> None:
        values = list(asdict(self).values())
        if not all(math.isfinite(v) for v in values):
            raise ValueError("P1 policy values must be finite")
        if not 0 < self.conf_floor <= min(self.thresholds) or max(self.thresholds) > 1:
            raise ValueError("require 0 < conf_floor <= all thresholds <= 1")
        if not 0 < self.small_area < self.large_area:
            raise ValueError("require 0 < small_area < large_area")

    @property
    def thresholds(self) -> tuple[float, float, float]:
        return self.small_threshold, self.medium_threshold, self.large_threshold

    @classmethod
    def load(cls, path: Path) -> ScalePolicy:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("P1 policy must be a YAML mapping")
        return cls(**raw)

    def select(self, classes, confidences, xywhn, shape) -> list[dict]:
        """Partition post-NMS teacher candidates; scale is in native tile pixels."""
        boxes = np.asarray(xywhn, dtype=float).reshape(-1, 4)
        if len(classes) != len(confidences) or len(classes) != len(boxes):
            raise ValueError("candidate arrays must have equal length")
        height, width = shape
        selected = []
        for cls, conf, box in zip(classes, confidences, boxes, strict=True):
            if not np.isfinite(box).all() or not math.isfinite(float(conf)):
                raise ValueError("non-finite teacher candidate")
            if not 0 <= conf <= 1 or int(cls) != cls or cls < 0:
                raise ValueError("invalid teacher class or confidence")
            if conf < self.conf_floor or min(box[2:]) <= 0:
                continue
            if np.any(box < 0) or np.any(box > 1):
                raise ValueError("teacher box must be normalized")
            area = float(box[2] * width * box[3] * height)
            scale = "small" if area < self.small_area else "medium" if area < self.large_area else "large"
            threshold = dict(zip(("small", "medium", "large"), self.thresholds))[scale]
            selected.append({"class_id": int(cls), "confidence": float(conf),
                             "xywhn": box.tolist(), "area": area, "scale": scale,
                             "state": "positive" if conf >= threshold else "ignore"})
        return selected


def generate_scale_labels(weights: Path, images_list: Path, output_dir: Path, *,
                          policy_path: Path, imgsz: int = 960, batch: int = 8,
                          device: str = "0", max_det: int = 500,
                          limit: int | None = None) -> dict:
    from ultralytics import YOLO
    import ultralytics

    policy = ScalePolicy.load(policy_path)
    weights = weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if batch <= 0 or imgsz <= 0 or max_det <= 0 or (limit is not None and limit <= 0):
        raise ValueError("batch, imgsz, max_det and limit must be positive")
    images = read_image_list(images_list)
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError("no images for P1 inference")
    output_dir = output_dir.resolve()
    _prepare_output(output_dir)
    for kind in ("images", "labels"):
        (output_dir / kind / "train").mkdir(parents=True)
    prediction_list = output_dir / "prediction_images.txt"
    prediction_list.write_text("\n".join(map(str, images)) + "\n", encoding="utf-8")
    summary = {"method": "P1", "policy": asdict(policy), "weights": str(weights),
               "teacher_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
               "ultralytics_version": ultralytics.__version__, "imgsz": imgsz,
               "batch": batch, "device": device, "max_det": max_det, "iou": 0.7,
               "requested_images": len(images), "processed_images": 0,
               "accepted_images": 0, "accepted_boxes": 0, "ignore_boxes": 0,
               "excluded_images_without_positives": 0,
               "per_scale": {s: {"positive": 0, "ignore": 0} for s in ("small", "medium", "large")},
               "per_class": {}, "metadata": str(output_dir / "metadata.jsonl")}
    accepted_images = []
    predictions = YOLO(str(weights)).predict(
        source=str(prediction_list), imgsz=imgsz, conf=policy.conf_floor, iou=0.7,
        batch=batch, device=device, max_det=max_det, stream=True, verbose=False)
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as metadata, \
            (output_dir / "candidates.jsonl").open("w", encoding="utf-8") as candidates:
        for index, result in enumerate(predictions):
            source = Path(result.path).resolve()
            if index >= len(images) or source != images[index]:
                raise RuntimeError("teacher output order differs from frozen image list")
            summary["processed_images"] += 1
            boxes = result.boxes
            items = [] if boxes is None else policy.select(
                boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(),
                boxes.xywhn.cpu().numpy(), result.orig_shape)
            positives = [item for item in items if item["state"] == "positive"]
            ignores = [item for item in items if item["state"] == "ignore"]
            candidates.write(json.dumps({"source_image": str(source), "shape": list(result.orig_shape),
                                         "candidates": items}) + "\n")
            if not positives:
                summary["excluded_images_without_positives"] += 1
                continue
            image = output_dir / "images" / "train" / f"{index:07d}_{source.name}"
            _link_or_copy(source, image)
            lines = [f"{p['class_id']} " + " ".join(f"{v:.10g}" for v in p["xywhn"]) for p in positives]
            (output_dir / "labels" / "train" / f"{image.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            metadata.write(json.dumps({"image": str(image), "source_image": str(source),
                                       "positive": positives, "ignore": ignores}) + "\n")
            accepted_images.append(str(image))
            summary["accepted_boxes"] += len(positives)
            summary["ignore_boxes"] += len(ignores)
            for item in items:
                summary["per_scale"][item["scale"]][item["state"]] += 1
                if item["state"] == "positive":
                    key = str(item["class_id"])
                    summary["per_class"][key] = summary["per_class"].get(key, 0) + 1
    if summary["processed_images"] != len(images):
        raise RuntimeError("teacher did not return every requested image")
    summary["accepted_images"] = len(accepted_images)
    (output_dir / "pseudo_train.txt").write_text(
        "\n".join(accepted_images) + ("\n" if accepted_images else ""), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
