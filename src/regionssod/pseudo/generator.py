from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np

from regionssod.data.annotations import IMAGE_SUFFIXES
from regionssod.data.dataset import load_source_dataset, write_dataset_yaml


def _prepare_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"output directory is not empty; choose a new path: {output}"
            )
    output.mkdir(parents=True, exist_ok=True)


def read_image_list(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_dir():
        images = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        )
    elif path.suffix.lower() == ".txt":
        images = []
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            images.append(candidate.resolve())
    elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        images = [path]
    else:
        raise FileNotFoundError(f"image list, directory, or image not found: {path}")

    deduplicated = list(dict.fromkeys(images))
    missing = [image for image in deduplicated if not image.is_file()]
    if missing:
        raise FileNotFoundError(f"image not found: {missing[0]}")
    return deduplicated


def filter_predictions(
    classes: Iterable[int],
    confidences: Iterable[float],
    xywhn: np.ndarray,
    threshold: float,
) -> list[tuple[int, float, float, float, float, float]]:
    classes_array = np.asarray(list(classes), dtype=np.int64).reshape(-1)
    confidence_array = np.asarray(list(confidences), dtype=np.float64).reshape(-1)
    box_array = np.asarray(xywhn, dtype=np.float64)
    if box_array.ndim != 2 or box_array.shape[1] != 4:
        raise ValueError("xywhn must have shape [N, 4]")
    if not (
        classes_array.shape[0]
        == confidence_array.shape[0]
        == box_array.shape[0]
    ):
        raise ValueError("classes, confidences, and xywhn must have equal length")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    accepted: list[tuple[int, float, float, float, float, float]] = []
    for class_id, confidence, box in zip(
        classes_array, confidence_array, box_array, strict=True
    ):
        x, y, w, h = (float(value) for value in box)
        if confidence < threshold or w <= 0 or h <= 0:
            continue
        accepted.append((int(class_id), float(confidence), x, y, w, h))
    accepted.sort(key=lambda item: item[1], reverse=True)
    return accepted


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def generate_pseudo_labels(
    weights: Path,
    images_list: Path,
    output_dir: Path,
    *,
    threshold: float = 0.5,
    conf_floor: float = 0.05,
    imgsz: int = 960,
    batch: int = 8,
    device: str = "0",
    max_det: int = 500,
    limit: int | None = None,
) -> dict[str, object]:
    from ultralytics import YOLO

    weights = weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if not 0 <= conf_floor <= threshold:
        raise ValueError("conf_floor must be in [0, threshold]")

    image_paths = read_image_list(images_list)
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError("no images were found for pseudo-label generation")

    output_dir = output_dir.resolve()
    _prepare_output(output_dir)
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        conf=conf_floor,
        iou=0.7,
        batch=batch,
        device=device,
        max_det=max_det,
        stream=True,
        verbose=False,
    )

    accepted_images: list[str] = []
    accepted_boxes = 0
    per_class: dict[str, int] = {}
    processed = 0
    empty_predictions = 0
    for result in predictions:
        source_path = (
            Path(result.path)
            if getattr(result, "path", None)
            else image_paths[processed]
        )
        processed += 1
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            empty_predictions += 1
            continue

        classes = boxes.cls.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        xywhn = boxes.xywhn.detach().cpu().numpy()
        accepted = filter_predictions(
            classes=classes,
            confidences=confidences,
            xywhn=xywhn,
            threshold=threshold,
        )
        if not accepted:
            empty_predictions += 1
            continue

        unique_name = (
            f"{processed - 1:07d}_{source_path.stem}{source_path.suffix.lower()}"
        )
        destination_image = images_dir / unique_name
        _link_or_copy(source_path, destination_image)
        label_lines = []
        for class_id, _confidence, x, y, w, h in accepted:
            label_lines.append(
                f"{class_id} {x:.10g} {y:.10g} {w:.10g} {h:.10g}"
            )
            class_key = str(class_id)
            per_class[class_key] = per_class.get(class_key, 0) + 1
        (labels_dir / f"{destination_image.stem}.txt").write_text(
            "\n".join(label_lines) + "\n", encoding="utf-8"
        )
        accepted_images.append(str(destination_image))
        accepted_boxes += len(accepted)

    pseudo_list_path = output_dir / "pseudo_train.txt"
    pseudo_list_path.write_text(
        "\n".join(accepted_images) + ("\n" if accepted_images else ""),
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "weights": str(weights),
        "images_list": str(images_list.resolve()),
        "output_dir": str(output_dir),
        "threshold": threshold,
        "conf_floor": conf_floor,
        "imgsz": imgsz,
        "requested_images": len(image_paths),
        "processed_images": processed,
        "accepted_images": len(accepted_images),
        "empty_predictions": empty_predictions,
        "accepted_boxes": accepted_boxes,
        "per_class": per_class,
        "pseudo_train_list": str(pseudo_list_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def combine_image_lists(
    inputs: list[Path],
    output: Path,
    dataset_yaml: Path,
    source_data: Path,
) -> dict[str, object]:
    all_images: list[Path] = []
    for input_path in inputs:
        all_images.extend(read_image_list(input_path))
    deduplicated = list(dict.fromkeys(all_images))

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(str(path) for path in deduplicated) + "\n",
        encoding="utf-8",
    )

    source = load_source_dataset(source_data)
    write_dataset_yaml(
        dataset_yaml.resolve(),
        train_spec=output,
        val_spec=source.images_val,
        names=source.names,
    )
    summary = {
        "inputs": [str(path.resolve()) for path in inputs],
        "deduplicated_images": len(deduplicated),
        "output_list": str(output),
        "dataset_yaml": str(dataset_yaml.resolve()),
    }
    (dataset_yaml.resolve().parent / "combine_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
