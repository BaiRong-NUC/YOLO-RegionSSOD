from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from regionssod.data.annotations import (
    Tile,
    YoloBox,
    clip_box_to_tile,
    iter_images,
    label_path_for_image,
    read_label_file,
    write_label_file,
)
from regionssod.data.dataset import SourceDataset, write_dataset_yaml


def _tile_from_center(
    center_x: float,
    center_y: float,
    side: int,
    image_width: int,
    image_height: int,
) -> Tile:
    side = min(side, image_width, image_height)
    if side <= 0:
        raise ValueError("tile side must be positive")

    x1 = int(round(center_x - side / 2.0))
    y1 = int(round(center_y - side / 2.0))
    x1 = min(max(x1, 0), image_width - side)
    y1 = min(max(y1, 0), image_height - side)
    return Tile(x1=x1, y1=y1, x2=x1 + side, y2=y1 + side)


def _tile_iou(left: Tile, right: Tile) -> float:
    intersection_width = max(
        0, min(left.x2, right.x2) - max(left.x1, right.x1)
    )
    intersection_height = max(
        0, min(left.y2, right.y2) - max(left.y1, right.y1)
    )
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union


def _clip_labels(
    labels: list[YoloBox],
    tile: Tile,
    image_width: int,
    image_height: int,
    min_visibility: float,
) -> list[YoloBox]:
    clipped: list[YoloBox] = []
    for label in labels:
        candidate = clip_box_to_tile(
            label,
            tile,
            image_width=image_width,
            image_height=image_height,
            min_visibility=min_visibility,
        )
        if candidate is not None:
            clipped.append(candidate)
    return clipped


def _sample_unlabeled_tile(
    image_width: int,
    image_height: int,
    side: int,
    labeled_tile: Tile,
    rng: np.random.Generator,
) -> Tile:
    max_x = max(0, image_width - side)
    max_y = max(0, image_height - side)
    best = Tile(x1=0, y1=0, x2=side, y2=side)
    best_iou = 1.0
    for _ in range(50):
        x1 = int(rng.integers(0, max_x + 1))
        y1 = int(rng.integers(0, max_y + 1))
        candidate = Tile(x1=x1, y1=y1, x2=x1 + side, y2=y1 + side)
        overlap = _tile_iou(candidate, labeled_tile)
        if overlap < best_iou:
            best = candidate
            best_iou = overlap
        if overlap <= 0.2:
            return candidate
    return best


def _choose_labeled_tile(
    labels: list[YoloBox],
    image_width: int,
    image_height: int,
    side: int,
    jitter_fraction: float,
    min_visibility: float,
    rng: np.random.Generator,
) -> tuple[Tile, list[YoloBox]]:
    if labels:
        for _ in range(12):
            anchor = labels[int(rng.integers(0, len(labels)))]
            center_x = anchor.x * image_width
            center_y = anchor.y * image_height
            center_x += rng.uniform(-jitter_fraction, jitter_fraction) * side
            center_y += rng.uniform(-jitter_fraction, jitter_fraction) * side
            tile = _tile_from_center(
                center_x, center_y, side, image_width, image_height
            )
            clipped = _clip_labels(
                labels,
                tile,
                image_width,
                image_height,
                min_visibility,
            )
            if clipped:
                return tile, clipped

        anchor = labels[0]
        tile = _tile_from_center(
            anchor.x * image_width,
            anchor.y * image_height,
            side,
            image_width,
            image_height,
        )
        return tile, _clip_labels(
            labels,
            tile,
            image_width,
            image_height,
            min_visibility,
        )

    center_x = float(rng.uniform(0, image_width))
    center_y = float(rng.uniform(0, image_height))
    tile = _tile_from_center(center_x, center_y, side, image_width, image_height)
    return tile, []


def _prepare_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"output directory is not empty; choose a new path: {output}"
            )
    output.mkdir(parents=True, exist_ok=True)


def build_partial_dataset(
    source: SourceDataset,
    output_dir: Path,
    area_ratio: float,
    *,
    seed: int = 0,
    unlabeled_per_image: int = 1,
    min_box_visibility: float = 0.35,
    jitter_fraction: float = 0.2,
    limit: int | None = None,
) -> dict[str, object]:
    if not 0 < area_ratio <= 1:
        raise ValueError("area_ratio must be in (0, 1]")
    if unlabeled_per_image < 0:
        raise ValueError("unlabeled_per_image must be non-negative")
    if not 0 <= min_box_visibility <= 1:
        raise ValueError("min_box_visibility must be in [0, 1]")
    if not 0 <= jitter_fraction <= 1:
        raise ValueError("jitter_fraction must be in [0, 1]")

    output_dir = output_dir.resolve()
    _prepare_output(output_dir)
    rng = np.random.default_rng(seed)

    labeled_images_dir = output_dir / "labeled" / "images" / "train"
    labeled_labels_dir = output_dir / "labeled" / "labels" / "train"
    unlabeled_images_dir = output_dir / "unlabeled" / "images" / "train"
    unlabeled_labels_dir = output_dir / "unlabeled" / "labels" / "train"
    splits_dir = output_dir / "splits"
    for directory in (
        labeled_images_dir,
        labeled_labels_dir,
        unlabeled_images_dir,
        unlabeled_labels_dir,
        splits_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    all_images = iter_images(source.images_train)
    if limit is not None:
        all_images = all_images[:limit]
    if not all_images:
        raise ValueError(f"no images found under {source.images_train}")

    labeled_list: list[str] = []
    unlabeled_list: list[str] = []
    manifest_path = output_dir / "manifest.jsonl"
    labeled_boxes = 0
    hidden_boxes = 0
    skipped: list[dict[str, str]] = []

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for image_path in all_images:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                skipped.append({"image": str(image_path), "reason": "decode_failed"})
                continue

            image_height, image_width = image.shape[:2]
            side = int(round(math.sqrt(area_ratio * image_width * image_height)))
            side = max(64, min(side, image_width, image_height))
            source_labels = read_label_file(
                label_path_for_image(
                    image_path, source.images_train, source.labels_train
                )
            )
            labeled_tile, tile_labels = _choose_labeled_tile(
                labels=source_labels,
                image_width=image_width,
                image_height=image_height,
                side=side,
                jitter_fraction=jitter_fraction,
                min_visibility=min_box_visibility,
                rng=rng,
            )

            safe_stem = image_path.stem.replace(" ", "_")
            labeled_name = (
                f"{safe_stem}__lab_{labeled_tile.x1}_{labeled_tile.y1}_"
                f"{labeled_tile.width}.jpg"
            )
            labeled_image_path = labeled_images_dir / labeled_name
            labeled_label_path = labeled_labels_dir / f"{Path(labeled_name).stem}.txt"
            labeled_crop = image[
                labeled_tile.y1 : labeled_tile.y2,
                labeled_tile.x1 : labeled_tile.x2,
            ]
            if not cv2.imwrite(
                str(labeled_image_path),
                labeled_crop,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise OSError(f"failed to write {labeled_image_path}")
            write_label_file(labeled_label_path, tile_labels)
            labeled_list.append(str(labeled_image_path))
            labeled_boxes += len(tile_labels)

            unlabeled_records: list[dict[str, object]] = []
            for unlabeled_index in range(unlabeled_per_image):
                unlabeled_tile = _sample_unlabeled_tile(
                    image_width=image_width,
                    image_height=image_height,
                    side=side,
                    labeled_tile=labeled_tile,
                    rng=rng,
                )
                unlabeled_name = (
                    f"{safe_stem}__unlab_{unlabeled_index}_"
                    f"{unlabeled_tile.x1}_{unlabeled_tile.y1}_{unlabeled_tile.width}.jpg"
                )
                unlabeled_image_path = unlabeled_images_dir / unlabeled_name
                unlabeled_crop = image[
                    unlabeled_tile.y1 : unlabeled_tile.y2,
                    unlabeled_tile.x1 : unlabeled_tile.x2,
                ]
                if not cv2.imwrite(
                    str(unlabeled_image_path),
                    unlabeled_crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise OSError(f"failed to write {unlabeled_image_path}")
                unlabeled_list.append(str(unlabeled_image_path))

                visible_hidden = _clip_labels(
                    source_labels,
                    unlabeled_tile,
                    image_width,
                    image_height,
                    min_visibility=min_box_visibility,
                )
                hidden_boxes += len(visible_hidden)
                unlabeled_records.append(
                    {
                        "image": str(unlabeled_image_path),
                        "tile": [
                            unlabeled_tile.x1,
                            unlabeled_tile.y1,
                            unlabeled_tile.x2,
                            unlabeled_tile.y2,
                        ],
                        "visible_gt_boxes_hidden_from_training": len(visible_hidden),
                    }
                )

            manifest_file.write(
                json.dumps(
                    {
                        "source_image": str(image_path),
                        "source_size": [image_width, image_height],
                        "requested_area_ratio": area_ratio,
                        "actual_area_ratio": (side * side)
                        / (image_width * image_height),
                        "labeled_tile": {
                            "image": str(labeled_image_path),
                            "tile": [
                                labeled_tile.x1,
                                labeled_tile.y1,
                                labeled_tile.x2,
                                labeled_tile.y2,
                            ],
                            "labels": len(tile_labels),
                            "source_gt_boxes": len(source_labels),
                        },
                        "unlabeled_tiles": unlabeled_records,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    labeled_list_path = splits_dir / "labeled_train.txt"
    unlabeled_list_path = splits_dir / "unlabeled_train.txt"
    val_list_path = splits_dir / "val.txt"
    labeled_list_path.write_text("\n".join(labeled_list) + "\n", encoding="utf-8")
    unlabeled_list_path.write_text(
        "\n".join(unlabeled_list) + ("\n" if unlabeled_list else ""),
        encoding="utf-8",
    )
    val_images = iter_images(source.images_val)
    val_list_path.write_text(
        "\n".join(str(path) for path in val_images) + "\n", encoding="utf-8"
    )

    write_dataset_yaml(
        output_dir / "dataset.yaml",
        train_spec=labeled_list_path,
        val_spec=source.images_val,
        names=source.names,
    )
    (output_dir / "source.json").write_text(
        json.dumps(
            {
                "source_yaml": str(source.yaml_path),
                "images_train": str(source.images_train),
                "labels_train": str(source.labels_train),
                "images_val": str(source.images_val),
                "names": source.names,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary: dict[str, object] = {
        "source_yaml": str(source.yaml_path),
        "output_dir": str(output_dir),
        "area_ratio": area_ratio,
        "seed": seed,
        "unlabeled_per_image": unlabeled_per_image,
        "min_box_visibility": min_box_visibility,
        "jitter_fraction": jitter_fraction,
        "images_discovered": len(all_images),
        "images_used": len(labeled_list),
        "images_skipped": skipped,
        "labeled_tiles": len(labeled_list),
        "unlabeled_tiles": len(unlabeled_list),
        "labeled_boxes": labeled_boxes,
        "visible_gt_boxes_in_unlabeled_tiles": hidden_boxes,
        "train_list": str(labeled_list_path),
        "unlabeled_list": str(unlabeled_list_path),
        "val_list": str(val_list_path),
        "dataset_yaml": str(output_dir / "dataset.yaml"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
