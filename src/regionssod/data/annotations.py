from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class YoloBox:
    class_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True, slots=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def iter_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_label_file(path: Path) -> list[YoloBox]:
    if not path.exists():
        return []

    boxes: list[YoloBox] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 columns, got {len(parts)}")

        class_id = int(parts[0])
        x, y, w, h = (float(value) for value in parts[1:])
        if class_id < 0:
            raise ValueError(f"{path}:{line_number}: negative class id")
        if not all(math.isfinite(value) for value in (x, y, w, h)):
            raise ValueError(f"{path}:{line_number}: non-finite coordinate")
        if w <= 0 or h <= 0:
            raise ValueError(f"{path}:{line_number}: non-positive box size")
        if not all(0.0 <= value <= 1.0 for value in (x, y, w, h)):
            raise ValueError(f"{path}:{line_number}: normalized box outside [0, 1]")
        boxes.append(YoloBox(class_id=class_id, x=x, y=y, w=w, h=h))
    return boxes


def write_label_file(path: Path, boxes: list[YoloBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{box.class_id} {box.x:.10g} {box.y:.10g} {box.w:.10g} {box.h:.10g}"
        for box in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def label_path_for_image(
    image_path: Path, images_root: Path, labels_root: Path
) -> Path:
    relative = image_path.relative_to(images_root).with_suffix(".txt")
    return labels_root / relative


def replace_images_component(path: Path) -> Path:
    parts = list(path.parts)
    image_indices = [
        index for index, part in enumerate(parts) if part.lower() == "images"
    ]
    if not image_indices:
        raise ValueError(f"cannot infer labels path from {path}")
    parts[image_indices[-1]] = "labels"
    return Path(*parts)


def clip_box_to_tile(
    box: YoloBox,
    tile: Tile,
    image_width: int,
    image_height: int,
    min_visibility: float = 0.35,
) -> YoloBox | None:
    original_width = box.w * image_width
    original_height = box.h * image_height
    original_area = original_width * original_height
    if original_area <= 0:
        return None

    x1 = (box.x - box.w / 2.0) * image_width
    y1 = (box.y - box.h / 2.0) * image_height
    x2 = (box.x + box.w / 2.0) * image_width
    y2 = (box.y + box.h / 2.0) * image_height

    clipped_x1 = max(x1, float(tile.x1))
    clipped_y1 = max(y1, float(tile.y1))
    clipped_x2 = min(x2, float(tile.x2))
    clipped_y2 = min(y2, float(tile.y2))
    clipped_width = clipped_x2 - clipped_x1
    clipped_height = clipped_y2 - clipped_y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    visibility = (clipped_width * clipped_height) / original_area
    if visibility + 1e-12 < min_visibility:
        return None

    center_x = (clipped_x1 + clipped_x2) / 2.0 - tile.x1
    center_y = (clipped_y1 + clipped_y2) / 2.0 - tile.y1
    normalized_w = clipped_width / tile.width
    normalized_h = clipped_height / tile.height
    return YoloBox(
        class_id=box.class_id,
        x=min(max(center_x / tile.width, 0.0), 1.0),
        y=min(max(center_y / tile.height, 0.0), 1.0),
        w=min(max(normalized_w, 0.0), 1.0),
        h=min(max(normalized_h, 0.0), 1.0),
    )
