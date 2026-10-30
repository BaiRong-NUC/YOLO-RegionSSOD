from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from regionssod.data.annotations import replace_images_component


@dataclass(frozen=True, slots=True)
class SourceDataset:
    yaml_path: Path
    base: Path
    images_train: Path
    labels_train: Path
    images_val: Path
    labels_val: Path
    names: dict[int, str]


def _resolve_split_path(base: Path, spec: Any, key: str) -> Path:
    if not isinstance(spec, str):
        raise ValueError(f"{key} must be a path string for this repository")
    path = Path(spec)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _normalize_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(index): str(name) for index, name in raw_names.items()}
    raise ValueError("dataset names must be a list or mapping")


def load_source_dataset(yaml_path: Path) -> SourceDataset:
    yaml_path = yaml_path.resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{yaml_path}: dataset YAML must contain a mapping")

    raw_base = data.get("path")
    if raw_base is None:
        base = yaml_path.parent
    else:
        base = Path(str(raw_base))
        if not base.is_absolute():
            base = yaml_path.parent / base
        base = base.resolve()

    images_train = _resolve_split_path(base, data.get("train"), "train")
    images_val = _resolve_split_path(base, data.get("val"), "val")
    labels_train = replace_images_component(images_train)
    labels_val = replace_images_component(images_val)
    names = _normalize_names(data.get("names"))

    if not images_train.is_dir():
        raise FileNotFoundError(f"train image directory not found: {images_train}")
    if not labels_train.is_dir():
        raise FileNotFoundError(f"train label directory not found: {labels_train}")
    if not images_val.is_dir():
        raise FileNotFoundError(f"val image directory not found: {images_val}")
    if not labels_val.is_dir():
        raise FileNotFoundError(f"val label directory not found: {labels_val}")

    return SourceDataset(
        yaml_path=yaml_path,
        base=base,
        images_train=images_train,
        labels_train=labels_train,
        images_val=images_val,
        labels_val=labels_val,
        names=names,
    )


def write_dataset_yaml(
    output_path: Path,
    train_spec: Path | str,
    val_spec: Path | str,
    names: dict[int, str],
) -> None:
    payload = {
        "path": str(output_path.parent.resolve()),
        "train": str(train_spec),
        "val": str(val_spec),
        "names": {index: names[index] for index in sorted(names)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
