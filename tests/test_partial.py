from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from regionssod.data.dataset import load_source_dataset
from regionssod.data.partial import build_partial_dataset


def _make_source_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    image = np.full((100, 200, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(root / "images" / "train" / "sample.jpg"), image)
    cv2.imwrite(str(root / "images" / "val" / "sample.jpg"), image)
    (root / "labels" / "train" / "sample.txt").write_text(
        "3 0.50 0.50 0.10 0.10\n", encoding="utf-8"
    )
    (root / "labels" / "val" / "sample.txt").write_text(
        "3 0.50 0.50 0.10 0.10\n", encoding="utf-8"
    )

    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "person", 3: "car"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def _read_manifest(output: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def test_build_partial_dataset_writes_trainable_layout(tmp_path: Path) -> None:
    source_yaml = _make_source_dataset(tmp_path / "source")
    source = load_source_dataset(source_yaml)
    output = tmp_path / "partial20"

    summary = build_partial_dataset(
        source,
        output,
        area_ratio=0.2,
        seed=7,
        unlabeled_per_image=1,
    )

    assert summary["labeled_tiles"] == 1
    assert summary["unlabeled_tiles"] == 1
    assert (output / "dataset.yaml").is_file()
    assert (output / "splits" / "labeled_train.txt").is_file()
    assert (output / "splits" / "unlabeled_train.txt").is_file()
    labeled_image = Path(
        (output / "splits" / "labeled_train.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert labeled_image.is_file()
    label_path = (
        output / "labeled" / "labels" / "train" / f"{labeled_image.stem}.txt"
    )
    assert label_path.is_file()
    assert label_path.read_text(encoding="utf-8").startswith("3 ")
    manifest = _read_manifest(output)
    assert manifest[0]["labeled_tile"]["labels"] == 1


def test_build_partial_dataset_is_seed_deterministic(tmp_path: Path) -> None:
    source_yaml = _make_source_dataset(tmp_path / "source")
    source = load_source_dataset(source_yaml)
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_partial_dataset(source, first, 0.2, seed=11, unlabeled_per_image=1)
    build_partial_dataset(source, second, 0.2, seed=11, unlabeled_per_image=1)

    first_record = _read_manifest(first)[0]
    second_record = _read_manifest(second)[0]
    assert first_record["labeled_tile"]["tile"] == second_record["labeled_tile"]["tile"]
    assert [
        (record["tile"], record["visible_gt_boxes_hidden_from_training"])
        for record in first_record["unlabeled_tiles"]
    ] == [
        (record["tile"], record["visible_gt_boxes_hidden_from_training"])
        for record in second_record["unlabeled_tiles"]
    ]
