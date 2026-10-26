from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from regionssod.pseudo.generator import combine_image_lists, filter_predictions


def test_filter_predictions_applies_global_threshold() -> None:
    accepted = filter_predictions(
        classes=[0, 1, 2],
        confidences=[0.9, 0.2, 0.7],
        xywhn=np.asarray(
            [
                [0.2, 0.2, 0.1, 0.1],
                [0.5, 0.5, 0.2, 0.2],
                [0.8, 0.8, 0.05, 0.05],
            ]
        ),
        threshold=0.5,
    )

    assert [item[0] for item in accepted] == [0, 2]
    assert accepted[0][1] == 0.9


def test_combine_image_lists_deduplicates_and_writes_yaml(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    list_a = tmp_path / "a.txt"
    list_b = tmp_path / "b.txt"
    list_a.write_text(f"{image_a}\n{image_b}\n", encoding="utf-8")
    list_b.write_text(f"{image_b}\n", encoding="utf-8")

    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    source_yaml = source / "dataset.yaml"
    source_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(source),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output_list = tmp_path / "combined.txt"
    output_yaml = tmp_path / "combined.yaml"
    summary = combine_image_lists(
        inputs=[list_a, list_b],
        output=output_list,
        dataset_yaml=output_yaml,
        source_data=source_yaml,
    )

    assert summary["deduplicated_images"] == 2
    assert len(output_list.read_text(encoding="utf-8").splitlines()) == 2
    data = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
    assert data["names"] == {0: "person"}
    assert data["train"] == str(output_list.resolve())
