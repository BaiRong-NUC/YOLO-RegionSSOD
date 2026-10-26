from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from regionssod.pseudo.generator import (
    combine_image_lists,
    filter_predictions,
    generate_pseudo_labels,
)


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


@pytest.mark.parametrize("limit, expected_batches", [(None, [2, 2, 1]), (3, [2, 1])])
def test_pseudo_generation_respects_inference_batch_and_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: int | None,
    expected_batches: list[int],
) -> None:
    import torch
    import ultralytics
    from ultralytics.data.build import load_inference_source
    from ultralytics.engine.results import Results

    images = []
    for index in range(5):
        path = tmp_path / f"image_{index}.jpg"
        assert cv2.imwrite(str(path), np.full((24, 24, 3), 127, dtype=np.uint8))
        images.append(path)
    images_list = tmp_path / "images.txt"
    images_list.write_text("\n".join(map(str, images)) + "\n", encoding="utf-8")
    weights = tmp_path / "teacher.pt"
    weights.write_bytes(b"unused by the fake model")
    observed_batches = []

    class FakeYOLO:
        def __init__(self, model: str) -> None:
            assert Path(model) == weights

        def predict(self, *, source, batch, **kwargs):
            # Use the installed Ultralytics loader so this catches its treatment
            # of a Python list as one batch, even when batch=2 is requested.
            for paths, frames, _ in load_inference_source(source, batch=batch):
                observed_batches.append(len(paths))
                assert len(paths) <= batch
                for path, frame in zip(paths, frames, strict=True):
                    yield Results(
                        orig_img=frame,
                        path=path,
                        names={0: "person"},
                        boxes=torch.tensor([[4, 4, 16, 16, 0.9, 0]]),
                    )

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    output = tmp_path / "pseudo"
    summary = generate_pseudo_labels(
        weights, images_list, output, batch=2, limit=limit, device="cpu"
    )

    expected_count = sum(expected_batches)
    assert observed_batches == expected_batches
    assert summary["processed_images"] == expected_count
    assert summary["accepted_images"] == expected_count
    assert summary["accepted_boxes"] == expected_count
    assert summary["empty_predictions"] == 0
    assert summary["per_class"] == {"0": expected_count}
    accepted_images = (output / "pseudo_train.txt").read_text(encoding="utf-8").splitlines()
    assert len(accepted_images) == expected_count
    assert all(Path(path).is_file() for path in accepted_images)
    assert len(list((output / "labels/train").glob("*.txt"))) == expected_count
