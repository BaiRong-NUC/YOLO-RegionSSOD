from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss

from regionssod.pseudo.scale import ScalePolicy, generate_scale_labels
from regionssod.training.p1 import P1Dataset, P1DetectionLoss, classification_factors


def test_scale_boundaries_native_pixels_and_uncertain_partition():
    policy = ScalePolicy()
    items = policy.select([0] * 6, [.30, .30, .49, .50, .05, .049],
                          [[.5, .5, .031, .032], [.5, .5, .032, .032],
                           [.5, .5, .096, .096], [.5, .5, .096, .096],
                           [.5, .5, .01, .01], [.5, .5, .01, .01]], (1000, 1000))
    assert [(p["scale"], p["state"]) for p in items] == [
        ("small", "positive"), ("medium", "ignore"), ("large", "ignore"),
        ("large", "positive"), ("small", "ignore")]
    assert ScalePolicy().select([0], [.35], [[.5, .5, .1, .1]], (200, 200))[0]["state"] == "positive"
    assert ScalePolicy().select([0], [.35], [[.5, .5, .1, .1]], (400, 400))[0]["state"] == "ignore"


@pytest.mark.parametrize("kwargs", [{"small_threshold": .01}, {"large_area": 10},
                                    {"conf_floor": 0}, {"medium_threshold": float("nan")}])
def test_policy_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ScalePolicy(**kwargs)


def test_ignore_and_soft_weight_gradients_positive_precedence_and_batch_isolation():
    anchors = torch.tensor([[10., 10.], [30., 30.], [50., 50.], [90., 90.]])
    fg = torch.tensor([[True, False, False, False], [False, True, False, False]])
    gt_idx = torch.zeros((2, 4), dtype=torch.long)
    batch = {"pseudo_weights": torch.tensor([.4, 1.]), "batch_idx": torch.tensor([0, 1]),
             "ignore_bboxes": torch.tensor([[.3, .3, .6, .6]]),
             "ignore_batch_idx": torch.tensor([0])}
    factors = classification_factors(anchors, fg, gt_idx, batch, 2, "cpu", torch.tensor([100, 100]))
    assert torch.allclose(factors[..., 0], torch.tensor([[.4, 0, 0, 1], [1, 1, 1, 1]]))
    logits = torch.zeros((2, 4, 2), requires_grad=True)
    (torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none") * factors).sum().backward()
    assert torch.allclose(logits.grad[0, :, 0], torch.tensor([.2, 0., 0., .5]))
    assert (logits.grad[1] == .5).all()
    ablated = classification_factors(anchors, fg, gt_idx, batch, 2, "cpu", torch.tensor([100, 100]),
                                     soft_weights=False, uncertain_ignore=False)
    assert (ablated == 1).all()


def loss_fixture():
    model = torch.nn.Module()
    model.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1)))
    model.model = [SimpleNamespace(stride=torch.tensor([8.]), nc=2, reg_max=4)]
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    generator = torch.Generator().manual_seed(12)
    preds = {"boxes": torch.randn(2, 16, 64, generator=generator, requires_grad=True),
             "scores": torch.randn(2, 2, 64, generator=generator, requires_grad=True),
             "feats": [torch.zeros(2, 1, 8, 8)]}
    batch = {"batch_idx": torch.tensor([0., 1.]), "cls": torch.tensor([[0.], [1.]]),
             "bboxes": torch.tensor([[.5, .5, .5, .5], [.5, .5, .5, .5]])}
    return model, preds, batch


@pytest.mark.parametrize("empty", [False, True])
def test_p1_loss_matches_ultralytics_with_unit_weights_and_no_ignore(empty):
    model, preds, batch = loss_fixture()
    if empty:
        batch = {k: v[:0] for k, v in batch.items()}
    batch["pseudo_weights"] = torch.ones(len(batch["cls"]))
    original, _ = v8DetectionLoss(model)(preds, batch)
    modified, _ = P1DetectionLoss(model)(preds, batch)
    assert torch.equal(original, modified)
    modified.sum().backward()
    assert all(torch.isfinite(preds[key].grad).all() for key in ("boxes", "scores"))


def test_full_loss_confidence_changes_only_classification_and_preserves_regression():
    model, preds, batch = loss_fixture()
    criterion = P1DetectionLoss(model)
    ordinary, _ = criterion(preds, batch)
    batch["pseudo_weights"] = torch.tensor([.3, .3])
    batch["ignore_bboxes"] = torch.tensor([[.5, .5, 1., 1.]])
    batch["ignore_batch_idx"] = torch.tensor([0])
    modified, _ = criterion(preds, batch)
    assert torch.equal(ordinary[[0, 2]], modified[[0, 2]])
    assert 0 < modified[1] < ordinary[1]
    modified.sum().backward()
    assert torch.isfinite(preds["scores"].grad).all()


def make_p1_dataset(tmp_path, *, augment=False, corrupt=False):
    paths, rows = [], []
    # Distinct confidence/class pairs expose cross-image metadata mixups in mosaic.
    for index in range(4):
        image = tmp_path / "pseudo" / "images" / "train" / f"{index}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image), np.full((100, 100, 3), 128, dtype=np.uint8))
        label = tmp_path / "pseudo" / "labels" / "train" / f"{index}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(f"{index} 0.2 0.4 0.2 0.2\n", encoding="utf-8")
        positive = {"class_id": index, "confidence": .3 + index * .1, "xywhn": [.2, .4, .2, .2]}
        ignore = {"class_id": index, "confidence": .1, "xywhn": [.8, .6, .2, .2]}
        rows.append({"image": str(image), "positive": [positive], "ignore": [ignore]})
        paths.append(str(image))
    image = tmp_path / "labeled" / "images" / "train" / "human.jpg"
    image.parent.mkdir(parents=True)
    assert cv2.imwrite(str(image), np.full((100, 100, 3), 128, dtype=np.uint8))
    label = tmp_path / "labeled" / "labels" / "train" / "human.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.3 0.3 0.1 0.1\n", encoding="utf-8")
    paths.append(str(image))
    image_list = tmp_path / "images.txt"
    image_list.write_text("\n".join(paths) + "\n", encoding="utf-8")
    if corrupt:
        rows[0]["positive"][0]["xywhn"][0] = .9
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    hyp = get_cfg(overrides={"mosaic": 1. if augment else 0., "fliplr": 1., "scale": .5 if augment else 0.,
                              "translate": .1 if augment else 0., "mixup": 0., "cutmix": 0.})
    dataset = P1Dataset(img_path=str(image_list), metadata_path=metadata, imgsz=128, batch_size=2,
                        augment=augment, hyp=hyp, data={"names": {i: str(i) for i in range(4)}, "channels": 3})
    return dataset, hyp


def test_dataset_transports_ignore_geometry_and_confidence_and_collates(tmp_path):
    dataset, hyp = make_p1_dataset(tmp_path)
    pseudo_index = next(i for i, file in enumerate(dataset.im_files) if Path(file).stem == "0")
    second_index = next(i for i, file in enumerate(dataset.im_files) if Path(file).stem == "1")
    human_index = next(i for i, file in enumerate(dataset.im_files) if Path(file).stem == "human")
    first = dataset[pseudo_index]
    assert torch.allclose(first["bboxes"], torch.tensor([[.2, .4, .2, .2]]))
    assert torch.allclose(first["ignore_bboxes"], torch.tensor([[.8, .6, .2, .2]]))
    assert first["pseudo_weights"].item() == pytest.approx(.3)
    human = dataset[human_index]
    assert human["pseudo_weights"].tolist() == [1.]
    assert human["ignore_bboxes"].shape == (0, 4)
    collated = dataset.collate_fn([first, dataset[second_index], human])
    assert collated["ignore_batch_idx"].tolist() == [0, 1]
    assert collated["batch_idx"].tolist() == [0, 1, 2]
    # Force a horizontal flip through the actual Ultralytics pipeline.
    dataset.augment = True
    dataset.transforms = dataset.build_transforms(hyp)
    flipped = dataset[pseudo_index]
    assert flipped["bboxes"][0, 0].item() == pytest.approx(.8)
    assert flipped["ignore_bboxes"][0, 0].item() == pytest.approx(.2)


def test_mosaic_preserves_class_confidence_pairing_and_close_mosaic(tmp_path):
    dataset, hyp = make_p1_dataset(tmp_path, augment=True)
    for index in range(12):
        sample = dataset[index % len(dataset)]
        for cls, weight in zip(sample["cls"].view(-1), sample["pseudo_weights"], strict=True):
            assert weight.item() == pytest.approx(.3 + cls.item() * .1) or weight.item() == 1.
        assert len(sample["ignore_bboxes"]) == len(sample["ignore_batch_idx"])
        assert torch.isfinite(sample["ignore_bboxes"]).all()
    hyp.scale = hyp.translate = 0.
    dataset.close_mosaic(deepcopy(hyp))
    pseudo_index = next(i for i, file in enumerate(dataset.im_files) if Path(file).stem == "0")
    assert dataset[pseudo_index]["pseudo_weights"].item() == pytest.approx(.3)


def test_dataset_rejects_sidecar_label_mismatch(tmp_path):
    with pytest.raises(ValueError, match="metadata/positive label mismatch"):
        make_p1_dataset(tmp_path, corrupt=True)


def test_p1_generation_retains_candidates_ignores_empty_tiles_and_batches(tmp_path, monkeypatch):
    import ultralytics
    from ultralytics.data.build import load_inference_source
    from ultralytics.engine.results import Results

    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.jpg"
        assert cv2.imwrite(str(path), np.full((100, 100, 3), 128, dtype=np.uint8))
        paths.append(str(path))
    image_list = tmp_path / "images.txt"
    image_list.write_text("\n".join(paths), encoding="utf-8")
    weights = tmp_path / "teacher.pt"
    weights.write_bytes(b"fake teacher")
    observed = []

    class FakeYOLO:
        def __init__(self, path):
            pass

        def predict(self, *, source, batch, **kwargs):
            for names, images, _ in load_inference_source(source, batch=batch):
                observed.append(len(images))
                for name, image in zip(names, images, strict=True):
                    positive_conf = .4 if Path(name).stem != "2" else .2
                    yield Results(orig_img=image, path=name, names={0: "person"},
                                  boxes=torch.tensor([[10, 10, 30, 30, positive_conf, 0],
                                                      [60, 60, 80, 80, .1, 0]]))

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    policy = Path(__file__).resolve().parents[1] / "configs/pseudo_p1_10.yaml"
    output = tmp_path / "p1"
    summary = generate_scale_labels(weights, image_list, output, policy_path=policy, batch=2, device="cpu")
    assert observed == [2, 1]
    assert summary["accepted_images"] == 2
    assert summary["accepted_boxes"] == summary["ignore_boxes"] == 2
    assert summary["excluded_images_without_positives"] == 1
    rows = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()]
    assert rows[0]["positive"][0]["confidence"] == pytest.approx(.4)
    assert len(rows[0]["ignore"]) == 1
    assert len((output / "candidates.jsonl").read_text().splitlines()) == 3
    assert all(len(line.split()) == 5 for file in (output / "labels/train").glob("*.txt") for line in file.read_text().splitlines())
