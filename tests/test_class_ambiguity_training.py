from copy import deepcopy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg

from regionssod.training.class_ambiguity import (
    ClassAmbiguityDataset, ClassAmbiguityLoss, ClassAmbiguityModel, ClassAmbiguityTrainer,
    EpochClassificationDiagnostics, run_class_ambiguity_training,
)
from regionssod.training.p1 import P1Dataset, P1DetectionLoss, P1Trainer
from test_p1 import loss_fixture


def augmented_batch(batch, *, onehot=False, confidence=.4):
    scores = torch.tensor([[confidence, 0. if onehot else confidence * .9], [0., 0.]])
    return batch | {
        "teacher_class_scores": scores,
        "is_pseudo": torch.tensor([True, False]),
        "pseudo_weights": torch.tensor([confidence, 1.]),
        "ignore_bboxes": torch.tensor([[.1, .1, .2, .2]]),
        "ignore_batch_idx": torch.tensor([0.]),
    }


def losses_and_gradients(criterion, preds, batch):
    assigned, losses, _ = criterion.get_assigned_targets_and_loss(preds, batch)
    gradients = torch.autograd.grad(losses.sum(), (preds["boxes"], preds["scores"]))
    return assigned, losses.detach(), gradients


@pytest.mark.parametrize("mode", ["disabled", "onehot"])
def test_original_loss_and_gradients_exactly_recovered(mode):
    model, preds, batch = loss_fixture()
    batch = augmented_batch(batch, onehot=True)
    if mode == "disabled":
        batch.pop("teacher_class_scores")
        batch.pop("is_pseudo")
    before = losses_and_gradients(P1DetectionLoss(model), preds, batch)
    criterion = ClassAmbiguityLoss(model, enabled=mode != "disabled")
    after = losses_and_gradients(criterion, preds, batch)
    for left, right in zip(before[0], after[0], strict=True):
        assert torch.equal(left, right)
    assert torch.equal(before[1], after[1])
    assert all(torch.equal(a, b) for a, b in zip(before[2], after[2], strict=True))


@pytest.mark.parametrize("confidence", [.4, 1.])
def test_nononehot_only_changes_pseudo_classification_and_preserves_mass(confidence):
    model, preds, batch = loss_fixture()
    batch = augmented_batch(batch, confidence=confidence)
    before = losses_and_gradients(P1DetectionLoss(model), preds, batch)
    criterion = ClassAmbiguityLoss(model, diagnostics=True)
    after = losses_and_gradients(criterion, preds, batch)
    assert all(torch.equal(a, b) for a, b in zip(before[0], after[0], strict=True))
    assert torch.equal(before[1][[0, 2]], after[1][[0, 2]])
    assert torch.equal(before[2][0], after[2][0])
    assert not torch.equal(before[2][1][0], after[2][1][0])
    assert torch.equal(before[2][1][1], after[2][1][1])
    fg = after[0][0]
    assert fg[0].any()
    assert torch.equal(before[2][1][0, :, ~fg[0]], after[2][1][0, :, ~fg[0]])
    diag = criterion.last_diagnostics
    pseudo = diag["pseudo_positive"]
    assert pseudo["anchors"] > 0
    assert diag["human_positive"]["anchors"] > 0
    for name in ("all", "pseudo_positive", "human_positive", "active_negative", "ignored_negative"):
        row = diag[name]
        assert torch.allclose(row["old_mass"], row["new_mass"], atol=1e-6)
        assert torch.allclose(row["old_weighted_mass"], row["new_weighted_mass"], atol=1e-6)
        assert torch.allclose(row["old_gradient_sum"], row["new_gradient_sum"], atol=1e-5)
        assert all(not value.requires_grad for value in row.values())
    assert pseudo["new_weighted_mass"] == pytest.approx(float(pseudo["new_mass"]) * confidence)
    assert diag["ignored_negative"]["new_bce"] == 0
    assert diag["ignored_negative"]["new_gradient_l1"] == 0
    actual = after[2][1] * preds["scores"].shape[0]
    assert torch.allclose(diag["all"]["new_gradient_l1"], actual.abs().sum(), atol=1e-5)
    assert torch.allclose(diag["all"]["new_gradient_l2"], actual.square().sum().sqrt(), atol=1e-5)


def test_analytic_bce_gradient_redistributes_mass_without_extra_confidence_factor():
    model, preds, batch = loss_fixture()
    batch = augmented_batch(batch, confidence=.4)
    old = P1DetectionLoss(model)
    new = ClassAmbiguityLoss(model)
    captured = {}
    original_assign = new.assigner.forward

    def capture(*args, **kwargs):
        result = original_assign(*args, **kwargs)
        captured["targets"] = result[2].detach().clone()
        return result

    new.assigner.forward = capture
    before = losses_and_gradients(old, preds, batch)
    after = losses_and_gradients(new, preds, batch)
    targets = captured["targets"]
    denominator = targets.sum().clamp(min=1)
    fg = after[0][0][0]
    mass = targets[0, fg].sum(-1)
    q = torch.tensor([.4, .36]) / .76
    expected = .4 * (targets[0, fg] - mass[:, None] * q) * model.args.cls / denominator
    actual = (after[2][1] - before[2][1])[0, :, fg].T
    assert torch.allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize("field", ["is_pseudo", "teacher_class_scores", "pseudo_weights", "ignore_bboxes"])
def test_training_rejects_missing_provenance(field):
    model, preds, batch = loss_fixture()
    batch = augmented_batch(batch)
    batch.pop(field)
    with pytest.raises(ValueError, match="provenance"):
        ClassAmbiguityLoss(model)(preds, batch)


@pytest.mark.parametrize("case", ["flag_dtype", "nan", "zero", "top1", "confidence", "human_weight"])
def test_training_rejects_invalid_provenance(case):
    model, preds, batch = loss_fixture()
    batch = augmented_batch(batch)
    if case == "flag_dtype":
        batch["is_pseudo"] = batch["is_pseudo"].long()
    elif case == "nan":
        batch["teacher_class_scores"][0, 1] = float("nan")
    elif case == "zero":
        batch["teacher_class_scores"].zero_()
    elif case == "top1":
        batch["teacher_class_scores"][0] = torch.tensor([.3, .4])
    elif case == "confidence":
        batch["pseudo_weights"][0] = .8
    else:
        batch["pseudo_weights"][1] = .8
    with pytest.raises(ValueError):
        ClassAmbiguityLoss(model)(preds, batch)


def test_empty_augmented_batch_and_nonuniform_weights_guard():
    model, preds, batch = loss_fixture()
    batch = {key: value[:0] for key, value in augmented_batch(batch).items()}
    before = losses_and_gradients(P1DetectionLoss(model), preds, batch)
    after = losses_and_gradients(ClassAmbiguityLoss(model), preds, batch)
    assert torch.equal(before[1], after[1])
    assert all(torch.equal(a, b) for a, b in zip(before[2], after[2], strict=True))
    model.class_weights = torch.tensor([1., 2.])
    with pytest.raises(ValueError, match="unit class weights"):
        ClassAmbiguityLoss(model)
    ClassAmbiguityLoss(model, enabled=False)


def write_receipt(metadata, sidecar):
    sidecar.with_name("receipt.json").write_text(json.dumps({
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }), encoding="utf-8")


def dataset_inputs(tmp_path):
    paths, metadata_rows, score_rows = [], [], []
    for index in range(4):
        image = tmp_path / "pseudo" / "images" / "train" / f"{index}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image), np.full((100, 100, 3), 128, dtype=np.uint8))
        label = tmp_path / "pseudo" / "labels" / "train" / f"{index}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        positives = []
        # Same class and confidence, different distributions: class/conf is not identity.
        for x in (.25, .65):
            positives.append({"class_id": index, "confidence": 1. if index == 0 else .6,
                              "xywhn": [x, .4, .2, .2]})
        label.write_text("\n".join(f"{index} {p['xywhn'][0]} 0.4 0.2 0.2" for p in positives),
                         encoding="utf-8")
        ignore = {"class_id": index, "confidence": .1, "xywhn": [.8, .8, .1, .1]}
        metadata_rows.append({"image": str(image), "positive": positives, "ignore": [ignore]})
        full = []
        for ordinal, p in enumerate(positives):
            scores = [0.] * 10
            scores[index] = p["confidence"]
            scores[(index + 1) % 10] = .1 + ordinal * .1
            full.append(p | {"teacher_class_scores": scores})
        score_rows.append({"image": str(image), "positive": full})
        paths.append(str(image))
    human = tmp_path / "labeled" / "images" / "train" / "human.jpg"
    human.parent.mkdir(parents=True)
    assert cv2.imwrite(str(human), np.full((100, 100, 3), 128, dtype=np.uint8))
    label = tmp_path / "labeled" / "labels" / "train" / "human.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.3 0.3 0.1 0.1\n", encoding="utf-8")
    paths.append(str(human))
    image_list = tmp_path / "images.txt"
    image_list.write_text("\n".join(paths), encoding="utf-8")
    metadata, sidecar = tmp_path / "metadata.jsonl", tmp_path / "scores.jsonl"
    metadata.write_text("\n".join(map(json.dumps, metadata_rows)), encoding="utf-8")
    sidecar.write_text("\n".join(map(json.dumps, score_rows)), encoding="utf-8")
    write_receipt(metadata, sidecar)
    return {
        "img_path": str(image_list), "metadata_path": metadata, "class_score_path": sidecar,
        "imgsz": 128, "batch_size": 2, "augment": True,
        "hyp": get_cfg(overrides={"mosaic": 0., "fliplr": 1., "flipud": 0., "scale": 0.,
                                 "translate": 0., "mixup": 0., "cutmix": 0.}),
        "data": {"names": {i: str(i) for i in range(10)}, "channels": 3},
    }


def test_dataset_flip_explicit_confidence_one_and_collation(tmp_path):
    dataset = ClassAmbiguityDataset(**dataset_inputs(tmp_path))
    pseudo_idx = next(i for i, file in enumerate(dataset.im_files) if Path(file).stem == "0")
    human_idx = next(i for i, file in enumerate(dataset.im_files) if Path(file).stem == "human")
    pseudo, human = dataset[pseudo_idx], dataset[human_idx]
    assert pseudo["is_pseudo"].tolist() == [True, True]
    assert pseudo["pseudo_weights"].tolist() == [1., 1.]
    assert human["is_pseudo"].tolist() == [False]
    assert human["pseudo_weights"].tolist() == [1.]
    assert torch.allclose(pseudo["bboxes"][:, 0], torch.tensor([.75, .35]))
    assert torch.allclose(pseudo["teacher_class_scores"][:, 1], torch.tensor([.1, .2]))
    result = dataset.collate_fn([pseudo, human, deepcopy(pseudo)])
    assert result["is_pseudo"].tolist() == [True, True, False, True, True]
    assert result["batch_idx"].tolist() == [0, 0, 1, 2, 2]
    assert result["ignore_batch_idx"].tolist() == [0, 2]
    assert torch.equal(result["teacher_class_scores"][:2], result["teacher_class_scores"][3:])


def seed_augmentation():
    random.seed(37)
    np.random.seed(37)
    torch.manual_seed(37)


def test_dataset_mosaic_filtering_and_close_mosaic_match_original_p1(tmp_path):
    inputs = dataset_inputs(tmp_path)
    inputs["hyp"].mosaic, inputs["hyp"].scale, inputs["hyp"].translate = 1., .7, .3
    changed = ClassAmbiguityDataset(**inputs)
    original = P1Dataset(**{k: v for k, v in inputs.items() if k != "class_score_path"})
    seed_augmentation()
    before = [original[i % len(original)] for i in range(12)]
    seed_augmentation()
    after = [changed[i % len(changed)] for i in range(12)]
    for a, b in zip(before, after, strict=True):
        for field in ("img", "cls", "bboxes", "batch_idx", "pseudo_weights",
                      "ignore_bboxes", "ignore_batch_idx"):
            assert torch.equal(a[field], b[field])
        for cls, conf, flag, scores in zip(b["cls"], b["pseudo_weights"], b["is_pseudo"],
                                           b["teacher_class_scores"], strict=True):
            if flag:
                assert scores.argmax() == cls.item()
                assert scores.max() == conf
                assert scores[(int(cls.item()) + 1) % 10].item() == pytest.approx(.1) or (
                    scores[(int(cls.item()) + 1) % 10].item() == pytest.approx(.2))
            else:
                assert conf == 1 and scores.sum() == 0
    hyp = deepcopy(inputs["hyp"])
    hyp.scale = hyp.translate = 0.
    changed.close_mosaic(hyp)
    assert changed[0]["teacher_class_scores"].shape[1] == 10


@pytest.mark.parametrize("case", [
    "metadata_hash", "sidecar_hash", "missing_receipt", "missing_image", "duplicate_image",
    "positive_count", "positive_order", "class_id", "confidence", "box", "relative_path",
    "score_shape", "score_nan", "score_negative", "score_zero", "score_top1", "score_confidence",
])
def test_dataset_rejects_stale_missing_or_misaligned_scores(tmp_path, case):
    inputs = dataset_inputs(tmp_path)
    metadata, sidecar = inputs["metadata_path"], inputs["class_score_path"]
    if case == "metadata_hash":
        metadata.write_bytes(metadata.read_bytes() + b"\n")
    elif case == "sidecar_hash":
        sidecar.write_bytes(sidecar.read_bytes() + b"\n")
    elif case == "missing_receipt":
        sidecar.with_name("receipt.json").unlink()
    else:
        rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
        positive = rows[0]["positive"][0]
        if case == "missing_image":
            rows.pop()
        elif case == "duplicate_image":
            rows.append(rows[0])
        elif case == "positive_count":
            rows[0]["positive"].pop()
        elif case == "positive_order":
            rows[0]["positive"].reverse()
        elif case == "class_id":
            positive["class_id"] = 1
        elif case == "confidence":
            positive["confidence"] = .9
        elif case == "box":
            positive["xywhn"][0] = .3
        elif case == "relative_path":
            rows[0]["image"] = "pseudo/images/train/0.jpg"
        elif case == "score_shape":
            positive["teacher_class_scores"].pop()
        elif case == "score_nan":
            positive["teacher_class_scores"][1] = float("nan")
        elif case == "score_negative":
            positive["teacher_class_scores"][1] = -.1
        elif case == "score_zero":
            positive["teacher_class_scores"] = [0.] * 10
        elif case == "score_top1":
            positive["teacher_class_scores"][0:2] = [.5, 1.]
        else:
            positive["teacher_class_scores"][0] = .9
        sidecar.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        write_receipt(metadata, sidecar)
    with pytest.raises((ValueError, FileNotFoundError)):
        ClassAmbiguityDataset(**inputs)


def test_all_objects_removed_collates_without_losing_field_shapes(tmp_path):
    dataset = ClassAmbiguityDataset(**dataset_inputs(tmp_path))
    sample = dataset[0]
    for key in ("cls", "bboxes", "batch_idx", "pseudo_weights", "teacher_class_scores",
                "is_pseudo", "ignore_bboxes", "ignore_batch_idx"):
        sample[key] = sample[key][:0]
    result = dataset.collate_fn([sample, sample])
    assert result["teacher_class_scores"].shape == (0, 10)
    assert result["is_pseudo"].shape == (0,) and result["is_pseudo"].dtype == torch.bool


def test_trainer_disabled_and_validation_route_to_p1(monkeypatch):
    trainer = object.__new__(ClassAmbiguityTrainer)
    trainer.class_ambiguity_enabled = False
    calls = []
    monkeypatch.setattr(P1Trainer, "build_dataset",
                        lambda self, *args: calls.append(args) or "original")
    assert trainer.build_dataset("train", batch=8) == "original"
    trainer.class_ambiguity_enabled = True
    assert trainer.build_dataset("val", mode="val", batch=8) == "original"
    assert calls == [("train", "train", 8), ("val", "val", 8)]
    with pytest.raises(ValueError, match="class_score_path"):
        ClassAmbiguityTrainer(enabled=True)


def test_trainer_enabled_builds_score_dataset_and_model(tmp_path):
    inputs = dataset_inputs(tmp_path)
    trainer = object.__new__(ClassAmbiguityTrainer)
    trainer.class_ambiguity_enabled = True
    trainer.class_ambiguity_diagnostics = True
    trainer.class_score_path = inputs["class_score_path"]
    trainer.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    trainer.args = inputs["hyp"]
    trainer.args.imgsz = 128
    trainer.data = inputs["data"] | {"nc": 10, "p1_metadata": str(inputs["metadata_path"])}
    config = {"nc": 10,
              "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [32, 3, 2]]],
              "head": [[[0, 1], 1, "Detect", [10]]]}
    trainer.model = trainer.get_model(config, verbose=False)
    assert isinstance(trainer.model, ClassAmbiguityModel)
    assert trainer.model.class_ambiguity_enabled and trainer.model.class_ambiguity_diagnostics
    dataset = trainer.build_dataset(inputs["img_path"], batch=2)
    assert isinstance(dataset, ClassAmbiguityDataset)
    assert dataset[0]["teacher_class_scores"].shape[1] == 10


@pytest.mark.parametrize("empty_validation", [False, True])
def test_real_detection_model_forward_backward_and_cached_validation_without_optimizer(empty_validation):
    config = {
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [32, 3, 2]]],
        "head": [[[0, 1], 1, "Detect", [2]]],
    }
    model = ClassAmbiguityModel(config, nc=2, ch=3, verbose=False)
    model.args = get_cfg()
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    batch = {"img": torch.rand(2, 3, 64, 64), "batch_idx": torch.tensor([0., 1.]),
             "cls": torch.tensor([[0.], [1.]]),
             "bboxes": torch.tensor([[.5, .5, .25, .25], [.5, .5, .25, .25]])}
    weighted = augmented_batch(batch, confidence=1.)
    model.train()
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    loss, _ = model.loss(weighted)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.equal(value, parameters[name]) for name, value in model.named_parameters())
    cached = model.criterion
    assert isinstance(cached, ClassAmbiguityLoss)
    if empty_validation:
        batch = {key: value if key == "img" else value[:0] for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        preds = model(batch["img"])
        expected, _ = P1DetectionLoss(model)(preds, batch)
        observed, _ = model.loss(batch, preds)
        implicit, _ = model.loss(batch)
    assert torch.equal(observed, expected) and torch.equal(implicit, expected)
    assert model.criterion is cached
    model.train()
    with pytest.raises(ValueError, match="provenance"):
        model.loss(batch)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_yolo11s_ten_classes_zero_optimizer_steps_with_amp_when_available(device):
    import ultralytics
    import yaml

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config_path = Path(ultralytics.__file__).parent / "cfg/models/11/yolo11.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["scale"] = "s"
    model = ClassAmbiguityModel(config, nc=10, ch=3, verbose=False).to(device)
    model.args = get_cfg()
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    model.class_ambiguity_diagnostics = True
    batch = {"img": torch.rand(2, 3, 64, 64, device=device),
             "batch_idx": torch.tensor([0., 1.], device=device),
             "cls": torch.tensor([[0.], [1.]], device=device),
             "bboxes": torch.tensor([[.5, .5, .3, .3], [.5, .5, .3, .3]], device=device)}
    training_batch = augmented_batch(batch, confidence=1.)
    training_batch["teacher_class_scores"] = torch.zeros(2, 10)
    training_batch["teacher_class_scores"][0, :2] = torch.tensor([1., .9])
    parameters = [parameter.detach().clone() for parameter in model.parameters()]
    context = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
    model.train()
    with context:
        losses, _ = model.loss(training_batch)
    losses.sum().backward()
    assert torch.isfinite(losses).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), parameters, strict=True))
    assert model.criterion.last_diagnostics["pseudo_positive"]["anchors"] > 0
    model.eval()
    with torch.no_grad():
        preds = model(batch["img"])
        expected, _ = P1DetectionLoss(model)(preds, batch)
        actual, _ = model.loss(batch, preds)
    assert torch.equal(actual, expected) and torch.isfinite(actual).all()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("diagnostics_option", [None, False, True])
def test_entry_records_original_runner_fields_and_sidecar_hashes_without_training(
        tmp_path, monkeypatch, enabled, diagnostics_option):
    import ultralytics
    import yaml
    from regionssod.training.runner import run_training

    inputs = dataset_inputs(tmp_path)
    summary = {"accepted_images": 4, "accepted_boxes": 8}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump({"p1_metadata": str(inputs["metadata_path"])}), encoding="utf-8")
    config = {"name": "candidate", "method": "p1", "model": "unused-local-model.pt",
              "data": str(dataset_yaml), "project": str(tmp_path / "runs"),
              "p1": {"soft_weights": True, "uncertain_ignore": True},
              "class_ambiguity": {"enabled": enabled},
              "train": {"epochs": 30, "batch": 8, "imgsz": 640, "seed": 0, "optimizer": "SGD"}}
    if enabled:
        config["class_ambiguity"]["class_score_path"] = str(inputs["class_score_path"])
    if diagnostics_option is not None:
        config["class_ambiguity"]["diagnostics"] = diagnostics_option
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    observed = []

    class FakeYOLO:
        def __init__(self, model):
            assert model == config["model"]
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **kwargs):
            observed.append(kwargs)
            result = SimpleNamespace(save_dir=Path(kwargs["project"]) / kwargs["name"])
            self.callbacks["on_pretrain_routine_end"](result)
            if kwargs["name"] == "candidate":
                expected_events = {"on_pretrain_routine_end"}
                if enabled:
                    expected_events |= {"on_train_epoch_start", "on_train_batch_end"}
                assert self.callbacks.keys() == expected_events
                if enabled:
                    model, preds, batch = loss_fixture()
                    result.model, result.epoch, result.ema = model, 0, None
                    self.callbacks["on_train_epoch_start"](result)
                    model.criterion = ClassAmbiguityLoss(model)
                    assert model.criterion.diagnostics
                    model.criterion(preds, augmented_batch(batch))
                    self.callbacks["on_train_batch_end"](result)
                    assert not model.criterion.diagnostics
            return result

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    candidate = run_class_ambiguity_training(config_path)
    recorded = json.loads((candidate / "experiment.json").read_text(encoding="utf-8"))
    reference_config = config | {"name": "reference"}
    reference_config.pop("class_ambiguity")
    reference_path = tmp_path / "reference.yaml"
    reference_path.write_text(yaml.safe_dump(reference_config), encoding="utf-8")
    reference = run_training(reference_path)
    expected = json.loads((reference / "experiment.json").read_text(encoding="utf-8"))
    for field in ("model", "data", "train_args", "p1", "pseudo_summary", "ultralytics_version"):
        assert recorded[field] == expected[field]
    for path, digest in expected["input_sha256"].items():
        if path != str(reference_path):
            assert recorded["input_sha256"][path] == digest
    if enabled:
        for path in (inputs["class_score_path"], inputs["class_score_path"].with_name("receipt.json")):
            assert recorded["input_sha256"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed[0]["trainer"].func is ClassAmbiguityTrainer
    assert observed[0]["trainer"].keywords["enabled"] is enabled
    assert observed[0]["trainer"].keywords["diagnostics"] is False
    assert recorded["class_ambiguity"]["diagnostics"] is enabled
    diagnostic_path = candidate / "classification_diagnostics.jsonl"
    assert diagnostic_path.exists() is enabled
    if enabled:
        rows = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["status"] == "recorded"
    assert observed[0]["exist_ok"] is False
    with pytest.raises(FileExistsError):
        run_class_ambiguity_training(config_path)
    assert len(observed) == 2


@pytest.mark.parametrize("cached", [False, True])
def test_epoch_callbacks_sample_once_rearm_and_do_not_use_ema_diagnostics(tmp_path, cached):
    model, preds, batch = loss_fixture()
    batch = augmented_batch(batch)
    ema = deepcopy(model)
    ema.class_ambiguity_diagnostics = True
    if cached:
        model.criterion = ClassAmbiguityLoss(model)
        model.criterion.last_diagnostics = {"stale": torch.tensor(-1.)}
        ema.criterion = ClassAmbiguityLoss(ema)
        ema.criterion.last_diagnostics = {"stale_ema": torch.tensor(-2.)}
    # Exercise unwrapping without initializing distributed training.
    wrapped = torch.nn.Module()
    wrapped.module = model
    trainer = SimpleNamespace(model=wrapped, ema=SimpleNamespace(ema=ema),
                              epoch=0, save_dir=tmp_path, world_size=1)
    callback = EpochClassificationDiagnostics()
    diagnostic_calls = []
    for epoch in range(2):
        trainer.epoch = epoch
        callback.on_train_epoch_start(trainer)
        assert model.class_ambiguity_diagnostics
        assert not ema.class_ambiguity_diagnostics
        if not cached and epoch == 0:
            model.criterion = ClassAmbiguityLoss(model)
        assert model.criterion.diagnostics
        assert model.criterion.last_diagnostics == {}
        if cached:
            assert not ema.criterion.diagnostics and ema.criterion.last_diagnostics == {}
        record_diagnostics = model.criterion._record_diagnostics

        def count(*args):
            diagnostic_calls.append(epoch)
            return record_diagnostics(*args)

        model.criterion._record_diagnostics = count
        expected = losses_and_gradients(ClassAmbiguityLoss(model, diagnostics=False), preds, batch)
        first = losses_and_gradients(model.criterion, preds, batch)
        assert torch.equal(first[1], expected[1])
        assert all(torch.equal(a, b) for a, b in zip(first[2], expected[2], strict=True))
        callback.on_train_batch_end(trainer)
        assert not model.class_ambiguity_diagnostics and not model.criterion.diagnostics
        assert model.criterion.last_diagnostics == {}
        losses_and_gradients(model.criterion, preds, batch)
        callback.on_train_batch_end(trainer)
        model.criterion._record_diagnostics = record_diagnostics
        # Duplicate epoch-start delivery must not append another sample.
        callback.on_train_epoch_start(trainer)
        assert not model.criterion.diagnostics
    rows = [json.loads(line) for line in (tmp_path / "classification_diagnostics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert diagnostic_calls == [0, 1]
    for row in rows:
        assert row["batch_index"] == 0 and row["status"] == "recorded"
        assert row["epoch_gradient_estimate"] is False
        assert row["scope"] == "engineering_mechanism_snapshot"
        assert row["diagnostics"]["pseudo_positive"]["anchors"] > 0
        assert isinstance(row["diagnostics"]["all"]["new_gradient_l1"], float)


def test_epoch_callback_missing_criterion_logs_missing_and_disables_lazy_initialization(tmp_path):
    model, _, _ = loss_fixture()
    trainer = SimpleNamespace(model=model, epoch=3, save_dir=tmp_path, ema=None)
    callback = EpochClassificationDiagnostics()
    callback.on_train_epoch_start(trainer)
    assert model.class_ambiguity_diagnostics
    callback.on_train_batch_end(trainer)
    assert not model.class_ambiguity_diagnostics
    assert not ClassAmbiguityLoss(model).diagnostics
    row = json.loads((tmp_path / "classification_diagnostics.jsonl").read_text())
    assert row["epoch"] == 4 and row["status"] == "missing_diagnostics"
    assert row["diagnostics"] == {}


def test_epoch_callbacks_nonprimary_rank_do_not_collect_or_write(tmp_path, monkeypatch):
    import regionssod.training.class_ambiguity as module

    monkeypatch.setattr(module, "RANK", 1)
    model, _, _ = loss_fixture()
    model.criterion = ClassAmbiguityLoss(model, diagnostics=True)
    trainer = SimpleNamespace(model=model, epoch=0, save_dir=tmp_path, ema=None)
    callback = EpochClassificationDiagnostics()
    callback.on_train_epoch_start(trainer)
    callback.on_train_batch_end(trainer)
    assert not model.class_ambiguity_diagnostics and not model.criterion.diagnostics
    assert not (tmp_path / "classification_diagnostics.jsonl").exists()
