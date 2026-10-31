import pytest
import torch
from typing import Any

from regionssod.training.p1 import P1DetectionLoss
from regionssod.training.regression_confidence import RegressionConfidenceLoss
from regionssod.training.regression_confidence import RegressionConfidenceModel
from test_p1 import loss_fixture


@pytest.mark.parametrize("confidence", [1., .4])
def test_regression_scales_once_and_classification_assignment_remain_unchanged(confidence):
    model, preds, batch = loss_fixture()
    batch["bboxes"][:, 2:] = .25
    batch["pseudo_weights"] = torch.full((2,), confidence)
    before = P1DetectionLoss(model).get_assigned_targets_and_loss(preds, batch)
    after = RegressionConfidenceLoss(model).get_assigned_targets_and_loss(preds, batch)
    assert all(torch.equal(x, y) for x, y in zip(before[0], after[0]))
    assert torch.equal(before[1][1], after[1][1])
    assert torch.allclose(after[1][[0, 2]], before[1][[0, 2]] * confidence, atol=1e-6)
    after[1].sum().backward()
    assert all(torch.isfinite(preds[k].grad).all() for k in ("boxes", "scores"))


def test_mixed_human_and_pseudo_weights_only_change_the_associated_image_regression():
    model, preds, batch = loss_fixture()
    batch["bboxes"][:, 2:] = .25
    batch["pseudo_weights"] = torch.tensor([.3, 1.])
    changed = RegressionConfidenceLoss(model).get_assigned_targets_and_loss(preds, batch)
    base = P1DetectionLoss(model).get_assigned_targets_and_loss(preds, batch)
    assert torch.equal(changed[1][1], base[1][1])
    assert (changed[1][[0, 2]] < base[1][[0, 2]]).all()
    assert (changed[1][[0, 2]] > .3 * base[1][[0, 2]]).all()


def test_missing_provenance_is_rejected_even_if_the_loss_can_otherwise_run():
    model, preds, batch = loss_fixture()
    with pytest.raises(ValueError, match="provenance"):
        RegressionConfidenceLoss(model).get_assigned_targets_and_loss(preds, batch)


def test_augmentation_removing_all_positives_remains_finite_and_equivalent():
    model, preds, batch = loss_fixture()
    batch = {key: value[:0] for key, value in batch.items()}
    batch["pseudo_weights"] = torch.empty(0)
    base = P1DetectionLoss(model).get_assigned_targets_and_loss(preds, batch)
    changed = RegressionConfidenceLoss(model).get_assigned_targets_and_loss(preds, batch)
    assert torch.equal(base[1], changed[1])
    assert torch.isfinite(changed[1]).all()


@pytest.mark.parametrize("empty", [False, True])
def test_model_validation_with_cached_training_loss_and_no_pseudo_metadata(empty):
    from ultralytics.cfg import get_cfg

    config: Any = {
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [32, 3, 2]]],
        "head": [[[0, 1], 1, "Detect", [2]]],
    }
    model: Any = RegressionConfidenceModel(config, nc=2, ch=3, verbose=False)
    model.args = get_cfg()
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    batch = {"img": torch.rand(2, 3, 64, 64),
             "batch_idx": torch.tensor([0., 1.]),
             "cls": torch.tensor([[0.], [1.]]),
             "bboxes": torch.tensor([[.5, .5, .25, .25], [.5, .5, .25, .25]])}
    weighted = batch | {"pseudo_weights": torch.tensor([.4, 1.])}
    model.train()
    training_loss, _ = model.loss(weighted)
    assert torch.isfinite(training_loss).all()
    cached = model.criterion
    assert isinstance(cached, RegressionConfidenceLoss)
    if empty:
        batch = {key: value if key == "img" else value[:0] for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        preds = model(batch["img"])
        expected, _ = P1DetectionLoss(model)(preds, batch)
        observed, _ = model.loss(batch, preds)
    assert torch.equal(observed, expected)
    assert torch.isfinite(observed).all()
    assert model.criterion is cached
    model.train()
    with pytest.raises(ValueError, match="provenance"):
        model.loss(batch)
    restored, _ = model.loss(weighted)
    assert torch.isfinite(restored).all()
