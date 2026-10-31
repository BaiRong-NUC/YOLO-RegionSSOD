import pytest
import torch

from regionssod.training.p1 import P1DetectionLoss
from regionssod.training.regression_confidence import RegressionConfidenceLoss
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
