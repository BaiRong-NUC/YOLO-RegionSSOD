from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

from regionssod.training.competitive_negative import CompetitiveNegativeLoss, negative_multipliers
from regionssod.training.p1 import P1DetectionLoss
from test_p1 import loss_fixture
from test_class_ambiguity_training import augmented_batch, losses_and_gradients


def test_control_matches_autograd_magnitude_but_not_semantic_direction():
    logits = torch.tensor([[.1, -.3, 1., -2.], [1., 2., 3., 4.]], requires_grad=True)
    target = torch.tensor([[.6, 0., 0., 0.], [0., .8, 0., 0.]])
    teacher = torch.tensor([[.6, .4, .02, .01], [.01, .8, .01, .01]])
    pseudo = torch.tensor([True, False])
    gradients = {}
    for mode in ("semantic", "strength"):
        w, active = negative_multipliers(logits, target, teacher, pseudo, mode=mode)
        assert active.tolist() == [True, False]
        assert not w.requires_grad
        grad, = torch.autograd.grad((F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).sum(), logits)
        gradients[mode] = grad
        assert torch.equal(w[target > 0], torch.ones_like(w[target > 0]))
        assert torch.equal(w[1], torch.ones(4))
    assert torch.allclose(gradients["semantic"][0, 1:].abs().sum(), gradients["strength"][0, 1:].abs().sum())
    assert not torch.allclose(gradients["semantic"][0, 1:], gradients["strength"][0, 1:])
    assert torch.equal(gradients["semantic"][target > 0], gradients["strength"][target > 0])


@pytest.mark.parametrize("mode", ["semantic", "strength"])
@pytest.mark.parametrize("trigger", [False, True])
def test_real_p1_loss_preserves_assignment_regression_and_untouched_gradients(mode, trigger):
    model, preds, batch = loss_fixture()
    model.competitive_negative_mode = mode
    batch = augmented_batch(batch, onehot=not trigger)
    before = losses_and_gradients(P1DetectionLoss(model), preds, batch)
    criterion = CompetitiveNegativeLoss(model)
    criterion.diagnostics = True
    after = losses_and_gradients(criterion, preds, batch)
    assert all(torch.equal(a, b) for a, b in zip(before[0], after[0]))
    assert torch.equal(before[1][[0, 2]], after[1][[0, 2]])
    assert torch.equal(before[2][0], after[2][0])
    fg = after[0][0][0]
    assert fg.any()
    assert torch.equal(before[2][1][1], after[2][1][1])
    assert torch.equal(before[2][1][0, :, ~fg], after[2][1][0, :, ~fg])
    assert torch.equal(before[2][1][0, 0, fg], after[2][1][0, 0, fg])
    if trigger:
        assert torch.allclose(after[2][1][0, 1, fg], .5 * before[2][1][0, 1, fg])
    else:
        assert torch.equal(before[1], after[1])
        assert all(torch.equal(a, b) for a, b in zip(before[2], after[2]))
    assert criterion.last_diagnostics["changed_positive_target_weights"] == 0
    assert criterion.current_batch is None and criterion.assigner.assigned is None
    copied = deepcopy(criterion)
    assert copied.bce.owner is copied and copied.bce.capture is copied.assigner


def test_rejects_soft_pseudo_targets_and_invalid_mode():
    logits = torch.zeros(1, 3)
    scores = torch.tensor([[.6, .3, .1]])
    with pytest.raises(ValueError, match="hard"):
        negative_multipliers(logits, torch.tensor([[.5, .1, 0.]]), scores, torch.tensor([True]), mode="semantic")
    with pytest.raises(ValueError, match="mode"):
        negative_multipliers(logits, torch.tensor([[.5, 0., 0.]]), scores, torch.tensor([True]), mode="other")


def test_lazy_snapshot_and_validation_do_not_use_pseudo_supervision():
    from regionssod.training.competitive_negative import _Snapshots, CompetitiveNegativeModel
    model, preds, batch = loss_fixture()
    model.competitive_negative_mode = "semantic"
    _Snapshots._set_diagnostics(model, True)
    criterion = CompetitiveNegativeLoss(model)
    assert criterion.diagnostics is True
    model.criterion = criterion
    _Snapshots._set_diagnostics(model, False)
    assert criterion.diagnostics is False
    model.eval()
    actual = CompetitiveNegativeModel.loss(model, batch, preds)[0]
    expected = P1DetectionLoss(model)(preds, batch)[0]
    assert torch.equal(actual, expected)


def test_three_arm_gate_rejects_equal_strength_gain_and_fp_inflation():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("compneg_runner_test", scripts / "run_p1_competitive_negative.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def metrics(recall):
            return {"ultralytics": {"metrics/mAP50-95(B)": .25},
                "coco": {"metrics": {"AP_small": .15}},
                "precision_constraints": {iou: {p: {"small_recall": recall} for p in ("0.7", "0.9")} for iou in ("0.5", "0.75")},
                "conf025": {"0.5": {"small_recall": recall, "fp": 1000}}}
        reference, semantic, strength = metrics(.2), metrics(.204), metrics(.202)
        result = module.decision({"reference": reference, "candidate": semantic, "strength": strength})
        assert result["recommend_second_seed30"]
        assert not module.decision({"reference": reference, "candidate": semantic, "strength": semantic})["recommend_second_seed30"]
        semantic["conf025"]["0.5"]["fp"] = 1021
        assert not module.decision({"reference": reference, "candidate": semantic, "strength": strength})["recommend_second_seed30"]
    finally:
        sys.path.remove(str(scripts))
