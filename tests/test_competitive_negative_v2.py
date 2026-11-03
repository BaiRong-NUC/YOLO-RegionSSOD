from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from regionssod.training.competitive_negative import negative_multipliers
from regionssod.training.competitive_negative_v2 import CompetitiveNegativeLossV2, negative_multipliers_v2
from regionssod.training.p1 import P1DetectionLoss
from test_p1 import loss_fixture
from test_class_ambiguity_training import augmented_batch, losses_and_gradients


@pytest.mark.parametrize("mode", ["semantic", "strength"])
@pytest.mark.parametrize("mass", [0., 8.708838983295664e-9, .5])
def test_zero_underflow_and_ordinary_targets_protect_assigned_class(mode, mass):
    logits = torch.zeros(1, 10, dtype=torch.float16)
    raw = torch.zeros(1, 10)
    raw[0, 9] = mass
    teacher = torch.zeros(1, 10)
    teacher[0, 9], teacher[0, 1] = .6, .3  # trigger even on zero mass
    target = raw.half()
    snapshots = [t.clone() for t in (target, raw, teacher)]
    weights, active = negative_multipliers_v2(logits, target, teacher, torch.tensor([True]),
        torch.tensor([9]), raw, mode=mode)
    assert weights[0, 9] == 1
    assert active.item() == (mass == .5)
    if mass != .5:
        assert torch.equal(weights, torch.ones_like(weights))
    else:
        old, old_active = negative_multipliers(logits, target, teacher, torch.tensor([True]), mode=mode)
        assert torch.equal(weights, old) and torch.equal(active, old_active)
    assert all(torch.equal(a, b) for a, b in zip((target, raw, teacher), snapshots))


@pytest.mark.parametrize("mutation", ["soft", "wrong_teacher", "changed_target", "invalid_class", "nonfinite"])
def test_v2_does_not_silence_real_errors(mutation):
    raw = torch.tensor([[.5, 0., 0.]])
    target = raw.clone()
    teacher = torch.tensor([[.6, .3, .1]])
    labels = torch.tensor([0])
    if mutation == "soft":
        raw[0, 1] = .1
        target = raw.clone()
    elif mutation == "wrong_teacher":
        teacher = teacher.flip(1)
    elif mutation == "changed_target":
        target[0, 0] = .4
    elif mutation == "invalid_class":
        labels[0] = 3
    else:
        raw[0, 0] = float("nan")
    with pytest.raises(ValueError):
        negative_multipliers_v2(torch.zeros(1, 3), target, teacher, torch.tensor([True]), labels, raw, mode="semantic")


def test_v2_strength_keeps_same_negative_logit_gradient_l1():
    logits = torch.tensor([[.2, -.3, 1.]], requires_grad=True)
    target = torch.tensor([[.6, 0., 0.]])
    teacher = torch.tensor([[.6, .3, .1]])
    gradients = []
    for mode in ("semantic", "strength"):
        w, _ = negative_multipliers_v2(logits, target, teacher, torch.tensor([True]), torch.tensor([0]), target, mode=mode)
        gradient, = torch.autograd.grad((F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).sum(), logits)
        gradients.append(gradient)
    assert torch.equal(gradients[0][:, 0], gradients[1][:, 0])
    assert torch.allclose(gradients[0][:, 1:].abs().sum(), gradients[1][:, 1:].abs().sum())
    assert not torch.equal(gradients[0], gradients[1])


@pytest.mark.parametrize("mode", ["semantic", "strength"])
@pytest.mark.parametrize("trigger", [False, True])
def test_v2_real_loss_keeps_assignment_box_DFL_positive_and_unaffected_gradients(mode, trigger):
    model, preds, batch = loss_fixture()
    model.competitive_negative_mode = mode
    batch = augmented_batch(batch, onehot=not trigger)
    before = losses_and_gradients(P1DetectionLoss(model), preds, batch)
    criterion = CompetitiveNegativeLossV2(model)
    criterion.diagnostics = True
    after = losses_and_gradients(criterion, preds, batch)
    assert all(torch.equal(a, b) for a, b in zip(before[0], after[0]))
    assert torch.equal(before[1][[0, 2]], after[1][[0, 2]])
    assert torch.equal(before[2][0], after[2][0])
    fg = after[0][0][0]
    assert torch.equal(before[2][1][1], after[2][1][1])
    assert torch.equal(before[2][1][0, :, ~fg], after[2][1][0, :, ~fg])
    assert torch.equal(before[2][1][0, 0, fg], after[2][1][0, 0, fg])
    if not trigger:
        assert torch.equal(before[1], after[1])
    assert criterion.last_diagnostics["changed_assigned_class_weights"] == 0
    copied = deepcopy(criterion)
    assert copied.bce.owner is copied and copied.bce.capture is copied.assigner
