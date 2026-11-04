"""Semantic protection and AMP math, without datasets, models or training."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from regionssod.training.positive_quality_transport import rearrange, PositiveBCE


@pytest.mark.parametrize("mode", ["candidate", "control"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_control_has_same_mass_support_and_added_gradient_l1(mode, dtype):
    old = torch.tensor([.8, .2, 0., .5, .3], dtype=dtype)
    quality = torch.tensor([.2, .9, .99, .4, .7])
    ids = torch.arange(5)
    candidate = rearrange(old, quality, ids, 25, "candidate")
    actual = rearrange(old, quality, ids, 25, mode)
    assert torch.equal(actual > 0, old > 0)
    torch.testing.assert_close(actual.double().sum(), old.double().sum(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close((actual.double()-old.double()).abs().sum(),
        (candidate.double()-old.double()).abs().sum(), rtol=2e-6, atol=1e-6)
    logits = torch.tensor([2., -2., 100., -100., .1], requires_grad=True)
    base = torch.autograd.grad(F.binary_cross_entropy_with_logits(logits, old.float(), reduction="sum"), logits)[0]
    observed = torch.autograd.grad(F.binary_cross_entropy_with_logits(logits, actual.float(), reduction="sum"), logits)[0]
    torch.testing.assert_close((observed-base).double().abs().sum(),
        (candidate.double()-old.double()).abs().sum(), rtol=2e-6, atol=1e-6)


def test_control_noop_when_candidate_noop_and_does_not_consume_rng():
    old = torch.tensor([.2, .4, .8])
    rng = torch.random.get_rng_state().clone()
    assert torch.equal(rearrange(old, old, torch.arange(3), 12, "control"), old)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_control_preserves_tiny_fp32_positive_when_full_budget_is_transported():
    old = torch.tensor([1., 1e-30])
    actual = rearrange(old, torch.tensor([.1, .9]), torch.arange(2), 0, "control")
    assert torch.equal(actual, old.flip(0))
    assert (actual > 0).all()


@pytest.mark.parametrize("mode", ["reference", "candidate", "control"])
def test_zero_underflow_singleton_and_flat_quality_are_identity(mode):
    old = torch.tensor([8.7e-9, 0., .5]).half()
    assert torch.equal(rearrange(old, torch.tensor([1., .5, .1]), torch.arange(3), 1, mode), old)
    flat = torch.tensor([.3, .8, .1])
    assert torch.equal(rearrange(flat, torch.ones(3), torch.arange(3), 1, mode), flat)


@pytest.mark.parametrize("mode", ["reference", "candidate", "control"])
def test_adapter_protects_pseudo_confidence_one_and_negative_classes(mode):
    # Two repeated human source tokens are separate augmented local GT objects.
    labels = torch.tensor([[1, 1, 1, 1, 1, 1]])
    local = torch.tensor([[0, 0, 1, 1, 2, 2]])
    truth = torch.tensor([[[0., 0., 10., 10.]] * 6])
    predicted = truth.clone()
    predicted[:, [0, 2, 4], 2:] = 5.
    raw = torch.zeros(1, 6, 3)
    raw[0, :, 1] = torch.tensor([.8, .2, .8, .2, .8, .2])
    batch = {"batch_idx": torch.zeros(3), "object_class": torch.ones(3, dtype=torch.long),
        "object_is_human": torch.tensor([True, True, False]), "object_id": torch.tensor([10, 10, 11])}
    owner = SimpleNamespace(current_batch=batch, mode=mode, diagnostics=True,
        assigner=SimpleNamespace(assigned=(labels, truth, raw, torch.ones(1, 6, dtype=torch.bool), local),
                                 predicted_boxes=predicted))
    logits = torch.zeros(1, 6, 3, requires_grad=True)
    actual = PositiveBCE(torch.nn.BCEWithLogitsLoss(reduction="none"), owner)(logits, raw)
    gradient = torch.autograd.grad(actual.sum(), logits)[0]
    assert torch.equal(owner.last_targets[:, 4:], raw[:, 4:])
    assert torch.equal(owner.last_targets[..., [0, 2]], raw[..., [0, 2]])
    assert torch.equal(gradient[:, 4:], .5 - raw[:, 4:])
    if mode == "candidate":
        assert torch.equal(owner.last_targets[0, :4, 1], torch.tensor([.2, .8, .2, .8]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1, 1.1])
def test_illegal_geometry_rejected(bad):
    with pytest.raises(ValueError):
        rearrange(torch.tensor([.3, .5]), torch.tensor([bad, .5]), torch.arange(2), 1, "candidate")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA AMP")
def test_amp_targets_do_not_create_zero_quality_and_gradients_remain_finite():
    old = torch.tensor([.8, 8.7e-9, .2], device="cuda", dtype=torch.float16)
    quality = torch.tensor([.2, 1., .9], device="cuda")
    ids = torch.arange(3, device="cuda")
    for mode in ("candidate", "control"):
        changed = rearrange(old, quality, ids, 10, mode)
        logits = torch.tensor([100., -100., 0.], device="cuda", dtype=torch.float16, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(logits, changed, reduction="sum")
        gradient, = torch.autograd.grad(loss, logits)
        assert torch.isfinite(gradient).all() and torch.isfinite(loss)
        assert changed[1] == 0
