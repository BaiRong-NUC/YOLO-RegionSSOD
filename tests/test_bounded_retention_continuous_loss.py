"""Complete-loss regressions for V4; synthetic head and prior public witness.

These tests do not replace a real Trainer/dataloader/AMP integration receipt.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from ultralytics.utils.loss import DFLoss
from regionssod.training.bounded_retention_observation import build_observation, build_ideal_observation
from regionssod.training.bounded_retention_objective_v4 import (
    ContinuousDensityLoss, continuous_support, partial_dfl,
)
from regionssod.training.p1 import P1DetectionLoss
from test_bounded_retention_p1_encoding import known_head_fixture, stopped_e_witness


@pytest.mark.parametrize("case", ["negative", "overflow"])
@pytest.mark.parametrize("mode", ["reference", "candidate", "control"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA AMP requires the installed GPU"))])
def test_all_known_full_loss_and_gradients_remain_original_p1(case, mode, device):
    model, pred, batch = known_head_fixture(case, device)
    # The old fixture disabled U only in a mock mask. V4 source identity checks
    # require an actually all-known source observation as well.
    batch["retention_observations"][0]["physical"] = build_ideal_observation(
        ["1/2", "1/2", "1/5", "1/5"])
    model.bounded_retention_mode = mode
    native, candidate = P1DetectionLoss(model), ContinuousDensityLoss(model)
    with torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
        a, b = candidate(pred, batch)[0], native(pred, batch)[0]
    assert torch.equal(a, b)
    ga = torch.autograd.grad(a.sum(), (pred["boxes"], pred["scores"]), retain_graph=True)
    gb = torch.autograd.grad(b.sum(), (pred["boxes"], pred["scores"]))
    assert all(torch.equal(x, y) and torch.isfinite(x).all() for x, y in zip(ga, gb))


def previous_empty_bin_witness():
    physical = build_observation(
        ["0.3067885117", "0.9569190608", "0.07571801671", "0.08616187839"],
        source_ratio_upper=10.069343278363734, tile_size_lower=131, tile_size_upper=548)
    record = {"physical": physical, "original_unknown": physical.unknown,
        "source_object_id": 164630, "occurrence_id": 399,
        "affine": np.array([-496.50463104248047, 496.50463104248047,
                            600.9212662577629, 141.04174208641052]),
        "affine_bounds": np.array([[-496.50463104248075, -496.5046310424802],
            [496.5046310424802, 496.50463104248075],
            [600.9212662577627, 600.9212662577631],
            [141.04174208641032, 141.04174208641072]]),
        "augmentation_clip_window_bounds": np.array([
            [-1.1368683772161605e-13, 1.1368683772161605e-13], [-1e-323, 1e-323],
            [639.9999999999999, 640.0000000000001],
            [639.9999999999998, 640.0000000000002]])}
    box = torch.tensor([429.80218505859375, 594.7666015625,
                        467.39654541015625, 637.54638671875])
    return record, box, torch.tensor([[444., 636.]]), torch.tensor([[8.]])


@pytest.mark.parametrize("which", ["negative_e", "empty_bins"])
def test_prior_stops_have_continuous_witness_without_moving_e_or_dropping_object(which):
    record, box, anchors, strides = stopped_e_witness() if which == "negative_e" else previous_empty_bin_witness()
    anchors, strides = anchors[:1], strides[:1]
    old_box, old_mask = box.clone(), record["original_unknown"].copy()
    old_bounds = record["affine_bounds"].copy()
    pred = torch.linspace(-1., 1., 64).reshape(1, 4, 16).requires_grad_()
    support = continuous_support(record, box, anchors, strides, (640, 640), pred)
    mask = support["mask"]
    assert support["valid"].all() and len(support["point_certificates"]) == 1
    assert torch.equal(support["geometry"][0, 0, ~mask], old_box[~mask].double())
    a, b = (partial_dfl(pred[0], support, 0, mode) for mode in ("candidate", "control"))
    assert torch.isfinite(a) and torch.isfinite(b)
    ga = torch.autograd.grad(a, pred, retain_graph=True)[0]
    gb = torch.autograd.grad(b, pred, retain_graph=True)[0]
    raw = torch.cat((anchors - box[:2], box[2:] - anchors), 1) / strides
    native = DFLoss(16)(pred[0], raw.clone()).sum()
    gn = torch.autograd.grad(native, pred)[0]
    assert torch.equal(ga[0, ~mask], gn[0, ~mask])
    assert torch.equal(gb[0, ~mask], gn[0, ~mask])
    assert not torch.equal(ga[0, mask], gb[0, mask])
    assert torch.equal(box, old_box)
    np.testing.assert_array_equal(record["original_unknown"], old_mask)
    np.testing.assert_array_equal(record["affine_bounds"], old_bounds)


def partial_fixture(device="cpu"):
    model = torch.nn.Module()
    model.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1, device=device)))
    model.model = [SimpleNamespace(stride=torch.tensor([8.], device=device), nc=2, reg_max=16)]
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    model.bounded_retention_mode = "candidate"
    dtype = torch.float16 if device == "cuda" else torch.float32
    raw = torch.full((1, 4, 16, 256), -4., dtype=dtype, device=device)
    raw[:, :, :3] = 1.
    pred = {"boxes": raw.reshape(1, 64, 256).requires_grad_(),
        "scores": torch.zeros(1, 2, 256, device=device, dtype=dtype, requires_grad=True),
        "feats": [torch.empty(1, 1, 16, 16, device=device, dtype=dtype)]}
    physical = build_ideal_observation(["9/10", "1/2", "1/5", "1/5"])
    record = {"physical": physical, "original_unknown": physical.unknown,
        "source_object_id": 17, "occurrence_id": 23,
        "affine": np.array([64., 64., 0., 0.]),
        "affine_bounds": np.array([[64., 64.], [64., 64.], [0., 0.], [0., 0.]]),
        "augmentation_clip_window_bounds": np.array([[0., 0.], [0., 0.], [128., 128.], [128., 128.]])}
    batch = {"batch_idx": torch.tensor([0.], device=device), "cls": torch.tensor([[0.]], device=device),
        "bboxes": torch.tensor([[.45, .25, .1, .1]], device=device),
        "pseudo_weights": torch.ones(1, device=device),
        "ignore_bboxes": torch.empty(0, 4, device=device), "ignore_batch_idx": torch.empty(0, device=device),
        "object_is_human": torch.tensor([True], device=device), "object_id": torch.tensor([17], device=device),
        "obs": {"occurrence_id": np.array([23])}, "retention_observations": [record]}
    return model, pred, batch


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA AMP requires the installed GPU"))])
def test_partial_complete_loss_shares_assignment_ciou_cls_and_preserves_identity(device):
    model, pred, batch = partial_fixture(device)
    saved = batch["bboxes"].clone()
    a, b = ContinuousDensityLoss(model), ContinuousDensityLoss(model)
    b.mode = "control"
    with torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
        aa, la, _ = a.get_assigned_targets_and_loss(pred, batch)
        ab, lb, _ = b.get_assigned_targets_and_loss(pred, batch)
    assert aa[0].any(), "a real positive assignment is needed"
    assert all(torch.equal(x, y) for x, y in zip(aa, ab))
    assert torch.equal(la[:2], lb[:2])
    ga = torch.autograd.grad(la.sum(), pred["boxes"], retain_graph=True)[0]
    gb = torch.autograd.grad(lb.sum(), pred["boxes"])[0]
    assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
    assert not torch.equal(ga, gb)
    assert torch.equal(batch["bboxes"], saved)
    assert a.diagnostics["effective_anchors"] > 0


def test_identity_mismatch_fails_before_using_any_geometry():
    model, pred, batch = partial_fixture()
    batch["obs"]["occurrence_id"][0] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        ContinuousDensityLoss(model)(pred, batch)
