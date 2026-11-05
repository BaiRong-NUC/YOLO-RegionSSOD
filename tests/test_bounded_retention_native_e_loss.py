"""V6 full-loss checks; no dataset, checkpoint or hidden labels."""
from contextlib import nullcontext
import random

import numpy as np
import pytest
import torch

from regionssod.training.bounded_retention_observation import build_ideal_observation
from regionssod.training.bounded_retention_objective_v6 import (
    ContinuousNativeELoss, continuous_support, partial_dfl, private_seed,
)
from regionssod.training.p1 import P1DetectionLoss
from test_bounded_retention_continuous_loss import partial_fixture, previous_empty_bin_witness
from test_bounded_retention_p1_encoding import known_head_fixture, stopped_e_witness
from test_bounded_retention_canvas_witness import disclosed


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA AMP needs the installed GPU"))]


@pytest.mark.parametrize("case", ["negative", "overflow"])
@pytest.mark.parametrize("mode", ["reference", "candidate", "control"])
@pytest.mark.parametrize("device", DEVICES)
def test_wholly_known_native_loss_and_gradient_parity(case, mode, device):
    model, pred, batch = known_head_fixture(case, device)
    batch["retention_observations"][0]["physical"] = build_ideal_observation(
        ["1/2", "1/2", "1/5", "1/5"])
    model.bounded_retention_mode = mode
    native, candidate = P1DetectionLoss(model), ContinuousNativeELoss(model)
    with torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
        a, b = candidate(pred, batch)[0], native(pred, batch)[0]
    assert torch.equal(a, b)
    ga = torch.autograd.grad(a.sum(), (pred["boxes"], pred["scores"]), retain_graph=True)
    gb = torch.autograd.grad(b.sum(), (pred["boxes"], pred["scores"]))
    assert all(torch.equal(x, y) for x, y in zip(ga, gb))


@pytest.mark.parametrize("which", ["negative_e", "empty_bins"])
def test_both_old_blockers_use_actual_continuous_interval_loss(which):
    record, box, anchor, stride = stopped_e_witness() if which == "negative_e" else previous_empty_bin_witness()
    anchor, stride = anchor[:1], stride[:1]
    z = torch.linspace(-1., 1., 64).reshape(1, 4, 16).requires_grad_()
    before = box.clone()
    support = continuous_support(record, box, anchor, stride, (640, 640), z)
    candidate = partial_dfl(z[0], support, 0, "candidate", audit=True)
    control = partial_dfl(z[0], support, 0, "control")
    gc = torch.autograd.grad(candidate, z, retain_graph=True)[0]
    gp = torch.autograd.grad(control, z)[0]
    assert torch.isfinite(candidate) and torch.isfinite(gc).all()
    assert torch.equal(gc[0, ~support["mask"]], gp[0, ~support["mask"]])
    assert not torch.equal(gc, gp)
    assert torch.equal(box, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dimensions", [1, 2])
def test_full_partial_loss_same_assignment_geometry_and_independent_rng(device, dimensions):
    model, pred, batch = partial_fixture(device)
    if dimensions == 2:
        p = build_ideal_observation(["9/10", "9/10", "1/5", "1/5"])
        batch["retention_observations"][0]["physical"] = p
        batch["retention_observations"][0]["original_unknown"] = p.unknown
        batch["bboxes"][0] = batch["bboxes"].new_tensor([.45, .45, .1, .1])
    a, b = ContinuousNativeELoss(model), ContinuousNativeELoss(model)
    b.mode = "control"
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
        aa, la, _ = a.get_assigned_targets_and_loss(pred, batch)
        ab, lb, _ = b.get_assigned_targets_and_loss(pred, batch)
    assert aa[0].any()
    assert all(torch.equal(x, y) for x, y in zip(aa, ab))
    assert torch.equal(la[:2], lb[:2])
    ga = torch.autograd.grad(la.sum(), pred["boxes"], retain_graph=True)[0]
    gb = torch.autograd.grad(lb.sum(), pred["boxes"])[0]
    assert torch.isfinite(ga).all() and torch.isfinite(gb).all() and not torch.equal(ga, gb)
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    if dimensions == 2:
        assert a.diagnostics["posterior_density_anchors"] > 0
        assert a.diagnostics["posterior_nll_max_interval_width"] >= 0
    else:
        assert a.diagnostics["analytic_density_anchors"] > 0


def test_private_sampling_stream_identity_and_call_scope():
    a = private_seed((17, 23), (4., 12.), 8., 1)
    assert a == private_seed((17, 23), (4., 12.), 8., 1)
    assert len({a, private_seed((17, 24), (4., 12.), 8., 1),
        private_seed((18, 23), (4., 12.), 8., 1),
        private_seed((17, 23), (12., 12.), 8., 1),
        private_seed((17, 23), (4., 12.), 8., 2)}) == 5
    with pytest.raises(ValueError, match="occurrence"):
        private_seed((17, None), (4., 12.), 8., 1)


@pytest.mark.parametrize("device", DEVICES)
def test_actual_stopped_nominal_E_outside_canvas_is_preserved_in_loss(disclosed, device):
    from ultralytics.utils.loss import DFLoss
    record, values = disclosed
    record = {**record, "source_object_id": 181722, "occurrence_id": 662}
    box = torch.tensor(values, device=device)
    anchor, stride = torch.tensor([[236., 124.]], device=device), torch.tensor([[8.]], device=device)
    z = torch.linspace(-1., 1., 64, device=device).reshape(1, 4, 16).requires_grad_()
    support = continuous_support(record, box, anchor, stride, (640, 640), z)
    assert support["domain"].active_sides == (2,)
    assert torch.equal(support["geometry"][0, 0, ~support["mask"]], box[~support["mask"]].double())
    assert box[0] == -1/65536
    a = partial_dfl(z[0], support, 0, "candidate", audit=True)
    b = partial_dfl(z[0], support, 0, "control")
    native = DFLoss(16)(z[0], support["target"].clone()).sum()
    ga = torch.autograd.grad(a, z, retain_graph=True)[0]
    gb = torch.autograd.grad(b, z, retain_graph=True)[0]
    gn = torch.autograd.grad(native, z)[0]
    assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
    assert torch.equal(ga[0, ~support["mask"]], gn[0, ~support["mask"]])
    assert torch.equal(gb[0, ~support["mask"]], gn[0, ~support["mask"]])
    assert not torch.equal(ga, gb)
    assert torch.equal(box, torch.tensor(values, device=device))
