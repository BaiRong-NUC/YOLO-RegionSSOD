"""Synthetic fixed graphs only: no detector, real batch, weights or optimizer."""
import json
import random

import numpy as np
import pytest
import torch

from regionssod.training.bounded_retention_amp_audit import ARMS, SCALES, AmpAuditError, audit_backward


class TensorModule(torch.nn.Module):
    def __init__(self, device="cpu", unused=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., 2.], device=device))
        if unused:
            self.unused = torch.nn.Parameter(torch.tensor([7.], device=device))
        self.register_buffer("fixed_buffer", torch.tensor(3, device=device))

    def forward(self, *args, **kwargs):
        raise AssertionError("audit must not run a forward")


def fixed_graph(device="cpu", coefficients=(1., 4., 2.), unused=False, head_dtype=torch.float16, loss_dtype=torch.float32):
    model = TensorModule(device, unused)
    boxes = model.weight.to(head_dtype)*1
    losses = {arm: boxes.to(loss_dtype)*coefficient for arm, coefficient in zip(ARMS, coefficients)}
    return model, boxes, losses


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
def test_initial_overflow_native_backoff_first_common_scale_and_populated_backward(device):
    model, boxes, losses = fixed_graph(device)
    report = {"existing_evidence": "preserved"}
    result = audit_backward(losses, boxes, model, report)
    assert set(result) == {"scale", "head_gradients"}
    assert result["scale"] == 8192.
    audit = report["amp_backward_audit"]
    assert [a["scale"] for a in audit["attempts"]] == [65536., 32768., 16384., 8192.]
    assert audit["attempts"][0]["arms"]["reference"]["head_scaled"]["inf_count"] == 2
    assert not audit["attempts"][1]["common_finite"]
    assert audit["attempts"][1]["arms"]["reference"]["finite_observed_backward"]
    assert not audit["attempts"][1]["arms"]["candidate"]["finite_observed_backward"]
    assert audit["attempts"][2]["arms"]["control"]["finite_observed_backward"]
    assert audit["attempts"][-1]["common_finite"]
    for arm, coefficient in zip(ARMS, (1., 4., 2.)):
        head = result["head_gradients"][arm]
        assert head.dtype == torch.float64 and not head.requires_grad
        assert torch.equal(head, torch.full_like(head, coefficient))
    assert torch.equal(model.weight.grad, torch.full_like(model.weight, 4*8192.))
    p = audit["candidate_populated_backward"]
    assert p["complete_parameter_coverage"]
    assert p["by_name"]["weight"]["unscaled_native_dtype"]["max_finite_abs"] == 4.
    assert audit["parameters_before_sha256"] == audit["parameters_after_sha256"]
    assert audit["buffers_before_sha256"] == audit["buffers_after_sha256"]
    assert report["existing_evidence"] == "preserved"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("device", DEVICES)
def test_final_candidate_backward_retains_shared_graph_for_genuine_p1_comparison(device):
    model, boxes, losses = fixed_graph(device)
    # A separate equivalent reference wrapper graph shares the single head.
    equivalent_reference = boxes.float()*1.
    result = audit_backward(losses, boxes, model, {})
    scaler = torch.amp.GradScaler(device, init_scale=result["scale"])
    again = torch.autograd.grad(scaler.scale(equivalent_reference.sum()), boxes, retain_graph=True)[0].double()/result["scale"]
    original_again = torch.autograd.grad(scaler.scale(losses["reference"].sum()), boxes, retain_graph=True)[0].double()/result["scale"]
    assert torch.equal(again, original_again)
    assert torch.equal(again, result["head_gradients"]["reference"])


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_unscaled_nonfinite_loss_immediately_rejected_never_backoff(monkeypatch, arm, bad):
    model, boxes, losses = fixed_graph()
    losses[arm] = losses[arm]*bad
    def forbidden(*args, **kwargs):
        raise AssertionError("raw-loss rejection must precede gradient probes")
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    report = {}
    with pytest.raises(AmpAuditError, match="UNSCALED loss"):
        audit_backward(losses, boxes, model, report)
    audit = report["amp_backward_audit"]
    assert audit["attempts"] == [] and audit["selected_scale"] is None
    assert set(audit["unscaled_losses"]) == set(ARMS)
    assert model.weight.grad is None
    json.dumps(report, allow_nan=False)


def test_finite_components_with_nonfinite_unscaled_sum_are_rejected():
    model, boxes, losses = fixed_graph(head_dtype=torch.float32)
    for arm in ARMS:
        losses[arm] = boxes*0+torch.finfo(torch.float32).max
    report = {}
    with pytest.raises(AmpAuditError, match="UNSCALED loss"):
        audit_backward(losses, boxes, model, report)
    raw = report["amp_backward_audit"]["unscaled_losses"]["reference"]
    assert raw["components"]["finite"] and not raw["sum"]["finite"]


@pytest.mark.parametrize("device", DEVICES)
def test_missing_parameter_gradients_are_recorded_without_false_complete_coverage(device):
    model, boxes, losses = fixed_graph(device, unused=True)
    report = {}
    audit_backward(losses, boxes, model, report)
    audit = report["amp_backward_audit"]
    for arm in ARMS:
        p = audit["attempts"][-1]["arms"][arm]["parameters"]
        assert p["unused_parameter_names"] == ["unused"]
        assert p["observed_parameter_tensors"] == 1 and p["trainable_parameter_tensors"] == 2
        assert not p["complete_parameter_coverage"]
        assert p["by_name"]["unused"]["scaled"]["nonfinite_count"] is None
        assert p["by_name"]["unused"]["scaled"]["finite"] is None
    assert model.unused.grad is None


@pytest.mark.parametrize("device", DEVICES)
def test_all_33_scales_unrecoverable_failure_not_individual_arm_success(device):
    model, boxes, losses = fixed_graph(device, coefficients=(1., 2.**32, 2.))
    report = {}
    with pytest.raises(AmpAuditError, match="no common finite"):
        audit_backward(losses, boxes, model, report)
    audit = report["amp_backward_audit"]
    assert len(audit["attempts"]) == 33
    assert tuple(a["scale"] for a in audit["attempts"]) == SCALES
    assert audit["selected_scale"] is None
    assert audit["attempts"][-1]["arms"]["reference"]["finite_observed_backward"]
    assert audit["attempts"][-1]["arms"]["control"]["finite_observed_backward"]
    assert not audit["attempts"][-1]["common_finite"]
    assert model.weight.grad is None
    json.dumps(report, allow_nan=False)


def test_prescribed_lowest_scale_is_inclusive_and_underflow_limit_disclosed():
    model, boxes, losses = fixed_graph(coefficients=(1., 2.**31, 2.))
    report = {}
    result = audit_backward(losses, boxes, model, report)
    assert result["scale"] == 2.**-16
    assert report["amp_backward_audit"]["selected_scale_below_one"]
    assert "no underflow/precision certification" in report["amp_backward_audit"]["underflow_limit"]
    assert torch.equal(result["head_gradients"]["candidate"], torch.full((2,), 2.**31, dtype=torch.float64))


@pytest.mark.parametrize("loss_dtype", [torch.float16, torch.float32, torch.float64])
def test_native_scaler_scalar_dtype_promotion_not_manual_fp16_multiplier(loss_dtype):
    model, boxes, losses = fixed_graph(coefficients=(1., 1., 1.), loss_dtype=loss_dtype)
    report = {}
    audit_backward(losses, boxes, model, report)
    audit = report["amp_backward_audit"]
    first = audit["attempts"][0]["arms"]["reference"]
    assert first["scaled_loss"]["finite"]  # no fake half scalar 65536 -> inf
    expected = torch.float64 if loss_dtype == torch.float64 else torch.float32
    assert first["scaled_loss"]["dtype"] == str(expected)
    assert audit["unscaled_losses"]["reference"]["components"]["dtype"] == str(loss_dtype)


@pytest.mark.parametrize("device", DEVICES)
def test_parameters_buffers_and_all_rng_unchanged_no_forward_optimizer(device, monkeypatch):
    model, boxes, losses = fixed_graph(device)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer creation is forbidden")
    monkeypatch.setattr(torch.optim, "SGD", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else None
    result = audit_backward(losses, boxes, model, {})
    assert random.getstate() == py_rng
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng))
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert set(result["head_gradients"]) == set(ARMS)


def test_missing_head_not_accepted_as_finite():
    model, boxes, _ = fixed_graph()
    losses = {arm: model.weight*2 for arm in ARMS}
    report = {}
    with pytest.raises(AmpAuditError, match="no common finite"):
        audit_backward(losses, boxes, model, report)
    head = report["amp_backward_audit"]["attempts"][0]["arms"]["reference"]["head_scaled"]
    assert head["status"] == "unused" and head["finite"] is None


def test_no_connected_model_parameter_cannot_vacuously_pass():
    model = TensorModule()
    boxes = torch.tensor([1., 2.], requires_grad=True)
    losses = {arm: boxes*2 for arm in ARMS}
    report = {}
    with pytest.raises(AmpAuditError, match="no common finite"):
        audit_backward(losses, boxes, model, report)
    p = report["amp_backward_audit"]["attempts"][0]["arms"]["reference"]["parameters"]
    assert p["observed_scaled_finite"] is None
    assert p["unused_parameter_names"] == ["weight"]


def test_existing_gradients_and_existing_audit_are_never_cleared_or_overwritten():
    model, boxes, losses = fixed_graph()
    model.weight.grad = torch.tensor([3., 4.])
    with pytest.raises(AmpAuditError, match="pre-existing"):
        audit_backward(losses, boxes, model, {})
    assert torch.equal(model.weight.grad, torch.tensor([3., 4.]))
    report = {"amp_backward_audit": {"prior_evidence": True}}
    with pytest.raises(ValueError, match="overwrite"):
        audit_backward(losses, boxes, model, report)
    assert report == {"amp_backward_audit": {"prior_evidence": True}}


@pytest.mark.parametrize("case", ["missing_arm", "extra_arm", "detached_loss", "detached_head"])
def test_malformed_graph_inputs_fail_explicitly(case):
    model, boxes, losses = fixed_graph()
    if case == "missing_arm":
        losses.pop("control")
    elif case == "extra_arm":
        losses["other"] = losses["candidate"]
    elif case == "detached_loss":
        losses["reference"] = losses["reference"].detach()
    else:
        boxes = boxes.detach()
    with pytest.raises(AmpAuditError):
        audit_backward(losses, boxes, model, {})
