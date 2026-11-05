"""Independent zero-optimizer AMP backward audit, contract 2026-11-05.

The caller constructs genuine P1/candidate/control losses ONCE from the same
forward graph. This helper does not construct criteria, forwards or optimizers,
sample posteriors, change parameters/buffers, or configure training's scaler.
The selected scale is an audit result, never a new training hyperparameter.
Original nonfinite losses cannot be repaired by scale backoff. Unused parameter
gradients are reported as unavailable, not as evidence of full-model coverage.
"""
from __future__ import annotations

import hashlib
import math

import torch


ARMS = ("reference", "candidate", "control")
SCALES = tuple(2.0**exponent for exponent in range(16, -17, -1))


class AmpAuditError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _stats(tensor):
    if tensor is None:
        return {"status": "unused", "finite": None, "numel": None,
                "nonfinite_count": None, "zero_count": None}
    value = tensor.detach()
    if value.layout != torch.strided:
        raise ValueError("AMP audit requires dense gradients")
    finite = torch.isfinite(value)
    counts = torch.stack((torch.isnan(value).sum(), torch.isinf(value).sum(), (value == 0).sum())).cpu().tolist()
    maximum = torch.where(finite, value.abs(), 0).max().item() if value.numel() else None
    return {"status": "observed", "dtype": str(value.dtype), "shape": list(value.shape),
            "numel": value.numel(), "finite": bool(finite.all()),
            "nan_count": counts[0], "inf_count": counts[1],
            "nonfinite_count": counts[0]+counts[1], "zero_count": counts[2],
            "max_finite_abs": maximum}


def _values(tensor):
    def safe(value):
        if isinstance(value, list):
            return [safe(v) for v in value]
        return value if math.isfinite(value) else str(value)
    return safe(tensor.detach().cpu().tolist())


def _digest(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(repr((name, str(value.dtype), tuple(value.shape))).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _parameter_stats(named_params, gradients, scale):
    records = {}
    for (name, _), gradient in zip(named_params, gradients):
        scaled = _stats(gradient)
        if gradient is None:
            records[name] = {"scaled": scaled, "unscaled_native_dtype": _stats(None)}
            continue
        # Same reciprocal construction as GradScaler.unscale_, on a detached
        # temporary; never unscale p.grad in place. Powers of two are exact.
        inverse = torch.tensor(scale, device=gradient.device, dtype=torch.float32).double().reciprocal().float()
        records[name] = {"scaled": scaled,
                         "unscaled_native_dtype": _stats(gradient.detach()*inverse)}
    missing = [name for name, record in records.items() if record["scaled"]["status"] == "unused"]
    present = len(records)-len(missing)
    finite = all(record["scaled"]["finite"] is True for record in records.values() if record["scaled"]["status"] == "observed")
    unscaled_finite = all(record["unscaled_native_dtype"]["finite"] is True for record in records.values() if record["scaled"]["status"] == "observed")
    return {"by_name": records, "trainable_parameter_tensors": len(records),
        "observed_parameter_tensors": present, "unused_parameter_names": missing,
        "complete_parameter_coverage": not missing,
        "observed_scaled_nonfinite_count": sum(record["scaled"]["nonfinite_count"] or 0 for record in records.values()),
        "observed_unscaled_nonfinite_count": sum(record["unscaled_native_dtype"]["nonfinite_count"] or 0 for record in records.values()),
        "observed_scaled_finite": finite if present else None,
        "observed_unscaled_finite": unscaled_finite if present else None}


def audit_backward(losses: dict[str, torch.Tensor], boxes: torch.Tensor,
                   model: torch.nn.Module, report: dict) -> dict:
    """Audit a fixed graph, returning {'scale': s, 'head_gradients': arm->FP64}.

    Each arm's loss must retain the genuine caller's dtype and graph. All model
    parameter .grad fields must initially be None: this helper never erases or
    accumulates into pre-existing gradients. On success candidate's SCALED
    gradients remain populated in p.grad; unscaled statistics use copies only.
    ``report['amp_backward_audit']`` contains JSON-safe evidence even on failure.
    Graphs are retained for caller-side comparisons, with no extra forward.
    Missing head or completely absent parameter coverage cannot qualify. Some
    unused parameters are allowed but explicitly listed, without a full-coverage
    claim. Finiteness alone does not certify FP16 accuracy or absence of underflow.
    """
    if "amp_backward_audit" in report:
        raise ValueError("refusing to overwrite an existing AMP audit report")
    audit = {"status": "RUNNING", "arms": list(ARMS), "scale_sequence": list(SCALES),
        "initial_scale": SCALES[0], "selected_scale": None, "attempts": [],
        "selection": "first common finite head and observed parameter gradients, including native-dtype parameter unscale",
        "forward_calls_by_helper": 0, "optimizer_created": False, "optimizer_steps": 0,
        "underflow_limit": "zero counts are descriptive; finite FP16 gradients need not equal real gradients; no underflow/precision certification",
        "training_scaler_unchanged": True}
    report["amp_backward_audit"] = audit
    try:
        if not isinstance(losses, dict) or set(losses) != set(ARMS):
            raise ValueError("exactly reference/candidate/control losses are required")
        if not isinstance(boxes, torch.Tensor) or not boxes.is_floating_point() or not boxes.requires_grad:
            raise ValueError("boxes must be a floating graph-connected tensor requiring grad")
        if not torch.is_grad_enabled():
            raise ValueError("AMP backward audit requires grad mode")
        named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not named_params:
            raise ValueError("no trainable model parameters to audit")
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("pre-existing model gradients must not be accumulated into or erased")
        if any(p.device != boxes.device for _, p in named_params):
            raise ValueError("audit expects model parameters and head on the same device")
        audit["parameters_before_sha256"] = _digest(model.named_parameters())
        audit["buffers_before_sha256"] = _digest(model.named_buffers())
        raw = audit["unscaled_losses"] = {}
        totals = {}
        # Record all three raw losses before rejecting any nonfinite one.
        for arm in ARMS:
            loss = losses[arm]
            if not isinstance(loss, torch.Tensor) or not loss.is_floating_point() or not loss.numel():
                raise ValueError(f"{arm} loss must be a nonempty floating tensor")
            if loss.device != boxes.device or not loss.requires_grad:
                raise ValueError(f"{arm} loss must retain the same-device graph")
            total = totals[arm] = loss.sum()
            raw[arm] = {"components": _stats(loss), "values": _values(loss),
                        "sum": _stats(total), "sum_value": _values(total)}
        bad = [arm for arm in ARMS if not raw[arm]["components"]["finite"] or not raw[arm]["sum"]["finite"]]
        if bad:
            raise ValueError(f"nonfinite UNSCALED loss ({', '.join(bad)}); not a scaled-gradient overflow")
        audit["unscaled_head_values"] = _stats(boxes)
        if not audit["unscaled_head_values"]["finite"]:
            raise ValueError("nonfinite original head values")
        selected = None
        inputs = (boxes, *(p for _, p in named_params))
        for scale in SCALES:
            scaler = torch.amp.GradScaler(boxes.device.type, init_scale=scale, enabled=True)
            if not scaler.is_enabled() or scaler.get_scale() != scale:
                raise ValueError("native GradScaler did not preserve the prescribed audit scale")
            attempt = {"scale": scale, "arms": {}}
            audit["attempts"].append(attempt)
            heads = {}
            for arm in ARMS:
                scaled_loss = scaler.scale(totals[arm])
                gradients = torch.autograd.grad(scaled_loss, inputs, retain_graph=True, allow_unused=True)
                head = gradients[0]
                head_stats = _stats(head)
                params = _parameter_stats(named_params, gradients[1:], scale)
                heads[arm] = None if head is None else head.detach().double()/scale
                arm_report = {"scaled_loss": _stats(scaled_loss), "head_scaled": head_stats,
                    "head_unscaled_fp64": _stats(heads[arm]), "parameters": params}
                arm_report["finite_observed_backward"] = bool(head_stats["finite"] is True
                    and arm_report["head_unscaled_fp64"]["finite"] is True
                    and params["observed_parameter_tensors"] > 0
                    and params["observed_scaled_finite"] is True
                    and params["observed_unscaled_finite"] is True)
                attempt["arms"][arm] = arm_report
                del gradients
            attempt["common_finite"] = all(attempt["arms"][arm]["finite_observed_backward"] for arm in ARMS)
            if attempt["common_finite"]:
                selected = (scale, scaler, heads, attempt)
                break
        if selected is None:
            raise ValueError("no common finite backward scale in the frozen 2^16..2^-16 sequence")
        scale, scaler, heads, attempt = selected
        audit["selected_scale"] = scale
        audit["selected_scale_below_one"] = scale < 1
        # No zero_grad/optimizer/forward: probes above used autograd.grad and
        # must not have populated any parameter's accumulation buffer.
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("gradient probes unexpectedly populated model .grad")
        scaler.scale(totals["candidate"]).backward(retain_graph=True)
        populated = _parameter_stats(named_params, [p.grad for _, p in named_params], scale)
        audit["candidate_populated_backward"] = populated
        audit["candidate_parameter_grad_storage"] = "scaled; unscaled diagnostics computed on detached copies"
        if (populated["observed_scaled_finite"] is not True or populated["observed_unscaled_finite"] is not True
                or populated["unused_parameter_names"] != attempt["arms"]["candidate"]["parameters"]["unused_parameter_names"]):
            raise ValueError("candidate populated backward is nonfinite or changed parameter coverage")
        audit["parameters_after_sha256"] = _digest(model.named_parameters())
        audit["buffers_after_sha256"] = _digest(model.named_buffers())
        if (audit["parameters_after_sha256"] != audit["parameters_before_sha256"]
                or audit["buffers_after_sha256"] != audit["buffers_before_sha256"]):
            raise ValueError("model parameters/buffers changed during backward audit")
        audit["status"] = "PASSED_FIXED_GRAPH_BACKWARD_ONLY"
        audit["scope"] = "not optimizer-step stability, full-training trajectory, or student permission; main checker owns CIoU/class parity and candidate-control differentiation"
        return {"scale": scale, "head_gradients": heads}
    except Exception as exc:
        audit["status"] = "STOP"
        audit["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise AmpAuditError(str(exc), audit) from exc
