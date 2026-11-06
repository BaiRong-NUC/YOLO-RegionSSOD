"""Fixed 16 open + 16 closed real-batch audit, INITIAL weights, zero optimization.

Run only after the owning session accepts synthetic_03 and full pytest for the
same solver sources. This script neither runs those gates nor grants a training permit.
--output must name a fresh real_NN directory under
outputs/bounded_retention_point_review_20261106/.
No resume, epoch-9 weights, validation loader/inference, optimizer or scale retry.

One owned worker preserves the seed-0 augmentation stream. Source setup reports
real progress every 5000 bound objects; inactive stages stop at 120 seconds.
EACH loss and EACH full backward have an uninterrupted 120-second wall budget.
Requests are saved before any loss. Killing this worker never kills by process
name, process group, or a training PID. An error stops the whole audit.

CPU request tensors replay criterion/head calculations; full parameter-gradient
replay additionally needs initial_state.pt and one TRAIN forward on that batch.
Only load these local trusted torch/pickle artifacts with weights_only=False.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/candidates/positive_quality_transport_v2/reference30.yaml"
FREEZE = ROOT / "outputs/bounded_retention_continuous_freeze_20261105_v2/freeze.json"
INITIAL_WEIGHTS = Path("D:/Python/YOLO/yolo11s.pt")
REAL_OUTPUT = ROOT / "outputs/bounded_retention_point_review_20261106/real_01"
TOKEN = "REGIONSSOD_POINT_REAL_WORKER_TOKEN"
TIMEOUT = 120
WINDOW = 16
ARMS = ("p1", "old_candidate", "old_control", "new_candidate", "new_control")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_value(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    # Tensor/NumPy conversion is available only in the worker, after imports.
    if hasattr(value, "detach"):
        return json_value(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    raise TypeError(f"unserializable diagnostic: {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(json_value(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def exception_info(exc):
    """Keep the primary error reportable even if its diagnostics are unfamiliar."""
    diagnostics = getattr(exc, "diagnostics", None)
    try:
        diagnostics = json_value(diagnostics)
    except Exception as diagnostic_error:
        diagnostics = {"repr": repr(diagnostics), "serialization_error": repr(diagnostic_error)}
    return {"type": type(exc).__name__, "message": str(exc),
            "diagnostics": diagnostics, "notes": getattr(exc, "__notes__", [])}


class Heartbeat:
    def __init__(self, output):
        self.output, self.sequence = output, 0
        self.events = output / "heartbeats"
        self.events.mkdir(exist_ok=False)
        self.context = {}

    def __call__(self, stage, **context):
        self.context.update(context)
        self.sequence += 1
        self.last = dict(sequence=self.sequence, stage=stage, pid=os.getpid(),
                         wall_timestamp=time.time(),
                         utc=datetime.now(timezone.utc).isoformat(),
                         stage_budget_seconds=TIMEOUT, **self.context)
        # Never replace an event a Windows reader may still have open.
        write_json(self.events / f"{self.sequence:08d}.json", self.last)


def latest_heartbeat(output):
    """Read only immutable, fully published events; ignore .json.tmp files."""
    events = sorted((output / "heartbeats").glob("[0-9]" * 8 + ".json"))
    return json.loads(events[-1].read_text(encoding="utf-8")) if events else None


def verify_sources(manifest):
    for name, expected in manifest["source_sha256"].items():
        require(sha(name) == expected, f"source changed: {name}")


def snapshot_sources(output):
    # Include transitive local dependencies, not just the two entry solvers.
    paths = [Path(__file__).resolve(), CONFIG,
             *sorted((ROOT / "src/regionssod").rglob("*.py"))]
    spec = importlib.util.find_spec("ultralytics")
    require(spec is not None and spec.origin, "installed Ultralytics is required")
    package = Path(spec.origin).parent
    installed = [*sorted(package.rglob("*.py")), package / "cfg/default.yaml"]
    hashes = {}
    for path in paths + installed:
        raw = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        relative = (Path("repo") / path.relative_to(ROOT) if path.is_relative_to(ROOT)
                    else Path("ultralytics") / path.relative_to(package))
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return hashes


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return deepcopy(value)


def save_tensor_file(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(repr((str(value.dtype), tuple(value.shape))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def named_hash(items):
    h = hashlib.sha256()
    for name, value in items:
        h.update(name.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


def bn_buffers(model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for name, value in module.named_buffers(recurse=False):
                yield f"{module_name}.{name}", value


def model_hashes(model):
    return {"parameters": named_hash(model.named_parameters()),
            "buffers": named_hash(model.named_buffers()),
            "bn_buffers": named_hash(bn_buffers(model)),
            "state_dict": named_hash(model.state_dict().items())}


def finite_stats(tensor):
    if tensor is None:
        return {"status": "UNUSED", "finite": None}
    value = tensor.detach().cpu()
    finite = torch.isfinite(value)
    return {"status": "OBSERVED", "dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": tensor_hash(value), "numel": value.numel(),
            "finite": bool(finite.all()), "nonfinite": int((~finite).sum()),
            "zeros": int((value == 0).sum())}


def tensor_difference(a, b):
    if a is None or b is None:
        return {"equal": a is None and b is None, "unused_mismatch": (a is None) != (b is None)}
    if a.shape != b.shape:
        return {"equal": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    delta = a.double() - b.double()
    return {"equal": torch.equal(a, b), "dtype_equal": a.dtype == b.dtype,
            "changed_elements": int((a != b).sum()), "numel": a.numel(),
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.,
            "l1": float(delta.abs().sum()), "l2_squared": float(delta.square().sum())}


def mapping_difference(a, b):
    keys = sorted(set(a) | set(b))
    details = {k: tensor_difference(a.get(k), b.get(k)) for k in keys}
    missing = sorted(set(a) ^ set(b))
    return {"equal": not missing and all(v["equal"] for v in details.values()),
            "missing_keys": missing, "by_tensor": details,
            "max_abs": max((v.get("max_abs", 0.) for v in details.values()), default=0.),
            "l1": sum(v.get("l1", 0.) for v in details.values())}


def clear_numerical_caches():
    cleared, seen = [], set()
    for name, module in sorted(tuple(sys.modules.items())):
        if module is None or not name.startswith("regionssod.training.bounded_retention"):
            continue
        for key, value in tuple(vars(module).items()):
            clear = getattr(value, "cache_clear", None)
            if callable(clear) and id(value) not in seen:
                seen.add(id(value))
                clear()
                cleared.append(f"{name}.{key}")
    return cleared


def geometry_snapshot(criterion):
    tensors, certificates = {}, {}
    for key, support in sorted(getattr(criterion, "supports", {}).items()):
        name = f"{key[0]}:{key[1]}"
        for field in ("geometry", "indices", "chosen", "valid", "mask", "nominal_box", "anchors", "strides"):
            tensors[f"{name}/{field}"] = cpu_tree(support[field])
        certificates[name] = {"identity": support["identity"], "points": [
            {"point": c.point, "gap": c.gap, "diagnostics": c.diagnostics}
            for c in support["point_certificates"]]}
    return tensors, certificates


def certificate_comparison(a, b):
    conflicts, compared = [], 0
    points_identical = True
    require(set(a) == set(b), "old/new point certificate object identities differ")
    for key in a:
        require(a[key]["identity"] == b[key]["identity"], "point source identity changed")
        aa, bb = a[key]["points"], b[key]["points"]
        require(len(aa) == len(bb), "old/new native anchor pools differ")
        for row, (x, y) in enumerate(zip(aa, bb)):
            points_identical = points_identical and tuple(x["point"]) == tuple(y["point"])
            dx, dy = x["diagnostics"], y["diagnostics"]
            require(all(k in d for d in (dx, dy) for k in ("score_lower", "score_upper")),
                    "point certificate is missing score enclosure")
            compared += 1
            if dx["score_lower"] > dy["score_upper"] or dy["score_lower"] > dx["score_upper"]:
                conflicts.append({"object": key, "anchor_row": row})
    return {"compared": compared, "conflicts": conflicts, "overlap": not conflicts,
            "point_identical": points_identical,
            "scope": "score-enclosure consistency, not proof of assignment/geometry parity"}


def run_arm(arm, model, preds, batch, ordinal, directory, beat, manifest):
    from regionssod.training import bounded_retention_objective_v6 as objective
    from regionssod.training.bounded_retention_continuous_numerics import optimize_point as old
    from regionssod.training.bounded_retention_point_search_v2 import optimize_point as new
    from regionssod.training.p1 import P1DetectionLoss

    beat("prepare_loss", arm=arm)
    verify_sources(manifest)
    cleared = clear_numerical_caches()
    criterion = P1DetectionLoss(model) if arm == "p1" else objective.ContinuousNativeELoss(model)
    if arm != "p1":
        criterion.mode = arm.split("_")[1]
        criterion.posterior_call_number = ordinal - 1
    assigned = {}

    def capture_assignment(module, inputs, output):
        # Detached GPU references only; CPU copies and hashing are outside loss timing.
        assigned.update({name: value.detach() for name, value in zip(
            ("labels", "boxes", "scores", "foreground", "gt_index"), output, strict=True)})

    point_stats = {"calls": 0, "seconds": 0., "by_dimension": {}, "cells_sum": 0}
    solver = new if arm.startswith("new_") else old

    def observed_point(domain, score, **kwargs):
        require(kwargs.get("max_cells", 20000) == 20000 and kwargs.get("tolerance", 1e-5) == 1e-5,
                "point resource/precision contract changed")
        start = time.perf_counter()
        try:
            result = solver(domain, score, **kwargs)
            require(result.diagnostics.get("qualified") is True and
                    0 <= result.gap <= Fraction.from_float(1e-5), "unqualified point result")
            require(result.diagnostics.get("cells", 0) <= 20000, "point cell budget exceeded")
            # Independent original point check, including for any proposed mean witness.
            require(domain.point(result.point) is not None, "returned point fails original oracle")
            point_stats["cells_sum"] += result.diagnostics.get("cells", 0)
            return result
        except Exception as exc:
            try:
                # DFLScore defines both logits and logits_U. Guard collection/IO
                # nevertheless: a failed diagnostic must never replace the solver error.
                save_tensor_file(directory / f"{arm}.point_failure.pt", {
                    "record_physical": domain.physical, "nominal_box": domain.box,
                    "canvas": domain.canvas, "active_sides": domain.active_sides,
                    "affine_bounds": domain.affine_bounds, "clip_bounds": domain.clip_bounds,
                    "logits_U": cpu_tree(score.logits), "offsets": score.offsets,
                    "signs": score.signs, "stride": score.stride,
                    "max_cells": 20000, "tolerance": 1e-5,
                    "error": repr(exc), "diagnostics": getattr(exc, "diagnostics", None)})
            except Exception as diagnostic_error:
                exc.add_note(f"point_failure artifact could not be saved: {diagnostic_error!r}; "
                             "the complete pre-loss request.pt remains the replay source")
            raise
        finally:
            elapsed = time.perf_counter() - start
            point_stats["calls"] += 1
            point_stats["seconds"] += elapsed
            group = point_stats["by_dimension"].setdefault(str(domain.dimension), {"calls": 0, "seconds": 0.})
            group["calls"] += 1
            group["seconds"] += elapsed

    arm_report = {"arm": arm, "cleared_numerical_caches": cleared,
                  "posterior_call_number": ordinal, "gradient_scale": 1.,
                  "gradient_scope": "both heads and every requires_grad model parameter; no .grad accumulation"}
    write_json(directory / f"{arm}.started.json", arm_report)
    with ExitStack() as stack:
        handle = criterion.assigner.register_forward_hook(capture_assignment)
        stack.callback(handle.remove)
        if arm != "p1":
            stack.enter_context(patch.object(objective, "optimize_point", observed_point))
        beat("loss", arm=arm)
        torch.cuda.synchronize(model.device if hasattr(model, "device") else None)
        torch.cuda.reset_peak_memory_stats()
        start, cpu_start = time.perf_counter(), time.process_time()
        with torch.autocast("cuda", dtype=torch.float16):
            full_loss, loss_items = criterion(preds, batch)
        torch.cuda.synchronize()
        arm_report.update(loss_seconds=time.perf_counter() - start,
                          loss_cpu_seconds=time.process_time() - cpu_start,
                          loss_cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                          loss_cuda_peak_reserved=torch.cuda.max_memory_reserved())
    beat("save_loss", arm=arm)
    require(assigned, "assigner hook did not observe the full assignment")
    assignment = cpu_tree(assigned)
    geometry, certificates = geometry_snapshot(criterion)
    loss_cpu = cpu_tree(full_loss)
    arm_report.update(full_loss=loss_cpu.tolist(), loss_items=cpu_tree(loss_items),
                      full_loss_stats=finite_stats(loss_cpu), point_search=point_stats,
                      criterion_diagnostics=cpu_tree(getattr(criterion, "diagnostics", {})))
    arm_report["observation_artifact"] = save_tensor_file(directory / f"{arm}.loss.pt", {
        "full_loss": loss_cpu, "loss_items": cpu_tree(loss_items),
        "assignment": assignment, "geometry": geometry, "certificates": certificates,
        "criterion_diagnostics": arm_report["criterion_diagnostics"]})
    write_json(directory / f"{arm}.json", arm_report)
    require(arm_report["full_loss_stats"]["finite"], f"nonfinite {arm} unscaled full loss")
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    require(named and all(p.grad is None for p in model.parameters()), "unexpected model gradient storage")
    inputs = (preds["boxes"], preds["scores"], *(p for _, p in named))
    beat("backward", arm=arm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, cpu_start = time.perf_counter(), time.process_time()
    gradients = torch.autograd.grad(full_loss.sum(), inputs, retain_graph=True, allow_unused=True)
    torch.cuda.synchronize()
    arm_report.update(backward_seconds=time.perf_counter() - start,
                      backward_cpu_seconds=time.process_time() - cpu_start,
                      backward_cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                      backward_cuda_peak_reserved=torch.cuda.max_memory_reserved())
    beat("save_gradients", arm=arm)
    names = ["head/boxes", "head/scores", *(f"parameter/{name}" for name, _ in named)]
    grads = dict(zip(names, cpu_tree(gradients), strict=True))
    stats = {name: finite_stats(value) for name, value in grads.items()}
    arm_report.update(gradients=stats,
                      unused_parameters=[k for k, v in stats.items() if k.startswith("parameter/") and v["status"] == "UNUSED"],
                      gradient_finite=all(v["finite"] is True for v in stats.values() if v["status"] == "OBSERVED"),
                      model_grad_storage_untouched=all(p.grad is None for p in model.parameters()),
                      status="COMPLETED")
    write_json(directory / f"{arm}.json", arm_report)
    require(grads["head/boxes"] is not None and grads["head/scores"] is not None, "head gradient missing")
    require(any(v is not None for k, v in grads.items() if k.startswith("parameter/")), "no model parameter gradients")
    require(arm_report["gradient_finite"] and arm_report["model_grad_storage_untouched"],
            f"nonfinite or accumulated {arm} gradients; no scaling retry")
    verify_sources(manifest)
    return {"report": arm_report, "loss": loss_cpu, "assignment": assignment,
            "geometry": geometry, "certificates": certificates, "gradients": grads}


def compare_arms(results):
    shared, cross = {}, {}
    for version in ("old", "new"):
        a, b = results[f"{version}_candidate"], results[f"{version}_control"]
        shared[version] = {"assignment": mapping_difference(a["assignment"], b["assignment"]),
                           "geometry": mapping_difference(a["geometry"], b["geometry"]),
                           "ciou_classification": tensor_difference(a["loss"][:2], b["loss"][:2]),
                           "candidate_control_gradient": mapping_difference(a["gradients"], b["gradients"])}
    for mode in ("candidate", "control"):
        a, b = results[f"old_{mode}"], results[f"new_{mode}"]
        cross[mode] = {"full_loss": tensor_difference(a["loss"], b["loss"]),
                       "assignment": mapping_difference(a["assignment"], b["assignment"]),
                       "geometry": mapping_difference(a["geometry"], b["geometry"]),
                       "gradients": mapping_difference(a["gradients"], b["gradients"]),
                       "point_certificates": certificate_comparison(a["certificates"], b["certificates"])}
    shared_ok = all(v[k]["equal"] for v in shared.values() for k in ("assignment", "geometry", "ciou_classification"))
    return {"within_version_shared": shared, "cross_version": cross,
            "shared_tal_ciou_passed": shared_ok,
            "certificate_conflicts": sum(len(v["point_certificates"]["conflicts"]) for v in cross.values()),
            "cross_version_equality_required": False}


def projected_cost(rows, complete_passed):
    report = {"complete_32_batches": len(rows) == 32,
              "formula": "30 * 1414 * ((20/30) * open_mean + (10/30) * closed_mean)",
              "scope": "loader + preprocess + TRAIN forward + full loss + full model autograd.grad; excludes optimizer/validation/checkpoint/audit IO/setup",
              "common_cost_accounting": "same measured loader/preprocess/forward charged once per projected independent arm; actual audit shares these across all five losses",
              "limits": "initial weights, first 16 each, order/runtime cache bias; not a tail bound or training forecast",
              "full_loss_speed_gate": {"threshold_old_over_new": 2., "passed": False,
                                       "complete_32_passed": complete_passed,
                                       "no_new_failures_certified": complete_passed},
              "arms": {}}
    if len(rows) != 32:
        report["extrapolation_status"] = "UNAVAILABLE_INCOMPLETE_FIXED_WINDOW"
        return report
    for arm in ARMS:
        phases = {}
        for phase in ("open", "closed"):
            sample = [r for r in rows if r["phase"] == phase]
            require(len(sample) == WINDOW, "wrong fixed-window population")
            components = {component: sum(r["arms"][arm][component] for r in sample) / WINDOW
                          for component in ("loss_seconds", "backward_seconds")}
            for component in ("loader_seconds", "preprocess_seconds", "forward_seconds"):
                components[component] = sum(r[component] for r in sample) / WINDOW
            components["total_seconds"] = sum(components.values())
            phases[phase] = components
        weighted = (20 * phases["open"]["total_seconds"] + 10 * phases["closed"]["total_seconds"]) / 30
        report["arms"][arm] = {"phases": phases, "weighted_seconds_per_batch": weighted,
                               "projected_30_epoch_seconds": 30 * 1414 * weighted,
                               "projected_30_epoch_hours": 30 * 1414 * weighted / 3600}
    report["three_arm_sum"] = {}
    for version in ("old", "new"):
        arms = ("p1", f"{version}_candidate", f"{version}_control")
        seconds = sum(report["arms"][a]["projected_30_epoch_seconds"] for a in arms)
        report["three_arm_sum"][version] = {"arms": arms,
            "projected_seconds": seconds, "projected_hours": seconds / 3600}
    report["paired_totals"] = {}
    for scope in ("full_loss", "data_forward_loss_backward"):
        totals = {}
        for version in ("old", "new"):
            total = 0.
            for row in rows:
                for mode in ("candidate", "control"):
                    arm = row["arms"][f"{version}_{mode}"]
                    total += arm["loss_seconds"]
                    if scope == "data_forward_loss_backward":
                        total += arm["backward_seconds"] + sum(row[c] for c in
                            ("loader_seconds", "preprocess_seconds", "forward_seconds"))
            totals[f"{version}_seconds"] = total
        ratio = totals["old_seconds"] / totals["new_seconds"] if totals["new_seconds"] > 0 else None
        report["paired_totals"][scope] = {**totals, "old_over_new": ratio,
            "paired_batches": 32, "paired_loss_calls": 64, "p1_included": False}
    ratio = report["paired_totals"]["full_loss"]["old_over_new"]
    report["full_loss_speed_gate"].update(old_over_new=ratio,
        passed=bool(complete_passed and ratio is not None and ratio >= 2.),
        definition="sum of all 64 old candidate/control full criterion calls divided by the same 64 new calls; no failed or skipped calls accepted",
        not_a_training_speed_claim=True)
    report["extrapolation_status"] = "DESCRIPTIVE_INITIAL_STATE_ONLY"
    return report


def worker(output):
    global torch, np
    beat = Heartbeat(output)
    beat("authenticate", phase=None, batch_index=None, arm=None)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    token = os.environ.pop(TOKEN, "")
    require(os.getppid() == manifest["parent_pid"] and token and
            hashlib.sha256(token.encode()).hexdigest() == manifest["worker_token_sha256"],
            "worker requires its owning parent and one-shot token")
    rows, trainer, current_state = [], None, None
    started = time.perf_counter()
    report = {"status": "RUNNING", "training_started": False, "optimizer_steps": 0,
              "validation_loader_created": False, "validation_inference_calls": 0,
              "explicit_model_forwards": 0, "forward_attempts": 0, "completed_batches": 0,
              "prerequisites": "owner must accept synthetic_03 + full pytest before launch; not auto-verified here"}
    try:
        beat("imports")
        verify_sources(manifest)
        sys.path.insert(0, str(ROOT / "src"))
        import torch
        import numpy as np
        import yaml
        import ultralytics
        from ultralytics.utils import callbacks
        from ultralytics.data import dataset as dataset_module
        from regionssod.training.bounded_retention_source import BoundSourceFactory, AUDIT, AUDIT_SHA
        from regionssod.training.bounded_retention_training_v6 import ContinuousNativeETrainer
        # Import the NEW independent path before clearing/timing numerical caches.
        from regionssod.training.bounded_retention_point_search_v2 import optimize_point as revision
        require(revision.__module__.endswith("bounded_retention_point_search_v2"), "wrong revised solver")
        verify_sources(manifest)
        require(torch.cuda.is_available(), "frozen device=0 requires CUDA")
        require(os.environ.get("RANK", "-1") == "-1" and os.environ.get("WORLD_SIZE", "1") == "1",
                "standalone single-process audit required")
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        require(Path(config["model"]).resolve() == INITIAL_WEIGHTS.resolve(), "initial weights only")
        require(all(config["train"][k] == v for k, v in
                    {"batch": 8, "imgsz": 640, "workers": 0, "seed": 0, "amp": True}.items()),
                "fixed real protocol changed")
        # Only inspect named config/weight bindings; do not open other freeze inputs.
        freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
        frozen = {str(Path(k).resolve()): v for k, v in freeze["input_sha256"].items()}
        for path in (CONFIG, INITIAL_WEIGHTS):
            require(frozen.get(str(path.resolve())) == sha(path), f"frozen initialization input changed: {path}")
        report["initial_weights"] = {"path": str(INITIAL_WEIGHTS), "sha256": sha(INITIAL_WEIGHTS)}
        report["freeze_sha256"] = sha(FREEZE)
        report["runtime"] = {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "thread_policy": "inherited native frozen GPU setup; no set_num_threads(1) override",
            "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")}}
        beat("source_factory")
        factory = BoundSourceFactory()

        def progress_factory(**kwargs):
            observation = factory(**kwargs)
            if factory.counts['objects'] % 5000 == 0:
                # Initialization work has real progress. This callback runs
                # only in source binding, NEVER inside a loss/backward stage.
                beat('train_loader_setup', legal_objects_bound=factory.counts['objects'])
            return observation

        class AuditTrainer(ContinuousNativeETrainer):
            def get_dataloader(self, dataset_path, batch_size=8, rank=-1, mode="train"):
                require(mode == "train" and batch_size == 8, "only batch8 train loader permitted")
                return super().get_dataloader(dataset_path, batch_size, rank, mode)

            def train(self, *args, **kwargs):
                raise RuntimeError("student training is forbidden")

            def build_optimizer(self, *args, **kwargs):
                raise RuntimeError("optimizer construction is forbidden")

            def get_validator(self, *args, **kwargs):
                raise RuntimeError("validation is forbidden")

        beat("trainer_setup")
        with patch.object(callbacks, "add_integration_callbacks", lambda *a, **k: None):
            trainer = AuditTrainer(mode="candidate", source_factory=progress_factory,
                source_audit_path=AUDIT, source_audit_sha256=AUDIT_SHA, p1_options=config["p1"],
                overrides={**config["train"], "model": str(INITIAL_WEIGHTS),
                    "data": str(ROOT / config["data"]), "project": str(output), "name": "trainer",
                    "save": False, "plots": False, "val": False, "resume": False},
                _callbacks={"on_pretrain_routine_start": []})
        require(trainer.device.type == "cuda" and getattr(trainer, "optimizer", None) is None,
                "unexpected device/optimizer")
        trainer.setup_model()
        trainer.model = trainer.model.to(trainer.device)
        trainer.set_model_attributes()
        model = trainer.model
        model.train()
        require(not trainer.args.multi_scale and not trainer.args.compile, "fixed input/noncompiled graph required")
        report["native_constructor_stride_probe_excluded_from_explicit_forward_count"] = True
        report["initial_hashes"] = model_hashes(model)
        report["initial_state"] = save_tensor_file(output / "initial_state.pt", {
            "state_dict": cpu_tree(model.state_dict()), "args": vars(trainer.args),
            "model_yaml": deepcopy(model.yaml), "p1_options": config["p1"],
            "hashes": report["initial_hashes"], "weights": report["initial_weights"]})
        beat("train_loader_setup")
        cache_writes = []
        original_cache_save = dataset_module.save_dataset_cache_file

        def redirected_cache(prefix, path, data, version):
            target = output / "dataset_cache" / (hashlib.sha256(str(path).encode()).hexdigest() + ".cache")
            target.parent.mkdir(exist_ok=True)
            cache_writes.append({"requested": str(path), "redirected": str(target)})
            return original_cache_save(prefix, target, data, version)

        with patch.object(dataset_module, "save_dataset_cache_file", redirected_cache):
            loader = trainer.get_dataloader(trainer.data["train"], batch_size=8, rank=-1, mode="train")
        trainer.train_loader = loader
        require(len(loader.dataset) == 11310 and len(loader) == 1414, "frozen training population changed")
        binding = {"reads": factory.reads, "bindings": factory.bindings,
                   "counts": dict(factory.counts), "premises": factory.premises,
                   "source_audit_sha256": AUDIT_SHA, "redirected_cache_writes": cache_writes}
        write_json(output / "source_factory.json", binding)
        binding_pointer = {"path": str(output / "source_factory.json"), "sha256": sha(output / "source_factory.json")}
        report["source_factory"] = binding_pointer
        report["cache_policy"] = ("clear discoverable bounded-retention functools caches before each loss; fresh criterion/domain; "
            "one persistent process/model, CUDA allocator/kernels and OS caches warm; no warmup forward")
        report["order_policy"] = ("odd ordinal: P1,old_candidate,old_control,new_candidate,new_control; "
            "even ordinal: new_control,new_candidate,old_control,old_candidate,P1")
        report["gradient_policy"] = "unscaled autograd.grad on both heads + all requires_grad parameters; no scale retry; finite does not rule out FP16 underflow"
        write_json(output / "progress.json", report)
        for phase in ("open", "closed"):
            beat("phase_setup", phase=phase, batch_index=None, arm=None)
            if phase == "closed":
                trainer._close_dataloader_mosaic()
                loader.reset()
            # Explicit next() exactly 16 times: never fetch a seventeenth batch.
            iterator = iter(loader)
            for batch_index in range(WINDOW):
                ordinal = len(rows) + 1
                beat("load_batch", phase=phase, batch_index=batch_index, ordinal=ordinal, arm=None)
                start = time.perf_counter()
                raw_batch = next(iterator)
                load_seconds = time.perf_counter() - start
                require(raw_batch["img"].shape[0] == 8, "batch8 required throughout fixed window")
                directory = output / f"{phase}_{batch_index:02d}"
                directory.mkdir()
                row = {"phase": phase, "batch_index": batch_index, "ordinal": ordinal,
                       "loader_seconds": load_seconds, "status": "RUNNING"}
                beat("save_cpu_batch")
                raw_cpu = cpu_tree(raw_batch)
                row["raw_batch"] = save_tensor_file(directory / "batch.pt", raw_cpu)
                beat("preprocess_batch")
                torch.cuda.synchronize()
                start = time.perf_counter()
                batch = trainer.preprocess_batch(raw_batch)
                torch.cuda.synchronize()
                row["preprocess_seconds"] = time.perf_counter() - start
                row["state_before"] = model_hashes(model)
                require(row["state_before"] == report["initial_hashes"], "initial parameters/buffers drifted between batches")
                current_state = {name: value.detach().clone() for name, value in model.named_buffers()}
                rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
                results = {}
                try:
                    beat("train_forward")
                    report["forward_attempts"] += 1
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.float16):
                        preds = model(batch["img"])
                    torch.cuda.synchronize()
                    row["forward_seconds"] = time.perf_counter() - start
                    report["explicit_model_forwards"] += 1
                    require(isinstance(preds, dict) and {"boxes", "scores", "feats"} <= set(preds), "unexpected TRAIN head schema")
                    beat("save_request")
                    row["state_after_forward"] = model_hashes(model)
                    require(row["state_after_forward"]["parameters"] == row["state_before"]["parameters"], "forward changed parameters")
                    order = list(ARMS if ordinal % 2 else reversed(ARMS))
                    request = {"schema": "bounded_retention_point_real_v1", "phase": phase,
                        "batch_index": batch_index, "ordinal": ordinal, "order": order,
                        "cpu_batch": cpu_tree(batch), "cpu_predictions": cpu_tree(preds),
                        "raw_batch": row["raw_batch"], "initial_state": report["initial_state"],
                        "state_before": row["state_before"], "state_after_forward": row["state_after_forward"],
                        "rng_before_forward": rng, "source_factory": binding_pointer,
                        "source_factory_reads": dict(factory.reads),
                        "source_sha256": manifest["source_sha256"], "max_cells": 20000, "tolerance": 1e-5,
                        "gradient_scale": 1., "optimizer_steps": 0,
                        "replay_limit": "detached CPU predictions retain values, not model autograd graph; use saved initial_state + raw batch for full backward"}
                    row["request"] = save_tensor_file(directory / "request.pt", request)
                    del request, raw_cpu
                    row["order"] = order
                    write_json(directory / "request.json", {**row, "source_sha256": manifest["source_sha256"],
                                                              "source_factory": binding_pointer})
                    # Restore this RNG state after comparisons so loss order cannot alter the next augmentation.
                    loss_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
                    for arm in order:
                        random.setstate(loss_rng[0]); np.random.set_state(loss_rng[1])
                        torch.set_rng_state(loss_rng[2]); torch.cuda.set_rng_state_all(loss_rng[3])
                        results[arm] = run_arm(arm, model, preds, batch, ordinal, directory, beat, manifest)
                    beat("compare_batch", arm=None)
                    row["comparisons"] = compare_arms(results)
                    row["arms"] = {arm: value["report"] for arm, value in results.items()}
                    write_json(directory / "comparisons.json", row["comparisons"])
                    require(row["comparisons"]["shared_tal_ciou_passed"], "candidate/control shared TAL/geometry/CIoU mismatch")
                    require(row["comparisons"]["certificate_conflicts"] == 0, "old/new point score certificate conflict")
                    random.setstate(loss_rng[0]); np.random.set_state(loss_rng[1])
                    torch.set_rng_state(loss_rng[2]); torch.cuda.set_rng_state_all(loss_rng[3])
                finally:
                    # No restoration until ALL attempted backwards have finished/raised.
                    beat("restore_buffers", arm=None)
                    row["state_before_restore"] = model_hashes(model)
                    with torch.no_grad():
                        for name, value in model.named_buffers():
                            value.copy_(current_state[name])
                    torch.cuda.synchronize()
                    row["state_after_restore"] = model_hashes(model)
                    row["state_restored"] = row["state_after_restore"] == row["state_before"]
                    row["parameter_grad_storage_empty"] = all(p.grad is None for p in model.parameters())
                    current_state = None
                    write_json(directory / "state.json", row)
                require(row["state_restored"] and row["parameter_grad_storage_empty"], "parameter/buffer/gradient-state invariant failed")
                require(trainer.optimizer is None and trainer.ema is None, "optimizer/EMA unexpectedly constructed")
                verify_sources(manifest)
                row["status"] = "COMPLETED"
                write_json(directory / "result.json", row)
                rows.append(row)
                report["completed_batches"] = len(rows)
                report["last_batch"] = {"phase": phase, "batch_index": batch_index}
                write_json(output / "progress.json", report)
                del results, preds, batch, raw_batch
        beat("final_verification", arm=None)
        verify_sources(manifest)
        for name, expected in factory.reads.items():
            require(sha(name) == expected, f"source factory input changed: {name}")
        require(sha(INITIAL_WEIGHTS) == report["initial_weights"]["sha256"], "initial weights changed")
        report["final_hashes"] = model_hashes(model)
        require(report["final_hashes"] == report["initial_hashes"], "final model state changed")
        require(len(rows) == 32, "incomplete fixed 32-batch protocol")
        report.update(status="PASSED_FIXED_32_ZERO_OPTIMIZATION", complete_32_passed=True,
                      source_verified_after=True, parameter_buffers_unchanged=True)
        code = 0
    except BaseException as exc:
        report.update(status="STOPPED_NO_RETRY", complete_32_passed=False,
                      error=exception_info(exc),
                      last_stage=getattr(beat, "last", None))
        (output / "worker_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        code = 2
    report["elapsed_worker_seconds"] = time.perf_counter() - started
    report["cost"] = projected_cost(rows, report["complete_32_passed"])
    report["cost"]["complete_32_passed"] = report["complete_32_passed"]
    report["parameter_restore_after_forced_kill"] = "unavailable if watchdog kills; process is discarded, original weights never written"
    write_json(output / "worker_result.json", report)
    return code


def parent(output):
    output = output.resolve()
    require(output.parent == REAL_OUTPUT.resolve().parent and re.fullmatch(r'real_[0-9]{2}', output.name),
            "output must be a fresh real_NN directory under the point review contract root")
    output.mkdir(exist_ok=False)
    token = secrets.token_hex(32)
    manifest = {"schema": "bounded_retention_point_real_v1", "parent_pid": os.getpid(),
        "worker_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": snapshot_sources(output), "stage_timeout_seconds": TIMEOUT,
        "windows": {"open": list(range(16)), "closed": list(range(16))},
        "batch": 8, "seed": 0, "weights": "initial_only", "automatic_retry": False,
        "gates": "manual owner prerequisite: synthetic_03 and full pytest accepted for these sources"}
    write_json(output / "manifest.json", manifest)
    proc = None
    terminal = {"status": "STARTING", "complete_32_passed": False}
    try:
        with (output / "worker.log").open("x", encoding="utf-8") as log:
            env = {**os.environ, TOKEN: token, "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                                     "--worker", str(output)], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            write_json(output / "process.json", {"parent_pid": os.getpid(), "worker_pid": proc.pid,
                        "command": proc.args, "utc": datetime.now(timezone.utc).isoformat()})
            changed_at, sequence, last = time.monotonic(), -1, None
            while proc.poll() is None:
                item = latest_heartbeat(output)
                if item is not None:
                    require(item["pid"] == proc.pid, "heartbeat PID differs from owned worker")
                    if item["sequence"] != sequence:
                        sequence, last, changed_at = item["sequence"], item, time.monotonic()
                if time.monotonic() - changed_at >= TIMEOUT:
                    proc.kill()  # Only the Popen child handle created above. No PID/name lookup.
                    proc.wait(timeout=10)
                    terminal = {"status": "TIMEOUT", "complete_32_passed": False,
                                "last_heartbeat": last, "owned_worker_pid": proc.pid,
                                "stage_budget_seconds": TIMEOUT, "returncode": proc.returncode,
                                "state_after_kill": "not verified; no next batch/arm/retry"}
                    break
                try:
                    proc.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
            if terminal["status"] != "TIMEOUT":
                path = output / "worker_result.json"
                result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                verify_sources(manifest)
                passed = proc.returncode == 0 and result.get("complete_32_passed") is True
                terminal = {"status": "PASSED" if passed else "WORKER_FAILED", "complete_32_passed": passed,
                            "owned_worker_pid": proc.pid, "returncode": proc.returncode,
                            "worker_result": result, "source_verified_after": True}
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        terminal = {"status": "PARENT_FAILED", "complete_32_passed": False,
                    "error": repr(exc), "owned_worker_pid": None if proc is None else proc.pid}
    write_json(output / "terminal.json", terminal)
    print(json.dumps({"output": str(output), "status": terminal["status"],
                      "complete_32_passed": terminal["complete_32_passed"]}), flush=True)
    return 0 if terminal["complete_32_passed"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="fresh outputs/bounded_retention_point_review_20261106/real_NN")
    mode.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(worker(args.worker.resolve()) if args.worker else parent(args.output))
