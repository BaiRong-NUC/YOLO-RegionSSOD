"""Read-only run observers, not loss inputs, tuning signals or stopping rules.

Only existing criterion sidecars are inspected. No detector/criterion forward,
source oracle, RNG, backward, parameter or optimizer operation is performed.
Every partial-object occurrence and its original native TAL pool is included;
these pools are NOT foreground/effective training counts. Each object contributes
one pool-mean to its U-dimensional group, regardless of candidate-pool size.

Native categorical expected-bin geometry is recomputed in FP64 from detached
logits (bins 0..15, not continuous-density means or qclip). This is the native
decoder formula, not a bitwise replay of AMP coordinate arithmetic. Nominal
visible area uses the unmodified nominal xyxy, including native E outside canvas.
Reports are rank-local, exclusive-create JSON files, one per completed epoch.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path

import torch


class DiagnosticsError(RuntimeError):
    """A malformed/missing signal or observer lifecycle failure; never skipped."""


METRICS = ("active_u_entropy_nats", "native_decoded_area_over_nominal",
           "shared_point_area_over_nominal")
COUNTERS = ("supported_objects", "effective_anchors", "negative_known_distances",
            "overflow_known_distances", "empty_anchor_events")
NLL_SCOPE = "unknown_event_before_quarter_normalization_TAL_weight_and_gain"


def _integer(value, name, minimum=0):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise DiagnosticsError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name):
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
        raise DiagnosticsError(f"{name} must be finite")
    return float(value)


def _snapshot(value):
    """JSON-compatible detached copy; no repr fallback hiding missing fields."""
    if isinstance(value, torch.Tensor):
        return _snapshot(value.detach().cpu().tolist())
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return _number(value, "diagnostic value")
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise DiagnosticsError("diagnostic mappings require string keys")
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_snapshot(v) for v in value]
    raise DiagnosticsError(f"unsupported diagnostic value type: {type(value).__name__}")


def _unwrap(model):
    seen = set()
    while True:
        if id(model) in seen:
            raise DiagnosticsError("cyclic model wrapper")
        seen.add(id(model))
        wrapped = getattr(model, "module", None)
        if wrapped is None:
            wrapped = getattr(model, "_orig_mod", None)
        if wrapped is None:
            return model
        model = wrapped


def _floating(value, shape, name, device=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or tuple(value.shape) != tuple(shape):
        raise DiagnosticsError(f"{name} requires floating tensor shape {shape}")
    result = value.detach().to(device=device or value.device, dtype=torch.float64)
    if not torch.isfinite(result).all():
        raise DiagnosticsError(f"nonfinite {name}")
    return result


def _indices(value, count, name, device):
    if (not isinstance(value, torch.Tensor) or value.shape != (count,)
            or value.dtype not in (torch.int32, torch.int64)):
        raise DiagnosticsError(f"{name} requires integer vector length {count}")
    return value.detach().to(device=device, dtype=torch.long)


def _posterior_events(support, count, dimension):
    events = []
    details = support.get("density_diagnostics", {})
    if not isinstance(details, Mapping):
        raise DiagnosticsError("density_diagnostics must be a mapping")
    for row, raw in details.items():
        row = _integer(row, "density anchor_row")
        if row >= count or not isinstance(raw, Mapping):
            raise DiagnosticsError("invalid density event row or record")
        if raw.get("method") == "analytic_1u":
            if dimension != 1 or raw.get("qualified") is not True:
                raise DiagnosticsError("invalid analytic 1U diagnostic")
            continue
        if dimension < 2 or raw.get("method") != "posterior_gradient" or raw.get("qualified") is not True:
            raise DiagnosticsError("unrecognized/unqualified posterior diagnostic")
        d = _snapshot(raw)
        k = _integer(d["samples"], "posterior K", 1)
        accepted = _integer(d["accepted"], "posterior accepted", 1)
        proposals = _integer(d["proposals"], "posterior proposals", k)
        cells = _integer(d["cells"], "posterior cells", 1)
        if k != accepted:
            raise DiagnosticsError("posterior accepted count differs from K")
        lower, upper = map(float, d["nll_interval"])
        if not 0 <= lower <= upper or not all(map(math.isfinite, (lower, upper))):
            raise DiagnosticsError("invalid posterior NLL interval")
        variance = torch.as_tensor(d["variance_of_mean"], dtype=torch.float64)
        bound = _number(d["variance_bound_per_component"], "variance bound")
        if variance.shape != (dimension, 16) or not torch.isfinite(variance).all() or (variance < 0).any() or bound < 0:
            raise DiagnosticsError("invalid posterior variance diagnostics")
        if d["reported_value"] != "detached_interval_midpoint_not_exact_NLL":
            raise DiagnosticsError("unrecognized posterior display-value semantics")
        events.append({"anchor_row": row, "K": k, "accepted": accepted, "proposals": proposals,
            "partition_cells": cells, "nll_interval": [lower, upper], "nll_scope": NLL_SCOPE,
            "reported_value": d["reported_value"], "variance_of_mean": d["variance_of_mean"],
            "variance_bound_per_component": bound, "variance_semantics": d["variance_semantics"],
            "variance_scope": "per active-U posterior q-mean component, before /4, TAL weights and gain; not parameter-gradient variance",
            "seed": d.get("seed"), "q_mean_error_bound": d.get("q_mean_error_bound"),
            "gradient_semantics": d.get("gradient_semantics"),
            "sampling_arithmetic": {k: d[k] for k in ("log_mass_lower", "log_mass_upper", "cover_cells", "inside_cells", "mixed_cells",
                "acceptance_lower_bound", "acceptance_observed", "q_approximation_bias_bound", "rng_assumption") if k in d}})
    return events


@torch.no_grad()
def _object_record(key, support, distances):
    if not isinstance(key, tuple) or len(key) != 2 or not isinstance(support, Mapping):
        raise DiagnosticsError("support must have (image, gt_row) key and mapping value")
    image, gt_row = (_integer(v, "support index") for v in key)
    if image >= distances.shape[0]:
        raise DiagnosticsError("support image exceeds detached logits")
    identity = support["identity"]
    if not isinstance(identity, (tuple, list)) or len(identity) != 2:
        raise DiagnosticsError("source/occurrence identity required")
    identity = [_integer(v, "object identity") for v in identity]
    mask = support["mask"]
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (4,):
        raise DiagnosticsError("active U mask must be bool[4]")
    dimension = int(mask.sum())
    if not 1 <= dimension <= 3:
        raise DiagnosticsError("partial object must have 1..3 active U")
    count = support["indices"].numel()
    if count < 1:
        raise DiagnosticsError("partial object's native pool is empty")
    indices = _indices(support["indices"], count, "native pool indices", distances.device)
    if (indices < 0).any() or (indices >= distances.shape[1]).any() or indices.unique().numel() != count:
        raise DiagnosticsError("native pool indices invalid or duplicated")
    logits = distances[image, indices].detach().to(torch.float64).reshape(count, 4, 16)
    if not torch.isfinite(logits).all():
        raise DiagnosticsError("nonfinite native pool logits")
    nominal = _floating(support["nominal_box"], (4,), "nominal box", logits.device)
    if not ((nominal[2:]-nominal[:2]) > 0).all():
        raise DiagnosticsError("nonpositive nominal visible area")
    nominal_area = (nominal[2:]-nominal[:2]).prod()
    strides = _floating(support["strides"], (count, 1), "strides", logits.device)
    if not (strides > 0).all():
        raise DiagnosticsError("nonpositive stride")
    _floating(support["anchors"], (count, 2), "anchors", logits.device)
    geometry = support["geometry"]
    if not isinstance(geometry, torch.Tensor) or geometry.ndim != 3 or geometry.shape[0] != count or geometry.shape[2] != 4:
        raise DiagnosticsError("shared point geometry must have shape [pool,witness,4]")
    geometry = _floating(geometry, geometry.shape, "shared point geometry", logits.device)
    chosen = _indices(support["chosen"], count, "chosen witness", logits.device)
    if (chosen < 0).any() or (chosen >= geometry.shape[1]).any():
        raise DiagnosticsError("chosen witness outside geometry")
    valid = support["valid"]
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.shape != geometry.shape[:2]:
        raise DiagnosticsError("invalid geometry validity mask")
    rows = torch.arange(count, device=logits.device)
    if not valid.detach().to(logits.device)[rows, chosen].all():
        raise DiagnosticsError("chosen shared point is not valid")
    points = geometry[rows, chosen]
    if not ((points[:, 2:]-points[:, :2]) > 0).all():
        raise DiagnosticsError("nonpositive shared point area")
    # Functional FP64 operations only. Do not invoke the decoder/criterion or
    # use the continuous-density distribution in place of native categorical p.
    logp = logits.log_softmax(-1)
    probability = logp.exp()
    entropy = -(probability*logp).sum(-1)[:, mask.detach().to(logits.device)].mean()
    expected_bins = (probability*torch.arange(16, dtype=torch.float64, device=logits.device)).sum(-1)
    decoded_area = (expected_bins[:, 0]+expected_bins[:, 2])*(expected_bins[:, 1]+expected_bins[:, 3])*strides[:, 0].square()
    point_area = (points[:, 2:]-points[:, :2]).prod(-1)
    metrics = torch.stack((entropy, (decoded_area/nominal_area).mean(), (point_area/nominal_area).mean()))
    if not torch.isfinite(metrics).all():
        raise DiagnosticsError("nonfinite object diagnostic")
    return {"image_index": image, "gt_row": gt_row, "identity": identity,
        "active_u": dimension, "native_candidate_count": count, "nominal_visible_area": float(nominal_area),
        **dict(zip(METRICS, metrics.cpu().tolist())),
        "posterior_events": _posterior_events(support, count, dimension)}


class BoundedRetentionRunDiagnostics:
    """One observer instance per trainer; lifecycle mistakes fail explicitly."""

    def __init__(self, output_dir=None, mode=None):
        if mode not in (None, "reference", "candidate", "control"):
            raise ValueError("diagnostic mode must be reference/candidate/control")
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.expected_mode = mode
        self._state = None

    def callbacks(self):
        return {name: getattr(self, name) for name in
                ("on_train_epoch_start", "on_train_batch_end", "on_train_epoch_end")}

    def _active(self, trainer):
        if self._state is None or self._state["trainer_id"] != id(trainer) or self._state["epoch"] != trainer.epoch:
            raise DiagnosticsError("callback outside its active trainer/epoch")
        if self._state["failure"] is not None:
            raise DiagnosticsError(f"earlier diagnostic callback failed: {self._state['failure']}")
        return self._state

    def on_train_epoch_start(self, trainer):
        if self._state is not None:
            raise DiagnosticsError("previous diagnostic epoch has not been saved")
        epoch = _integer(trainer.epoch, "epoch")
        rank = _integer(getattr(trainer, "rank", int(os.environ.get("RANK", "-1"))), "rank", -1)
        directory = (self.output_dir if self.output_dir is not None else Path(trainer.save_dir)/"bounded_retention_diagnostics").resolve()
        root = Path(__file__).resolve().parents[3]
        if directory.is_relative_to(root) and directory.relative_to(root).parts[0:1] not in [
                (name,) for name in ("outputs", "runs", "output", "predictions", "tmp", "temp")]:
            raise DiagnosticsError("repository diagnostic output must be under an ignored artifact directory")
        self._state = {"trainer_id": id(trainer), "epoch": epoch, "rank": rank,
            "directory": directory, "batches": [], "mode": None, "availability": None,
            "last_call_number": None, "failure": None}

    @torch.no_grad()
    def on_train_batch_end(self, trainer):
        state = self._active(trainer)
        try:
            model = _unwrap(trainer.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None:
                raise DiagnosticsError("training criterion unavailable at batch end")
            mode = getattr(criterion, "mode", getattr(model, "bounded_retention_mode", None))
            if mode is not None and self.expected_mode is not None and mode != self.expected_mode:
                raise DiagnosticsError("criterion mode differs from callback's declared run mode")
            mode = mode if mode is not None else self.expected_mode
            has_supports = hasattr(criterion, "supports")
            unavailable = mode == "reference" or (not has_supports and mode not in {"candidate", "control"})
            status = "not_available" if unavailable else "available"
            if state["batches"] and (state["mode"] != mode or state["availability"] != status):
                raise DiagnosticsError("criterion mode/availability changed within an epoch")
            if unavailable:
                record = {"availability": status, "reason": "reference_original_P1_has_no_partial_observation_sidecar",
                          "objects": None, "criterion_diagnostics": None}
            else:
                if not has_supports or not isinstance(criterion.supports, Mapping):
                    raise DiagnosticsError("candidate/control missing supports mapping")
                diagnostics = _snapshot(criterion.diagnostics)
                for key in COUNTERS:
                    _integer(diagnostics[key], key)
                if diagnostics["supported_objects"] != len(criterion.supports):
                    raise DiagnosticsError("supported-object counter differs from complete supports")
                call_number = _integer(diagnostics["posterior_call_number"], "criterion call number", 1)
                if state["last_call_number"] is not None and call_number <= state["last_call_number"]:
                    raise DiagnosticsError("stale/repeated criterion sidecar at batch end")
                objects = []
                if criterion.supports:
                    distances = criterion.detached_dist
                    if (not isinstance(distances, torch.Tensor) or not distances.is_floating_point()
                            or distances.ndim != 3 or distances.shape[-1] != 64):
                        raise DiagnosticsError("detached_dist must be floating [batch,anchors,64]")
                    objects = [_object_record(key, support, distances) for key, support in criterion.supports.items()]
                actual_events = sum(len(o["posterior_events"]) for o in objects)
                if actual_events != diagnostics["posterior_density_anchors"]:
                    raise DiagnosticsError("posterior event count differs from actual sidecars")
                events = [e for o in objects for e in o["posterior_events"]]
                for counter, field in (("posterior_proposals", "proposals"), ("posterior_partition_cells", "partition_cells")):
                    if _integer(diagnostics[counter], counter) != sum(e[field] for e in events):
                        raise DiagnosticsError(f"{counter} differs from actual sidecars")
                details = [{"identity": o["identity"], "anchor_row": e["anchor_row"],
                    "nll_interval": e["nll_interval"], "reported_value": e["reported_value"]}
                    for o in objects for e in o["posterior_events"]]
                if diagnostics["posterior_event_details"] != details:
                    raise DiagnosticsError("posterior identity/row/NLL details differ from actual sidecars")
                if diagnostics["posterior_nll_interval_scope"] != NLL_SCOPE:
                    raise DiagnosticsError("unrecognized posterior NLL interval scope")
                record = {"availability": status, "objects": objects, "criterion_diagnostics": diagnostics}
                state["last_call_number"] = call_number
            state["mode"], state["availability"] = mode, status
            state["batches"].append(record)
        except Exception as exc:
            state["failure"] = f"{type(exc).__name__}: {exc}"
            raise DiagnosticsError(f"batch diagnostic failed in epoch {state['epoch']}: {exc}") from exc

    def on_train_epoch_end(self, trainer):
        state = self._active(trainer)
        if not state["batches"]:
            raise DiagnosticsError("cannot report an epoch with no observed batches")
        groups, counters, posterior = None, None, None
        if state["availability"] == "available":
            groups = {str(d): {"object_occurrences": 0, "native_candidate_object_pairs": 0,
                "sum_of_object_means": dict.fromkeys(METRICS, 0.)} for d in (1, 2, 3)}
            counters = dict.fromkeys(COUNTERS, 0)
            events = []
            for batch in state["batches"]:
                for key in COUNTERS:
                    counters[key] += batch["criterion_diagnostics"][key]
                for obj in batch["objects"]:
                    group = groups[str(obj["active_u"])]
                    group["object_occurrences"] += 1
                    group["native_candidate_object_pairs"] += obj["native_candidate_count"]
                    for key in METRICS:
                        group["sum_of_object_means"][key] += obj[key]
                    events.extend(obj["posterior_events"])
            for group in groups.values():
                n = group["object_occurrences"]
                group["mean_over_objects"] = {k: v/n if n else None for k, v in group["sum_of_object_means"].items()}
            posterior = {"event_count": len(events), "K_histogram": dict(Counter(str(e["K"]) for e in events)),
                "accepted_total": sum(e["accepted"] for e in events),
                "proposals_total": sum(e["proposals"] for e in events),
                "partition_cells_total": sum(e["partition_cells"] for e in events),
                "nll_interval_extrema": [min(e["nll_interval"][0] for e in events), max(e["nll_interval"][1] for e in events)] if events else None,
                "nll_scope": NLL_SCOPE, "nll_extrema_are_not_an_epoch_loss_interval": True,
                "reported_value_semantics": sorted({e["reported_value"] for e in events}),
                "variance_bound_max": max((e["variance_bound_per_component"] for e in events), default=None),
                "empirical_variance_of_mean_max": max((v for e in events for row in e["variance_of_mean"] for v in row), default=None)}
        report = {"schema": "bounded_retention_read_only_diagnostics_v1", "epoch_index": state["epoch"],
            "epoch_number": state["epoch"]+1, "rank": state["rank"], "mode": state["mode"],
            "availability": state["availability"], "batch_count": len(state["batches"]),
            "population": "all partial-object occurrences and original native TAL candidate pools; not FG effective training counts",
            "aggregation": "mean within each object's pool/active edges, then equal weight per object occurrence; no subsampling",
            "decode": "native categorical expected bins 0..15, recomputed FP64; not bitwise AMP replay or continuous-density decoding",
            "area_denominator": "raw nominal visible xyxy area, without canvas clipping",
            "counter_scopes": {"effective_anchors": "actual partial foreground regression anchors",
                "negative_known_distances": "known edge/native-candidate pairs", "overflow_known_distances": "known edge/native-candidate pairs"},
            "diagnostic_only": True, "participates_in_loss_tuning_or_stopping": False,
            "distributed_scope": "rank_local_no_reduction", "groups_by_active_u": groups,
            "criterion_counter_totals": counters, "posterior_summary": posterior, "batches": state["batches"]}
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        directory = state["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/f"epoch_{state['epoch']+1:04d}.rank_{state['rank']}.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        self._state = None


def build_callbacks(output_dir=None, mode=None):
    """Return event->callable; register each with trainer.add_callback(event, fn).

    Default: trainer.save_dir/bounded_retention_diagnostics. Repository paths
    must be under ignored artifact roots; external destinations (e.g. pytest
    temporary directories) are caller-owned. Existing epoch files are never
    overwritten. Observers do not register themselves or start training.
    """
    return BoundedRetentionRunDiagnostics(output_dir, mode).callbacks()


def make_callbacks(out, mode):
    """Stable runner API: make_callbacks(out, mode) -> {event: callback}.

    Use mode='reference' for the genuine original P1Trainer (no sidecars).
    candidate/control inspect V6's existing sidecars without forwarding.
    """
    if mode not in ("reference", "candidate", "control"):
        raise ValueError("mode must be reference/candidate/control")
    return build_callbacks(out, mode)
