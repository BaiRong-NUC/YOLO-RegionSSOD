"""Fresh sequential P1 / V6 candidate / V6 control, exactly 30 epochs each.

Import is inert: no detector, dataset, cache or old runner/evaluator is imported.
The final JSON spec is supplied by the owning session AFTER engineering passes;
this module never creates that permit, its configs, or its scientific contract.

Spec schema (all pointers are {path, sha256}):
  schema='bounded_retention_continuous_three30_v1',
  permit='FINAL_THREE_ARM_30_FROZEN', baseline, configs[reference/candidate/control],
  geometry_engineering (immutable G0 STOP), engineering (independent AMP pass),
  full_suite, contract, failure_policy=FAILURE_POLICY,
  input_sha256[path]=sha256.
full_suite: status='PASSED', scope='full_repository', exit_code=0, passed>=1862,
failed=errors=0, input_sha256 covering the final Python implementation/tests.
The suite/freeze also pins installed ultralytics/engine/trainer.py. The owning
scientific contract must disclose the identical fault-only FAILURE_POLICY.
input_sha256 must cover all engineering/suite inputs and the legal materialized
training/validation population. No arbitrary input list is opened unchecked.
CLI --check is read-only; --train alone dispatches authenticated child workers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import traceback
from types import MethodType

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("reference", "candidate", "control")
OUT = ROOT / "outputs/bounded_retention_continuous_three30_s0"
BASE = ROOT / "configs/candidates/positive_quality_transport_v2/reference30.yaml"
GEOMETRY_ENGINEERING = ROOT / "outputs/bounded_retention_continuous_20261105/native_e_integration_01/result.json"
GEOMETRY_ENGINEERING_SHA = "39c03aa3a4243a0f3169abe174c4701d964f905df749c208148fc0bccee7d259"
ENGINEERING = ROOT / "outputs/bounded_retention_continuous_20261105/native_amp_audit_01/result.json"
AMP_CONTRACT = ROOT / "docs/paper/candidates/bounded_retention/amp_audit_revision_contract.md"
AUDIT = ROOT / "outputs/bounded_retention_engineering_20261105/source_audit.json"
AUDIT_SHA = "c8f2d57e5b652c1bfd79893404175aea7ce1e8aa64610b49f3690084e375f95a"
SCHEMA = "bounded_retention_continuous_three30_v1"
WORKER_TOKEN = "REGIONSSOD_CONTINUOUS_WORKER_TOKEN"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FINAL_INFERENCE = {"imgsz": 960, "batch": 8, "workers": 0, "device": "0", "conf": .001,
    "iou": .7, "max_det": 500, "rect": True, "quantize": None, "augment": False,
    "plots": False, "save_json": False, "verbose": False, "split": "val"}
FAILURE_POLICY = {
    "all_arms": list(ARMS), "oom_auto_batch_retry": False, "checkpoint_nan_recovery": False,
    "batch_end_check": "finite unscaled loss and loss items",
    "epoch_end_check": "finite actual model and EMA floating parameters/buffers",
    "recovery_hook_check": "finite loss/fitness and actual model/EMA state; return False or raise",
    "scaled_gradient_overflow_or_skip_is_failure": False,
    "automatic_worker_termination": False,
}


class WorkerStillLive(RuntimeError):
    """Parent interruption is not terminal evidence for a still-running worker."""


def evaluator_api():
    from evaluate_bounded_retention import evaluate, audit, POPULATION, INFERENCE
    require(POPULATION == 548 and INFERENCE == FINAL_INFERENCE, "final 548/960 inference settings differ")
    return evaluate, audit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, value, *, exclusive=False):
    with Path(p).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def hashes(mapping):
    require(isinstance(mapping, dict) and bool(mapping), "empty hash inventory")
    result = {}
    for name, value in mapping.items():
        p = str(path(name))
        require(isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value), "invalid SHA256")
        require(p not in result or result[p] == value, "conflicting input aliases")
        result[p] = value
    return result


def check_subset(child, parent):
    for p, h in hashes(child).items():
        require(parent.get(p) == h, f"missing/conflicting frozen input: {p}")


def pointer(item):
    require(isinstance(item, dict) and set(item) == {"path", "sha256"}, "invalid frozen pointer")
    p = path(item["path"])
    require("manifest" not in p.name.lower() and p.suffix.lower() in {".json", ".md", ".yaml", ".yml"},
            "forbidden pointer")
    require(digest(p) == item["sha256"], f"pointer hash changed: {p}")
    return p


def verify_common_engineering(report):
    require(report.get("objective") == "V6_native_E_same_continuous_density", "wrong objective")
    require(report.get("stage") == "real_model_amp_forward_backward", "AMP stage incomplete")
    require(report.get("optimizer_steps") == 0 and report.get("training_started") is False
            and report.get("validation_inference_calls") == 0, "engineering executed training/inference")
    d = report.get("dataloaders", {})
    require(d.get("train_images") == 11310 and d.get("val_images") == 548
            and d.get("workers") == 0 and d.get("image_size") == 640
            and d.get("validation_inference") is False, "loader population/settings differ")
    require(report.get("model_forward_calls") == report.get("model_forward_attempts") == 1,
            "exactly one explicit AMP forward required")
    require(report.get("forward_batch_selection") == {
        "rule": "first_maximum_active_dimension", "phase": "mosaic_closed",
        "batch_index": 1316, "active_dimension": 3}, "fixed pressure batch selection changed")
    hashes(report.get("input_sha256"))


def verify_geometry_engineering(report):
    """G0 remains STOP: inherit only its completed geometry/numerical evidence."""
    require(report.get("status") == "STOP_BEFORE_STUDENT_TRAINING"
            and report.get("error", {}).get("message") == "nonfinite original P1 loss/gradient",
            "immutable G0 STOP state changed")
    verify_common_engineering(report)
    # G0 writes all_stream_geometry_checked only AFTER its backward checks.
    # Its immutable STOP therefore lacks that success-only summary; inspect
    # the complete ordered records and final loop position instead.
    require("--scan-all-batches" in report.get("command", []), "full geometry scan absent")
    rows = report.get("geometry_batch_checks", [])
    require(Counter(r.get("phase") for r in rows) == {"mosaic_open": 1414, "mosaic_closed": 1414},
            "both complete 1414-batch phases required")
    require([r.get("phase") for r in rows] == ["mosaic_open"] * 1414 + ["mosaic_closed"] * 1414
            and report.get("geometry_phase") == "mosaic_closed"
            and report.get("geometry_batch_index") == 1413, "geometry stream ordering/final position differs")
    for row in rows:
        require(row.get("supported_objects") == row.get("partial_objects_checked")
                and isinstance(row.get("supported_objects"), int)
                and row["supported_objects"] >= 0 and row.get("empty_anchor_events") == 0,
                "unsupported geometry in complete scan")
    require(report.get("amp_initial_gradient_scale") == 65536, "original AMP scale missing")
    audits = report.get("actual_head_numeric_audits")
    require(isinstance(audits, list) and len(audits) == 7
            and any(e.get("dimension") == 3 for e in audits), "seven actual head numeric audits including 3U required")
    for entry in audits:
        diag = entry.get("diagnostics", {})
        require(entry.get("dimension") in (1, 2, 3) and len(entry.get("identity", [])) == 2
                and diag.get("qualified") is True
                and diag.get("denominator", {}).get("qualified") is True,
                "actual head arithmetic not qualified")
        if entry["dimension"] == 1:
            require(diag.get("numerator", {}).get("qualified") is True, "1U numerator unqualified")


def finite_number(value):
    return type(value) in (float, int) and math.isfinite(value)


def finite_stats(stats):
    return (stats.get("status") == "observed" and stats.get("finite") is True
            and stats.get("nonfinite_count") == stats.get("nan_count") == stats.get("inf_count") == 0
            and type(stats.get("numel")) is int and stats["numel"] >= 0)


def verify_parameter_stats(record):
    """Check every named record, preserving explicitly unused parameters."""
    entries = record.get("by_name", {})
    require(bool(entries) and record.get("trainable_parameter_tensors") == len(entries),
            "named trainable parameter gradient records missing")
    unused = [name for name, e in entries.items() if e.get("scaled", {}).get("status") == "unused"]
    observed = [e for name, e in entries.items() if name not in unused]
    require(sorted(record.get("unused_parameter_names", [])) == sorted(unused)
            and record.get("observed_parameter_tensors") == len(observed)
            and record.get("complete_parameter_coverage") is (not unused), "parameter coverage summary differs")
    for name in unused:
        require(entries[name].get("unscaled_native_dtype", {}).get("status") == "unused",
                "unused parameter was relabelled observed")
    valid = []
    for field, flag, count in (("scaled", "observed_scaled_finite", "observed_scaled_nonfinite_count"),
                               ("unscaled_native_dtype", "observed_unscaled_finite", "observed_unscaled_nonfinite_count")):
        require(all(e.get(field, {}).get("status") == "observed" for e in observed), "parameter gradient unavailable")
        counts = [e[field].get("nonfinite_count") for e in observed]
        require(all(type(n) is int and n >= 0 for n in counts)
                and record.get(count) == sum(counts), "parameter nonfinite count differs")
        finite = all(finite_stats(e[field]) for e in observed) if observed else None
        require(record.get(flag) is finite, "parameter finite summary differs")
        valid.append(finite is True)
    return all(valid)


def verify_amp_backward(audit, losses):
    scales = [2.**e for e in range(16, -17, -1)]
    require(audit.get("status") == "PASSED_FIXED_GRAPH_BACKWARD_ONLY"
            and audit.get("arms") == list(ARMS) and audit.get("scale_sequence") == scales
            and audit.get("initial_scale") == scales[0]
            and audit.get("forward_calls_by_helper") == audit.get("optimizer_steps") == 0
            and audit.get("optimizer_created") is False and audit.get("training_scaler_unchanged") is True,
            "AMP backward protocol not satisfied")
    for arm in ARMS:
        raw = audit.get("unscaled_losses", {}).get(arm, {})
        require(raw.get("values") == losses[arm] and finite_stats(raw.get("components", {}))
                and finite_stats(raw.get("sum", {})) and finite_number(raw.get("sum_value")),
                "unscaled loss evidence differs/nonfinite")
    require(finite_stats(audit.get("unscaled_head_values", {})), "original head values nonfinite/unavailable")
    attempts = audit.get("attempts", [])
    require(0 < len(attempts) <= len(scales)
            and [a.get("scale") for a in attempts] == scales[:len(attempts)]
            and audit.get("selected_scale") == attempts[-1]["scale"], "AMP scale sequence skipped/reordered")
    names = None
    for index, attempt in enumerate(attempts):
        require(set(attempt.get("arms", {})) == set(ARMS), "AMP attempt missing arm")
        flags = []
        for arm in ARMS:
            record = attempt["arms"][arm]
            params = record.get("parameters", {})
            param_ok = verify_parameter_stats(params)
            current_names = set(params["by_name"])
            if names is None:
                names = current_names
            require(current_names == names, "trainable parameter identities changed across arms/scales")
            ok = param_ok and finite_stats(record.get("head_scaled", {})) and finite_stats(record.get("head_unscaled_fp64", {}))
            require(record.get("finite_observed_backward") is ok, "AMP arm summary contradicts gradients")
            flags.append(ok)
        common = all(flags)
        require(attempt.get("common_finite") is common and common == (index == len(attempts)-1),
                "selected scale is not first common finite backward")
    populated = audit.get("candidate_populated_backward", {})
    require(verify_parameter_stats(populated) and set(populated["by_name"]) == names
            and populated["unused_parameter_names"] == attempts[-1]["arms"]["candidate"]["parameters"]["unused_parameter_names"],
            "candidate populated backward not finite or coverage changed")
    for field in ("parameters", "buffers"):
        before = audit.get(f"{field}_before_sha256")
        require(isinstance(before, str) and len(before) == 64
                and before == audit.get(f"{field}_after_sha256"), "backward changed parameter/buffer state")


def verify_engineering(report, geometry):
    """Joint gate; neither a STOP relabel nor an `accepted` flag is sufficient."""
    verify_geometry_engineering(geometry)
    verify_common_engineering(report)
    require(report.get("status") == "AMP_INTEGRATION_COMPONENT_PASSED_NOT_STUDENT_PERMIT"
            and report.get("audit_protocol") == "native_dynamic_scaling_v1", "independent AMP component not passed")
    require(path(report.get("parent_geometry_receipt", "")) == GEOMETRY_ENGINEERING.resolve()
            and report.get("parent_geometry_sha256") == GEOMETRY_ENGINEERING_SHA,
            "AMP parent geometry binding differs")
    require(report.get("replayed_batches") == {"mosaic_open": 1414, "mosaic_closed": 1414}, "both 1414-batch replays required")
    require(isinstance(geometry.get("source_factory_counts"), dict) and bool(geometry["source_factory_counts"])
            and report.get("source_factory_counts") == geometry["source_factory_counts"], "source population changed")
    before = report.get("parameters_before_forward_sha256")
    require(isinstance(before, str) and len(before) == 64
            and before == geometry.get("parameters_before_forward_sha256") == report.get("parameters_after_backward_sha256")
            and report.get("parameter_hash_unchanged") is True, "initial/current parameters changed")
    require(report.get("actual_head_numeric_audits") == geometry["actual_head_numeric_audits"]
            and report.get("parent_actual_head_numeric_audits_equal") is True, "parent head receipts differ")
    losses = report.get("unscaled_losses", {})
    require(set(losses) == set(ARMS) and all(isinstance(v, list) and len(v) == 3
            and all(finite_number(x) for x in v) for v in losses.values()), "three finite unscaled losses required")
    require(losses["candidate"][:2] == losses["control"][:2] and report.get("real_shared_ciou_classification") is True,
            "shared CIoU/classification differs")
    gap = report.get("real_control_gradient_difference_l1")
    require(finite_number(gap) and gap > 0, "candidate/control gradients indistinguishable")
    require(report.get("legacy_reference_value_equal") is True and report.get("legacy_reference_gradient_equal") is True,
            "legacy original P1 value/gradient parity missing")
    verify_amp_backward(report.get("amp_backward_audit", {}), losses)
    previews = report.get("diagnostic_previews", {})
    require(set(previews) == set(ARMS), "three callback previews required")
    for arm in ARMS:
        require(path(previews[arm].get("path", "")) == preview_path(arm)
                and isinstance(previews[arm].get("sha256"), str) and len(previews[arm]["sha256"]) == 64,
                "callback preview pointer differs")


def preview_path(arm):
    return ENGINEERING.parent / "zero_step_diagnostic_preview" / arm / "epoch_0001.rank_-1.json"


def verify_previews(report):
    for arm in ARMS:
        p = preview_path(arm)
        require(digest(p) == report["diagnostic_previews"][arm]["sha256"], "callback preview hash differs")
        value = read(p)
        require(value.get("schema") == "bounded_retention_read_only_diagnostics_v1"
                and value.get("mode") == arm and value.get("rank") == -1
                and value.get("epoch_index") == 0 and value.get("epoch_number") == value.get("batch_count") == 1
                and value.get("diagnostic_only") is True
                and value.get("participates_in_loss_tuning_or_stopping") is False, "callback preview not zero-step evidence")


def verify_suite(report):
    require(report.get("status") == "PASSED" and report.get("scope") == "full_repository"
            and report.get("exit_code") == 0 and report.get("failed") == report.get("errors") == 0
            and isinstance(report.get("passed"), int) and report["passed"] >= 1862,
            "final full-suite receipt missing/not passed")
    return hashes(report.get("input_sha256"))


def verify_snapshots(engineering, receipt_path):
    for name, expected in hashes(engineering["input_sha256"]).items():
        p = Path(name)
        if p.is_relative_to(ROOT) and p.suffix in {".py", ".md", ".yaml", ".json"}:
            snapshot = Path(receipt_path).parent / "source_snapshot" / p.relative_to(ROOT)
            require(digest(snapshot) == expected, f"engineering snapshot differs: {p}")


def configurations(spec):
    require(pointer(spec["baseline"]) == BASE.resolve(), "wrong baseline")
    baseline = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    base = {k: v for k, v in baseline.items() if k != "positive_quality_transport"}
    require(base.get("method") == "p1" and base["train"].get("epochs") == 30
            and base["train"].get("seed") == 0 and base["train"].get("batch") == 8
            and base["train"].get("workers") == 0 and base["train"].get("amp") is True,
            "baseline protocol differs")
    require(set(spec["configs"]) == set(ARMS), "all three configs required")
    result = {}
    for arm in ARMS:
        config = yaml.safe_load(pointer(spec["configs"][arm]).read_text(encoding="utf-8"))
        expected = base | {"project": str(OUT), "name": arm, "bounded_retention": {"mode": arm}}
        # Resolve only the project spelling; all other training/config fields are fixed.
        require(path(config.get("project", "")) == OUT.resolve(), "independent output root required")
        require(config | {"project": str(OUT)} == expected, f"unexpected {arm} config change")
        result[arm] = expected
    return result


def image_label(image):
    parts = list(Path(image).parts)
    require("images" in parts, "image path has no images component")
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def legal_population(config, frozen):
    """Read only fixed dataset/list/metadata; return legal image/label paths.

    Labels/images are subsequently hashed, not decoded. Source-image references
    inside pseudo metadata are never followed. No aggregate cache is opened.
    """
    def bound(p):
        p = path(p)
        require(frozen.get(str(p)) == digest(p), f"unfrozen population index: {p}")
        return p
    dataset_path = bound(config["data"])
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    listing = path(dataset["train"])
    require(listing.suffix == ".txt" and "manifest" not in listing.name.lower(), "not a legal train list")
    metadata = path(dataset["p1_metadata"])
    require(metadata.name == "metadata.jsonl", "expected materialized pseudo metadata.jsonl")
    bound(listing); bound(metadata)
    images = [path(s) for s in listing.read_text(encoding="utf-8").splitlines() if s.strip()]
    human_images = (ROOT / "data/visdrone_area10/labeled/images/train").resolve()
    pseudo_images = (metadata.parent / "images/train").resolve()
    require(len(images) == len(set(images)) == 11310, "training inventory not 11310 unique images")
    require(all(p.suffix.lower() in IMAGE_SUFFIXES and p.parent in (human_images, pseudo_images)
                for p in images), "training image outside materialized human/pseudo population")
    human = {p for p in images if p.parent == human_images}
    pseudo = {p for p in images if p.parent == pseudo_images}
    require(len(human) == 6471 and len(pseudo) == 4839, "human/pseudo inventory differs")
    records = [json.loads(s) for s in metadata.read_text(encoding="utf-8").splitlines() if s.strip()]
    require(len(records) == len(pseudo) and {path(r["image"]) for r in records} == pseudo,
            "pseudo metadata identities differ")
    val = path(dataset["val"])
    require(val.name == "val" and val.parent.name == "images", "full val image directory required")
    val_images = sorted(p.resolve() for p in val.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    require(len(val_images) == 548, "full validation inventory not 548")
    all_images = images + val_images
    labels = [image_label(p).resolve() for p in all_images]
    return {dataset_path, listing, metadata, *all_images, *labels}


def required_code():
    return {*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").glob("*.py"),
            *(ROOT / "scripts").glob("*.py")}


def required_engineering_code():
    # The post-engineering runner/evaluator/diagnostic hooks belong to the final
    # full-suite freeze; the entire already-executed geometric implementation
    # must be present in the integration receipt itself.
    files = {p.resolve() for p in (ROOT / "src/regionssod/training").glob("bounded_retention*.py")
             if p.name not in {"bounded_retention_run_diagnostics.py", "bounded_retention_amp_audit.py"}}
    files.update(ROOT / p for p in ("src/regionssod/training/p1.py", "src/regionssod/data/annotations.py",
                                   "src/regionssod/data/partial.py", "scripts/check_bounded_retention_native_e.py"))
    return files


def installed_trainer_path():
    return Path(find_spec("ultralytics").origin).parent / "engine/trainer.py"


def required_amp_code():
    return {ROOT / "scripts/check_bounded_retention_amp.py", AMP_CONTRACT,
            ROOT / "src/regionssod/training/bounded_retention_amp_audit.py",
            ROOT / "src/regionssod/training/bounded_retention_run_diagnostics.py",
            Path(find_spec("torch").origin).parent / "amp/grad_scaler.py"}


def assert_finite(value, name):
    """Read-only check of unscaled observed values; never traverses .grad."""
    import torch
    if value is None:
        return  # native fitness/loss fields may not yet be populated
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{name}[{index}]")
    else:
        finite = bool(torch.isfinite(value.detach()).all()) if isinstance(value, torch.Tensor) else math.isfinite(float(value))
        if not finite:
            raise FloatingPointError(f"nonfinite {name}; automatic recovery forbidden")


def assert_actual_states_finite(trainer):
    """Inspect actual state before native EMA repair/nan_to_num serialization."""
    import torch
    ema = getattr(getattr(trainer, "ema", None), "ema", None)
    for label, module in (("model", getattr(trainer, "model", None)), ("EMA", ema)):
        if module is not None:
            for name, value in module.state_dict().items():
                if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
                    assert_finite(value, f"{label}.{name}")


def no_oom_retry(trainer):
    # Installed 8.4.172 checks >=3 before batch reduction/rebuild. Its successful
    # for/else resets this to zero; re-arm at every epoch start, not every batch.
    trainer._oom_retries = 3


def finite_batch_end(trainer):
    assert_finite(getattr(trainer, "loss", None), "unscaled loss")
    assert_finite(getattr(trainer, "loss_items", None), "unscaled loss_items")


def strict_nan_recovery(trainer, epoch):
    """Replacement, never a wrapper around the native checkpoint loader."""
    finite_batch_end(trainer)
    assert_finite(getattr(trainer, "fitness", None), "fitness")
    # Validation occurs after on_train_epoch_end. Recheck immediately before
    # native save_model as well, so validation cannot hide poisoned EMA state.
    assert_actual_states_finite(trainer)
    return False


def install_failure_guards(trainer):
    """Identical fault-only policy for original P1, candidate and control."""
    trainer._handle_nan_recovery = MethodType(strict_nan_recovery, trainer)
    trainer.add_callback("on_train_epoch_start", no_oom_retry)
    trainer.add_callback("on_train_batch_end", finite_batch_end)
    trainer.add_callback("on_train_epoch_end", assert_actual_states_finite)


def allowed_input(p, *, legal, pointers, model):
    """Deny unapproved data BEFORE hashing; historical nested hash maps stay opaque."""
    if "manifest" in p.name.lower() or p.suffix.lower() in {".cache", ".npy", ".npz", ".pkl", ".pickle"}:
        return False
    if p in legal or p in pointers or p == model:
        return True
    if p.is_relative_to(ROOT):
        rel = p.relative_to(ROOT)
        return ((rel.parts[0] in {"src", "scripts", "tests"} and p.suffix == ".py")
                or (rel.parts[0] in {"configs", "docs"} and p.suffix in {".yaml", ".yml", ".md"})
                or rel.as_posix() in {"pyproject.toml", "AGENTS.md"})
    # Installed runtime inputs are code/binaries, never arbitrary data or weights.
    return p.is_relative_to(Path(sys.prefix).resolve()) and p.suffix.lower() in {
        ".py", ".yaml", ".yml", ".pyd", ".dll", ".exe", ".so"}


def expected_args(config):
    from ultralytics.cfg import DEFAULT_CFG_DICT
    overrides = config["train"] | {"model": str(path(config["model"])), "data": str(path(config["data"])),
        "project": str(OUT), "name": config["name"], "exist_ok": False, "resume": False,
        "task": "detect", "mode": "train"}
    return canonical(DEFAULT_CFG_DICT | overrides | {"save_dir": str(OUT / config["name"])})


def preflight(spec_path, *, fresh=True):
    spec_path = path(spec_path)
    require(spec_path.suffix == ".json" and "manifest" not in spec_path.name.lower(), "invalid run spec path")
    spec = read(spec_path)
    require(spec.get("schema") == SCHEMA and spec.get("permit") == "FINAL_THREE_ARM_30_FROZEN",
            "final frozen run permit absent")
    require(spec.get("failure_policy") == FAILURE_POLICY, "shared fault-only guards must be declared in final freeze")
    if fresh and OUT.exists():
        raise FileExistsError("output already exists; no overwrite, retry or resume")
    configs = configurations(spec)
    require(spec.get("geometry_engineering", {}).get("sha256") == GEOMETRY_ENGINEERING_SHA,
            "G0 geometry SHA must match immutable STOP pin")
    gp, ep, sp, cp = (pointer(spec[k]) for k in ("geometry_engineering", "engineering", "full_suite", "contract"))
    require(gp == GEOMETRY_ENGINEERING.resolve() and ep == ENGINEERING.resolve(), "wrong joint integration receipts")
    geometry, engineering, suite = read(gp), read(ep), read(sp)
    verify_engineering(engineering, geometry)
    suite_hashes = verify_suite(suite)
    frozen = hashes(spec["input_sha256"])
    check_subset(engineering["input_sha256"], frozen)
    check_subset(geometry["input_sha256"], engineering["input_sha256"])
    require(hashes(engineering["input_sha256"]).get(str(gp)) == GEOMETRY_ENGINEERING_SHA,
            "AMP inputs do not bind parent STOP receipt")
    check_subset(suite_hashes, frozen)
    required = {p.resolve() for p in required_code()}
    required.update(ROOT / p for p in (
        "scripts/run_bounded_retention_continuous.py", "scripts/evaluate_bounded_retention.py",
        "src/regionssod/training/bounded_retention_run_diagnostics.py",
        "tests/test_bounded_retention_continuous_runner.py"))
    required.add(installed_trainer_path().resolve())
    require(required <= {Path(p) for p in suite_hashes}, "suite does not bind final code/tests")
    reports = {AUDIT.resolve(), gp, ep, sp,
        ROOT / "outputs/reliable_region_ranking_feasibility_20261105/results.json",
        ROOT / "outputs/reliable_region_ranking_feasibility_20261105/receipt.json",
        ROOT / "outputs/human_multiscale_observation_review_20261105/geometry.json",
        ROOT / "outputs/human_multiscale_observation_review_20261105/geometry_receipt.json"}
    pointers = reports | {cp, BASE.resolve(), *(preview_path(a) for a in ARMS),
                          *(path(x["path"]) for x in spec["configs"].values())}
    integration_required = required_engineering_code() | (reports - {gp, ep, sp}) | {BASE.resolve(), path(configs["reference"]["model"])}
    require(integration_required <= {Path(p) for p in hashes(geometry["input_sha256"])},
            "geometry integration does not bind executed source/reports/baseline/initial weights")
    require(required_amp_code() <= {Path(p) for p in hashes(engineering["input_sha256"])},
            "AMP integration does not bind new checker/helper/callback/scaler/contract")
    require(frozen.get(str(AUDIT.resolve())) == AUDIT_SHA, "pinned source audit missing")
    legal = legal_population(configs["reference"], frozen)
    model = path(configs["reference"]["model"])
    require(legal | pointers | required | {model} <= {Path(p) for p in frozen}, "incomplete frozen inputs")
    # Validate the entire read set first, before opening any inventory entry.
    for name in frozen:
        require(allowed_input(Path(name), legal=legal, pointers=pointers, model=model),
                f"forbidden/unclassified input: {name}")
    for name, expected in frozen.items():
        require(digest(name) == expected, f"frozen input changed: {name}")
    verify_snapshots(engineering, ep)
    verify_snapshots(geometry, gp)
    verify_previews(engineering)
    require(engineering["runtime"] == geometry["runtime"] == {"torch": version("torch"), "ultralytics": version("ultralytics")},
            "engineering runtime changed")
    require(version("ultralytics") == "8.4.172" and Path(sys.prefix).name == "yolo", "pinned yolo runtime required")
    evaluator_api()  # validate only; never invokes evaluation here
    return {"schema": SCHEMA, "spec": str(spec_path), "spec_sha256": digest(spec_path),
        "configs": configs, "expected_args": {a: expected_args(c) for a, c in configs.items()},
        "input_sha256": frozen, "epochs_per_arm": 30, "arms": list(ARMS),
        "failure_policy": canonical(FAILURE_POLICY),
        "evaluation": {"images": 548, "imgsz": 960}, "automatic_retry": False,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}


def check_epochs(run):
    with (Path(run) / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
    require(len(rows) == 30, "30 complete epoch rows required")
    require([float(r["epoch"]) for r in rows] == list(range(1, 31)), "epoch sequence differs")
    require(all(math.isfinite(float(v)) for r in rows for v in r.values()), "nonfinite/incomplete epoch CSV")
    return rows


def make_trainer(arm, config, *, classes=None, source=None):
    """Lazy constructor; reference is the original P1Trainer, never reference mode."""
    require(arm in ARMS, "unknown arm")
    if classes is None:
        from regionssod.training.p1 import P1Trainer
        from regionssod.training.bounded_retention_training_v6 import ContinuousNativeETrainer
        classes = (P1Trainer, ContinuousNativeETrainer)
    overrides = {k: v for k, v in expected_args(config).items() if k != "save_dir"}
    kwargs = {"overrides": overrides, "p1_options": config["p1"]}
    if arm == "reference":
        return classes[0](**kwargs)
    if source is None:
        from regionssod.training.bounded_retention_source import BoundSourceFactory, AUDIT as bound_audit, AUDIT_SHA as bound_sha
        require(bound_audit.resolve() == AUDIT.resolve() and bound_sha == AUDIT_SHA, "source constants changed")
        source = BoundSourceFactory()
    return classes[1](**kwargs, mode=arm, source_factory=source,
                      source_audit_path=AUDIT, source_audit_sha256=AUDIT_SHA)


def verify_dispatch():
    receipt, dispatch = read(OUT / "preflight.json"), read(OUT / "dispatch.json")
    require(dispatch["preflight_sha256"] == digest(OUT / "preflight.json"), "dispatch changed")
    require(digest(receipt["spec"]) == receipt["spec_sha256"], "frozen spec changed")
    replay = preflight(receipt["spec"], fresh=False)
    # Commit identity is recorded provenance; unchanged pinned inputs remain
    # valid across an unrelated commit. File hashes are the execution binding.
    replay["source_commit"] = receipt["source_commit"]
    require(replay == receipt, "preflight replay differs")
    return receipt, dispatch


def artifact_hashes(directory, *, exclude=()):
    return {str(p.resolve()): digest(p) for p in sorted(Path(directory).rglob("*"))
            if p.is_file() and p not in exclude}


def audit_arm(arm, receipt):
    run = OUT / arm
    claim, seal, exited = (read(OUT / f"{arm}_{name}.json") for name in ("claim", "seal", "exit"))
    process = read(OUT / f"{arm}_process.json")
    require(exited["exit_code"] == 0 and exited["pid"] == claim["pid"] and claim["arm"] == arm,
            "arm process did not finish successfully")
    require(process["pid"] == claim["pid"] and process["parent_pid"] == claim["parent_pid"]
            and process["command"][-2:] == ["--worker", arm], "process command/PID binding differs")
    require(claim["preflight_sha256"] == digest(OUT / "preflight.json")
            and seal["claim_sha256"] == digest(OUT / f"{arm}_claim.json"), "arm identity changed")
    require(seal["artifact_sha256"] == artifact_hashes(run), "arm output seal differs")
    for relative in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "experiment.json"):
        require(str((run / relative).resolve()) in seal["artifact_sha256"], "arm artifact missing")
    rows = check_epochs(run)
    actual = canonical(yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8")))
    require(actual == receipt["expected_args"][arm], "actual trainer args changed")
    experiment = read(run / "experiment.json")
    require(experiment["arm"] == arm and experiment["config"] == receipt["configs"][arm]
            and experiment["preflight_sha256"] == digest(OUT / "preflight.json")
            and experiment["input_sha256"] == receipt["input_sha256"]
            and experiment["trainer"] == ("P1Trainer" if arm == "reference" else "ContinuousNativeETrainer"),
            "experiment provenance differs")
    return {"epochs": len(rows), "best_sha256": digest(run / "weights/best.pt"),
            "last_sha256": digest(run / "weights/last.pt"), "pid": claim["pid"]}


def worker(arm):
    receipt, dispatch = verify_dispatch()
    state = read(OUT / "status.json")
    token = os.environ.pop(WORKER_TOKEN, "")
    require(os.getppid() == dispatch["parent_pid"] and bool(token)
            and hashlib.sha256(token.encode()).hexdigest() == dispatch["token_sha256"][arm]
            and state.get("stage") == "training" and state.get("arm") == arm,
            "worker needs matching live one-shot parent")
    for earlier in ARMS[:ARMS.index(arm)]:
        audit_arm(earlier, receipt)
    run = OUT / arm
    if run.exists():
        raise FileExistsError("arm directory already exists; no resume")
    claim = {"arm": arm, "pid": os.getpid(), "parent_pid": os.getppid(),
             "preflight_sha256": digest(OUT / "preflight.json"), "command": [sys.executable, *sys.argv]}
    write(OUT / f"{arm}_claim.json", claim, exclusive=True)
    # This import is intentionally lazy and independent of old runners.
    from regionssod.training.bounded_retention_run_diagnostics import make_callbacks
    trainer = make_trainer(arm, receipt["configs"][arm])
    require(Path(trainer.save_dir).resolve() == run.resolve(), "trainer destination changed")
    install_failure_guards(trainer)
    for event, callback in make_callbacks(run / "diagnostics", arm).items():
        trainer.add_callback(event, callback)
    write(run / "experiment.json", {"arm": arm, "trainer": type(trainer).__name__,
        "config": receipt["configs"][arm], "preflight_sha256": digest(OUT / "preflight.json"),
        "input_sha256": receipt["input_sha256"]}, exclusive=True)
    trainer.train()
    check_epochs(run)
    verify_dispatch()
    write(OUT / f"{arm}_seal.json", {"claim_sha256": digest(OUT / f"{arm}_claim.json"),
          "artifact_sha256": artifact_hashes(run)}, exclusive=True)


def execute_arms(receipt, *, popen=subprocess.Popen, evaluator=None, evaluation_audit=None):
    """One dispatch per arm, no retry. Evaluation starts only after all seals pass."""
    tokens = {a: secrets.token_hex(32) for a in ARMS}
    write(OUT / "dispatch.json", {"parent_pid": os.getpid(), "arms": list(ARMS),
        "preflight_sha256": digest(OUT / "preflight.json"),
        "token_sha256": {a: hashlib.sha256(t.encode()).hexdigest() for a, t in tokens.items()}}, exclusive=True)
    audits = {}
    for arm in ARMS:
        write(OUT / "status.json", {"stage": "training", "arm": arm, "epochs": 30, "automatic_retry": False})
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", arm]
        start = time.monotonic()
        with (OUT / f"{arm}_stdout.log").open("x", encoding="utf-8") as log:
            child = None
            try:
                child = popen(command, cwd=ROOT, env=os.environ | {WORKER_TOKEN: tokens[arm], "PYTHONDONTWRITEBYTECODE": "1"},
                              stdout=log, stderr=subprocess.STDOUT)
                write(OUT / f"{arm}_process.json", {"pid": child.pid, "parent_pid": os.getpid(), "command": command,
                      "started_utc": datetime.now(timezone.utc).isoformat()}, exclusive=True)
                code = child.wait()
            except BaseException as exc:
                try:
                    live = child is None or child.poll() is None
                except BaseException:
                    live = True  # unknown liveness must never authorize a terminal seal
                if live:
                    details = {"stage": "parent_interrupted_worker_live", "arm": arm,
                        "pid": child.pid if child is not None else None,
                        "parent_pid": os.getpid(), "command": command,
                        "automatic_retry": False, "dispatch_stopped": True,
                        "worker_termination_requested": False, "terminal": False,
                        "error": repr(exc)}
                    try:
                        write(OUT / "status.json", details)
                    finally:
                        # Preserve the live-worker signal even if recording the
                        # interruption itself fails. Do not terminate or seal.
                        raise WorkerStillLive(f"worker {details['pid']} retained/unknown; parent dispatch interrupted") from exc
                raise
        write(OUT / f"{arm}_exit.json", {"pid": child.pid, "exit_code": code,
              "seconds": time.monotonic() - start}, exclusive=True)
        require(code == 0, f"{arm} failed; automatic retry forbidden")
        audits[arm] = audit_arm(arm, receipt)
    write(OUT / "training_audit.json", {"accepted": True, "arms": audits}, exclusive=True)
    verify_dispatch()
    if evaluator is None or evaluation_audit is None:
        evaluator, evaluation_audit = evaluator_api()
    write(OUT / "status.json", {"stage": "evaluating", "automatic_retry": False})
    evaluator({a: OUT / a / "weights/best.pt" for a in ARMS}, OUT / "evaluation")
    evaluated = evaluation_audit(OUT / "evaluation")
    require(evaluated.get("audit_ok") is True and evaluated.get("numeric_replay") is True
            and evaluated.get("images") == 548 and set(evaluated.get("arms", [])) == set(ARMS)
            and evaluated.get("inference_calls") == 0
            and evaluated.get("external_data_or_weights_read") is False, "final evaluator audit failed")
    write(OUT / "evaluation_audit.json", evaluated, exclusive=True)
    # Evaluator may not mutate frozen training artifacts.
    for arm in ARMS:
        audit_arm(arm, receipt)
    write(OUT / "status.json", {"stage": "completed", "automatic_retry": False,
          "performance_decision": "see frozen contract and final evaluation; no early selection"})


def train(spec_path):
    receipt = preflight(spec_path)
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / "preflight.json", receipt, exclusive=True)
    live_worker = False
    try:
        execute_arms(receipt)
    except WorkerStillLive:
        live_worker = True
        raise
    except BaseException:
        write(OUT / "status.json", {"stage": "failed", "error": traceback.format_exc(), "automatic_retry": False})
        raise
    finally:
        if not live_worker:
            seal = OUT / "artifact_sha256.json"
            write(seal, artifact_hashes(OUT, exclude=(seal,)), exclusive=True)


def audit():
    status = read(OUT / "status.json")
    require(status["stage"] in {"completed", "failed"}, "not terminal; live worker must not be sealed")
    seal = OUT / "artifact_sha256.json"
    require(read(seal) == artifact_hashes(OUT, exclude=(seal,)), "terminal output seal differs")
    receipt, _ = verify_dispatch()
    if status["stage"] == "failed":
        return {"accepted": False, "terminal_evidence_verified": True, "automatic_retry": False}
    for arm in ARMS:
        audit_arm(arm, receipt)
    from evaluate_bounded_retention import audit as evaluation_audit
    require(evaluation_audit(OUT / "evaluation") == read(OUT / "evaluation_audit.json"), "evaluation replay differs")
    return {"accepted": True, "epochs_per_arm": 30, "automatic_retry": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.audit:
        print(json.dumps(audit(), ensure_ascii=False))
    else:
        if args.spec is None:
            parser.error("--spec is required")
        if args.train:
            train(args.spec)
        else:
            print(json.dumps(preflight(args.spec), ensure_ascii=False))


if __name__ == "__main__":
    main()
