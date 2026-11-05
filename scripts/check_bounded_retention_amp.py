"""Independent AMP audit; inherit verified V6 geometry, replay one fixed batch.

No optimizer, student training, validation inference, or failed-run restart.
The former failed receipt is immutable. One new forward diagnoses AMP scaling.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import ultralytics
import yaml

from check_bounded_retention_native_e import audit_actual_logits, snapshot_inputs
from regionssod.training.bounded_retention_objective_v6 import ContinuousNativeELoss, current_unknown
from regionssod.training.bounded_retention_source import BoundSourceFactory, AUDIT, AUDIT_SHA
from regionssod.training.bounded_retention_training_v6 import ContinuousNativeETrainer
from regionssod.training.p1 import P1DetectionLoss

PARENT = ROOT / "outputs/bounded_retention_continuous_20261105/native_e_integration_01/result.json"
PARENT_SHA = "39c03aa3a4243a0f3169abe174c4701d964f905df749c208148fc0bccee7d259"
CONFIG = ROOT / "configs/candidates/positive_quality_transport_v2/reference30.yaml"
ARMS = ("reference", "candidate", "control")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_parent(parent=PARENT):
    if sha(parent) != PARENT_SHA:
        raise ValueError("frozen V6 stop receipt changed")
    report = json.loads(Path(parent).read_text(encoding="utf-8"))
    if (report["status"] != "STOP_BEFORE_STUDENT_TRAINING"
            or report["error"]["message"] != "nonfinite original P1 loss/gradient"
            or report["stage"] != "real_model_amp_forward_backward"
            or report["optimizer_steps"] != 0 or report["model_forward_calls"] != 1
            or report["validation_inference_calls"] != 0):
        raise ValueError("unexpected parent terminal state")
    rows = report["geometry_batch_checks"]
    if Counter(r["phase"] for r in rows) != {"mosaic_open": 1414, "mosaic_closed": 1414}:
        raise ValueError("parent does not cover both complete augmentation streams")
    if any(r["supported_objects"] != r["partial_objects_checked"] or r["empty_anchor_events"] for r in rows):
        raise ValueError("parent geometry has an unsupported object")
    for name, expected in report["input_sha256"].items():
        p = Path(name)
        if sha(p) != expected:
            raise ValueError(f"parent input changed: {p}")
        if p.is_relative_to(ROOT) and p.suffix in {".py", ".md", ".yaml", ".json"}:
            if sha(Path(parent).parent / "source_snapshot" / p.relative_to(ROOT)) != expected:
                raise ValueError(f"parent source snapshot changed: {p}")
    return report


def parameter_digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_fingerprints(mapping):
    result = {}
    for name, value in mapping.items():
        if isinstance(value, torch.Tensor):
            h = hashlib.sha256()
            h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
            result[name] = h.hexdigest()
    return result


def replay_streams(loader, trainer, parent, report):
    selected, maximum, position = None, -1, None
    reference = iter(parent["geometry_batch_checks"])
    report["replayed_batches"] = {}
    for phase in ("mosaic_open", "mosaic_closed"):
        if phase == "mosaic_closed":
            trainer.train_loader = loader
            trainer._close_dataloader_mosaic()
            loader.reset()
        count = 0
        for batch_index, batch in enumerate(loader):
            count += 1
            height, width = batch["img"].shape[-2:]
            dimensions = Counter()
            for record in batch["retention_observations"]:
                if record is not None:
                    n = int(current_unknown(record, width, height)[0].sum())
                    if n:
                        dimensions[str(n)] += 1
            expected = next(reference)
            if (expected["phase"] != phase or dict(dimensions) != expected["active_dimensions"]
                    or sum(dimensions.values()) != expected["partial_objects_checked"]):
                raise ValueError(f"transport activity replay differs at {phase}/{batch_index}")
            dimension = max(map(int, dimensions), default=0)
            if dimension > maximum:
                selected, maximum = batch, dimension
                position = dict(rule="first_maximum_active_dimension", phase=phase,
                                batch_index=batch_index, active_dimension=dimension)
            if batch_index % 300 == 0:
                print(json.dumps(dict(stage="transport_replay", phase=phase,
                                      batches=count, optimizer_steps=0)), flush=True)
        report["replayed_batches"][phase] = count
    if next(reference, None) is not None or report["replayed_batches"] != {"mosaic_open": 1414, "mosaic_closed": 1414}:
        raise ValueError("transport replay incomplete")
    if position != parent["forward_batch_selection"]:
        raise ValueError("fixed maximum-U pressure batch changed")
    report["forward_batch_selection"] = position
    report["batch_tensor_sha256"] = tensor_fingerprints(selected)
    report["batch_positive_rows"] = len(selected["cls"])
    report["batch_identity"] = [None if r is None else [r["source_object_id"], r["occurrence_id"]]
                                for r in selected["retention_observations"]]
    return selected


def preview_callbacks(criteria, out, report):
    from regionssod.training.bounded_retention_run_diagnostics import make_callbacks
    previews = {}
    for arm, criterion in criteria.items():
        directory = out / "zero_step_diagnostic_preview" / arm
        context = SimpleNamespace(epoch=0, rank=-1, save_dir=out,
                                  model=SimpleNamespace(criterion=criterion))
        callbacks = make_callbacks(directory, arm)
        for event in ("on_train_epoch_start", "on_train_batch_end", "on_train_epoch_end"):
            callbacks[event](context)
        files = list(directory.glob("*.json"))
        if len(files) != 1:
            raise ValueError("diagnostic preview must produce one zero-step batch report")
        previews[arm] = {"path": str(files[0]), "sha256": sha(files[0])}
    report["diagnostic_previews"] = previews
    report["diagnostic_preview_scope"] = "one real criterion sidecar; synthetic callback context; ZERO student epochs"


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="native_amp_audit_01")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("fresh single-directory audit name required")
    out = ROOT / "outputs/bounded_retention_continuous_20261105" / args.name
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status="RUNNING", stage="verify_parent", started_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), command=[sys.executable, *sys.argv], optimizer_steps=0, training_started=False,
        validation_inference_calls=0, model_forward_calls=0, model_forward_attempts=0,
        forward_count_scope="explicit augmented batch only; excludes native constructor stride probe",
        model_initialization_may_run_native_stride_probe=True,
        native_train_forward_may_update_BN_buffers=True,
        objective="V6_native_E_same_continuous_density", audit_protocol="native_dynamic_scaling_v1",
        parent_geometry_receipt=str(PARENT), parent_geometry_sha256=PARENT_SHA,
        geometry_scope="inherited verified full V6 scan; replay does not rerun flat-density numerical checks",
        runtime={"torch": torch.__version__, "ultralytics": ultralytics.__version__})
    code = [Path(__file__).resolve(), *sorted((ROOT / "src/regionssod/training").glob("bounded_retention*.py")),
            ROOT / "docs/paper/candidates/bounded_retention/amp_audit_revision_contract.md",
            Path(torch.amp.grad_scaler.__file__)]
    report["input_sha256"] = {str(p): sha(p) for p in code}
    (out / "started.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    exit_code = 2
    try:
        parent = verify_parent()
        report["input_sha256"].update(parent["input_sha256"])
        report["input_sha256"][str(PARENT)] = PARENT_SHA
        factory = BoundSourceFactory()
        report["input_sha256"].update(factory.reads)
        snapshot_inputs(report, out)
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        report["stage"] = "real_trainer_and_loader_replay"
        trainer = ContinuousNativeETrainer(mode="candidate", source_factory=factory,
            source_audit_path=AUDIT, source_audit_sha256=AUDIT_SHA, p1_options=config["p1"],
            overrides={**config["train"], "model": config["model"], "data": str(ROOT / config["data"]),
                "project": str(out), "name": "trainer", "save": False, "plots": False}, _callbacks={})
        trainer.setup_model()
        trainer.model = trainer.model.to(trainer.device)
        trainer.set_model_attributes()
        train_loader = trainer.get_dataloader(trainer.data["train"], batch_size=8, rank=-1, mode="train")
        val_loader = trainer.get_dataloader(trainer.data["val"], batch_size=8, rank=-1, mode="val")
        if (len(train_loader.dataset), len(val_loader.dataset)) != (11310, 548):
            raise ValueError("actual loader population changed")
        report["dataloaders"] = dict(train_images=11310, val_images=548, workers=0, image_size=640,
                                      validation_inference=False)
        if dict(factory.counts) != parent["source_factory_counts"]:
            raise ValueError("source binding population changed")
        report["source_factory_counts"] = dict(factory.counts)
        report["stage"] = "real_transport_replay"
        batch = replay_streams(train_loader, trainer, parent, report)
        report["parameters_before_forward_sha256"] = parameter_digest(trainer.model)
        if report["parameters_before_forward_sha256"] != parent["parameters_before_forward_sha256"]:
            raise ValueError("initial model parameters differ from parent check")
        batch = trainer.preprocess_batch(batch)
        trainer.model.train()
        report["stage"] = "real_model_amp_forward_backward"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            report["model_forward_attempts"] += 1
            preds = trainer.model(batch["img"])
            report["model_forward_calls"] += 1
            candidate, control = ContinuousNativeELoss(trainer.model), ContinuousNativeELoss(trainer.model)
            control.mode = "control"
            reference = P1DetectionLoss(trainer.model)
            criteria = dict(reference=reference, candidate=candidate, control=control)
            losses = {a: criteria[a](preds, batch)[0] for a in ("candidate", "control", "reference")}
            legacy = ContinuousNativeELoss(trainer.model)
            legacy.mode = "reference"
            legacy_loss = legacy(preds, batch)[0]
        report["prediction_tensor_sha256"] = tensor_fingerprints(preds)
        report["unscaled_losses"] = {a: v.detach().cpu().tolist() for a, v in losses.items()}
        if any(not torch.isfinite(v).all() for v in losses.values()):
            raise ValueError("nonfinite unscaled loss; changing gradient scale cannot qualify it")
        report["legacy_reference_value_equal"] = torch.equal(legacy_loss, losses["reference"])
        if not report["legacy_reference_value_equal"]:
            raise ValueError("legacy reference wrapper and original P1 loss differ")
        report["real_shared_ciou_classification"] = torch.equal(losses["candidate"][:2], losses["control"][:2])
        if not report["real_shared_ciou_classification"]:
            raise ValueError("candidate/control shared losses differ")
        audit_actual_logits(candidate, preds, report)
        report["parent_actual_head_numeric_audits_equal"] = json_safe(report["actual_head_numeric_audits"]) == parent["actual_head_numeric_audits"]
        if not report["parent_actual_head_numeric_audits_equal"]:
            raise ValueError("actual partial-head numerical receipts differ from parent")
        from regionssod.training.bounded_retention_amp_audit import audit_backward
        result = audit_backward(losses, preds["boxes"], trainer.model, report)
        grads, scale = result["head_gradients"], result["scale"]
        legacy_scaler = torch.amp.GradScaler("cuda", init_scale=scale)
        legacy_grad = torch.autograd.grad(legacy_scaler.scale(legacy_loss.sum()), preds["boxes"], retain_graph=True)[0].double()/scale
        report["legacy_reference_gradient_equal"] = torch.equal(legacy_grad, grads["reference"])
        if not report["legacy_reference_gradient_equal"]:
            raise ValueError("legacy reference wrapper and original P1 gradient differ")
        difference = (grads["candidate"]-grads["control"]).abs().sum()
        if not torch.isfinite(difference) or float(difference) <= 0:
            raise ValueError("candidate/control gradients not distinguishable")
        report["real_control_gradient_difference_l1"] = float(difference)
        preview_callbacks(criteria, out, report)
        report["parameters_after_backward_sha256"] = parameter_digest(trainer.model)
        report["parameter_hash_unchanged"] = report["parameters_before_forward_sha256"] == report["parameters_after_backward_sha256"]
        if not report["parameter_hash_unchanged"] or getattr(trainer, "optimizer", None) is not None:
            raise ValueError("zero-optimizer parameter-preserving contract violated")
        for name, expected in report["input_sha256"].items():
            if sha(name) != expected:
                raise ValueError(f"input changed during audit: {name}")
        report["status"] = "AMP_INTEGRATION_COMPONENT_PASSED_NOT_STUDENT_PERMIT"
        exit_code = 0
    except Exception as exc:
        report["status"] = "STOP_BEFORE_STUDENT_TRAINING"
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "diagnostics": getattr(exc, "diagnostics", None)}
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        report["elapsed_seconds"] = time.monotonic()-start
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "result.json").write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False,
                                                    allow_nan=False, default=str), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("status", "stage", "optimizer_steps", "model_forward_calls", "elapsed_seconds")}))
        print(str(out / "result.json"))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
