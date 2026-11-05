"""Independent V6 native-E integration; zero optimizer, no training/retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from itertools import chain
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml
import ultralytics
from ultralytics.utils.loss import DFLoss, BboxLoss

from regionssod.training.bounded_retention import UnsupportedObservation
from regionssod.training.bounded_retention_objective_v6 import current_unknown, continuous_support, partial_dfl, ContinuousNativeELoss
from regionssod.training.bounded_retention_source import BoundSourceFactory, AUDIT, AUDIT_SHA
from regionssod.training.bounded_retention_training_v6 import ContinuousNativeETrainer
from ultralytics.utils.tal import make_anchors, TaskAlignedAssigner
from ultralytics.utils.ops import xywh2xyxy


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_inputs(report, out):
    for filename, expected in report["input_sha256"].items():
        path = Path(filename)
        if path.is_relative_to(ROOT) and path.suffix in {".py", ".md", ".yaml", ".json"}:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("input changed during engineering snapshot")
            target = out / "source_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)


def portable(record, box):
    return {"source_object_id": record["source_object_id"], "occurrence_id": record["occurrence_id"],
        "source_xywh_tokens": list(record["physical"].tokens), "nominal_output_xyxy": box.tolist(),
        "original_unknown": record["original_unknown"].tolist(), "affine": record["affine"].tolist(),
        "affine_bounds": record["affine_bounds"].tolist(),
        "augmentation_clip_window_bounds": record["augmentation_clip_window_bounds"].tolist(),
        "physical_visible_bounds": record["physical"].visible_edge_bounds.tolist()}


def preflight(batch, model, report, source_factory, partial_loss=partial_dfl):
    """Continuous-domain/flat-density coverage before a real detector forward."""
    height, width = batch["img"].shape[-2:]
    stride = model.stride.cpu()
    feats = [torch.empty(1, 1, height // int(s), width // int(s)) for s in stride]
    anchors, strides = make_anchors(feats, stride, .5)
    anchors *= strides
    report["geometry_preflight"] = {"partial_objects_checked": 0, "supported_objects": 0,
        "empty_anchor_events": 0, "effective_training_examples": "not established", "active_dimensions": {}}
    stats = report["geometry_preflight"]
    boxes = xywh2xyxy(batch["bboxes"].float() * torch.tensor([width, height, width, height]))
    for i, (record, box) in enumerate(zip(batch["retention_observations"], boxes)):
        if record is None or not current_unknown(record, width, height)[0].any():
            continue
        stats["partial_objects_checked"] += 1
        dimension = str(int(current_unknown(record, width, height)[0].sum()))
        stats["active_dimensions"][dimension] = stats["active_dimensions"].get(dimension, 0) + 1
        visible = TaskAlignedAssigner(stride=stride.tolist()).select_candidates_in_gts(anchors, box[None, None], torch.ones(1, 1, 1)).reshape(-1).bool()
        indices = visible.nonzero().flatten()
        witness = {**portable(record, box), "batch_row": i, "batch_image": int(batch["batch_idx"][i]),
                   "native_candidate_anchor_count": len(indices),
                   "source_binding": source_factory.bindings[record["source_object_id"]],
                   "source_aliases": record["source_aliases"]}
        if not len(indices):
            report["blocking_witness"] = witness
            raise UnsupportedObservation("partially unknown object has no native TAL anchor")
        report["current_geometry_witness"] = witness
        flat = torch.zeros(len(indices), 4, 16)
        support = continuous_support(record, box, anchors[indices], strides[indices], (width, height), flat)
        # Flat logits certify volume/geometry, not the later prediction-dependent
        # mass/gradient calculation. No detector is called in this pass.
        for mode in ("candidate", "control"):
            value = partial_loss(flat[0], support, 0, mode)
            if not torch.isfinite(value):
                raise UnsupportedObservation("nonfinite continuous flat-density preflight")
        count = support["valid"].sum(1)
        stats["empty_anchor_events"] += int((count == 0).sum())
        if not count.any():
            report["blocking_witness"] = {**witness, "anchors": anchors[indices].tolist(),
                "strides": strides[indices].tolist(), "unclamped_nominal_distances": support["target"].tolist(),
                "certified_continuous_witness_counts": count.tolist()}
            raise UnsupportedObservation("partially unknown object has no certified continuous anchor event")
        stats["supported_objects"] += 1


def main(*, Trainer=ContinuousNativeETrainer, Loss=ContinuousNativeELoss,
         partial_loss=partial_dfl, default_name="native_e_integration_01",
         objective="V6_native_E_same_continuous_density", extra_sources=(), numeric_audit=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=default_name)
    parser.add_argument("--scan-all-batches", action="store_true")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("fresh single directory name required")
    out = ROOT / "outputs/bounded_retention_continuous_20261105" / args.name
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING",
        "pid": os.getpid(), "command": [sys.executable, *sys.argv],
        "optimizer_steps": 0, "training_started": False, "objective": objective, "control": "globally_certified_native_continuous_point_DFL", "model_forward_calls": 0, "model_forward_attempts": 0,
        "forward_count_scope": "explicit augmented-batch calls; excludes native model-constructor stride probe",
        "validation_inference_calls": 0, "stage": "source_binding", "input_sha256": {}}
    code = [Path(__file__).resolve(), *map(Path, extra_sources), *sorted((ROOT / "src/regionssod/training").glob("bounded_retention*.py")),
            ROOT / "src/regionssod/training/p1.py", *sorted((ROOT / "docs/paper/candidates/bounded_retention").glob("*.md")),
            Path(inspect.getfile(DFLoss)), Path(inspect.getfile(TaskAlignedAssigner))]
    report["runtime"] = {"ultralytics": ultralytics.__version__, "torch": torch.__version__}
    report["input_sha256"] = {str(p): fingerprint(p) for p in code}
    (out / "started.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    exit_code = 0
    try:
        factory = BoundSourceFactory()
        report["input_sha256"].update(factory.reads)
        report["source_premises"] = factory.premises
        config_path = ROOT / "configs/candidates/positive_quality_transport_v2/reference30.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        report["input_sha256"][str(config_path)] = fingerprint(config_path)
        report["input_sha256"][config["model"]] = fingerprint(Path(config["model"]))
        snapshot_inputs(report, out)
        report["stage"] = "real_trainer_constructor"
        trainer = Trainer(mode="candidate", source_factory=factory,
            source_audit_path=AUDIT, source_audit_sha256=AUDIT_SHA, p1_options=config["p1"],
            overrides={**config["train"], "model": config["model"], "data": str(ROOT / config["data"]),
                "project": str(out), "name": "trainer", "save": False, "plots": False}, _callbacks={})
        report["stage"] = "real_model_initialization"
        report["model_initialization_may_run_native_stride_probe"] = True
        trainer.setup_model()
        trainer.model = trainer.model.to(trainer.device)
        trainer.set_model_attributes()
        report["stage"] = "real_train_dataloader"
        train_loader = trainer.get_dataloader(trainer.data["train"], batch_size=8, rank=-1, mode="train")
        if len(train_loader.dataset) != 11310:
            raise ValueError("fixed training set is not 11310 images")
        report["source_factory_counts"] = dict(factory.counts)
        bindings = train_loader.dataset.human_labels
        report["human_row_binding_counts"] = dict(bindings.counts)
        report["native_duplicate_aliases"] = bindings.duplicate_aliases
        if (bindings.counts["raw_rows"] != 105058
                or bindings.counts["retained_rows"] != factory.counts["objects"]
                or bindings.counts["retained_rows"] + bindings.counts["native_duplicate_rows"] != 105058):
            raise ValueError("audited human population count differs")
        report["stage"] = "real_val_dataloader"
        val_loader = trainer.get_dataloader(trainer.data["val"], batch_size=8, rank=-1, mode="val")
        if len(val_loader.dataset) != 548:
            raise ValueError("full validation set is not 548 images")
        report["dataloaders"] = {"train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
            "workers": 0, "image_size": 640, "validation_inference": False}
        report["stage"] = "first_real_augmented_batch"
        iterator = iter(train_loader)
        batch = next(iterator)
        report["batch_positive_rows"] = len(batch["cls"])
        report["stage"] = "real_augmented_geometry_preflight"
        report["geometry_batch_checks"] = []
        selected_dimension = -1
        def check_stream(stream, phase):
            nonlocal batch, selected_dimension
            for batch_index, geometry_batch in enumerate(stream):
                report["geometry_phase"], report["geometry_batch_index"] = phase, batch_index
                preflight(geometry_batch, trainer.model, report, factory, partial_loss)
                report["geometry_batch_checks"].append({"phase": phase, **report["geometry_preflight"]})
                dimension = max(map(int, report["geometry_preflight"]["active_dimensions"]), default=0)
                if dimension > selected_dimension:
                    # One detector forward, on the first batch attaining the
                    # maximum observed U dimension. This is an engineering
                    # stress choice from public metadata, never a val selection.
                    batch, selected_dimension = geometry_batch, dimension
                    report["forward_batch_selection"] = {"rule": "first_maximum_active_dimension",
                        "phase": phase, "batch_index": batch_index, "active_dimension": dimension}
                if batch_index % 100 == 0:
                    print(json.dumps({"geometry_phase": phase, "geometry_batches_checked": batch_index + 1,
                                      "optimizer_steps": 0}), flush=True)
        check_stream(chain([batch], iterator) if args.scan_all_batches else [batch], "mosaic_open")
        if args.scan_all_batches and config["train"].get("close_mosaic", 0):
            # The final ten baseline epochs use this actual native transition.
            # Inspect both augmentation phases, with no detector/optimizer calls.
            trainer.train_loader = train_loader
            trainer._close_dataloader_mosaic()
            train_loader.reset()
            check_stream(iter(train_loader), "mosaic_closed")
        report["stage"] = "real_model_amp_forward_backward"
        report["batch_positive_rows"] = len(batch["cls"])
        def parameter_digest():
            h = hashlib.sha256()
            for name, p in trainer.model.named_parameters():
                h.update(name.encode("utf-8"))
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        report["parameters_before_forward_sha256"] = parameter_digest()
        batch = trainer.preprocess_batch(batch)
        trainer.model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            report["model_forward_attempts"] += 1
            preds = trainer.model(batch["img"])
            report["model_forward_calls"] += 1
            candidate = Loss(trainer.model)
            loss, _ = candidate(preds, batch)
            if not torch.isfinite(loss).all():
                raise ValueError("nonfinite full candidate loss")
        control = Loss(trainer.model)
        control.mode = "control"
        reference = Loss(trainer.model)
        reference.mode = "reference"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            control_loss, _ = control(preds, batch)
            reference_loss, _ = reference(preds, batch)
        if numeric_audit is not None:
            numeric_audit(candidate, preds, report)
        scaler = torch.amp.GradScaler("cuda")
        amp_scale = scaler.get_scale()
        candidate_grad = torch.autograd.grad(scaler.scale(loss.sum()), preds["boxes"], retain_graph=True)[0].double() / amp_scale
        control_grad = torch.autograd.grad(scaler.scale(control_loss.sum()), preds["boxes"], retain_graph=True)[0].double() / amp_scale
        reference_grad = torch.autograd.grad(scaler.scale(reference_loss.sum()), preds["boxes"], retain_graph=True)[0].double() / amp_scale
        report["amp_initial_gradient_scale"] = amp_scale
        if not torch.isfinite(reference_loss).all() or not torch.isfinite(reference_grad).all():
            raise ValueError("nonfinite original P1 loss/gradient")
        if not torch.isfinite(control_loss).all() or not torch.isfinite(control_grad).all():
            raise ValueError("nonfinite real control loss/gradient")
        if not torch.equal(loss[:2], control_loss[:2]):
            raise ValueError("candidate/control CIoU or classification mismatch")
        if torch.equal(candidate_grad, control_grad):
            raise ValueError("real candidate/control gradients cannot be distinguished")
        report["real_control_loss"] = control_loss.detach().cpu().tolist()
        report["real_reference_loss"] = reference_loss.detach().cpu().tolist()
        report["real_control_gradient_difference_l1"] = float((candidate_grad-control_grad).abs().sum())
        report["real_shared_ciou_classification"] = True
        scaler.scale(loss.sum()).backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainer.model.parameters()):
            raise ValueError("nonfinite full candidate model gradients")
        report["full_model_loss"] = loss.detach().cpu().tolist()
        report["loss_diagnostics"] = candidate.diagnostics
        report["parameters_after_backward_sha256"] = parameter_digest()
        report["parameter_hash_unchanged"] = report["parameters_before_forward_sha256"] == report["parameters_after_backward_sha256"]
        if not report["parameter_hash_unchanged"]:
            raise ValueError("trainable parameters changed in zero-optimizer check")
        if getattr(trainer, "optimizer", None) is not None:
            raise AssertionError("zero-step check unexpectedly created an optimizer")
        for filename, expected in report["input_sha256"].items():
            if fingerprint(Path(filename)) != expected:
                raise ValueError(f"input changed during integration audit: {filename}")
        report["status"] = "INTEGRATION_COMPONENT_PASSED_NOT_STUDENT_PERMIT"
        report["all_stream_geometry_checked"] = bool(args.scan_all_batches)
    except Exception as exc:
        report["status"] = "STOP_BEFORE_STUDENT_TRAINING"
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "diagnostics": getattr(exc, "diagnostics", None)}
        if "current_geometry_witness" in report:
            report["blocking_witness"] = report["current_geometry_witness"]
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        exit_code = 2
    finally:
        if "factory" in locals():
            report["source_factory_counts"] = dict(factory.counts)
        report["elapsed_seconds"] = time.perf_counter() - start
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("status", "stage", "optimizer_steps", "model_forward_calls", "elapsed_seconds")}, ensure_ascii=False))
        print(str(out / "result.json"))
    return exit_code


def audit_actual_logits(criterion, preds, report):
    audits = []
    for (image, _), support in criterion.supports.items():
        index = support["indices"][0]
        z = preds["boxes"][image, :, index].detach().double().reshape(4, 16).requires_grad_()
        # Auditing is a separate draw/calculation, never the recorded training
        # draw. The remaining geometry/certificates are read-only here.
        audit_support = {**support, "density_diagnostics": {}}
        value = partial_dfl(z, audit_support, 0, "candidate", rng_seed=0, audit=True)
        gradient = torch.autograd.grad(value, z)[0]
        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            raise ValueError("independent audit of actual head values failed")
        audits.append({"identity": support["identity"], "dimension": support["domain"].dimension,
                       "diagnostics": audit_support["density_diagnostics"][0]})
    if not audits:
        raise ValueError("actual first batch has no partial observation for numerical audit")
    report["actual_head_numeric_audits"] = audits


if __name__ == "__main__":
    raise SystemExit(main(numeric_audit=audit_actual_logits))
