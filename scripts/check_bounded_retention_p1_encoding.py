"""Fresh, zero-optimizer integration attempt. Never launches/retries training."""
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
from regionssod.training.bounded_retention_objective_v3 import current_unknown, support_geometry, P1EncodingLoss
from regionssod.training.bounded_retention_source import BoundSourceFactory, AUDIT, AUDIT_SHA
from regionssod.training.bounded_retention_training_v3 import P1EncodingTrainer
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


def preflight(batch, model, report, source_factory):
    """Same native-P1 pool/event policy before a real detector forward."""
    height, width = batch["img"].shape[-2:]
    stride = model.stride.cpu()
    feats = [torch.empty(1, 1, height // int(s), width // int(s)) for s in stride]
    anchors, strides = make_anchors(feats, stride, .5)
    anchors *= strides
    report["geometry_preflight"] = {"partial_objects_checked": 0, "supported_objects": 0,
        "empty_anchor_events": 0, "effective_training_examples": "not established"}
    stats = report["geometry_preflight"]
    boxes = xywh2xyxy(batch["bboxes"].float() * torch.tensor([width, height, width, height]))
    for i, (record, box) in enumerate(zip(batch["retention_observations"], boxes)):
        if record is None or not current_unknown(record, width, height)[0].any():
            continue
        stats["partial_objects_checked"] += 1
        visible = TaskAlignedAssigner(stride=stride.tolist()).select_candidates_in_gts(anchors, box[None, None], torch.ones(1, 1, 1)).reshape(-1).bool()
        indices = visible.nonzero().flatten()
        witness = {**portable(record, box), "batch_row": i, "batch_image": int(batch["batch_idx"][i]),
                   "native_candidate_anchor_count": len(indices),
                   "source_binding": source_factory.bindings[record["source_object_id"]],
                   "source_aliases": record["source_aliases"]}
        if not len(indices):
            report["blocking_witness"] = witness
            raise UnsupportedObservation("partially unknown object has no native TAL anchor")
        support = support_geometry(record, box, anchors[indices], strides[indices], (width, height))
        count = support["valid"].sum(1)
        stats["empty_anchor_events"] += int((count == 0).sum())
        if not count.any():
            report["blocking_witness"] = {**witness, "anchors": anchors[indices].tolist(),
                "strides": strides[indices].tolist(), "unclamped_nominal_distances": support["target"].tolist(),
                "legal_atom_counts": count.tolist()}
            raise UnsupportedObservation("partially unknown object has no legal discrete anchor event")
        stats["supported_objects"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="p1_encoding_integration_01")
    parser.add_argument("--scan-all-batches", action="store_true")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("fresh single directory name required")
    out = ROOT / "outputs/bounded_retention_p1_encoding_review_20261105" / args.name
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING",
        "pid": os.getpid(), "command": [sys.executable, *sys.argv],
        "optimizer_steps": 0, "training_started": False, "model_forward_calls": 0, "model_forward_attempts": 0,
        "forward_count_scope": "explicit augmented-batch calls; excludes native model-constructor stride probe",
        "validation_inference_calls": 0, "stage": "source_binding", "input_sha256": {}}
    code = [Path(__file__).resolve(), *sorted((ROOT / "src/regionssod/training").glob("bounded_retention*.py")),
            ROOT / "src/regionssod/training/p1.py", *sorted((ROOT / "docs/paper/candidates/bounded_retention").glob("*.md")),
            Path(inspect.getfile(DFLoss)), Path(inspect.getfile(TaskAlignedAssigner))]
    report["runtime"] = {"ultralytics": ultralytics.__version__, "torch": torch.__version__}
    report["input_sha256"] = {str(p): fingerprint(p) for p in code}
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
        trainer = P1EncodingTrainer(mode="candidate", source_factory=factory,
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
        stream = chain([batch], iterator) if args.scan_all_batches else [batch]
        for batch_index, geometry_batch in enumerate(stream):
            report["geometry_batch_index"] = batch_index
            preflight(geometry_batch, trainer.model, report, factory)
            report["geometry_batch_checks"].append(dict(report["geometry_preflight"]))
            if batch_index % 100 == 0:
                print(json.dumps({"geometry_batches_checked": batch_index + 1, "optimizer_steps": 0}), flush=True)
        report["stage"] = "real_model_amp_forward_backward"
        batch = trainer.preprocess_batch(batch)
        trainer.model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            report["model_forward_attempts"] += 1
            preds = trainer.model(batch["img"])
            report["model_forward_calls"] += 1
            candidate = P1EncodingLoss(trainer.model)
            loss, _ = candidate(preds, batch)
            if not torch.isfinite(loss).all():
                raise ValueError("nonfinite full candidate loss")
        control = P1EncodingLoss(trainer.model)
        control.mode = "control"
        reference = P1EncodingLoss(trainer.model)
        reference.mode = "reference"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            control_loss, _ = control(preds, batch)
            reference_loss, _ = reference(preds, batch)
        candidate_grad = torch.autograd.grad(loss.sum(), preds["boxes"], retain_graph=True)[0]
        control_grad = torch.autograd.grad(control_loss.sum(), preds["boxes"], retain_graph=True)[0]
        reference_grad = torch.autograd.grad(reference_loss.sum(), preds["boxes"], retain_graph=True)[0]
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
        loss.sum().backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainer.model.parameters()):
            raise ValueError("nonfinite full candidate model gradients")
        report["full_model_loss"] = loss.detach().cpu().tolist()
        report["loss_diagnostics"] = candidate.diagnostics
        if trainer.optimizer is not None:
            raise AssertionError("zero-step check unexpectedly created an optimizer")
        report["status"] = "INTEGRATION_COMPONENT_PASSED_NOT_STUDENT_PERMIT"
    except Exception as exc:
        report["status"] = "STOP_BEFORE_STUDENT_TRAINING"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        exit_code = 2
    finally:
        if "factory" in locals():
            report["source_factory_counts"] = dict(factory.counts)
        report["elapsed_seconds"] = time.perf_counter() - start
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("status", "stage", "optimizer_steps", "model_forward_calls", "elapsed_seconds")}, ensure_ascii=False))
        print(str(out / "result.json"))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
