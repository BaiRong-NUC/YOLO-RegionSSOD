"""Permute human positive BCE targets by detached geometry; keep frozen P1 TAL.

This is a bounded experimental adapter, not a new quality-estimation method.
The original target tensor still feeds TAL normalization, box and DFL unchanged.
"""
from __future__ import annotations

from functools import partial
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou

from .p1 import P1DetectionLoss, P1DetectionModel, P1Trainer
from .reliable_competitor_negative import SourceDataset, validate_batch_provenance, ReliableCompetitorTrainer
from .runner import load_experiment_config, _optional_git_commit

IMPLEMENTATION = "positive_quality_transport_v1"
MODES = {"reference", "candidate", "control"}


def rearrange(values, quality, anchor_ids, object_id, mode):
    """Candidate permutes values; control matches target-displacement L1/mass.

    Signed BCE logit-gradient sum is preserved, not absolute gradient L1.
    Zero effective targets never acquire positive mass, including AMP underflow.
    """
    if (mode not in MODES or values.ndim != 1 or not values.is_floating_point()
            or not quality.is_floating_point() or quality.shape != values.shape
            or anchor_ids.shape != values.shape or anchor_ids.dtype != torch.long
            or not torch.isfinite(values).all() or not torch.isfinite(quality).all()
            or (values < 0).any() or (values > 1).any()
            or (quality < 0).any() or (quality > 1).any()):
        raise ValueError("invalid positive target/geometry/identity")
    result = values.clone()
    active = torch.where(values > 0)[0]
    if mode == "reference" or len(active) < 2 or quality[active].max() == quality[active].min():
        return result
    by_value = active[torch.argsort(values[active], stable=True)]
    order = by_value[torch.argsort(quality[by_value], stable=True)]
    result[order] = values[by_value]
    if mode == "control":
        original = values.float()
        budget = (result.float() - original).abs().sum()
        if budget == 0:
            return original
        # Fixed integer bijective residue ordering; does not consume any RNG.
        # Geometry determines the scalar budget only, not the control direction.
        key = ((anchor_ids[active] + 1) * 1103515245 + (int(object_id) + 1) * 12345) % 2147483647
        random_order = active[torch.argsort(key, stable=True)]
        direction = original.clone()
        direction[random_order] = original[by_value]
        available = (direction - original).abs().sum()
        if available < budget:
            direction[by_value] = original[by_value.flip(0)]
            available = (direction - original).abs().sum()
        if available == 0 or available + 1e-6 < budget:
            raise ValueError("target-displacement control budget unreachable")
        fraction = (budget / available).clamp(max=1)
        # Stable convex combination: x + (y-x) can erase tiny positive y when
        # fraction==1, violating the effective-support invariant in FP32.
        result = (1 - fraction) * original + fraction * direction
        # Roundoff may overshoot an endpoint by one FP32 ULP. Restrict only
        # active entries, leaving the exact original zero support untouched.
        result[active] = result[active].clamp(min=original[active].min(), max=original[active].max())
    return result


class GeometryAssignment(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original, self.assigned, self.predicted_boxes = original, None, None

    def forward(self, *args, **kwargs):
        self.predicted_boxes = args[1].detach()
        self.assigned = self.original(*args, **kwargs)
        return self.assigned


class PositiveBCE(nn.Module):
    def __init__(self, original, owner):
        super().__init__()
        self.original, self.owner = original, owner

    def forward(self, logits, targets):
        owner, batch = self.owner, self.owner.current_batch
        labels, boxes, raw, fg, local = owner.assigner.assigned
        if not torch.equal(targets, raw.to(targets.dtype)):
            raise ValueError("unexpected TAL target mutation")
        changed = targets.clone()
        if owner.mode == "control":
            changed = changed.float()
        image_ids = batch["batch_idx"].to(logits.device).long()
        classes = batch["object_class"].to(logits.device)
        human = batch["object_is_human"].to(logits.device)
        ids = batch["object_id"].to(logits.device)
        groups = 0
        geometries = torch.zeros_like(targets[..., 0], dtype=torch.float32) if owner.diagnostics else None
        for image in range(len(logits)):
            anchors = torch.where(fg[image])[0]
            selected, labels_here = local[image, anchors], labels[image, anchors]
            if (not torch.isfinite(selected).all() or (selected != selected.round()).any()
                    or not torch.isfinite(labels_here).all() or (labels_here != labels_here.round()).any()):
                raise ValueError("invalid TAL index before cast")
            selected = selected.long()
            objects = image_ids == image
            if (selected < 0).any() or (selected >= objects.sum()).any():
                raise ValueError("TAL source index out of range")
            assigned_class = classes[objects][selected]
            if not torch.equal(labels_here.long(), assigned_class):
                raise ValueError("TAL source/class mismatch")
            # local GT index, rather than source token, distinguishes repeated
            # mosaic appearances with their own augmented boxes.
            for gt in selected[human[objects][selected]].unique(sorted=True).tolist():
                at = anchors[selected == gt]
                cls = int(classes[objects][gt])
                quality = bbox_iou(owner.assigner.predicted_boxes[image, at].float(),
                    boxes[image, at].float(), xywh=False).flatten().detach()
                if geometries is not None:
                    geometries[image, at] = quality
                changed[image, at, cls] = rearrange(targets[image, at, cls], quality,
                    at, ids[objects][gt], owner.mode).to(changed.dtype)
                groups += 1
        if owner.diagnostics:
            owner.last_targets = changed.detach().clone()
            owner.last_original_targets = targets.detach().clone()
            owner.last_quality = geometries
            owner.last_predicted_boxes = owner.assigner.predicted_boxes.detach().clone()
            owner.last_raw_targets = raw.detach().clone()
            owner.last_assignment = tuple(t.detach().clone() for t in owner.assigner.assigned)
            owner.last_diagnostics = {"human_groups": groups,
                "changed_targets": int((changed != targets).sum()),
                "transport_L1": float((changed.double() - targets.double()).abs().sum()),
                "control_scope": "same per-object mass/support/target-displacement L1; not final or parameter gradient L1"}
        return self.original(logits, targets if owner.mode == "reference" else changed)


class PositiveQualityLoss(P1DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.mode = model.positive_quality_mode
        if self.mode not in MODES or self.p1_options != {"soft_weights": True, "uncertain_ignore": True}:
            raise ValueError("original P1 options and legal mode required")
        self.diagnostics = getattr(model, "positive_quality_diagnostics", False)
        self.current_batch = None
        self.assigner = GeometryAssignment(self.assigner)
        self.bce = PositiveBCE(self.bce, self)

    def get_assigned_targets_and_loss(self, preds, batch):
        validate_batch_provenance(batch, self.nc)
        n = preds["scores"].shape[0]
        for key in ("batch_idx", "ignore_batch_idx"):
            if len(batch[key]) and batch[key].max() >= n:
                raise ValueError("image/source index out of range")
        self.current_batch = batch
        try:
            return super().get_assigned_targets_and_loss(preds, batch)
        finally:
            self.current_batch = None
            self.assigner.assigned = self.assigner.predicted_boxes = None


class PositiveQualityModel(P1DetectionModel):
    def init_criterion(self):
        return PositiveQualityLoss(self)

    def loss(self, batch, preds=None):
        if self.training:
            return super().loss(batch, preds)
        if preds is None:
            preds = self.forward(batch["img"])
        return P1DetectionLoss(self)(preds, batch)


class PositiveQualityTrainer(P1Trainer):
    def __init__(self, *args, mode, **kwargs):
        import ultralytics
        if ultralytics.__version__ != "8.4.172" or mode not in MODES:
            raise ValueError("pinned runtime and legal mode required")
        self.mode = mode
        super().__init__(*args, **kwargs)
        if self.args.cls_pw != 0:
            raise ValueError("original P1 BCE required")

    # Same frozen source transport/augmentation already validated for P1.
    build_dataset = ReliableCompetitorTrainer.build_dataset

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(PositiveQualityModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options, model.positive_quality_mode = self.p1_options, self.mode
        if weights:
            model.load(weights)
        return model


def run_training(config_path):
    """Called by the explicit dispatch worker only, never by diagnostic/preflight."""
    import yaml
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    mode = config["positive_quality_transport"]["mode"]
    run = Path(config["project"]).resolve() / config["name"]
    if run.exists():
        raise FileExistsError(run)
    data = Path(config["data"]).resolve()
    metadata = Path(yaml.safe_load(data.read_text(encoding="utf-8"))["p1_metadata"])
    sources = [config_path, data, metadata, Path(config["model"]).resolve()]
    sources += list(Path(__file__).resolve().parents[1].rglob("*.py"))
    provenance = {"config": config, "implementation": IMPLEMENTATION, "mode": mode,
        "source_commit": _optional_git_commit(config_path.parent),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    yolo = YOLO(config["model"])
    def save_metadata(trainer):
        (Path(trainer.save_dir) / "experiment.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    yolo.add_callback("on_pretrain_routine_end", save_metadata)
    result = yolo.train(data=str(data), name=config["name"], project=str(run.parent), exist_ok=False,
        trainer=partial(PositiveQualityTrainer, mode=mode, p1_options=config["p1"]), **config["train"])
    return Path(result.save_dir).resolve()
