"""Isolated negative-only P1 extension; original targets/assignment/regression retained."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import unwrap_model

from .p1 import P1DetectionLoss
from .class_ambiguity import ClassAmbiguityLoss, ClassAmbiguityTrainer, EpochClassificationDiagnostics
from .runner import load_experiment_config, _optional_git_commit


def negative_multipliers(logits, targets, teacher, pseudo, *, mode):
    """Detached weights. Strength control matches removed negative-logit gradient L1.

    Inputs are flattened assigned foreground anchors, not all background anchors.
    Top1 target must stay nonzero and equal the teacher top1 class. The control
    matches a scalar magnitude at the current logits, NOT parameter gradients,
    BCE value, per-class gradients, or the future training trajectory.
    """
    if mode not in {"semantic", "strength"}:
        raise ValueError("mode must be semantic or strength")
    if logits.ndim != 2 or logits.shape != targets.shape or teacher.shape != logits.shape:
        raise ValueError("inconsistent anchor/class shapes")
    if pseudo.shape != logits.shape[:1] or pseudo.dtype != torch.bool:
        raise ValueError("explicit pseudo provenance required")
    if not (torch.isfinite(logits).all() and torch.isfinite(targets).all()
            and torch.isfinite(teacher).all()) or (teacher < 0).any() or (teacher > 1).any():
        raise ValueError("invalid logits/targets/scores")
    weights = torch.ones_like(logits, dtype=torch.float32)
    # Stable sort provides a deterministic class-id tie break.
    order = torch.argsort(teacher, dim=1, descending=True, stable=True)
    top1, top2 = order[:, 0], order[:, 1]
    mass, target_class = targets.max(1)
    if ((target_class[pseudo] != top1[pseudo]).any() or (mass[pseudo] <= 0).any()
            or ((targets[pseudo] > 0).sum(1) != 1).any()):
        raise ValueError("original hard pseudo-positive TAL target required")
    ratio = teacher.gather(1, top2[:, None]).squeeze(1) / teacher.gather(1, top1[:, None]).squeeze(1).clamp_min(1e-12)
    active = pseudo & (ratio >= .25)
    rows = active.nonzero(as_tuple=True)[0]
    probabilities = logits.detach().float().sigmoid()
    negative = targets == 0
    removed = .5 * probabilities[rows, top2[rows]]
    if mode == "semantic":
        weights[rows, top2[rows]] = .5
    else:
        total = (probabilities * negative).sum(1)[rows]
        fraction = removed / total.clamp_min(1e-12)
        weights[rows] = torch.where(negative[rows], 1 - fraction[:, None], 1.)
    return weights.detach(), active


class _CaptureAssignment(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.assigned = None

    def forward(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        self.assigned = result
        return result


class _NegativeBCE(nn.Module):
    def __init__(self, original, capture, owner):
        super().__init__()
        self.original = original
        self.capture = capture
        # Loss object is not an nn.Module; avoid registering a circular module.
        self.owner = owner

    def forward(self, logits, targets):
        owner, batch = self.owner, self.owner.current_batch
        _, _, assigned_scores, foreground, assigned_ids = self.capture.assigned
        if not torch.equal(targets.float(), assigned_scores.to(targets.dtype).float()):
            raise ValueError("TAL classification targets changed")
        values = self.original(logits, targets)
        batch_ids = batch["batch_idx"].to(logits.device).long()
        teacher = batch["teacher_class_scores"].to(logits.device)
        flags = batch["is_pseudo"].to(logits.device)
        weights = torch.ones_like(logits, dtype=torch.float32)
        active_total = torch.tensor(0, device=logits.device)
        for image_index in range(len(logits)):
            anchors = foreground[image_index].nonzero(as_tuple=True)[0]
            local = assigned_ids[image_index, anchors].long()
            image_objects = batch_ids == image_index
            w, active = negative_multipliers(logits[image_index, anchors], targets[image_index, anchors],
                teacher[image_objects][local], flags[image_objects][local], mode=owner.mode)
            weights[image_index, anchors] = w
            active_total += active.sum()
        # Original P1 multiplies confidence/ignore factors after this call.
        if owner.diagnostics:
            from .p1 import classification_factors
            from ultralytics.utils.tal import make_anchors
            pred = owner.current_predictions
            anchor, stride = make_anchors(pred["feats"], owner.stride, .5)
            size = torch.tensor(pred["feats"][0].shape[2:], device=logits.device) * owner.stride[0]
            factors = classification_factors(anchor * stride, foreground, assigned_ids, batch,
                len(logits), logits.device, size, **owner.p1_options)
            p = logits.detach().float().sigmoid()
            removed = ((1 - weights) * p * (targets == 0) * factors)
            owner.last_diagnostics = {
                "active_pseudo_anchors": active_total.detach(),
                "removed_negative_logit_gradient_l1_before_gain": removed.sum().detach(),
                "changed_positive_target_weights": ((weights != 1) & (targets > 0)).sum().detach(),
                "target_tensor_unchanged": True,
                "control_scope": "local logit negative gradient L1 only; not parameter gradient",
            }
        return values * weights


class CompetitiveNegativeLoss(P1DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.mode = model.competitive_negative_mode
        if self.p1_options != {"soft_weights": True, "uncertain_ignore": True}:
            raise ValueError("original P1 options required")
        if self.class_weights is not None and not torch.all(self.class_weights == 1):
            raise ValueError("unit class weights required for strength matching")
        self.diagnostics = getattr(model, "competitive_negative_diagnostics", False)
        self.last_diagnostics = {}
        self.current_batch = self.current_predictions = None
        self.assigner = _CaptureAssignment(self.assigner)
        self.bce = _NegativeBCE(self.bce, self.assigner, self)

    def get_assigned_targets_and_loss(self, preds, batch):
        ClassAmbiguityLoss._provenance(self, batch)
        self.current_batch, self.current_predictions = batch, preds
        self.last_diagnostics = {}
        try:
            return super().get_assigned_targets_and_loss(preds, batch)
        finally:
            self.current_batch = self.current_predictions = None
            self.assigner.assigned = None


class CompetitiveNegativeModel(DetectionModel):
    def init_criterion(self):
        return CompetitiveNegativeLoss(self)

    def loss(self, batch, preds=None):
        if self.training:
            return super().loss(batch, preds)
        if preds is None:
            preds = self.forward(batch["img"])
        return P1DetectionLoss(self)(preds, batch)


class CompetitiveNegativeTrainer(ClassAmbiguityTrainer):
    def __init__(self, *args, mode, **kwargs):
        self.competitive_negative_mode = mode
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(CompetitiveNegativeModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        model.competitive_negative_mode = self.competitive_negative_mode
        if weights:
            model.load(weights)
        return model


class _Snapshots(EpochClassificationDiagnostics):
    @staticmethod
    def _set_diagnostics(model, enabled):
        if model is not None:
            model = unwrap_model(model)
            criterion = getattr(model, "criterion", None)
            if isinstance(criterion, CompetitiveNegativeLoss):
                criterion.diagnostics = enabled
                criterion.last_diagnostics = {}
            # First batch creates the criterion lazily.
            model.competitive_negative_diagnostics = enabled


def run_competitive_negative_training(config_path):
    """Explicit user-run only; source/tests are preparation, not performance evidence."""
    import ultralytics
    from .class_ambiguity import _verified_scores
    import yaml
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    mode = config["competitive_negative"]["mode"]
    if mode not in {"semantic", "strength"} or config["p1"] != {"soft_weights": True, "uncertain_ignore": True}:
        raise ValueError("Original P1 required")
    if ultralytics.__version__ != "8.4.172":
        raise ValueError("Pinned Ultralytics required")
    sidecar = Path(config["competitive_negative"]["class_score_path"]).resolve()
    data = Path(config["data"]).resolve()
    metadata = Path(yaml.safe_load(data.read_text(encoding="utf-8"))["p1_metadata"])
    _verified_scores(metadata, sidecar)
    run = Path(config["project"]).resolve() / config["name"]
    if run.exists():
        raise FileExistsError(run)
    sources = [config_path, data, metadata, metadata.parent / "summary.json", sidecar, sidecar.with_name("receipt.json")]
    sources += list(Path(__file__).resolve().parents[1].rglob("*.py"))
    import hashlib
    provenance = {"config": config, "mode": mode, "source_commit": _optional_git_commit(config_path.parent),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    yolo = YOLO(config["model"])
    def save_metadata(trainer):
        (Path(trainer.save_dir) / "experiment.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    yolo.add_callback("on_pretrain_routine_end", save_metadata)
    snapshots = _Snapshots()
    yolo.add_callback("on_train_epoch_start", snapshots.on_train_epoch_start)
    yolo.add_callback("on_train_batch_end", snapshots.on_train_batch_end)
    result = yolo.train(data=str(data), name=config["name"], project=str(run.parent), exist_ok=False,
        trainer=partial(CompetitiveNegativeTrainer, mode=mode, p1_options=config["p1"],
                        class_score_path=sidecar, enabled=True, diagnostics=False), **config["train"])
    return Path(result.save_dir).resolve()
