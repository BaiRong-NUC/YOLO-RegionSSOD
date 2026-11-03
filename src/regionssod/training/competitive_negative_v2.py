"""AMP-safe candidate revision. Frozen P1 and competitive_negative.py stay intact."""
from __future__ import annotations

from functools import partial
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.utils.tal import make_anchors

from .competitive_negative import (
    CompetitiveNegativeLoss, CompetitiveNegativeModel, CompetitiveNegativeTrainer, _Snapshots,
)
from .p1 import classification_factors
from .runner import load_experiment_config, _optional_git_commit

IMPLEMENTATION = "competitive_negative_amp_safe_v2"


def negative_multipliers_v2(logits, targets, teacher, pseudo, labels, raw_targets, *, mode):
    """Validate provenance before casting; preserve P1 at zero effective target mass.

    The assigned class is explicit, never inferred from argmax of an all-zero
    half target. Neither raw TAL targets nor the BCE targets are modified.
    Strength matching remains local negative-logit gradient L1, not parameter L1.
    """
    if mode not in {"semantic", "strength"}:
        raise ValueError("mode must be semantic or strength")
    if (logits.ndim != 2 or logits.shape[1] < 2 or logits.shape != targets.shape
            or logits.shape != teacher.shape or raw_targets.shape != logits.shape):
        raise ValueError("inconsistent anchor/class shapes")
    if (pseudo.shape != logits.shape[:1] or pseudo.dtype != torch.bool
            or labels.shape != pseudo.shape or labels.dtype != torch.long):
        raise ValueError("explicit pseudo provenance and assigned class required")
    if (not all(torch.isfinite(x).all() for x in (logits, targets, raw_targets, teacher))
            or (teacher < 0).any() or (teacher > 1).any()
            or (raw_targets < 0).any() or (raw_targets > 1).any()
            or (labels < 0).any() or (labels >= logits.shape[1]).any()):
        raise ValueError("invalid logits/targets/scores/classes")
    if not torch.equal(targets, raw_targets.to(targets.dtype)):
        raise ValueError("TAL classification targets changed")
    positive_class = torch.zeros_like(targets, dtype=torch.bool)
    positive_class.scatter_(1, labels[:, None], True)
    if (raw_targets[~positive_class] != 0).any():
        raise ValueError("original hard pseudo-positive TAL target required")
    order = torch.argsort(teacher, dim=1, descending=True, stable=True)
    top1, top2 = order[:, 0], order[:, 1]
    if ((top1[pseudo] != labels[pseudo]).any() or (teacher[pseudo].sum(1) <= 0).any()):
        raise ValueError("teacher/assigned class mismatch")
    mass = targets.gather(1, labels[:, None]).squeeze(1)
    ratio = teacher.gather(1, top2[:, None]).squeeze(1) / teacher.gather(1, top1[:, None]).squeeze(1).clamp_min(1e-12)
    # Zero/underflow mass is a numerical boundary, not a new tunable threshold.
    active = pseudo & (mass > 0) & (ratio >= .25)
    rows = active.nonzero(as_tuple=True)[0]
    weights = torch.ones_like(logits, dtype=torch.float32)
    probabilities = logits.detach().float().sigmoid()
    removed = .5 * probabilities[rows, top2[rows]]
    if mode == "semantic":
        weights[rows, top2[rows]] = .5
    else:
        negative = ~positive_class
        total = (probabilities * negative).sum(1)[rows]
        fraction = removed / total.clamp_min(1e-12)
        weights[rows] = torch.where(negative[rows], 1 - fraction[:, None], 1.)
    return weights.detach(), active


class _NegativeBCEV2(nn.Module):
    def __init__(self, original, capture, owner):
        super().__init__()
        self.original, self.capture, self.owner = original, capture, owner

    def forward(self, logits, targets):
        owner, batch = self.owner, self.owner.current_batch
        labels, _, raw, foreground, assigned_ids = self.capture.assigned
        if not torch.equal(targets, raw.to(targets.dtype)):
            raise ValueError("TAL classification targets changed")
        values = self.original(logits, targets)
        weights = torch.ones_like(logits, dtype=torch.float32)
        batch_ids = batch["batch_idx"].to(logits.device).long()
        teacher, flags = batch["teacher_class_scores"].to(logits.device), batch["is_pseudo"].to(logits.device)
        classes = batch["cls"].to(logits.device).view(-1).long()
        active_total = torch.tensor(0, device=logits.device)
        zero_total = torch.tensor(0, device=logits.device)
        underflow_total = torch.tensor(0, device=logits.device)
        protected_changed = torch.tensor(0, device=logits.device)
        for image in range(len(logits)):
            anchors = foreground[image].nonzero(as_tuple=True)[0]
            local = assigned_ids[image, anchors].long()
            objects = batch_ids == image
            assigned_class = labels[image, anchors].long()
            if not torch.equal(assigned_class, classes[objects][local]):
                raise ValueError("TAL object/class binding changed")
            pseudo = flags[objects][local]
            actual, original = targets[image, anchors], raw[image, anchors]
            w, active = negative_multipliers_v2(logits[image, anchors], actual,
                teacher[objects][local], pseudo, assigned_class, original, mode=owner.mode)
            weights[image, anchors] = w
            mass = actual.gather(1, assigned_class[:, None]).squeeze(1)
            raw_mass = original.gather(1, assigned_class[:, None]).squeeze(1)
            active_total += active.sum()
            zero_total += (pseudo & (mass == 0)).sum()
            underflow_total += (pseudo & (mass == 0) & (raw_mass > 0)).sum()
            protected_changed += (w.gather(1, assigned_class[:, None]) != 1).sum()
        if owner.diagnostics:
            pred = owner.current_predictions
            anchor, stride = make_anchors(pred["feats"], owner.stride, .5)
            size = torch.tensor(pred["feats"][0].shape[2:], device=logits.device) * owner.stride[0]
            factors = classification_factors(anchor * stride, foreground, assigned_ids, batch,
                len(logits), logits.device, size, **owner.p1_options)
            removed = (1 - weights) * logits.detach().float().sigmoid() * (targets == 0) * factors
            owner.last_diagnostics = {
                "implementation": IMPLEMENTATION,
                "active_pseudo_anchors": active_total.detach(),
                "zero_mass_pseudo_anchors_unchanged": zero_total.detach(),
                "cast_underflow_pseudo_anchors": underflow_total.detach(),
                "changed_assigned_class_weights": protected_changed.detach(),
                "removed_negative_logit_gradient_l1_before_gain": removed.sum().detach(),
                "changed_positive_target_weights": ((weights != 1) & (targets > 0)).sum().detach(),
                "target_tensor_unchanged": True,
                "control_scope": "local negative logit gradient L1 only; not parameter gradient",
            }
        return values * weights


class CompetitiveNegativeLossV2(CompetitiveNegativeLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bce = _NegativeBCEV2(self.bce.original, self.assigner, self)


class CompetitiveNegativeModelV2(CompetitiveNegativeModel):
    def init_criterion(self):
        return CompetitiveNegativeLossV2(self)


class CompetitiveNegativeTrainerV2(CompetitiveNegativeTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(CompetitiveNegativeModelV2(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options, model.competitive_negative_mode = self.p1_options, self.competitive_negative_mode
        if weights:
            model.load(weights)
        return model


def run_competitive_negative_training_v2(config_path):
    import ultralytics
    import yaml
    from .class_ambiguity import _verified_scores
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    if ultralytics.__version__ != "8.4.172" or config["p1"] != {"soft_weights": True, "uncertain_ignore": True}:
        raise ValueError("Pinned Ultralytics and original P1 required")
    sidecar = Path(config["competitive_negative"]["class_score_path"]).resolve()
    data = Path(config["data"]).resolve()
    metadata = Path(yaml.safe_load(data.read_text(encoding="utf-8"))["p1_metadata"])
    _verified_scores(metadata, sidecar)
    run = Path(config["project"]).resolve() / config["name"]
    if run.exists():
        raise FileExistsError(run)
    sources = [config_path, data, metadata, metadata.parent / "summary.json", sidecar, sidecar.with_name("receipt.json")]
    sources += list(Path(__file__).resolve().parents[1].rglob("*.py"))
    provenance = {"config": config, "implementation": IMPLEMENTATION,
        "source_commit": _optional_git_commit(config_path.parent),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    yolo = YOLO(config["model"])
    def save_metadata(trainer):
        (Path(trainer.save_dir) / "experiment.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    yolo.add_callback("on_pretrain_routine_end", save_metadata)
    snapshots = _Snapshots()
    yolo.add_callback("on_train_epoch_start", snapshots.on_train_epoch_start)
    yolo.add_callback("on_train_batch_end", snapshots.on_train_batch_end)
    result = yolo.train(data=str(Path(config["data"]).resolve()), name=config["name"], project=str(run.parent),
        exist_ok=False, trainer=partial(CompetitiveNegativeTrainerV2,
            mode=config["competitive_negative"]["mode"], p1_options=config["p1"],
            class_score_path=sidecar, enabled=True, diagnostics=False), **config["train"])
    return Path(result.save_dir).resolve()
