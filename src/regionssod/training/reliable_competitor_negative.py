"""Human-positive negative BCE extension; delegates TAL/regression to frozen P1.

Source transport uses the same scalar token augmentation route as P1. No hidden
labels, teacher inference, or confidence-based source inference is used.
"""
from __future__ import annotations

from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset

from .p1 import P1Dataset, P1DetectionLoss, P1DetectionModel, P1Trainer
from .runner import load_experiment_config, _optional_git_commit

IMPLEMENTATION = "reliable_competitor_negative_v1"
MODES = {"reference", "candidate", "control"}
PROVENANCE = {"object_is_human", "object_id", "object_class", "object_table"}


class SourceDataset(P1Dataset):
    """Unique object tokens bind class/confidence/ignore/source through cropping.

    Table columns: class, confidence, ignore, human, token ID. Per-object table
    rows are collated with the corresponding object, never inferred from weights.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base = self.p1_lookup
        nc = len(self.data["names"])
        human = (torch.arange(len(base)) < nc).float()[:, None]
        rows = torch.cat((base, human, torch.arange(len(base)).float()[:, None]), 1).tolist()
        for label in self.labels:
            key = str(Path(label["im_file"]).resolve())
            if key in self.p1_tokens:
                continue
            ids = []
            for cls in label["cls"].reshape(-1):
                if not np.isfinite(cls) or int(cls) != cls or not 0 <= cls < nc:
                    raise ValueError("invalid human class")
                ids.append(len(rows))
                rows.append([int(cls), 1., 0., 1., len(rows)])
            self.p1_tokens[key] = np.asarray(ids, dtype=np.float32).reshape(-1, 1)
            self.p1_ignores[key] = np.empty((0, 4), dtype=np.float32)
        if len(rows) >= 2**24:
            raise ValueError("too many source tokens for exact float32 transport")
        self.p1_lookup = torch.tensor(rows, dtype=torch.float32)

    def __getitem__(self, index):
        sample = YOLODataset.__getitem__(self, index)
        tokens = sample["cls"].view(-1)
        if (not torch.isfinite(tokens).all() or (tokens != tokens.round()).any()
                or (tokens < len(self.data["names"])).any() or (tokens >= len(self.p1_lookup)).any()):
            raise ValueError("damaged augmented object token")
        ids = tokens.long()
        decoded = self.p1_lookup[ids]
        nc = len(self.data["names"])
        if (not torch.isfinite(decoded).all() or (decoded[:, 0] != decoded[:, 0].round()).any()
                or (decoded[:, 0] < 0).any() or (decoded[:, 0] >= nc).any()
                or (decoded[:, 1] <= 0).any() or (decoded[:, 1] > 1).any()
                or ((decoded[:, 2:4] != 0) & (decoded[:, 2:4] != 1)).any()
                or not torch.equal(decoded[:, 4], tokens)
                or ((decoded[:, 3] == 1) & ((decoded[:, 1] != 1) | (decoded[:, 2] != 0))).any()):
            raise ValueError("damaged decoded object source/class/token")
        ignore = decoded[:, 2].bool()
        sample["ignore_bboxes"] = sample["bboxes"][ignore]
        sample["ignore_batch_idx"] = sample["batch_idx"][ignore]
        selected = decoded[~ignore]
        sample["pseudo_weights"] = selected[:, 1]
        sample["cls"] = selected[:, :1]
        sample["bboxes"] = sample["bboxes"][~ignore]
        sample["batch_idx"] = sample["batch_idx"][~ignore]
        sample["object_is_human"] = selected[:, 3].bool()
        sample["object_id"] = ids[~ignore]
        sample["object_class"] = selected[:, 0].long()
        sample["object_table"] = selected
        return sample

    @staticmethod
    def collate_fn(batch):
        ordinary = [{k: v for k, v in sample.items() if k not in PROVENANCE} for sample in batch]
        result = P1Dataset.collate_fn(ordinary)
        for key in PROVENANCE:
            result[key] = torch.cat([sample[key] for sample in batch])
        return result


def validate_batch_provenance(batch, nc):
    """Validate real object identities independently of numerical TAL quality."""
    if not PROVENANCE.issubset(batch) or "pseudo_weights" not in batch:
        raise ValueError("explicit object provenance required")
    classes = batch["cls"].reshape(-1)
    n = len(classes)
    human, ids, saved, table = (batch[k] for k in
        ("object_is_human", "object_id", "object_class", "object_table"))
    if (human.dtype != torch.bool or ids.dtype != torch.long or saved.dtype != torch.long
            or any(v.shape != (n,) for v in (human, ids, saved)) or table.shape != (n, 5)
            or table.dtype != torch.float32):
        raise ValueError("damaged source types/shapes")
    if (not torch.isfinite(table).all() or not torch.isfinite(classes).all()
            or (classes != classes.round()).any() or (classes < 0).any() or (classes >= nc).any()
            or (ids < nc).any() or (ids >= 2**24).any()
            or not torch.equal(classes.long(), saved.to(classes.device))):
        raise ValueError("damaged object identity/class")
    expected = torch.stack((saved.float(), batch["pseudo_weights"].to(saved.device).float(),
        torch.zeros_like(saved).float(), human.to(saved.device).float(), ids.to(saved.device).float()), 1)
    if not torch.equal(table.to(expected.device), expected):
        raise ValueError("object token/source/class binding changed")
    conf = table[:, 1]
    if (conf <= 0).any() or (conf > 1).any() or (conf[human.to(conf.device)] != 1).any():
        raise ValueError("invalid source confidence")
    # Repeated mosaic objects may share IDs, but must carry identical metadata.
    order = torch.argsort(ids, stable=True)
    if len(order) > 1:
        repeated = ids[order][1:] == ids[order][:-1]
        rows = table.to(order.device)[order]
        if ((rows[1:] != rows[:-1]).any(1) & repeated).any():
            raise ValueError("conflicting repeated object source")
    indices = batch["batch_idx"].reshape(-1)
    boxes = batch["bboxes"]
    if (indices.shape != (n,) or boxes.shape != (n, 4) or not torch.isfinite(indices).all()
            or (indices != indices.round()).any() or (indices < 0).any()
            or not torch.isfinite(boxes).all()):
        raise ValueError("damaged object batch binding")
    if "ignore_bboxes" not in batch or "ignore_batch_idx" not in batch:
        raise ValueError("explicit ignore geometry required")
    ignore, ignore_ids = batch["ignore_bboxes"], batch["ignore_batch_idx"]
    if (ignore.ndim != 2 or ignore.shape[1] != 4 or ignore_ids.shape != (len(ignore),)
            or not torch.isfinite(ignore).all() or not torch.isfinite(ignore_ids).all()
            or (ignore_ids != ignore_ids.round()).any() or (ignore_ids < 0).any()):
        raise ValueError("damaged ignore geometry/image binding")


def negative_multipliers(logits, targets, is_human, labels, raw_targets, *, mode):
    """Only effective human positive anchors; all selection/probabilities detached.

    fp64 sigmoid and max-rescaled negative sums avoid 0/0, overflow and unstable
    tiny denominators. When every sigmoid truly underflows, added gradient is 0.
    Returned fp32 multipliers match local negative-logit L1 to rounding tolerance,
    not shared-parameter gradients or subsequent optimization trajectories.
    """
    if mode not in MODES:
        raise ValueError("invalid mode")
    if (logits.ndim != 2 or logits.shape[1] < 2 or targets.shape != logits.shape
            or raw_targets.shape != logits.shape or not logits.is_floating_point()
            or not targets.is_floating_point() or not raw_targets.is_floating_point()):
        raise ValueError("inconsistent anchor/class tensors")
    if (is_human.dtype != torch.bool or labels.dtype != torch.long
            or is_human.shape != logits.shape[:1] or labels.shape != logits.shape[:1]
            or (labels < 0).any() or (labels >= logits.shape[1]).any()):
        raise ValueError("explicit source and legal assigned class required")
    positive = torch.zeros_like(targets, dtype=torch.bool).scatter_(1, labels[:, None], True)
    if ((raw_targets[~positive] != 0).any() or not torch.equal(targets, raw_targets.to(targets.dtype))):
        raise ValueError("assigned class/unaltered TAL target required")
    if (not all(torch.isfinite(x).all() for x in (logits, targets, raw_targets))
            or (raw_targets < 0).any() or (raw_targets > 1).any()):
        raise ValueError("nonfinite or invalid numerical targets/logits")
    active = is_human & (targets.gather(1, labels[:, None]).squeeze(1) > 0)
    weights = torch.ones_like(logits, dtype=torch.float32)
    if mode == "reference":
        return weights, active
    rows = active.nonzero(as_tuple=True)[0]
    p = logits.detach().double().sigmoid()
    negative_p = p.masked_fill(positive, -1.)
    # torch.max returns the first index at ties: class ID is the fixed tie break.
    pk, k = negative_p.max(1)
    if mode == "candidate":
        weights[rows, k[rows]] = 2.
    else:
        values = p.masked_fill(positive, 0.)
        maximum = values.max(1).values
        scaled = values / torch.where(maximum > 0, maximum, torch.ones_like(maximum))[:, None]
        total = scaled.sum(1)
        fraction = torch.where(total > 0, 1. / total.clamp_min(1.), 0.)
        weights[rows] = torch.where(positive[rows], 1., 1. + fraction[rows, None]).float()
    return weights.detach(), active


class CaptureAssignment(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original, self.assigned = original, None

    def forward(self, *args, **kwargs):
        self.assigned = self.original(*args, **kwargs)
        return self.assigned


class SourceBCE(nn.Module):
    def __init__(self, original, owner):
        super().__init__()
        self.original, self.owner = original, owner

    def forward(self, logits, targets):
        owner, batch = self.owner, self.owner.current_batch
        labels, _, raw, foreground, assigned_ids = owner.assigner.assigned
        if not torch.equal(targets, raw.to(targets.dtype)):
            raise ValueError("TAL targets changed")
        values = self.original(logits, targets)
        weights = torch.ones_like(logits, dtype=torch.float32)
        batch_ids = batch["batch_idx"].to(logits.device).long()
        classes = batch["object_class"].to(logits.device)
        human = batch["object_is_human"].to(logits.device)
        active_count = 0
        for image in range(len(logits)):
            anchors = foreground[image].nonzero(as_tuple=True)[0]
            selected_ids, selected_labels = assigned_ids[image, anchors], labels[image, anchors]
            if (not torch.isfinite(selected_ids).all() or not torch.isfinite(selected_labels).all()
                    or (selected_ids != selected_ids.round()).any()
                    or (selected_labels != selected_labels.round()).any()):
                raise ValueError("damaged TAL source/class before integer cast")
            local = selected_ids.long()
            objects = batch_ids == image
            if (local < 0).any() or (local >= objects.sum()).any():
                raise ValueError("TAL object binding out of range")
            assigned_class = classes[objects][local]
            if not torch.equal(labels[image, anchors].long(), assigned_class):
                raise ValueError("TAL object/class binding changed")
            w, active = negative_multipliers(logits[image, anchors], targets[image, anchors],
                human[objects][local], assigned_class, raw[image, anchors], mode=owner.mode)
            weights[image, anchors] = w
            active_count += int(active.sum())
        if owner.diagnostics:
            owner.last_diagnostics = {
                "implementation": IMPLEMENTATION, "mode": owner.mode,
                "eligible_human_anchors": active_count,
                "changed_positive_target_weights": int(((weights != 1) & (targets > 0)).sum()),
                "target_tensor_unchanged": True,
                "control_scope": "local negative logit gradient L1 only",
            }
            owner.last_weights = weights.detach()
            owner.last_targets = targets.detach().clone()
            owner.last_raw_targets = raw.detach().clone()
            owner.last_assignment = tuple(t.detach().clone() for t in owner.assigner.assigned)
        # P1 applies its original confidence/ignore factors and denominator later.
        return values if owner.mode == "reference" else values * weights


class ReliableCompetitorLoss(P1DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.mode = model.reliable_competitor_mode
        if self.mode not in MODES or self.p1_options != {"soft_weights": True, "uncertain_ignore": True}:
            raise ValueError("original P1 options and legal mode required")
        if self.class_weights is not None and not torch.all(self.class_weights == 1):
            raise ValueError("unit class weights required for local L1 matching")
        self.diagnostics = getattr(model, "reliable_competitor_diagnostics", False)
        self.last_diagnostics = {}
        self.current_batch = None
        self.assigner = CaptureAssignment(self.assigner)
        self.bce = SourceBCE(self.bce, self)

    def get_assigned_targets_and_loss(self, preds, batch):
        validate_batch_provenance(batch, self.nc)
        if len(batch["batch_idx"]) and batch["batch_idx"].max() >= preds["scores"].shape[0]:
            raise ValueError("object image binding out of range")
        if len(batch["ignore_batch_idx"]) and batch["ignore_batch_idx"].max() >= preds["scores"].shape[0]:
            raise ValueError("ignore image binding out of range")
        self.current_batch = batch
        try:
            return super().get_assigned_targets_and_loss(preds, batch)
        finally:
            self.current_batch = None
            self.assigner.assigned = None


class ReliableCompetitorModel(P1DetectionModel):
    def init_criterion(self):
        return ReliableCompetitorLoss(self)

    def loss(self, batch, preds=None):
        if self.training:
            return super().loss(batch, preds)
        # EMA can cache the training criterion; ordinary validation has no source
        # sidecar and always uses the unmodified P1 loss through this explicit path.
        if preds is None:
            preds = self.forward(batch["img"])
        return P1DetectionLoss(self)(preds, batch)


class ReliableCompetitorTrainer(P1Trainer):
    def __init__(self, *args, mode, **kwargs):
        import ultralytics
        if ultralytics.__version__ != "8.4.172" or mode not in MODES:
            raise ValueError("pinned Ultralytics and legal mode required")
        self.mode = mode
        super().__init__(*args, **kwargs)
        if self.args.cls_pw != 0:
            raise ValueError("original P1 cls_pw required")

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(ReliableCompetitorModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options, model.reliable_competitor_mode = self.p1_options, self.mode
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        if self.args.single_cls or self.args.classes is not None or self.args.fraction != 1:
            raise ValueError("all classes and fraction=1 required")
        return SourceDataset(img_path=img_path, metadata_path=Path(self.data["p1_metadata"]),
            imgsz=self.args.imgsz, batch_size=batch, augment=True, hyp=self.args,
            rect=self.args.rect, cache=self.args.cache or None, single_cls=False,
            stride=max(int(self.model.stride.max()), 32), pad=0., prefix="train: ",
            task="detect", data=self.data, fraction=1.)


def run_training(config_path):
    """Explicit worker only. No execution occurs on module import or precheck."""
    import yaml
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    mode = config["reliable_competitor_negative"]["mode"]
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
        trainer=partial(ReliableCompetitorTrainer, mode=mode, p1_options=config["p1"]), **config["train"])
    return Path(result.save_dir).resolve()
