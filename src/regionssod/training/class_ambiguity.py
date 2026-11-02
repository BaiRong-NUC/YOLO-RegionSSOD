"""Isolated P1 classification targets: retain TAL mass and distribute it by teacher scores.

The enabled loss follows the pinned P1/Ultralytics 8.4.172 path (AGPL-3.0).
Only pseudo-positive classification targets change; regression uses original TAL scores.
"""
from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model

from regionssod.training.p1 import P1Dataset, P1DetectionLoss, P1Trainer, classification_factors
from regionssod.training.runner import _optional_git_commit, load_experiment_config


_EXTRA_FIELDS = {"teacher_class_scores", "is_pseudo"}
_P1_OPTIONS = {"soft_weights": True, "uncertain_ignore": True}


def _verified_scores(metadata_path: Path, class_score_path: Path) -> dict:
    metadata_bytes = metadata_path.read_bytes()
    sidecar_bytes = class_score_path.read_bytes()
    receipt = json.loads(class_score_path.with_name("receipt.json").read_text(encoding="utf-8"))
    for name, content in (("metadata_sha256", metadata_bytes), ("sidecar_sha256", sidecar_bytes)):
        if receipt.get(name) != hashlib.sha256(content).hexdigest():
            raise ValueError(f"class-score receipt {name} mismatch")

    def keyed_rows(content):
        rows = {}
        for line in content.decode("utf-8").splitlines():
            row = json.loads(line)
            image = Path(row["image"])
            if not image.is_absolute():
                raise ValueError("class-score identity requires absolute image paths")
            key = str(image.resolve())
            if key in rows:
                raise ValueError(f"duplicate image identity: {key}")
            rows[key] = row
        return rows

    metadata, sidecar = keyed_rows(metadata_bytes), keyed_rows(sidecar_bytes)
    if metadata.keys() != sidecar.keys():
        raise ValueError("class-score and metadata image sets differ")
    for image, row in sidecar.items():
        expected, observed = metadata[image]["positive"], row["positive"]
        if len(expected) != len(observed):
            raise ValueError(f"class-score positive count mismatch: {image}")
        for index, (old, new) in enumerate(zip(expected, observed, strict=True)):
            if any(old[field] != new[field] for field in ("class_id", "confidence", "xywhn")):
                raise ValueError(f"class-score positive identity mismatch: {image}:{index}")
            scores = np.asarray(new["teacher_class_scores"], dtype=np.float64)
            if (scores.shape != (10,) or not np.isfinite(scores).all()
                    or (scores < 0).any() or (scores > 1).any() or scores.sum() <= 0):
                raise ValueError(f"invalid teacher class scores: {image}:{index}")
            if (int(scores.argmax()) != old["class_id"]
                    or not np.isclose(scores.max(), old["confidence"], atol=1e-6, rtol=0)):
                raise ValueError(f"teacher top1/confidence mismatch: {image}:{index}")
    return sidecar


class ClassAmbiguityDataset(P1Dataset):
    """Decode full scores alongside P1 IDs after augmentation, before collation.

    ``class_score_path`` is JSONL with original pseudo image paths and ordered
    positive identities. Adjacent ``receipt.json`` binds it to the P1 metadata.
    Human rows carry zero score placeholders and an explicit false pseudo flag.
    """

    def __init__(self, *args, metadata_path, class_score_path, **kwargs):
        metadata_path, class_score_path = Path(metadata_path), Path(class_score_path)
        sidecar = _verified_scores(metadata_path, class_score_path)
        if len(kwargs["data"]["names"]) != 10:
            raise ValueError("class-score sidecar requires the original 10-class dataset")
        super().__init__(*args, metadata_path=metadata_path, **kwargs)
        self.class_score_path = class_score_path.resolve()
        self.teacher_score_lookup = torch.zeros((len(self.p1_lookup), 10), dtype=torch.float32)
        self.pseudo_lookup = torch.zeros(len(self.p1_lookup), dtype=torch.bool)
        for image, row in sidecar.items():
            count = len(row["positive"])
            ids = torch.from_numpy(self.p1_tokens[image][:count, 0].astype(np.int64))
            scores = torch.tensor([p["teacher_class_scores"] for p in row["positive"]],
                                  dtype=torch.float32).reshape(-1, 10)
            self.teacher_score_lookup[ids] = scores
            self.pseudo_lookup[ids] = True

    def __getitem__(self, index):
        # Skip P1's decoding, but retain its get_image_and_label and transforms.
        sample = YOLODataset.__getitem__(self, index)
        ids = sample["cls"].view(-1).long()
        decoded = self.p1_lookup[ids]
        ignore = decoded[:, 2].bool()
        sample["ignore_bboxes"] = sample["bboxes"][ignore]
        sample["ignore_batch_idx"] = sample["batch_idx"][ignore]
        sample["pseudo_weights"] = decoded[~ignore, 1]
        sample["teacher_class_scores"] = self.teacher_score_lookup[ids[~ignore]]
        sample["is_pseudo"] = self.pseudo_lookup[ids[~ignore]]
        sample["cls"] = decoded[~ignore, :1]
        sample["bboxes"] = sample["bboxes"][~ignore]
        sample["batch_idx"] = sample["batch_idx"][~ignore]
        return sample

    @staticmethod
    def collate_fn(batch):
        ordinary = [{k: v for k, v in sample.items() if k not in _EXTRA_FIELDS} for sample in batch]
        result = P1Dataset.collate_fn(ordinary)
        for field in _EXTRA_FIELDS:
            result[field] = torch.cat([sample[field] for sample in batch])
        return result


class ClassAmbiguityLoss(P1DetectionLoss):
    """Use m * (s / sum(s)) at explicit pseudo positives; disabled delegates to P1.

    Optional ``last_diagnostics`` contains detached per-batch scalar tensors.
    BCE is confidence-weighted but unnormalized. Gradient summaries describe
    class logits of the gain- and batch-scaled classification loss, not model
    parameter gradients. No teacher or validation labels are fitted.
    """

    def __init__(self, model, *, enabled=None, diagnostics=None):
        super().__init__(model)
        self.enabled = getattr(model, "class_ambiguity_enabled", True) if enabled is None else enabled
        self.diagnostics = (getattr(model, "class_ambiguity_diagnostics", False)
                            if diagnostics is None else diagnostics)
        self.last_diagnostics = {}
        if self.enabled:
            if self.p1_options != _P1_OPTIONS:
                raise ValueError("class ambiguity requires original P1 confidence and ignore")
            if self.class_weights is not None and not torch.all(self.class_weights == 1):
                raise ValueError("class ambiguity requires unit class weights to preserve mass")

    def _provenance(self, batch):
        required = _EXTRA_FIELDS | {"pseudo_weights", "ignore_bboxes", "ignore_batch_idx"}
        if not required.issubset(batch):
            raise ValueError(f"class ambiguity requires training provenance: {sorted(required - batch.keys())}")
        count = len(batch["cls"])
        scores = batch["teacher_class_scores"].detach().to(self.device, dtype=torch.float32)
        flags = batch["is_pseudo"].to(self.device)
        weights = batch["pseudo_weights"].detach().to(self.device)
        if (scores.shape != (count, self.nc) or flags.shape != (count,)
                or flags.dtype != torch.bool or weights.shape != (count,)):
            raise ValueError("class ambiguity provenance shape/dtype mismatch")
        if not torch.isfinite(scores).all() or not torch.isfinite(weights).all():
            raise ValueError("nonfinite class ambiguity provenance")
        if ((scores < 0).any() or (scores > 1).any() or (weights <= 0).any()
                or (weights > 1).any() or (weights[~flags] != 1).any()):
            raise ValueError("invalid class ambiguity scores or confidence")
        pseudo = scores[flags]
        if len(pseudo):
            top, cls = pseudo.max(1)
            original_cls = batch["cls"].to(self.device).view(-1)[flags]
            if ((pseudo.sum(1) <= 0).any() or not torch.equal(cls, original_cls)
                    or not torch.allclose(top, weights[flags].float(), atol=1e-6, rtol=0)):
                raise ValueError("class ambiguity teacher top1/confidence mismatch")
        return scores, flags

    def get_assigned_targets_and_loss(self, preds, batch):
        self.last_diagnostics = {}
        if not self.enabled:
            return super().get_assigned_targets_and_loss(preds, batch)
        scores, flags = self._provenance(batch)
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        denominator = target_scores.sum().clamp_(min=1)

        classification_targets = target_scores.clone()
        pseudo_anchors = torch.zeros_like(fg_mask)
        batch_idx = batch["batch_idx"].to(self.device).long()
        for image_index in range(batch_size):
            anchors = fg_mask[image_index].nonzero(as_tuple=True)[0]
            local_ids = target_gt_idx[image_index, anchors].long()
            image_objects = batch_idx == image_index
            pseudo = flags[image_objects][local_ids]
            anchors, local_ids = anchors[pseudo], local_ids[pseudo]
            teacher = scores[image_objects][local_ids]
            q = teacher / teacher.sum(-1, keepdim=True)
            mass = target_scores[image_index, anchors].sum(-1, keepdim=True)
            classification_targets[image_index, anchors] = mass * q
            pseudo_anchors[image_index, anchors] = True

        bce = self.bce(pred_scores, classification_targets.to(dtype))
        if self.class_weights is not None:
            bce = bce * self.class_weights
        factors = classification_factors(anchor_points * stride_tensor, fg_mask, target_gt_idx,
                                         batch, batch_size, self.device, imgsz, **self.p1_options)
        loss[1] = (bce * factors).sum() / denominator
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
            target_scores, denominator, fg_mask, imgsz, stride_tensor)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        if self.diagnostics:
            self._record_diagnostics(pred_scores, target_scores, classification_targets, bce,
                                     factors, denominator, fg_mask, pseudo_anchors)
        return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
                loss, dict(zip(self.loss_names, loss.detach())))

    @torch.no_grad()
    def _record_diagnostics(self, logits, old, new, new_bce, factors, denominator, fg, pseudo):
        old_bce = self.bce(logits, old.to(logits.dtype))
        if self.class_weights is not None:
            old_bce = old_bce * self.class_weights
        groups = {"all": torch.ones_like(fg), "pseudo_positive": pseudo,
                  "human_positive": fg & ~pseudo,
                  "active_negative": ~fg & (factors[..., 0] != 0),
                  "ignored_negative": ~fg & (factors[..., 0] == 0)}
        result = {"denominator": denominator.detach()}
        grad_scale = self.hyp.cls * logits.shape[0] / denominator
        for name, mask in groups.items():
            entry = {"anchors": mask.sum()}
            for prefix, target, bce in (("old", old, old_bce), ("new", new, new_bce)):
                gradient = ((logits.float().sigmoid() - target.to(logits.dtype).float())
                            * factors * grad_scale)[mask]
                entry.update({
                    f"{prefix}_mass": target[mask].sum(),
                    f"{prefix}_weighted_mass": (target * factors)[mask].sum(),
                    f"{prefix}_bce": (bce * factors)[mask].sum(),
                    f"{prefix}_gradient_sum": gradient.sum(),
                    f"{prefix}_gradient_l1": gradient.abs().sum(),
                    f"{prefix}_gradient_l2": gradient.square().sum().sqrt(),
                })
            result[name] = entry
        self.last_diagnostics = result


class ClassAmbiguityModel(DetectionModel):
    def init_criterion(self):
        return ClassAmbiguityLoss(self)

    def loss(self, batch, preds=None):
        if self.training:
            return super().loss(batch, preds)
        # EMA may carry a cached training criterion into ordinary GT validation.
        if preds is None:
            preds = self.forward(batch["img"])
        return P1DetectionLoss(self)(preds, batch)


class ClassAmbiguityTrainer(P1Trainer):
    def __init__(self, *args, class_score_path=None, enabled=True, diagnostics=False, **kwargs):
        import ultralytics

        if ultralytics.__version__ != "8.4.172":
            raise RuntimeError("use pinned ultralytics==8.4.172")
        if enabled and class_score_path is None:
            raise ValueError("enabled class ambiguity requires class_score_path")
        self.class_score_path = Path(class_score_path).resolve() if class_score_path is not None else None
        self.class_ambiguity_enabled = enabled
        self.class_ambiguity_diagnostics = diagnostics
        super().__init__(*args, **kwargs)
        if enabled and (self.p1_options != _P1_OPTIONS or self.args.cls_pw != 0):
            raise ValueError("class ambiguity requires original P1 options and cls_pw=0")

    def get_model(self, cfg=None, weights=None, verbose=True):
        if not self.class_ambiguity_enabled:
            return super().get_model(cfg, weights, verbose)
        model = self.set_model_names_for_load(ClassAmbiguityModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        model.class_ambiguity_enabled = self.class_ambiguity_enabled
        model.class_ambiguity_diagnostics = self.class_ambiguity_diagnostics
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train" or not self.class_ambiguity_enabled:
            return super().build_dataset(img_path, mode, batch)
        if self.args.single_cls or self.args.classes is not None or self.args.fraction != 1:
            raise ValueError("class ambiguity requires all classes and fraction=1")
        return ClassAmbiguityDataset(
            img_path=img_path, metadata_path=Path(self.data["p1_metadata"]),
            class_score_path=self.class_score_path, imgsz=self.args.imgsz, batch_size=batch,
            augment=True, hyp=self.args, rect=self.args.rect, cache=self.args.cache or None,
            single_cls=False, stride=max(int(self.model.stride.max()), 32),
            pad=0.0, prefix="train: ", task="detect", data=self.data, fraction=1.0)


class EpochClassificationDiagnostics:
    """Sample the first completed training batch, once per epoch on the main rank.

    These are engineering/mechanism snapshots, never full-epoch gradient estimates.
    EMA is kept disabled; its validation criterion is not a training observation.
    """

    def __init__(self):
        self._pending_epoch = None
        self._written_epochs = set()

    @staticmethod
    def _set_diagnostics(model, enabled):
        if model is None:
            return
        model = unwrap_model(model)
        # init_criterion reads this attribute if no cached criterion exists yet.
        model.class_ambiguity_diagnostics = enabled
        criterion = getattr(model, "criterion", None)
        if isinstance(criterion, ClassAmbiguityLoss):
            criterion.diagnostics = enabled
            criterion.last_diagnostics = {}

    @staticmethod
    def _ema_model(trainer):
        return getattr(getattr(trainer, "ema", None), "ema", None)

    def on_train_epoch_start(self, trainer):
        epoch = int(trainer.epoch)
        collect = RANK in {-1, 0} and epoch not in self._written_epochs
        self._pending_epoch = epoch if collect else None
        self._set_diagnostics(trainer.model, collect)
        self._set_diagnostics(self._ema_model(trainer), False)

    def on_train_batch_end(self, trainer):
        if self._pending_epoch is None:
            return
        epoch = self._pending_epoch
        criterion = getattr(unwrap_model(trainer.model), "criterion", None)
        diagnostics = getattr(criterion, "last_diagnostics", {})

        def json_values(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().item()
            if isinstance(value, dict):
                return {key: json_values(item) for key, item in value.items()}
            return value

        try:
            record = {
                "epoch": epoch + 1,
                "batch_index": 0,
                "sampling": "first_completed_training_batch_each_epoch",
                "scope": "engineering_mechanism_snapshot",
                "epoch_gradient_estimate": False,
                "rank": RANK,
                "world_size": int(getattr(trainer, "world_size", 1)),
                "gradient_definition": (
                    "analytic_class_logits_with_cls_gain_and_local_batch_size;"
                    "excludes_amp_scaler_optimizer_accumulation_and_ddp_multiplier"
                ),
                "status": "recorded" if diagnostics else "missing_diagnostics",
                "diagnostics": json_values(diagnostics),
            }
            line = json.dumps(record, allow_nan=False) + "\n"
            save_dir = Path(trainer.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            with (save_dir / "classification_diagnostics.jsonl").open("a", encoding="utf-8") as output:
                output.write(line)
            self._written_epochs.add(epoch)
        finally:
            self._pending_epoch = None
            self._set_diagnostics(trainer.model, False)
            self._set_diagnostics(self._ema_model(trainer), False)


def run_class_ambiguity_training(config_path) -> Path:
    """Explicit training entry; the caller owns paired dispatch and evaluation.

    Paths follow the original runner's working-directory convention. The
    ``class_ambiguity`` mapping accepts enabled, class_score_path, and optional
    diagnostics (retained for API compatibility). Enabled runs always sample the
    first batch each epoch, independent of that legacy flag. Disabled runs use
    P1's dataset/model with no sidecar dependency.
    """
    import ultralytics
    from ultralytics import YOLO
    import yaml

    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    options = config.get("class_ambiguity")
    if (not isinstance(options, dict)
            or set(options) - {"enabled", "class_score_path", "diagnostics"}
            or not isinstance(options.get("enabled"), bool)
            or not isinstance(options.get("diagnostics", False), bool)):
        raise ValueError("class_ambiguity requires boolean enabled and optional diagnostics/class_score_path")
    enabled = options["enabled"]
    p1_options = _P1_OPTIONS | config.get("p1", {})
    if config.get("method") != "p1" or (enabled and p1_options != _P1_OPTIONS):
        raise ValueError("class ambiguity requires method p1 and original P1 options")
    if ultralytics.__version__ != "8.4.172":
        raise RuntimeError("use pinned ultralytics==8.4.172")
    data_path = Path(config["data"]).resolve()
    dataset = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not dataset.get("p1_metadata"):
        raise ValueError("class ambiguity requires a P1 dataset")
    metadata_path = Path(dataset["p1_metadata"]).resolve()
    project = Path(config.get("project", "runs")).resolve()
    name = str(config["name"])
    run = project / name
    if run.exists():
        raise FileExistsError(run)
    sidecar = None
    if enabled:
        if not options.get("class_score_path"):
            raise ValueError("enabled class ambiguity requires class_score_path")
        sidecar = Path(options["class_score_path"]).resolve()
        _verified_scores(metadata_path, sidecar)
    sources = [config_path, data_path, metadata_path, metadata_path.parent / "summary.json"]
    sources += list(Path(__file__).resolve().parents[1].rglob("*.py"))
    if sidecar is not None:
        sources += [sidecar, sidecar.with_name("receipt.json")]
    model_path = str(config.get("model", "yolo11s.pt"))
    train_args = dict(config.get("train", {}))
    metadata = {
        "config_path": str(config_path), "model": model_path, "data": str(data_path),
        "project": str(project), "name": name, "train_args": train_args,
        "ultralytics_version": ultralytics.__version__,
        "git_commit": _optional_git_commit(config_path.parent),
        "method": "p1", "p1": p1_options,
        "class_ambiguity": {"enabled": enabled,
                            "class_score_path": str(sidecar) if sidecar is not None else None,
                            "diagnostics": enabled,
                            "diagnostics_sampling": "first_completed_training_batch_each_epoch" if enabled else None},
        "pseudo_summary": json.loads((metadata_path.parent / "summary.json").read_text(encoding="utf-8")),
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    }
    yolo = YOLO(model_path)

    def save_metadata(trainer):
        save_dir = Path(trainer.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "experiment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    yolo.add_callback("on_pretrain_routine_end", save_metadata)
    if enabled:
        diagnostic_callbacks = EpochClassificationDiagnostics()
        yolo.add_callback("on_train_epoch_start", diagnostic_callbacks.on_train_epoch_start)
        yolo.add_callback("on_train_batch_end", diagnostic_callbacks.on_train_batch_end)
    results = yolo.train(
        data=str(data_path), name=name, project=str(project), exist_ok=False,
        trainer=partial(ClassAmbiguityTrainer, p1_options=p1_options,
                        class_score_path=sidecar, enabled=enabled,
                        diagnostics=False),
        **train_args)
    return Path(getattr(results, "save_dir", run)).resolve()
