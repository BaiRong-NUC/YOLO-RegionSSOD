"""Isolated P1 adapter: original teacher confidence also weights box and DFL."""
from __future__ import annotations

from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tal import make_anchors

from regionssod.training.p1 import P1DetectionLoss, P1Trainer, classification_factors
from regionssod.training.runner import _optional_git_commit, load_experiment_config


class RegressionConfidenceLoss(P1DetectionLoss):
    def get_assigned_targets_and_loss(self, preds: dict[str, Any], batch: dict[str, torch.Tensor]):
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        strides = cast(torch.Tensor, self.stride)
        anchor_points, stride_tensor = make_anchors(preds["feats"], strides, .5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * strides[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        denominator = target_scores.sum().clamp_(min=1)
        bce = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce = bce * self.class_weights
        factors = classification_factors(anchor_points * stride_tensor, fg_mask, target_gt_idx,
            batch, batch_size, self.device, imgsz, **self.p1_options)
        loss[1] = (bce * factors).sum() / denominator
        if "pseudo_weights" not in batch:
            raise ValueError("confidence regression requires augmented per-object provenance")
        regression = classification_factors(anchor_points * stride_tensor, fg_mask, target_gt_idx,
            batch, batch_size, self.device, imgsz, soft_weights=True, uncertain_ignore=False)
        # Keep the original TAL denominator; confidence must not cancel by normalization.
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
            target_scores * regression, denominator, fg_mask, imgsz, stride_tensor)
        hyperparameters: Any = self.hyp
        loss[0] *= hyperparameters.box
        loss[1] *= hyperparameters.cls
        loss[2] *= hyperparameters.dfl
        return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
                loss, dict(zip(self.loss_names, loss.detach())))


class RegressionConfidenceModel(DetectionModel):
    def init_criterion(self):
        return RegressionConfidenceLoss(self)

    def loss(self, batch, preds: Any = None):
        if self.training:
            return super().loss(batch, preds)
        # The validator reuses EMA's cached training criterion, but its GT has no pseudo metadata.
        if preds is None:
            preds = self.forward(batch["img"])
        return P1DetectionLoss(self)(cast(Any, preds), batch)


class RegressionConfidenceTrainer(P1Trainer):
    def get_model(self, cfg: Any = None, weights=None, verbose=True):
        model = self.set_model_names_for_load(RegressionConfidenceModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        if weights:
            model.load(weights)
        return model


def run_regression_confidence_training(config_path):
    import ultralytics
    from ultralytics import YOLO
    import yaml

    if ultralytics.__version__ != "8.4.172":
        raise RuntimeError("use pinned ultralytics==8.4.172")
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    if (config["method"] != "p1" or config["regression_confidence"] != "teacher_confidence"
            or config["p1"] != {"soft_weights": True, "uncertain_ignore": True}):
        raise ValueError("expected the single confidence regression extension")
    data = Path(config["data"]).resolve()
    dataset = yaml.safe_load(data.read_text(encoding="utf-8"))
    metadata_path = Path(dataset["p1_metadata"])
    project = Path(config["project"]).resolve()
    run = project / config["name"]
    if run.exists():
        raise FileExistsError(run)
    sources = [config_path, data, metadata_path, metadata_path.parent / "summary.json"]
    sources += list(Path(__file__).resolve().parents[1].rglob("*.py"))
    metadata = {"config_path": str(config_path), "model": config["model"], "data": str(data),
                "project": str(project), "name": config["name"], "train_args": config["train"],
                "method": "p1", "p1": config["p1"], "regression_confidence": "teacher_confidence",
                "ultralytics_version": ultralytics.__version__,
                "git_commit": _optional_git_commit(config_path.parent),
                "pseudo_summary": json.loads((metadata_path.parent / "summary.json").read_text(encoding="utf-8")),
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    yolo = YOLO(config["model"])

    def save_metadata(trainer):
        (Path(trainer.save_dir) / "experiment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    yolo.add_callback("on_pretrain_routine_end", save_metadata)
    yolo.train(data=str(data), name=config["name"], project=str(project), exist_ok=False,
               trainer=partial(RegressionConfidenceTrainer, p1_options=config["p1"]),
               **config["train"])
    return run
