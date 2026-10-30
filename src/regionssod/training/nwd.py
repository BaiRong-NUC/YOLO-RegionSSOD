"""Isolated small-box hybrid NWD adapter; the frozen P1 path is unchanged."""
from __future__ import annotations

from functools import partial
import hashlib
import json
import math
from pathlib import Path

import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou

from regionssod.training.p1 import P1DetectionLoss, P1Trainer
from regionssod.training.runner import _optional_git_commit, load_experiment_config


def nwd_loss_pixels(predicted, target, normalizer):
    """Equation 9, using xyxy boxes and C in the same pixel coordinate system."""
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("NWD normalizer must be positive and finite")
    predicted, target = predicted.float(), target.float()
    center = (predicted[..., :2] + predicted[..., 2:]
              - target[..., :2] - target[..., 2:]) / 2
    half_size = (predicted[..., 2:] - predicted[..., :2]
                 - target[..., 2:] + target[..., :2]) / 2
    distance = torch.linalg.vector_norm(torch.cat((center, half_size), dim=-1), dim=-1,
                                        keepdim=True)
    return -torch.expm1(-distance / normalizer)


class SmallNWDBboxLoss(BboxLoss):
    def __init__(self, reg_max=16, *, normalizer, mix=.5, small_side=32.):
        super().__init__(reg_max)
        if (not all(math.isfinite(v) for v in (normalizer, mix, small_side))
                or normalizer <= 0 or not 0 <= mix <= 1 or small_side <= 0):
            raise ValueError("invalid NWD options")
        self.normalizer, self.mix, self.small_side = normalizer, mix, small_side

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        base, dfl = super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes,
                                    target_scores, target_scores_sum, fg_mask, imgsz, stride)
        if self.mix == 0:
            return base, dfl
        selected_stride = stride.view(1, -1, 1).expand(*fg_mask.shape, 1)[fg_mask]
        pred, target = pred_bboxes[fg_mask], target_bboxes[fg_mask]
        pixel_target = target.float() * selected_stride
        side = pixel_target[:, 2:] - pixel_target[:, :2]
        small = (side.prod(dim=-1, keepdim=True) < self.small_side**2).to(torch.float32)
        ciou = bbox_iou(pred, target, xywh=False, CIoU=True).float()
        nwd = 1 - nwd_loss_pixels(pred * selected_stride, pixel_target, self.normalizer)
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        correction = (small * weight * (ciou - nwd)).sum() / target_scores_sum
        return base + self.mix * correction, dfl


class NWDDetectionLoss(P1DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bbox_loss = SmallNWDBboxLoss(self.reg_max, **model.nwd_options).to(self.device)


class NWDDetectionModel(DetectionModel):
    def init_criterion(self):
        return NWDDetectionLoss(self)


class NWDTrainer(P1Trainer):
    def __init__(self, *args, nwd_options, **kwargs):
        self.nwd_options = nwd_options
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(NWDDetectionModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        model.nwd_options = self.nwd_options
        if weights:
            model.load(weights)
        return model


def estimate_normalizer(images, *, imgsz=640, small_side=32):
    """Use only materialized open labels, after ordinary aspect-preserving resize."""
    from PIL import Image
    from regionssod.data.annotations import read_label_file, replace_images_component

    sizes = []
    for image in images:
        with Image.open(image) as pixels:
            width, height = pixels.size
        scale = min(imgsz / width, imgsz / height)
        label = replace_images_component(Path(image)).with_suffix(".txt")
        for box in read_label_file(label):
            side = math.sqrt(box.w * width * box.h * height) * scale
            if side < small_side:
                sizes.append(side)
    if not sizes:
        raise ValueError("no open small boxes for NWD normalization")
    return {"normalizer": math.fsum(sizes) / len(sizes), "small_boxes": len(sizes),
            "open_tiles": len(images), "imgsz": imgsz, "small_side": small_side,
            "units": "training input pixels after aspect-preserving resize; no augmentation"}


def run_nwd_training(config_path, *, normalizer):
    """Mirror the pinned P1 training invocation with an explicit alternate trainer."""
    import ultralytics
    from ultralytics import YOLO
    import yaml

    if ultralytics.__version__ != "8.4.172":
        raise RuntimeError("use pinned ultralytics==8.4.172")
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    if (config["method"] != "p1" or config["p1"] != {
            "soft_weights": True, "uncertain_ignore": True}
            or config["train"]["imgsz"] != 640):
        raise ValueError("NWD screen requires unchanged P1 classification and 640 input")
    data = Path(config["data"]).resolve()
    dataset = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not dataset.get("p1_metadata"):
        raise ValueError("P1 metadata is required")
    options = dict(config["nwd"])
    if options.pop("normalizer") != "open_small_mean":
        raise ValueError("normalizer must be derived from open labels")
    options["normalizer"] = normalizer
    project = Path(config["project"]).resolve()
    run = project / config["name"]
    if run.exists():
        raise FileExistsError(run)
    metadata_file = Path(dataset["p1_metadata"])
    sources = [config_path, data, metadata_file, metadata_file.parent / "summary.json"]
    sources += list(Path(__file__).resolve().parents[1].rglob("*.py"))
    metadata = {
        "config_path": str(config_path), "model": config["model"], "data": str(data),
        "project": str(project), "name": config["name"], "train_args": config["train"],
        "method": "p1", "p1": config["p1"], "nwd": options,
        "ultralytics_version": ultralytics.__version__,
        "git_commit": _optional_git_commit(config_path.parent),
        "pseudo_summary": json.loads((metadata_file.parent / "summary.json").read_text(encoding="utf-8")),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    model = YOLO(config["model"])

    def save_metadata(trainer):
        (Path(trainer.save_dir) / "experiment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model.add_callback("on_pretrain_routine_end", save_metadata)
    model.train(data=str(data), name=config["name"], project=str(project), exist_ok=False,
                trainer=partial(NWDTrainer, nwd_options=options, p1_options=config["p1"]),
                **config["train"])
    return run
