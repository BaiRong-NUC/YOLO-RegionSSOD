"""P1 adapter for the pinned Ultralytics 8.4.172 detection interfaces.

The loss follows ultralytics.utils.loss.v8DetectionLoss (AGPL-3.0), adding
classification weights and negative-anchor ignore without changing assignment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.instance import Instances
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class P1Dataset(YOLODataset):
    """Transport object IDs through the unchanged scalar-class augmentation path.

    IDs are decoded before collation; neither IDs nor ignore boxes reach TAL.
    Standard YOLO label files contain only accepted positive boxes.
    """

    def __init__(self, *args, metadata_path: Path, **kwargs):
        self.p1_metadata = {}
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = str(Path(row["image"]).resolve())
            if key in self.p1_metadata:
                raise ValueError(f"duplicate P1 metadata: {key}")
            self.p1_metadata[key] = row
        super().__init__(*args, **kwargs)
        nc = len(self.data["names"])
        tokens = [[i, 1.0, 0] for i in range(nc)]
        self.p1_tokens = {}
        self.p1_ignores = {}
        used = set()
        for label in self.labels:
            key = str(Path(label["im_file"]).resolve())
            row = self.p1_metadata.get(key)
            if row is None:
                # Only materialized labeled tiles may omit P1 metadata. This
                # fails closed if a pseudo image loses its sidecar entry.
                if "labeled" not in Path(key).parts:
                    raise ValueError(f"training image has no P1 metadata and is not a labeled tile: {key}")
                continue
            used.add(key)
            positive = row["positive"]
            expected_cls = np.array([p["class_id"] for p in positive]).reshape(-1, 1)
            expected_boxes = np.array([p["xywhn"] for p in positive]).reshape(-1, 4)
            if (label["cls"].shape != expected_cls.shape or
                    not np.array_equal(label["cls"], expected_cls) or
                    not np.allclose(label["bboxes"], expected_boxes, atol=1e-6, rtol=0)):
                raise ValueError(f"P1 metadata/positive label mismatch: {key}")
            ids = []
            for ignore, items in ((False, positive), (True, row["ignore"])):
                for item in items:
                    cls, conf = item["class_id"], item["confidence"]
                    box = np.array(item["xywhn"], dtype=float)
                    if (not 0 <= cls < nc or int(cls) != cls or not 0 < conf <= 1 or
                            box.shape != (4,) or not np.isfinite(box).all() or
                            np.any(box < 0) or np.any(box > 1) or min(box[2:]) <= 0):
                        raise ValueError(f"invalid P1 metadata: {key}")
                    ids.append(len(tokens))
                    tokens.append([cls, conf, int(ignore)])
            self.p1_tokens[key] = np.array(ids, dtype=np.float32).reshape(-1, 1)
            self.p1_ignores[key] = np.array([p["xywhn"] for p in row["ignore"]], dtype=np.float32).reshape(-1, 4)
        if used != set(self.p1_metadata):
            raise ValueError("P1 metadata and training list must contain the same pseudo images (fraction=1 required)")
        if len(tokens) >= 2**24:
            raise ValueError("too many object IDs for exact float32 augmentation transport")
        self.p1_lookup = torch.tensor(tokens, dtype=torch.float32)

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        key = str(Path(label["im_file"]).resolve())
        if key in self.p1_tokens:
            label["cls"] = self.p1_tokens[key].copy()
            boxes = self.p1_ignores[key]
            if len(boxes):
                ignore = Instances(boxes.copy(), np.zeros((0, 1000, 2), dtype=np.float32),
                                   bbox_format="xywh", normalized=True)
                label["instances"] = Instances.concatenate([label["instances"], ignore])
        return label

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        ids = sample["cls"].view(-1).long()
        decoded = self.p1_lookup[ids]
        ignore = decoded[:, 2].bool()
        sample["ignore_bboxes"] = sample["bboxes"][ignore]
        sample["ignore_batch_idx"] = sample["batch_idx"][ignore]
        sample["pseudo_weights"] = decoded[~ignore, 1]
        sample["cls"] = decoded[~ignore, :1]
        sample["bboxes"] = sample["bboxes"][~ignore]
        sample["batch_idx"] = sample["batch_idx"][~ignore]
        return sample

    @staticmethod
    def collate_fn(batch):
        ordinary = [{k: v for k, v in sample.items()
                     if k not in {"ignore_bboxes", "ignore_batch_idx", "pseudo_weights"}} for sample in batch]
        result = YOLODataset.collate_fn(ordinary)
        result["pseudo_weights"] = torch.cat([s["pseudo_weights"] for s in batch])
        result["ignore_bboxes"] = torch.cat([s["ignore_bboxes"] for s in batch])
        result["ignore_batch_idx"] = torch.cat([s["ignore_batch_idx"] + i for i, s in enumerate(batch)])
        return result


def classification_factors(anchor_pixels, fg_mask, target_gt_idx, batch, batch_size,
                           device, imgsz, *, soft_weights=True, uncertain_ignore=True):
    """Positive anchor BCE gets teacher confidence; uncertain negative BCE is zero.

    Denominator is the original unweighted TAL target-score sum, so confidence
    scaling cannot cancel itself. All class logits at an ignored anchor are masked.
    """
    factors = torch.ones_like(fg_mask, dtype=torch.float32, device=device)
    if soft_weights and "pseudo_weights" in batch:
        weights = batch["pseudo_weights"].to(device)
        indices = batch["batch_idx"].to(device).long()
        for image_index in range(batch_size):
            image_weights = weights[indices == image_index]
            selected = fg_mask[image_index]
            factors[image_index, selected] = image_weights[target_gt_idx[image_index, selected].long()]
    if uncertain_ignore and "ignore_bboxes" in batch:
        boxes = batch["ignore_bboxes"].to(device)
        indices = batch["ignore_batch_idx"].to(device).long()
        scale = imgsz[[1, 0]]
        for image_index in range(batch_size):
            # Chunking bounds memory for crowded images and large feature maps.
            image_boxes = boxes[indices == image_index]
            inside = torch.zeros_like(fg_mask[image_index])
            for chunk in image_boxes.split(64):
                lo = (chunk[:, :2] - chunk[:, 2:] / 2) * scale
                hi = (chunk[:, :2] + chunk[:, 2:] / 2) * scale
                inside |= ((anchor_pixels[:, None] >= lo) & (anchor_pixels[:, None] <= hi)).all(-1).any(-1)
            factors[image_index, inside & ~fg_mask[image_index]] = 0
    return factors.unsqueeze(-1)


class P1DetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.p1_options = model.p1_options

    def get_assigned_targets_and_loss(self, preds, batch):
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
        bce = self.bce(pred_scores, target_scores.to(dtype))
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
        return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
                loss, dict(zip(self.loss_names, loss.detach())))


class P1DetectionModel(DetectionModel):
    def init_criterion(self):
        return P1DetectionLoss(self)


class P1Trainer(DetectionTrainer):
    def __init__(self, *args, p1_options=None, **kwargs):
        self.p1_options = p1_options or {"soft_weights": True, "uncertain_ignore": True}
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(P1DetectionModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        if self.args.single_cls or self.args.classes is not None or self.args.fraction != 1:
            raise ValueError("P1 requires all classes and fraction=1")
        metadata_path = Path(self.data["p1_metadata"])
        return P1Dataset(
            img_path=img_path, metadata_path=metadata_path, imgsz=self.args.imgsz,
            batch_size=batch, augment=True, hyp=self.args, rect=self.args.rect,
            cache=self.args.cache or None, single_cls=False, stride=max(int(self.model.stride.max()), 32),
            pad=0.0, prefix="train: ", task="detect", data=self.data, fraction=1.0)
