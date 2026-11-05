"""Native-P1 TAL event revision; v1 source and STOP remain protected.

Unknown sides use a certified physical observation oracle. Exact prediction
coordinates remain the native P1 nominal targets. The oracle never sees logits,
validation data, or hidden labels. Candidate/control differ only in sum vs max.
"""
from __future__ import annotations

from itertools import product
import weakref

import numpy as np
import torch
from torch.nn import functional as F
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.loss import DFLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors

from .bounded_retention import UnsupportedObservation
from .p1 import P1DetectionLoss, classification_factors


def _multiply_interval(a, b):
    products = np.stack([a[..., i] * b[..., j] for i in (0, 1) for j in (0, 1)], -1)
    if np.isnan(products).any():
        raise UnsupportedObservation("undefined interval multiplication")
    return np.stack((np.nextafter(products.min(-1), -np.inf),
                     np.nextafter(products.max(-1), np.inf)), -1)


def inverse_interval(coordinates, scale, offset):
    """Outward enclosure of exact binary-parameter affine inverse."""
    scale, offset = np.asarray(scale), np.asarray(offset)
    if scale[0] <= 0 <= scale[1] or not np.isfinite(scale).all():
        raise UnsupportedObservation("inverse affine scale crosses zero")
    difference = np.stack((np.nextafter(coordinates[..., 0] - offset[1], -np.inf),
                           np.nextafter(coordinates[..., 1] - offset[0], np.inf)), -1)
    reciprocal = np.array([np.nextafter(1 / scale[1], -np.inf),
                           np.nextafter(1 / scale[0], np.inf)])
    return _multiply_interval(difference, reciprocal)


def _transport_bounds(record):
    bounds = np.asarray(record["affine_bounds"], dtype=np.float64)
    clip = np.asarray(record["augmentation_clip_window_bounds"], dtype=np.float64)
    if (bounds.shape != (4, 2) or clip.shape != (4, 2) or not np.isfinite(bounds).all()
            or np.isnan(clip).any() or (bounds[:, 0] > bounds[:, 1]).any()
            or (clip[:, 0] > clip[:, 1]).any()):
        raise UnsupportedObservation("invalid certified transport bounds")
    if any(bounds[axis, 0] <= 0 <= bounds[axis, 1] for axis in (0, 1)):
        raise UnsupportedObservation("affine scale interval crosses zero")
    return bounds, clip


def current_unknown(record, width, height):
    """Original U only; resolve solely when accumulated clipping proves it.

    Uses outward source/transport bounds, never a final nominal-canvas test.
    The cumulative augmentation clipping domain is separate from the tile.
    """
    sx, sy, tx, ty = np.asarray(record["affine"], dtype=np.float64)
    if not np.isfinite([sx, sy, tx, ty]).all() or sx == 0 or sy == 0:
        raise UnsupportedObservation("invalid affine observation transport")
    original = np.asarray(record["original_unknown"], dtype=bool)
    bounds, clip = _transport_bounds(record)
    visible = np.asarray(record["physical"].visible_edge_bounds, dtype=np.float64)
    if visible.shape != (4, 2) or not np.isfinite(visible).all() or (visible[:, 0] > visible[:, 1]).any():
        raise UnsupportedObservation("physical source requires certified visible edge bounds")
    mask = np.zeros(4, dtype=bool)
    mapping = []
    for side in range(4):
        axis = side % 2
        scale, offset, limit = (sx, tx, width) if axis == 0 else (sy, ty, height)
        mapped = side if scale > 0 else (side + 2) % 4
        # An input left/top can extend down to -inf; its maximum is bounded
        # by the physical visible edge. Right/bottom has the dual minimum.
        extreme = visible[side, 1 if side < 2 else 0]
        value = _multiply_interval(np.array([extreme, extreme]), bounds[axis])
        value = np.array([np.nextafter(value[0] + bounds[axis + 2, 0], -np.inf),
                          np.nextafter(value[1] + bounds[axis + 2, 1], np.inf)])
        resolved = value[1] <= clip[mapped, 0] if mapped < 2 else value[0] >= clip[mapped, 1]
        active = bool(original[side] and not resolved)
        mask[mapped] = active
        if active:
            mapping.append((side, mapped, scale, offset))
    return mask, mapping


def support_geometry(record, box, anchors, strides, canvas, bins=16):
    """One object's common finite-support set for all candidate anchors.

    Geometry is label-only. Return valid atoms; empty anchors are explicit,
    never clamped. Caller must not silently discard an entire labeled object.
    CPU float64 queries keep source certification outside autocast.
    """
    width, height = canvas
    mask_np, mapping = current_unknown(record, width, height)
    mask = torch.as_tensor(mask_np, dtype=torch.bool, device=box.device)
    unknown_ids = mask.nonzero().flatten()
    if not len(unknown_ids):
        return None
    values = torch.tensor(list(product(range(bins), repeat=len(unknown_ids))),
                          dtype=torch.long, device=box.device)
    n, k = len(anchors), len(values)
    # All distances are pre-clamp, in native stride units.
    target = torch.cat((anchors - box[:2], box[2:] - anchors), 1) / strides
    safe = (target[:, ~mask] >= 0).all(1) & (target[:, ~mask] <= bins - 1 - .01).all(1)
    distances = target[:, None, :].expand(n, k, 4).clone()
    distances[:, :, unknown_ids] = values.to(distances.dtype)
    anchor4 = torch.cat((anchors, anchors), 1)[:, None, :]
    signs = torch.tensor([-1., -1., 1., 1.], device=box.device)
    geometry = anchor4 + distances * strides[:, None, :] * signs
    # Preserve fixed nominal E; distance subtraction/addition is not an
    # identity in floating point (e.g. 1e-6 -> 0 next to anchor=64).
    geometry[:, :, ~mask] = box[~mask]
    limits = torch.tensor([width, height, width, height], device=box.device)
    valid = safe[:, None] & ((geometry >= 0) & (geometry <= limits)).all(2)
    valid &= (geometry[:, :, 2:] > geometry[:, :, :2]).all(2)
    flat = geometry.detach().cpu().double().numpy().reshape(-1, 4)
    affine_bounds, clip_bounds = _transport_bounds(record)
    ranges = np.empty((n * k, 4, 2), dtype=np.float64)
    ranges[:, :, 0], ranges[:, :, 1] = -np.inf, np.inf
    for original_side, mapped_side, scale, offset in mapping:
        axis = original_side % 2
        coordinate = flat[:, mapped_side]
        lower, upper = clip_bounds[mapped_side % 2], clip_bounds[mapped_side % 2 + 2]
        inside = (coordinate >= lower[0]) & (coordinate <= upper[1])
        valid &= torch.as_tensor(inside.reshape(n, k), device=box.device)
        preclip = np.repeat(coordinate[:, None], 2, 1)
        preclip[(coordinate >= lower[0]) & (coordinate <= lower[1]), 0] = -np.inf
        preclip[(coordinate >= upper[0]) & (coordinate <= upper[1]), 1] = np.inf
        ranges[:, original_side] = inverse_interval(preclip, affine_bounds[axis], affine_bounds[axis + 2])
    eligible = valid.detach().cpu().numpy().reshape(-1)
    certificate = record["physical"].compatible_ranges(ranges[eligible])
    if not isinstance(certificate, np.ndarray) or certificate.dtype != np.bool_ or certificate.shape != (int(eligible.sum()),):
        raise ValueError("physical oracle must return explicit boolean compatibility for each joint atom")
    joint = np.zeros(n * k, dtype=bool)
    joint[eligible] = certificate
    valid &= torch.as_tensor(joint.reshape(n, k), device=box.device)
    return {"mask": mask, "values": values, "distances": distances, "geometry": geometry,
            "valid": valid, "target": target}


def partial_dfl(logits, support, anchor_row, mode):
    """Shared support, fixed exact nominal coordinates, original /4 scale."""
    valid = support["valid"][anchor_row]
    if not valid.any():
        raise UnsupportedObservation("assigned anchor has empty discrete support")
    mask, target = support["mask"], support["target"][anchor_row]
    logp = F.log_softmax(logits.float(), dim=-1)
    known = (~mask).nonzero().flatten()
    # Invoke native DFL on known edges at its original input precision. Its
    # clamp is inactive because support_geometry already rejected overflow.
    exact = (DFLoss(logits.shape[-1])(logits[known], target[known][None].clone()).squeeze()
             * len(known)) if len(known) else logits.sum() * 0
    atoms = support["values"][valid]
    lp = logp[mask.nonzero().flatten()[None], atoms].sum(1)
    event = torch.logsumexp(lp, 0) if mode == "candidate" else lp.max()
    return (exact - event) / 4


class ObservationAssigner(TaskAlignedAssigner):
    def __init__(self, original, owner):
        super().__init__(topk=original.topk, num_classes=original.num_classes,
            alpha=original.alpha, beta=original.beta, stride=original.stride,
            eps=original.eps, topk2=original.topk2)
        object.__setattr__(self, "owner_ref", weakref.ref(owner))

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        if not self.owner_ref().observations:
            return super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        # Native OOM fallback slices the batch without remapping sidecar IDs.
        # Fail the attempt instead of retrying with another object's evidence.
        self.bs, self.n_max_boxes = pd_scores.shape[0], gt_bboxes.shape[1]
        return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        owner = self.owner_ref()
        candidates = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        candidates = candidates * mask_gt
        metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, candidates)
        self.legal_assignment = torch.ones_like(candidates, dtype=torch.bool)
        for (batch_i, gt_i), record in owner.observations.items():
            box = gt_bboxes[batch_i, gt_i]
            effective, _ = current_unknown(record, *owner.canvas)
            if not effective.any():
                continue
            # Use only the original P1 pool, including its tiny-box expansion.
            # Unknown nominal distances do not pre-empt actual event atoms.
            indices = candidates[batch_i, gt_i].bool().nonzero().flatten()
            candidates[batch_i, gt_i] = False
            metric[batch_i, gt_i] = 0
            overlaps[batch_i, gt_i] = 0
            if not len(indices):
                raise UnsupportedObservation("partially unknown object has no native TAL candidate anchor")
            support = support_geometry(record, box, anc_points[indices], owner.strides[indices], owner.canvas, owner.reg_max)
            has_atoms = support["valid"].any(1)
            owner.diagnostics["empty_anchor_events"] += int((~has_atoms).sum())
            if not has_atoms.any():
                raise UnsupportedObservation("partially unknown object has no legal discrete anchor event")
            geometric_iou = bbox_iou(pd_bboxes[batch_i, indices, None, :], support["geometry"],
                                      xywh=False, CIoU=True).squeeze(-1)
            geometric_iou = geometric_iou.masked_fill(~support["valid"], -torch.inf)
            quality, chosen = geometric_iou.max(1)
            quality = torch.where(has_atoms, quality.clamp(min=0), torch.zeros_like(quality))
            cls = gt_labels[batch_i, gt_i, 0].long()
            overlaps[batch_i, gt_i, indices] = quality
            metric[batch_i, gt_i, indices] = pd_scores[batch_i, indices, cls].pow(self.alpha) * quality.pow(self.beta)
            candidates[batch_i, gt_i, indices] = has_atoms.to(candidates.dtype)
            support["indices"] = indices
            support["chosen"] = chosen
            self.legal_assignment[batch_i, gt_i] = False
            self.legal_assignment[batch_i, gt_i, indices] = has_atoms
            owner.supports[batch_i, gt_i] = support
            owner.diagnostics["supported_objects"] += 1
        topk = self.select_topk_candidates(metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        return topk * candidates * mask_gt.bool(), metric, overlaps

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        if not self.owner_ref().observations:
            return super().select_highest_overlaps(mask_pos, overlaps, n_max_boxes, align_metric)
        # Native tie-breaking can select a GT outside its candidate set. Do
        # not let an all-zero IoU tie revive an illegal partial observation.
        eligible_overlap = overlaps.masked_fill(~self.legal_assignment, -1)
        indices, fg, positives = super().select_highest_overlaps(
            mask_pos, eligible_overlap, n_max_boxes, align_metric)
        if (positives.bool() & ~self.legal_assignment).any():
            raise UnsupportedObservation("TAL conflict resolution selected an illegal observation")
        return indices, fg, positives


class NativeEventLoss(P1DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.mode = model.bounded_retention_mode
        if self.mode not in {"reference", "candidate", "control"}:
            raise ValueError("invalid retention mode")
        self.assigner = ObservationAssigner(self.assigner, self)
        self.observations = {}
        self.supports = {}
        self.diagnostics = {}

    def get_assigned_targets_and_loss(self, preds, batch):
        self.observations, self.supports = {}, {}
        self.diagnostics = {"supported_objects": 0, "empty_anchor_events": 0, "effective_anchors": 0}
        if self.mode == "reference":
            return super().get_assigned_targets_and_loss(preds, batch)
        records = batch.get("retention_observations")
        if records is None or len(records) != len(batch["cls"]):
            raise ValueError("explicit aligned retention observations required")
        human, object_ids = batch.get("object_is_human"), batch.get("object_id")
        occurrences = batch.get("obs", {}).get("occurrence_id")
        if (not isinstance(human, torch.Tensor) or human.dtype != torch.bool
                or human.shape != (len(records),) or not isinstance(object_ids, torch.Tensor)
                or object_ids.shape != (len(records),) or occurrences is None
                or np.asarray(occurrences).shape != (len(records),)):
            raise ValueError("explicit human/object/occurrence identity required")
        for is_human, object_id, occurrence, record in zip(human.tolist(), object_ids.tolist(), occurrences, records):
            if is_human != (record is not None):
                raise ValueError("human observation missing or pseudo carries human evidence")
            if record is not None and (record["source_object_id"] != object_id or record["occurrence_id"] != int(occurrence)):
                raise ValueError("observation identity mismatch")
        batch_ids = batch["batch_idx"].reshape(-1).long().tolist()
        local_counts = {}
        for image, record in zip(batch_ids, records):
            index = local_counts.get(image, 0)
            local_counts[image] = index + 1
            if record is not None:
                self.observations[image, index] = record
        # Exact/pseudo-only batches retain original implementation wholesale.
        if not self.observations:
            return super().get_assigned_targets_and_loss(preds, batch)
        pred_dist = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchors, strides = make_anchors(preds["feats"], self.stride, .5)
        self.strides = strides
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_scores.dtype) * self.stride[0]
        self.canvas = (float(imgsz[1]), float(imgsz[0]))
        if not any(current_unknown(record, *self.canvas)[0].any() for record in self.observations.values()):
            self.observations = {}
            return super().get_assigned_targets_and_loss(preds, batch)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), pred_scores.shape[0], scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_cls, gt_boxes = targets.split((1, 4), 2)
        pred_boxes = self.bbox_decode(anchors, pred_dist)
        _, target_boxes, target_scores, fg_mask, target_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_boxes.detach() * strides).type(gt_boxes.dtype),
            anchors * strides, gt_cls, gt_boxes, gt_boxes.sum(2, keepdim=True).gt_(0))
        denominator = target_scores.sum().clamp_(min=1)
        bce = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
        if self.class_weights is not None:
            bce = bce * self.class_weights
        factors = classification_factors(anchors * strides, fg_mask, target_idx, batch,
            pred_scores.shape[0], self.device, imgsz, **self.p1_options)
        cls_loss = (bce * factors).sum() / denominator
        # Baseline regression on ordinary anchors; no partial point targets
        # reach native CIoU/DFL and then get numerically subtracted afterwards.
        ordinary = fg_mask.clone()
        selections = []
        for (image, gt_index), support in self.supports.items():
            chosen_mask = fg_mask[image, support["indices"]] & (target_idx[image, support["indices"]] == gt_index)
            rows = chosen_mask.nonzero().flatten()
            if len(rows):
                indices = support["indices"][rows]
                ordinary[image, indices] = False
                selections.append((image, support, rows, indices))
        box_loss, dfl_loss = self.bbox_loss(pred_dist, pred_boxes, anchors, target_boxes / strides,
                                           target_scores, denominator, ordinary, imgsz, strides)
        for image, support, rows, indices in selections:
            chosen_boxes = support["geometry"][rows, support["chosen"][rows]].detach()
            target_boxes[image, indices] = chosen_boxes
            weight = target_scores[image, indices].sum(-1)
            iou = bbox_iou(pred_boxes[image, indices], chosen_boxes / strides[indices], xywh=False, CIoU=True).squeeze(-1)
            box_loss = box_loss + ((1 - iou) * weight).sum() / denominator
            terms = torch.stack([partial_dfl(pred_dist[image, index].view(4, self.reg_max), support, row, self.mode)
                                 for row, index in zip(rows, indices)])
            dfl_loss = dfl_loss + (terms * weight).sum() / denominator
            self.diagnostics["effective_anchors"] += len(indices)
        loss = torch.stack((box_loss * self.hyp.box, cls_loss * self.hyp.cls, dfl_loss * self.hyp.dfl))
        return ((fg_mask, target_idx, target_boxes, anchors, strides), loss, dict(zip(self.loss_names, loss.detach())))
