"""V4 normalized continuous event supervision and global point-DFL control.

The density is a new log-linear lift; its mean is not the native decoded mean.
Exact E and the all-known P1 branch are preserved. No training entry is exposed.
"""
from __future__ import annotations
import weakref
from fractions import Fraction
import numpy as np
import torch
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.loss import DFLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors
from .bounded_retention import UnsupportedObservation
from .p1 import P1DetectionLoss, classification_factors
from .bounded_retention_continuous_domain import ContinuousDomain, active_mask
from .bounded_retention_continuous_numerics import DFLScore, integrate_domain, optimize_point
from .bounded_retention_continuous_normalizer import normalizer


def current_unknown(record, width, height):
    # Exact-real transport semantics, not V3's extra outward-query rounding.
    # The second item keeps the coverage observer's existing read-only API.
    return np.asarray(active_mask(record), dtype=bool), None


def score_for(logits, mask, anchor, stride):
    ids = mask.nonzero().flatten().tolist()
    return DFLScore(logits[mask].double(),
        offsets=tuple(float(anchor[i % 2]) for i in ids),
        signs=tuple(-1 if i < 2 else 1 for i in ids), stride=float(stride))


def continuous_support(record, box, anchors, strides, canvas, logits):
    domain = ContinuousDomain(record, box.detach().cpu().tolist(), canvas)
    expected, _ = current_unknown(record, *canvas)
    mask = torch.as_tensor(expected, dtype=torch.bool, device=box.device)
    if not mask.any():
        raise UnsupportedObservation("continuous support called without active U")
    if not np.array_equal(np.asarray(domain.mask, dtype=bool), expected):
        raise UnsupportedObservation("continuous domain changed original active-edge semantics")
    target = torch.cat((anchors - box[:2], box[2:] - anchors), 1) / strides
    geometry = box.double()[None, None].expand(len(anchors), 1, 4).clone()
    certificates, rounding = [], []
    for row in range(len(anchors)):
        score = score_for(logits[row].detach(), mask, anchors[row], strides[row].item())
        certificate = optimize_point(domain, score)
        if not certificate.diagnostics.get("qualified", False) or certificate.gap > 1e-5:
            raise UnsupportedObservation("global continuous point DFL gap exceeds contract")
        geometry[row, 0, mask] = torch.tensor([float(v) for v in certificate.point],
                                             device=box.device, dtype=torch.float64)
        rounding.append(tuple(abs(Fraction(float(v)) - v) for v in certificate.point))
        certificates.append(certificate)
    # E comes directly from the nominal float32 target, never a distance roundtrip.
    geometry[..., ~mask] = box[~mask].double()
    if not ((geometry[..., 2:] - geometry[..., :2]) > 0).all():
        raise UnsupportedObservation("continuous witness has nonpositive nominal geometry")
    return {"mask": mask, "target": target, "domain": domain, "geometry": geometry,
        "valid": torch.ones(len(anchors), 1, dtype=torch.bool, device=box.device),
        "anchors": anchors, "strides": strides, "canvas": canvas, "nominal_box": box.clone(),
        "point_certificates": certificates, "geometry_rounding_error": rounding,
        "identity": (record.get("source_object_id"), record.get("occurrence_id"))}


def partial_dfl(logits, support, anchor_row, mode):
    mask, target = support["mask"], support["target"][anchor_row]
    known = (~mask).nonzero().flatten()
    exact = (DFLoss(logits.shape[-1])(logits[known], target[known][None].clone()).squeeze()
             * len(known)) if len(known) else logits.sum() * 0
    anchor = support["anchors"][anchor_row]
    stride = support["strides"][anchor_row].item()
    if mode == "control":
        # The certified physical target is rational. Its rounded CIoU geometry
        # is not another certified point and must never define this DFL target.
        score = score_for(logits, mask, anchor, stride)
        certificate = support["point_certificates"][anchor_row]
        logp = score.logits.log_softmax(-1)
        terms, weight_error = [], Fraction(0)
        for axis, coordinate in enumerate(certificate.point):
            k, w = score.weights(axis, coordinate)
            wf = float(w)
            terms.append(-logp[axis, k] - wf * (logp[axis, k+1] - logp[axis, k]))
            weight_error += abs(Fraction(wf)-w) * abs(score.exact_logits[axis][k+1]-score.exact_logits[axis][k])
        arithmetic_budget = 128 * torch.finfo(torch.float64).eps * (1 + float(score.logits.detach().abs().max())) * score.dimension
        if float(certificate.gap + weight_error) + arithmetic_budget > 1e-5:
            raise UnsupportedObservation("continuous point encoding arithmetic exceeds global gap budget")
        unknown = torch.stack(terms).sum()
    elif mode == "candidate":
        score = score_for(logits, mask, anchor, stride)
        numerator = integrate_domain(support["domain"], score)
        log_denominator, denominator_diagnostics = normalizer(
            logits, support["nominal_box"], anchor, stride, mask, support["canvas"])
        if not numerator.diagnostics.get("qualified", False) or numerator.relative_residual_bound > 1e-4:
            raise UnsupportedObservation("continuous mass/gradient residual exceeds contract")
        if not denominator_diagnostics.get("qualified", False):
            raise UnsupportedObservation("continuous normalizer arithmetic is not qualified")
        unknown = log_denominator - numerator.log_integral
        if not torch.isfinite(unknown) or float(unknown.detach()) < -1e-8:
            raise UnsupportedObservation("invalid normalized continuous event mass")
    else:
        raise ValueError("partial loss is only defined for candidate/control")
    return (exact + unknown) / 4


class ContinuousObservationAssigner(TaskAlignedAssigner):
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
            # Encoding saturation does not remove continuous physical coordinates.
            indices = candidates[batch_i, gt_i].bool().nonzero().flatten()
            candidates[batch_i, gt_i] = False
            metric[batch_i, gt_i] = 0
            overlaps[batch_i, gt_i] = 0
            if not len(indices):
                raise UnsupportedObservation("partially unknown object has no native TAL candidate anchor")
            support = continuous_support(record, box, anc_points[indices], owner.strides[indices], owner.canvas, owner.detached_dist[batch_i, indices].view(-1, 4, owner.reg_max))
            known = support["target"][:, ~support["mask"]]
            owner.diagnostics["negative_known_distances"] += int((known < 0).sum())
            owner.diagnostics["overflow_known_distances"] += int((known > owner.reg_max - 1 - .01).sum())
            has_atoms = support["valid"].any(1)
            owner.diagnostics["empty_anchor_events"] += int((~has_atoms).sum())
            if not has_atoms.any():
                raise UnsupportedObservation("partially unknown object has no certified continuous anchor event")
            geometric_iou = bbox_iou(pd_bboxes[batch_i, indices, None, :], support["geometry"],
                                      xywh=False, CIoU=True).squeeze(-1)
            geometric_iou = geometric_iou.masked_fill(~support["valid"], -torch.inf)
            quality, chosen = geometric_iou.max(1)
            quality = torch.where(has_atoms, quality.clamp(min=0), torch.zeros_like(quality)).to(overlaps.dtype)
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


class ContinuousDensityLoss(P1DetectionLoss):
    partial_dfl = staticmethod(partial_dfl)

    def __init__(self, model):
        super().__init__(model)
        if self.reg_max != 16:
            raise ValueError("V4 continuous lift is frozen to the original 16-bin P1 head")
        self.mode = model.bounded_retention_mode
        if self.mode not in {"reference", "candidate", "control"}:
            raise ValueError("invalid retention mode")
        self.assigner = ContinuousObservationAssigner(self.assigner, self)
        self.observations = {}
        self.supports = {}
        self.diagnostics = {}

    def get_assigned_targets_and_loss(self, preds, batch):
        self.observations, self.supports = {}, {}
        self.diagnostics = {"supported_objects": 0, "empty_anchor_events": 0, "effective_anchors": 0, "negative_known_distances": 0, "overflow_known_distances": 0}
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
        self.detached_dist = pred_dist.detach()
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
            target_boxes[image, indices] = chosen_boxes.to(target_boxes.dtype)
            weight = target_scores[image, indices].sum(-1)
            iou = bbox_iou(pred_boxes[image, indices], chosen_boxes / strides[indices], xywh=False, CIoU=True).squeeze(-1)
            box_loss = box_loss + ((1 - iou) * weight).sum() / denominator
            terms = torch.stack([self.partial_dfl(pred_dist[image, index].view(4, self.reg_max), support, row, self.mode)
                                 for row, index in zip(rows, indices)])
            dfl_loss = dfl_loss + (terms * weight).sum() / denominator
            self.diagnostics["effective_anchors"] += len(indices)
        loss = torch.stack((box_loss * self.hyp.box, cls_loss * self.hyp.cls, dfl_loss * self.hyp.dfl))
        return ((fg_mask, target_idx, target_boxes, anchors, strides), loss, dict(zip(self.loss_names, loss.detach())))
