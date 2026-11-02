"""Native-pixel COCO AP and realizable global-threshold precision diagnostics."""
from __future__ import annotations

from contextlib import redirect_stdout
import io

import numpy as np


def assign_predictions(classes, scores, xyxy, gt_classes, gt_xyxy, iou=0.5):
    """Return a GT index or -1 for each prediction; match all sizes jointly."""
    boxes = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt_xyxy, dtype=float).reshape(-1, 4)
    cls, gt_cls = np.asarray(classes), np.asarray(gt_classes)
    scores = np.asarray(scores, dtype=float)
    if len(boxes) != len(cls) or len(scores) != len(cls) or len(gt) != len(gt_cls):
        raise ValueError("inconsistent prediction/GT lengths")
    if not (0 < iou <= 1) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("invalid boxes, scores or matching IoU")
    result = np.full(len(boxes), -1, dtype=np.int64)
    if not len(gt) or not len(boxes):
        return result
    intersect = np.maximum(0, np.minimum(boxes[:, None, 2:], gt[None, :, 2:]) -
                           np.maximum(boxes[:, None, :2], gt[None, :, :2])).prod(2)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    gt_area = np.maximum(0, gt[:, 2:] - gt[:, :2]).prod(1)
    overlaps = intersect / np.maximum(area[:, None] + gt_area[None, :] - intersect, 1e-12)
    matched = np.zeros(len(gt), dtype=bool)
    for index in np.argsort(-scores, kind="stable"):
        eligible = (gt_cls == cls[index]) & ~matched & (overlaps[index] >= iou)
        if eligible.any():
            best = np.where(eligible, overlaps[index], -1).argmax()
            matched[best] = True
            result[index] = best
    return result


def threshold_curve(scores, matched_areas, gt_counts):
    """Include entire equal-score groups so each row is an actual threshold."""
    scores, areas = np.asarray(scores, dtype=float), np.asarray(matched_areas, dtype=float)
    if len(scores) != len(areas) or not np.isfinite(scores).all():
        raise ValueError("invalid score/area population")
    order = np.argsort(-scores, kind="stable")
    scores, areas = scores[order], areas[order]
    ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True]) if len(scores) else np.array([], int)
    tp = np.cumsum(areas >= 0, dtype=np.int64)[ends]
    n = ends + 1
    small = np.cumsum((areas >= 0) & (areas < 1024), dtype=np.int64)[ends]
    medium = np.cumsum((areas >= 1024) & (areas < 9216), dtype=np.int64)[ends]
    large = np.cumsum(areas >= 9216, dtype=np.int64)[ends]
    return {"threshold": scores[ends], "predictions": n, "tp": tp, "fp": n - tp,
            "precision": tp / n, "recall": tp / sum(gt_counts.values()),
            "small_tp": small, "small_recall": small / gt_counts["small"],
            "medium_tp": medium, "medium_recall": medium / gt_counts["medium"],
            "large_tp": large, "large_recall": large / gt_counts["large"]}


def curve_point(curve, index):
    return {key: value[index].item() for key, value in curve.items()}


def precision_constrained_point(curve, target):
    """Empirical maximum small recall subject to P>=target; no interpolation."""
    if not 0 < target <= 1:
        raise ValueError("precision constraint must be in (0,1]")
    eligible = np.flatnonzero(curve["precision"] >= target)
    if not len(eligible):
        return None
    # Recall is nondecreasing with prefix length. Last feasible prefix also
    # gives the broadest coverage if several prefixes have equal small recall.
    return curve_point(curve, eligible[-1])


def fixed_threshold_point(curve, threshold):
    selected = np.flatnonzero(curve["threshold"] >= threshold)
    return curve_point(curve, selected[-1]) if len(selected) else None


def coco_ap(gt_dataset, detections):
    """COCO defaults for area matching; report tensors at maxDet=500 explicitly."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = gt_dataset
        truth.createIndex()
        if detections:
            predicted = truth.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = {"images": gt_dataset["images"],
                                 "categories": gt_dataset["categories"], "annotations": []}
            predicted.createIndex()
        evaluator = COCOeval(truth, predicted, "bbox")
        evaluator.params.maxDets = [1, 100, 500]
        evaluator.evaluate()
        evaluator.accumulate()
    tensor = evaluator.eval["precision"]  # IoU x recall x class x area x maxDet
    max_recall = evaluator.eval["recall"]
    metrics, per_class, curves = {}, {}, {}
    for area_index, label in enumerate(evaluator.params.areaRngLbl):
        values = tensor[:, :, :, area_index, -1]
        valid = values[values >= 0]
        metrics[f"AP_{label}"] = float(valid.mean()) if len(valid) else None
        for t, tag in ((0, "50"), (5, "75")):
            valid = values[t][values[t] >= 0]
            metrics[f"AP{tag}_{label}"] = float(valid.mean()) if len(valid) else None
            per_recall = values[t]
            support = (per_recall >= 0).sum(axis=1)
            mean = np.divide(np.maximum(per_recall, 0).sum(axis=1), support,
                             out=np.zeros(len(support)), where=support > 0)
            curves[f"{label}_iou{tag}"] = mean
        valid = max_recall[:, :, area_index, -1]
        metrics[f"AR_{label}_500"] = float(valid[valid >= 0].mean()) if (valid >= 0).any() else None
    for index, category in enumerate(gt_dataset["categories"]):
        values = tensor[:, :, index, 1, -1]
        valid = values[values >= 0]
        per_class[category["name"]] = float(valid.mean()) if len(valid) else None
    return {"metrics": metrics, "per_class_AP_small": per_class,
            "area_ranges": evaluator.params.areaRng,
            "area_labels": evaluator.params.areaRngLbl,
            "recall_grid": evaluator.params.recThrs.tolist(),
            "max_dets": evaluator.params.maxDets}, curves
