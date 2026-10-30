"""Confidence-prefix diagnostics on saved predictions, without new inference."""
from __future__ import annotations

import numpy as np


def prediction_matches(classes, scores, boxes, gt_classes, gt_boxes, ious):
    """Return GT indices per prediction/IoU using the existing recall rule."""
    classes, scores = np.asarray(classes), np.asarray(scores, dtype=float)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    gt_classes = np.asarray(gt_classes)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    ious = np.asarray(ious, dtype=float)
    if (len(classes) != len(scores) or len(boxes) != len(scores)
            or len(gt_classes) != len(gt_boxes)):
        raise ValueError("inconsistent prediction or GT lengths")
    if (not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
            or not np.isfinite(boxes).all() or not np.isfinite(gt_boxes).all()
            or np.any(boxes[:, 2:] <= 0) or np.any(gt_boxes[:, 2:] <= 0)
            or not len(ious) or not np.isfinite(ious).all()
            or np.any((ious <= 0) | (ious > 1))):
        raise ValueError("invalid scores, boxes or matching IoUs")
    matches = np.full((len(ious), len(scores)), -1, dtype=int)
    if not len(scores) or not len(gt_boxes):
        return matches
    lo, hi = boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2
    gt_lo = gt_boxes[:, :2] - gt_boxes[:, 2:] / 2
    gt_hi = gt_boxes[:, :2] + gt_boxes[:, 2:] / 2
    intersection = np.maximum(
        0, np.minimum(hi[:, None], gt_hi) - np.maximum(lo[:, None], gt_lo)
    ).prod(axis=2)
    union = boxes[:, 2:].prod(axis=1)[:, None] + gt_boxes[:, 2:].prod(axis=1) - intersection
    overlaps = intersection / np.maximum(union, 1e-12)
    for row, threshold in enumerate(ious):
        used = np.zeros(len(gt_boxes), dtype=bool)
        for index in np.argsort(-scores, kind="stable"):
            eligible = (gt_classes == classes[index]) & ~used & (overlaps[index] >= threshold)
            if eligible.any():
                best = np.where(eligible, overlaps[index], -1).argmax()
                matches[row, index] = best
                used[best] = True
    return matches


def operating_curve(scores, matched_sizes, gt_counts, *, confidence_floor):
    """Aggregate exact attainable global thresholds; never split score ties.

    matched_sizes: -1 for FP; 0/1/2 for matched small/medium/large GT.
    Prefix matching is valid because removing lower-confidence predictions cannot
    change an earlier greedy assignment within the same image.
    """
    scores = np.asarray(scores, dtype=float)
    sizes = np.asarray(matched_sizes, dtype=int)
    gt_counts = np.asarray(gt_counts, dtype=int)
    if (scores.ndim != 1 or sizes.shape != scores.shape or gt_counts.shape != (3,)
            or np.any(gt_counts < 0) or np.any((sizes < -1) | (sizes > 2))
            or not np.isfinite(scores).all()
            or not 0 < confidence_floor <= 1
            or np.any((scores < confidence_floor) | (scores > 1))):
        raise ValueError("invalid confidence-prefix inputs")
    order = np.argsort(-scores, kind="stable")
    scores, sizes = scores[order], sizes[order]
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True]) if len(scores) else np.array([], dtype=int)
    # The empty prefix is attainable only if there are no confidence-one boxes.
    empty = not len(scores) or scores[0] < 1
    thresholds = np.r_[1., scores[ends]] if empty else scores[ends]
    predictions = np.r_[0, ends + 1] if empty else ends + 1
    counts = np.column_stack([
        np.r_[0, np.cumsum(sizes == size)[ends]] if empty
        else np.cumsum(sizes == size)[ends]
        for size in range(3)
    ])
    tp = counts.sum(axis=1)
    if np.any(counts > gt_counts):
        raise ValueError("matched counts exceed GT denominators")
    return {"thresholds": thresholds, "predictions": predictions, "matched": counts,
            "fp": predictions - tp, "gt": gt_counts, "floor": confidence_floor}


def curve_point(curve, *, threshold=None, fp_budget=None):
    """Select a fixed threshold or the widest prefix within a fixed FP budget."""
    if (threshold is None) == (fp_budget is None):
        raise ValueError("specify exactly one of threshold or fp_budget")
    if threshold is not None:
        if not curve["floor"] <= threshold <= 1:
            raise ValueError("cannot reconstruct thresholds below the saved confidence floor")
        available = np.flatnonzero(curve["thresholds"] >= threshold)
    elif fp_budget is not None:
        if fp_budget < 0:
            raise ValueError("FP budget must be nonnegative")
        available = np.flatnonzero(curve["fp"] <= fp_budget)
    else:
        raise ValueError("missing threshold or FP budget")
    if not len(available):
        return None
    index = available[-1]
    counts = curve["matched"][index]
    predictions = int(curve["predictions"][index])
    tp = int(counts.sum())
    return {
        "threshold": float(threshold if threshold is not None else curve["thresholds"][index]),
        "predictions": predictions, "tp": tp, "fp": int(curve["fp"][index]),
        "precision": tp / predictions if predictions else None,
        "matched": counts.tolist(), "gt": curve["gt"].tolist(),
        "recall": [int(n) / int(d) if d else None for n, d in zip(counts, curve["gt"])],
    }
