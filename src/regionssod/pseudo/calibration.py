"""Fit reliability maps from open-label binary matches, never hidden labels."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def monotone_means(values, weights):
    """Weighted pool-adjacent-violators projection onto increasing means."""
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if (values.ndim != 1 or values.shape != weights.shape
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights <= 0).any()):
        raise ValueError("invalid isotonic inputs")
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([index, index + 1, value * weight, weight])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    result = np.empty_like(values)
    for start, end, total, weight in blocks:
        result[start:end] = total / weight
    return result.tolist()


def score_bins(items, edges):
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0)):
        raise ValueError("invalid score edges")
    scores = np.asarray([p["confidence"] for p in items], dtype=float)
    if (not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1) | (scores < edges[0]) | (scores >= edges[-1])).any()):
        raise ValueError("score outside calibration domain")
    return np.searchsorted(edges, scores, side="right") - 1


def group_key(item):
    if (not isinstance(item["class_id"], int) or item["class_id"] < 0
            or item["scale"] not in ("small", "medium", "large")):
        raise ValueError("invalid class/scale")
    return f"{item['class_id']}:{item['scale']}"


def fit_calibrator(items, targets, *, edges, global_prior=20, group_prior=50, min_group=50):
    """Targets must come from open annotated tile labels, supplied by caller."""
    target = np.asarray(targets, dtype=float)
    if (not items or target.shape != (len(items),) or not np.isin(target, [0, 1]).all()
            or not np.isfinite([global_prior, group_prior, min_group]).all()
            or min(global_prior, group_prior, min_group) <= 0):
        raise ValueError("invalid calibration targets/prior")
    bins = score_bins(items, edges)
    count = np.bincount(bins, minlength=len(edges) - 1)
    true = np.bincount(bins, weights=target, minlength=len(edges) - 1)
    prior = np.clip((np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2, 0, 1)
    global_probs = monotone_means((true + global_prior * prior) / (count + global_prior),
                                  count + global_prior)
    indices = defaultdict(list)
    for i, item in enumerate(items):
        indices[group_key(item)].append(i)
    groups = {}
    for key, selected in sorted(indices.items()):
        if len(selected) < min_group:
            continue
        count = np.bincount(bins[selected], minlength=len(edges) - 1)
        true = np.bincount(bins[selected], weights=target[selected], minlength=len(edges) - 1)
        groups[key] = {
            "n": len(selected), "bin_count": count.tolist(),
            "probabilities": monotone_means(
                (true + group_prior * np.asarray(global_probs)) / (count + group_prior),
                count + group_prior)}
    return {"edges": list(edges), "fit_positives": len(items),
            "global_probabilities": global_probs, "groups": groups}


def calibrated_weights(items, model, *, conditional=True):
    bins = score_bins(items, model["edges"])
    weights, supported = [], []
    for item, band in zip(items, bins, strict=True):
        group = model["groups"].get(group_key(item)) if conditional else None
        values = group["probabilities"] if group is not None else model["global_probabilities"]
        weights.append(values[int(band)])
        supported.append(group is not None)
    return weights, supported


def reliability_metrics(targets, weights):
    targets, weights = np.asarray(targets, dtype=float), np.asarray(weights, dtype=float)
    if (targets.ndim != 1 or targets.shape != weights.shape or not np.isin(targets, [0, 1]).all()
            or not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any()):
        raise ValueError("invalid reliability inputs")
    if len(targets) == 0:
        return {"positives": 0, "brier": None, "mean_weight": None, "error_weight_share": None}
    return {"positives": len(targets), "precision": float(targets.mean()),
            "brier": float(np.mean((weights - targets) ** 2)), "mean_weight": float(weights.mean()),
            "error_weight_share": float(weights[targets == 0].sum() / weights.sum())
            if weights.sum() else None}
