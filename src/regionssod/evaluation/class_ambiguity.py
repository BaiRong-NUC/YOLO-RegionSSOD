"""Fixed-feature teacher class-error diagnostics; never produces training labels."""

from __future__ import annotations

import numpy as np


def ambiguity_ratio(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("Need at least two class scores")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid sigmoid scores")
    top = np.sort(scores, axis=1)[:, -2:]
    if np.any(top[:, 1] <= 0):
        raise ValueError("Top score must be positive")
    return top[:, 0] / top[:, 1]


def features(rows, with_ratio):
    c = np.clip([row["confidence"] for row in rows], 1e-6, 1 - 1e-6)
    area = np.asarray([row["area"] for row in rows], float)
    classes = np.asarray([row["class_id"] for row in rows], int)
    if np.any(area <= 0) or np.any((classes < 0) | (classes >= 10)):
        raise ValueError("Invalid area or class")
    columns = [np.log(c / (1 - c)), np.log(area)]
    columns += [(classes == cls).astype(float) for cls in range(1, 10)]
    if with_ratio:
        columns.append(np.asarray([row["ratio"] for row in rows], float))
    return np.column_stack(columns)


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def fit_logistic(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or set(np.unique(y)) != {0, 1} or not np.isfinite(x).all():
        raise ValueError("Fit requires finite features and both binary outcomes")
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    beta = np.zeros(design.shape[1])
    beta[0] = np.log(y.mean() / (1 - y.mean()))
    penalty = np.eye(len(beta))
    penalty[0, 0] = 0

    def objective(b):
        z = design @ b
        return float(np.sum(np.logaddexp(0, z) - y * z) + .5 * b @ penalty @ b)

    converged = False
    for iteration in range(80):
        p = sigmoid(design @ beta)
        gradient = design.T @ (p - y) + penalty @ beta
        if np.max(np.abs(gradient)) < 1e-7:
            converged = True
            break
        hessian = design.T @ ((p * (1 - p))[:, None] * design) + penalty
        step = np.linalg.solve(hessian, gradient)
        factor = 1.0
        old = objective(beta)
        while objective(beta - factor * step) > old and factor > 2**-20:
            factor *= .5
        beta -= factor * step
    return {"mean": mean.tolist(), "scale": scale.tolist(), "beta": beta.tolist(),
            "iterations": iteration + 1, "converged": converged, "l2": 1.0}


def predict(model, x):
    x = (np.asarray(x) - model["mean"]) / model["scale"]
    return sigmoid(np.column_stack([np.ones(len(x)), x]) @ model["beta"])


def metrics(y, p):
    y, p = np.asarray(y, int), np.clip(p, 1e-12, 1 - 1e-12)
    losses = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    positive, negative = int(y.sum()), int((1 - y).sum())
    auc = None
    if positive and negative:
        order = np.argsort(p, kind="stable")
        groups = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1, len(p)]
        rank_sum = 0.0
        for lo, hi in zip(groups[:-1], groups[1:]):
            rank_sum += float(y[order[lo:hi]].sum()) * (lo + 1 + hi) / 2
        auc = (rank_sum - positive * (positive + 1) / 2) / (positive * negative)
    return {"n": len(y), "errors": positive, "log_loss": float(losses.mean()),
            "brier": float(np.mean((p - y)**2)), "auc": None if auc is None else float(auc)}, losses


def image_bootstrap(image_ids, improvement, *, all_images, repetitions=1000, seed=20261104):
    image_ids, improvement = np.asarray(image_ids), np.asarray(improvement, float)
    if len(image_ids) != len(improvement) or not set(image_ids).issubset(set(all_images)):
        raise ValueError("Bootstrap image mismatch")
    totals = np.asarray([improvement[image_ids == i].sum() for i in all_images])
    counts = np.asarray([(image_ids == i).sum() for i in all_images])
    draws = np.random.default_rng(seed).integers(0, len(all_images), (repetitions, len(all_images)))
    denominator = counts[draws].sum(1)
    if np.any(denominator == 0):
        raise ValueError("Empty bootstrap replicate")
    values = totals[draws].sum(1) / denominator
    return {"repetitions": repetitions, "unit": "original training image",
            "ci95": np.quantile(values, [.025, .975]).tolist(), "seed": seed}
