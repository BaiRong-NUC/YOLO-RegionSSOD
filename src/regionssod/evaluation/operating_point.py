"""Cross-fitted global and prediction-scale confidence operating points."""
from __future__ import annotations

import numpy as np


def confidence_grid(n=201):
    if n < 2:
        raise ValueError("grid needs at least two thresholds")
    return np.exp(np.linspace(0, np.log(1000), n)) / 1000


def _group_stats(scores, matches, gt_areas, grid):
    scores, matches, areas = np.asarray(scores), np.asarray(matches), np.asarray(gt_areas)
    if len(scores) != len(matches) or len(matches) != len(areas):
        raise ValueError("score/match/area population differs")
    rows = []
    for threshold in np.asarray(grid):
        selected = scores >= threshold
        tp = selected & (matches >= 0)
        rows.append({"threshold": float(threshold), "predictions": int(selected.sum()),
            "tp": int(tp.sum()), "small_tp": int((tp & (areas >= 0) & (areas < 1024)).sum())})
    return rows


def select_global(scores, matches, gt_areas, grid, target_precision):
    if not 0 < target_precision <= 1:
        raise ValueError("invalid precision target")
    rows = _group_stats(scores, matches, gt_areas, grid)
    feasible = [row for row in rows if row["predictions"] and row["tp"] / row["predictions"] >= target_precision]
    if not feasible:
        return None
    # Max small GT coverage, then fewer predictions, then a more conservative
    # numerical threshold. This is fixed before any fold result is read.
    return max(feasible, key=lambda row: (row["small_tp"], -row["predictions"], row["threshold"]))


def select_scale(scores, matches, gt_areas, predicted_small, grid, target_precision):
    if not 0 < target_precision <= 1:
        raise ValueError("invalid precision target")
    scores, predicted_small = np.asarray(scores), np.asarray(predicted_small, dtype=bool)
    if len(scores) != len(predicted_small) or len(scores) != len(matches):
        raise ValueError("score/match/prediction-scale population differs")
    small = _group_stats(scores[predicted_small], np.asarray(matches)[predicted_small],
                         np.asarray(gt_areas)[predicted_small], grid)
    other = _group_stats(scores[~predicted_small], np.asarray(matches)[~predicted_small],
                         np.asarray(gt_areas)[~predicted_small], grid)
    ns, ts, ss = (np.asarray([row[name] for row in small]) for name in ("predictions", "tp", "small_tp"))
    no, to, so = (np.asarray([row[name] for row in other]) for name in ("predictions", "tp", "small_tp"))
    total = ns[:, None] + no[None, :]
    tp = ts[:, None] + to[None, :]
    small_tp = ss[:, None] + so[None, :]
    feasible = (total > 0) & (tp / np.maximum(total, 1) >= target_precision)
    if not feasible.any():
        return None
    # Explicit lexicographic tie-breaking: max small TP, then fewer selected
    # predictions, then the higher non-small/small threshold. The number of
    # feasible pairs is bounded by the fixed 201x201 grid.
    candidates = np.argwhere(feasible)
    i, j = max(candidates, key=lambda pair: (small_tp[tuple(pair)], -total[tuple(pair)],
                                              grid[pair[1]], grid[pair[0]]))
    return {"small_threshold": float(grid[i]), "other_threshold": float(grid[j]),
        "calibration_predictions": int(total[i, j]), "calibration_tp": int(tp[i, j]),
        "calibration_small_tp": int(small_tp[i, j]),
        "calibration_precision": float(tp[i, j] / total[i, j])}


def evaluate_policy(scores, matches, gt_areas, predicted_small, policy):
    scores, matches, areas = np.asarray(scores), np.asarray(matches), np.asarray(gt_areas)
    if "small_threshold" in policy:
        selected = ((scores >= policy["small_threshold"]) & predicted_small) | (
                   (scores >= policy["other_threshold"]) & ~predicted_small)
    else:
        selected = scores >= policy["threshold"]
    tp = selected & (matches >= 0)
    return {"predictions": int(selected.sum()), "tp": int(tp.sum()),
            "precision": float(tp.sum() / selected.sum()) if selected.any() else None,
            "small_tp": int((tp & (areas >= 0) & (areas < 1024)).sum()),
            "small_recall": float((tp & (areas >= 0) & (areas < 1024)).sum() / 26577),
            "selected": selected}
