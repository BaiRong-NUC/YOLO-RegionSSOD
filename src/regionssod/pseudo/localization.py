"""GT-free geometric agreement of accepted positives across teacher views."""
from __future__ import annotations

import numpy as np

from regionssod.pseudo.consistency import pairwise_iou


def localization_stability(positive, view, *, min_view_conf=.05, min_pair_iou=.1,
                           unstable_below=.5, stable_at=.7):
    """Return one feature row per unchanged positive in native tile coordinates.

    A missing view match is unknown, not an incorrect detection. Geometry and
    confidence are the only inputs; hidden labels are deliberately not accepted.
    """
    if not (0 < min_view_conf <= 1 and 0 < min_pair_iou < unstable_below < stable_at <= 1):
        raise ValueError("invalid stability thresholds")
    for item in [*positive, *view]:
        coords = np.asarray(item["xywhn"], dtype=float)
        if (coords.shape != (4,) or not np.isfinite(coords).all()
                or (coords[2:] <= 0).any()
                or not 0 <= item["confidence"] <= 1):
            raise ValueError("invalid prediction")
    if any(item.get("state", "positive") != "positive" for item in positive):
        raise ValueError("only accepted positives can be scored")
    eligible = [j for j, item in enumerate(view) if item["confidence"] >= min_view_conf]
    candidates = [view[j] for j in eligible]
    iou = pairwise_iou(positive, candidates)
    same = (np.asarray([p["class_id"] for p in positive])[:, None]
            == np.asarray([p["class_id"] for p in candidates])[None, :])
    edges = same & (iou >= min_pair_iou)
    pi, vi = np.where(edges)
    assignments, used = {}, set()
    for k in np.argsort(-iou[pi, vi], kind="stable"):
        p, v = int(pi[k]), int(vi[k])
        if p not in assignments and v not in used:
            assignments[p] = v
            used.add(v)
    rows = []
    for p, original in enumerate(positive):
        row = {"view_index": None, "iou": None, "center_shift": None,
               "log_shape_change": None, "group": "unsupported",
               "support_candidates": int(edges[p].sum()), "ambiguous": False}
        if p in assignments:
            v = assignments[p]
            before = np.asarray(original["xywhn"], dtype=float)
            after = np.asarray(candidates[v]["xywhn"], dtype=float)
            score = float(iou[p, v])
            row.update(view_index=eligible[v], iou=score,
                       center_shift=float(np.linalg.norm((after[:2]-before[:2])/before[2:])),
                       log_shape_change=float(np.abs(np.log(after[2:]/before[2:])).mean()),
                       group="stable" if score >= stable_at else
                             "unstable" if score < unstable_below else "middle",
                       ambiguous=bool(edges[p].sum() > 1 or edges[:, v].sum() > 1))
        elif edges[p].any():
            row["ambiguous"] = True
        rows.append(row)
    return rows
