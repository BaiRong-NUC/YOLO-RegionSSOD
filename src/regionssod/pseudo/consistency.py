"""GT-free promotion of uncertain boxes supported by another teacher view."""
from __future__ import annotations

import copy

import numpy as np


def pairwise_iou(first: list[dict], second: list[dict]) -> np.ndarray:
    def corners(items):
        boxes = np.asarray([item["xywhn"] for item in items], dtype=float).reshape(-1, 4)
        return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,
                               boxes[:, :2] + boxes[:, 2:] / 2), axis=1)

    a, b = corners(first), corners(second)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                              - np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    union = ((a[:, 2:] - a[:, :2]).prod(axis=1)[:, None]
             + (b[:, 2:] - b[:, :2]).prod(axis=1)[None, :] - intersection)
    return intersection / np.maximum(union, 1e-12)


def promote_consistent(original: list[dict], other_view: list[dict], *,
                       floor: float = .15, support_conf: float = .5,
                       match_iou: float = .5, duplicate_iou: float = .5) -> list[dict]:
    """Preserve original coordinates/confidence; assign distinct view supports.

    Both inputs must use the same native tile coordinate system. Existing
    positives are never changed; hidden labels are not accepted by this API.
    """
    if not all(0 < value <= 1 for value in (floor, support_conf, match_iou, duplicate_iou)):
        raise ValueError("confidence and IoU bounds must lie in (0, 1]")
    positive = [item for item in original if item["state"] == "positive"]
    uncertain = [item for item in original if item["state"] == "ignore"
                 and item["scale"] == "small" and item["confidence"] >= floor]
    views = [item for item in other_view if item["confidence"] >= support_conf]
    duplicate = pairwise_iou(uncertain, positive)
    same = (np.asarray([p["class_id"] for p in uncertain])[:, None]
            == np.asarray([p["class_id"] for p in positive])[None, :])
    uncertain = [item for i, item in enumerate(uncertain)
                 if not (same & (duplicate >= duplicate_iou)).any(axis=1)[i]]
    iou = pairwise_iou(uncertain, views)
    same = (np.asarray([p["class_id"] for p in uncertain])[:, None]
            == np.asarray([p["class_id"] for p in views])[None, :])
    a, b = np.where(same & (iou >= match_iou))
    used_a, used_b, promoted = set(), set(), []
    for k in np.argsort(-iou[a, b], kind="stable"):
        i, j = int(a[k]), int(b[k])
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        item = copy.deepcopy(uncertain[i])
        item.update(state="positive", support_confidence=views[j]["confidence"],
                    support_iou=float(iou[i, j]))
        promoted.append(item)
    return promoted
