"""Rectify low-confidence tile categories using a mapped contextual view."""
from __future__ import annotations

import copy

import numpy as np

from regionssod.pseudo.consistency import pairwise_iou
from regionssod.data.annotations import Tile, YoloBox, clip_box_to_tile


def validate_predictions(items):
    for item in items:
        coords = np.asarray(item["xywhn"], dtype=float)
        if (coords.shape != (4,) or not np.isfinite(coords).all()
                or (coords < 0).any() or (coords > 1).any() or (coords[2:] <= 0).any()
                or not 0 <= item["confidence"] <= 1
                or not isinstance(item["class_id"], int) or item["class_id"] < 0):
            raise ValueError("invalid prediction")


def map_context(context, tile: Tile, image_size, *, min_visibility=.35):
    """Clip source-image predictions into the native tile without labels."""
    validate_predictions(context)
    width, height = image_size
    if (width <= 0 or height <= 0 or not 0 < min_visibility <= 1
            or not 0 <= tile.x1 < tile.x2 <= width
            or not 0 <= tile.y1 < tile.y2 <= height):
        raise ValueError("invalid source/tile geometry")
    mapped = []
    for index, item in enumerate(context):
        clipped = clip_box_to_tile(
            YoloBox(item["class_id"], *item["xywhn"]), tile, width, height, min_visibility)
        if clipped is not None:
            mapped.append({**copy.deepcopy(item),
                           "xywhn": [clipped.x, clipped.y, clipped.w, clipped.h],
                           "source_prediction_index": index})
    return mapped


def rectify_categories(positive, context, *, context_conf=.5, original_below=.5,
                       match_iou=.5, only_scale="small"):
    """Change class only; preserve count, geometry, confidence and ordering.

    Context predictions must already be mapped into native tile coordinates.
    All positives reserve distinct supports before low-confidence corrections.
    There is no ground-truth input and unsupported positives remain unchanged.
    """
    if not all(0 < v <= 1 for v in (context_conf, original_below, match_iou)):
        raise ValueError("invalid correction thresholds")
    if only_scale not in ("small", "medium", "large"):
        raise ValueError("invalid scale")
    validate_predictions([*positive, *context])
    if any(p.get("state", "positive") != "positive" for p in positive):
        raise ValueError("only accepted positives can be rectified")
    view_indices = [j for j, item in enumerate(context) if item["confidence"] >= context_conf]
    view = [context[j] for j in view_indices]
    overlap = pairwise_iou(positive, view)
    pi, vi = np.where(overlap >= match_iou)
    corrected = copy.deepcopy(positive)
    used_p, used_v, changes = set(), set(), []
    for k in np.argsort(-overlap[pi, vi], kind="stable"):
        p, v = int(pi[k]), int(vi[k])
        if p in used_p or v in used_v:
            continue
        used_p.add(p)
        used_v.add(v)
        old, support = positive[p], view[v]
        if (old["scale"] == only_scale and old["confidence"] < original_below
                and old["class_id"] != support["class_id"]):
            corrected[p]["class_id"] = support["class_id"]
            changes.append({"positive_index": p, "context_index": view_indices[v],
                            "old_class_id": old["class_id"], "new_class_id": support["class_id"],
                            "context_confidence": support["confidence"],
                            "iou": float(overlap[p, v])})
    return corrected, changes
