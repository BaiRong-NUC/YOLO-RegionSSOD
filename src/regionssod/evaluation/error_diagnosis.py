"""Descriptive cache diagnostics; every GT oracle stays outside deployment."""
from __future__ import annotations

import numpy as np

ERROR_NAMES = ("matched", "duplicate_competition", "wrong_class", "localization_near",
               "near_other_class", "unexplained_background")
LOST_NAMES = ("active_matching_competition", "below_score_threshold", "coarse_not_strict",
              "wrong_class_candidate", "localization_near", "near_other_class", "saved_candidate_gap")


def box_iou_matrix(predictions, truth):
    boxes = np.asarray(predictions, float).reshape(-1, 4)
    gt = np.asarray(truth, float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(boxes[:, None, 2:], gt[None, :, 2:]) -
                              np.maximum(boxes[:, None, :2], gt[None, :, :2])).prod(2)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    gt_area = np.maximum(0, gt[:, 2:] - gt[:, :2]).prod(1)
    return intersection / np.maximum(area[:, None] + gt_area[None, :] - intersection, 1e-12)


def classify_false_predictions(matches, overlaps, classes, gt_classes, iou):
    """Mutually exclusive descriptive FP labels; no FP inherits a GT size."""
    matches = np.asarray(matches)
    overlaps = np.asarray(overlaps)
    same = np.asarray(classes)[:, None] == np.asarray(gt_classes)[None, :]
    best_same = np.max(np.where(same, overlaps, 0), axis=1, initial=0)
    best_other = np.max(np.where(~same, overlaps, 0), axis=1, initial=0)
    labels = np.full(len(matches), 5, dtype=np.int8)
    labels[best_other >= .1] = 4
    labels[best_same >= .1] = 3
    labels[best_other >= iou] = 2
    labels[best_same >= iou] = 1
    labels[matches >= 0] = 0
    return labels


def gt_candidate_summary(overlaps, classes, scores, gt_classes):
    overlaps = np.asarray(overlaps)
    same = np.asarray(classes)[:, None] == np.asarray(gt_classes)[None, :]
    scores = np.asarray(scores)[:, None]
    result = {
        "best_same_iou": np.max(np.where(same, overlaps, 0), axis=0, initial=0),
        "best_other_iou": np.max(np.where(~same, overlaps, 0), axis=0, initial=0),
        "score_iou0.5": np.max(np.where(same & (overlaps >= .5), scores, -1), axis=0, initial=-1),
        "score_iou0.75": np.max(np.where(same & (overlaps >= .75), scores, -1), axis=0, initial=-1),
    }
    for iou in (.5, .75):
        eligible = same & (overlaps >= iou)
        indices = np.argmax(np.where(eligible, scores, -1), axis=0) if len(overlaps) else np.full(len(gt_classes), -1)
        result[f"index_iou{iou}"] = np.where(eligible.any(axis=0), indices, -1)
    return result


def lost_gt_reason(summary, index, iou, threshold):
    """Prioritized candidate evidence, never an identified training cause."""
    score = summary[f"score_iou{iou}"][index]
    if score >= threshold:
        return LOST_NAMES[0]
    if score >= 0:
        return LOST_NAMES[1]
    if iou == .75 and summary["best_same_iou"][index] >= .5:
        return LOST_NAMES[2]
    if summary["best_other_iou"][index] >= iou:
        return LOST_NAMES[3]
    if summary["best_same_iou"][index] >= .1:
        return LOST_NAMES[4]
    if summary["best_other_iou"][index] >= .1:
        return LOST_NAMES[5]
    return LOST_NAMES[6]


def paired_geometry(predicted_boxes, gt_boxes):
    """Geometry for an explicit fixed GT association, not nearest-GT precision."""
    predictions = np.asarray(predicted_boxes, float).reshape(-1, 4)
    truth = np.asarray(gt_boxes, float).reshape(-1, 4)
    if len(predictions) != len(truth):
        raise ValueError("paired geometry requires one box per fixed GT")
    width_height = truth[:, 2:] - truth[:, :2]
    if (width_height <= 0).any():
        raise ValueError("nonpositive GT dimension")
    intersection = np.maximum(0, np.minimum(predictions[:, 2:], truth[:, 2:]) -
                              np.maximum(predictions[:, :2], truth[:, :2])).prod(1)
    pa = np.maximum(0, predictions[:, 2:] - predictions[:, :2]).prod(1)
    ga = width_height.prod(1)
    iou = intersection / np.maximum(pa + ga - intersection, 1e-12)
    centers = ((predictions[:, :2] + predictions[:, 2:]) - (truth[:, :2] + truth[:, 2:])) / 2
    center_error = np.sqrt(((centers / width_height) ** 2).sum(1))
    edge_error = (np.abs(predictions - truth) / np.tile(width_height, (1, 2))).mean(1)
    return {"iou": iou, "normalized_center_error": center_error, "normalized_edge_error": edge_error}


def transition_counts(b2_low, p1_low, b2_high, p1_high, mask):
    """All sixteen shared-GT states, including zero/impossible cells."""
    arrays = [np.asarray(value, bool) for value in (b2_low, p1_low, b2_high, p1_high)]
    if (arrays[2] & ~arrays[0]).any() or (arrays[3] & ~arrays[1]).any():
        raise ValueError("higher precision frontier has gained a non-prefix match")
    state = sum(array.astype(np.int8) * weight for array, weight in zip(arrays, (8, 4, 2, 1)))
    return np.bincount(state[np.asarray(mask, bool)], minlength=16).tolist()
