import numpy as np
import pytest

from regionssod.evaluation.precision_recall import (
    assign_predictions, coco_ap, precision_constrained_point, threshold_curve,
)
from regionssod.evaluation.size_recall import match_by_confidence


def test_joint_matching_duplicates_wrong_class_and_old_recall_agree():
    gt = np.array([[0, 0, 20, 20], [0, 0, 40, 40]])
    pred = np.array([[0, 0, 20, 20], [0, 0, 20, 20], [0, 0, 40, 40], [0, 0, 20, 20]])
    cls, scores = [0, 0, 0, 1], [.9, .8, .7, .99]
    matches = assign_predictions(cls, scores, pred, [0, 0], gt)
    assert matches.tolist() == [0, -1, 1, -1]
    def xywh(a):
        return np.c_[(a[:, :2] + a[:, 2:]) / 2, a[:, 2:] - a[:, :2]]
    old = match_by_confidence(cls, scores, xywh(pred), [0, 0], xywh(gt))
    assert np.array_equal(np.bincount(matches[matches >= 0], minlength=2) > 0, old)
    assert assign_predictions([0], [.8], pred[:1], [], []).tolist() == [-1]


def test_precision_constraint_never_splits_equal_scores_or_interpolates():
    counts = {"small": 2, "medium": 1, "large": 1}
    curve = threshold_curve([.9, .9, .8, .7], [100, -1, 200, 2000], counts)
    assert curve["predictions"].tolist() == [2, 3, 4]
    assert precision_constrained_point(curve, .8) is None
    point = precision_constrained_point(curve, .7)
    assert point["threshold"] == .7 and point["small_recall"] == 1
    assert point["precision"] == .75
    reversed_ties = threshold_curve([.9, .9, .8, .7], [-1, 100, 200, 2000], counts)
    assert np.array_equal(curve["precision"], reversed_ties["precision"])


def test_coco_small_ignores_matched_large_and_unmatched_large_predictions():
    gt = {"images": [{"id": 1}], "info": {}, "categories": [{"id": 1, "name": "a"}],
          "annotations": [
              {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
              {"id": 2, "image_id": 1, "category_id": 1, "bbox": [100, 100, 100, 100], "area": 10000, "iscrowd": 0}]}
    detections = [
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 100, 100], "score": .99},
        {"image_id": 1, "category_id": 1, "bbox": [300, 300, 100, 100], "score": .98},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .8}]
    result, _ = coco_ap(gt, detections)
    assert result["metrics"]["AP_small"] == pytest.approx(1)
    assert result["metrics"]["AP_all"] < 1
    assert result["max_dets"][-1] == 500
    empty, _ = coco_ap(gt, [])
    assert empty["metrics"]["AP_small"] == 0


def test_coco_area_boundary_is_distinct_from_strict_diagnostic_small():
    gt = {"images": [{"id": 1}], "info": {}, "categories": [{"id": 1, "name": "a"}],
          "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 32, 32],
                           "area": 1024, "iscrowd": 0}]}
    result, _ = coco_ap(gt, [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 32, 32], "score": .9}])
    assert result["metrics"]["AP_small"] == pytest.approx(1)
    curve = threshold_curve([.9], [1024], {"small": 1, "medium": 1, "large": 1})
    assert curve["small_tp"].tolist() == [0] and curve["medium_tp"].tolist() == [1]
