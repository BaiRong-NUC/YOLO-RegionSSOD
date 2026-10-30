import numpy as np
import pytest

from regionssod.evaluation.size_recall import match_by_confidence
from regionssod.evaluation.tradeoff import curve_point, operating_curve, prediction_matches


def test_duplicate_cross_class_and_size_competition_match_existing_recall():
    boxes = np.array([[.5, .5, .1, .1], [.5, .5, .11, .11], [.8, .8, .1, .1]])
    predictions = boxes[[1, 1, 2]]
    scores, classes = [.8, .9, .99], [0, 0, 1]
    matches = prediction_matches(classes, scores, predictions, [0, 0, 0], boxes, [.5, .9])
    for row, iou in zip(matches, [.5, .9]):
        matched_gt = np.zeros(3, dtype=bool)
        matched_gt[row[row >= 0]] = True
        assert np.array_equal(matched_gt, match_by_confidence(
            classes, scores, predictions, [0, 0, 0], boxes, iou))
    assert matches[0].tolist() == [0, 1, -1]
    assert matches[1].tolist() == [-1, 1, -1]


def test_prefix_matching_equals_rematching_after_threshold_and_handles_empty_gt():
    boxes = np.array([[.5, .5, .1, .1], [.5, .5, .11, .11]])
    scores = np.array([.4, .9])
    matches = prediction_matches([0, 0], scores, boxes, [0, 0], boxes, [.5])
    keep = scores >= .5
    assert np.array_equal(matches[:, keep], prediction_matches(
        [0], scores[keep], boxes[keep], [0, 0], boxes, [.5]))
    assert prediction_matches([0], [.9], boxes[:1], [], [], [.5]).tolist() == [[-1]]
    assert prediction_matches([], [], [], [0], boxes[:1], [.5]).shape == (1, 0)


def test_equal_fp_budget_never_splits_tied_scores_and_keeps_trailing_true_positives():
    # The .8 group crosses the one-FP budget even though one member is a TP.
    curve = operating_curve([.9, .8, .8, .7], [-1, -1, 0, 0], [2, 0, 0],
                            confidence_floor=.25)
    point = curve_point(curve, fp_budget=1)
    assert point["threshold"] == .9 and point["predictions"] == 1
    point = curve_point(curve, fp_budget=2)
    assert point["threshold"] == .7 and point["matched"] == [2, 0, 0]
    assert point["fp"] == 2 and point["recall"] == [1., None, None]
    assert curve_point(curve, threshold=.81)["predictions"] == 1
    assert curve_point(curve, threshold=.8)["predictions"] == 3
    with pytest.raises(ValueError, match="below"):
        curve_point(curve, threshold=.2)


def test_empty_and_confidence_one_prefixes_do_not_invent_an_attainable_threshold():
    curve = operating_curve([], [], [1, 0, 0], confidence_floor=.25)
    assert curve_point(curve, fp_budget=0)["predictions"] == 0
    curve = operating_curve([1.], [-1], [1, 0, 0], confidence_floor=.25)
    assert curve_point(curve, fp_budget=0) is None
    assert curve_point(curve, threshold=1.)["fp"] == 1


def test_strict_iou_detects_localization_loss_and_rejects_invalid_cache():
    gt = [[.5, .5, .1, .1]]
    shifted = [[.52, .5, .1, .1]]
    assert prediction_matches([0], [.8], shifted, [0], gt, [.5, .75]).tolist() == [[0], [-1]]
    with pytest.raises(ValueError):
        operating_curve([.2], [0], [1, 0, 0], confidence_floor=.25)
    with pytest.raises(ValueError):
        operating_curve([.8, .9], [0, 0], [1, 0, 0], confidence_floor=.25)
