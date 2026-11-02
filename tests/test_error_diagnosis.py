import numpy as np
import pytest

from regionssod.evaluation.error_diagnosis import (
    box_iou_matrix, classify_false_predictions, gt_candidate_summary, json_ready,
    lost_gt_reason, paired_geometry, transition_counts,
)
from regionssod.evaluation.precision_recall import assign_predictions


def test_error_priority_handles_duplicate_wrong_class_geometry_and_background():
    gt = [[0, 0, 10, 10], [100, 100, 110, 110]]
    pred = [[0, 0, 10, 10], [0, 0, 10, 10], [100, 100, 110, 110],
            [0, 0, 5, 10], [101, 100, 103, 110], [200, 200, 210, 210]]
    cls = [0, 0, 0, 0, 0, 0]
    overlap = box_iou_matrix(pred, gt)
    match = assign_predictions(cls, [.99, .98, .97, .96, .95, .94], pred, [0, 1], gt, .75)
    assert classify_false_predictions(match, overlap, cls, [0, 1], .75).tolist() == [0, 1, 2, 3, 4, 5]


def test_removing_gt_defined_false_predictions_never_changes_the_true_matches():
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]]
    cls, scores, truth = [0] * 4, [.9, .9, .8, .7], [boxes[0], boxes[2]]
    match = assign_predictions(cls, scores, boxes, [0, 0], truth)
    keep = match >= 0
    reduced = assign_predictions(np.array(cls)[keep], np.array(scores)[keep], np.array(boxes)[keep], [0, 0], truth)
    assert reduced.tolist() == match[keep].tolist()
    prefix = np.asarray(scores) >= .8
    assert np.array_equal(assign_predictions(np.array(cls)[prefix], np.array(scores)[prefix],
                          np.array(boxes)[prefix], [0, 0], truth), match[prefix])


def test_candidate_reasons_separate_low_score_and_geometry_and_competition():
    overlaps = np.array([[.8, .6, 0, 0], [0, .2, .8, 0]])
    summary = gt_candidate_summary(overlaps, [0, 1], [.3, .9], [0, 0, 0, 0])
    assert lost_gt_reason(summary, 0, .75, .5) == 'below_score_threshold'
    assert lost_gt_reason(summary, 0, .75, .2) == 'active_matching_competition'
    assert lost_gt_reason(summary, 1, .75, .5) == 'coarse_not_strict'
    assert lost_gt_reason(summary, 2, .75, .5) == 'wrong_class_candidate'
    assert lost_gt_reason(summary, 3, .75, .5) == 'saved_candidate_gap'
    empty = gt_candidate_summary(np.empty((0, 1)), [], [], [0])
    assert lost_gt_reason(empty, 0, .5, .7) == 'saved_candidate_gap'


def test_geometry_uses_gt_dimensions_and_exact_association():
    result = paired_geometry([[2, 0, 12, 10]], [[0, 0, 10, 10]])
    assert result['iou'][0] == pytest.approx(2/3)
    assert result['normalized_center_error'][0] == pytest.approx(.2)
    assert result['normalized_edge_error'][0] == pytest.approx(.1)


def test_transition_accounting_is_exact_and_rejects_non_nested_frontiers():
    cells = transition_counts([1, 1], [1, 0], [1, 0], [0, 0], [1, 1])
    assert sum(cells) == 2 and cells[14] == 1 and cells[8] == 1
    with pytest.raises(ValueError):
        transition_counts([0], [0], [1], [0], [1])


def test_json_conversion_handles_numpy_scalars_and_nested_arrays():
    converted = json_ready({"scalar": np.int64(3), "array": np.array([np.float64(1.5)])})
    assert converted == {"scalar": 3, "array": [1.5]}
