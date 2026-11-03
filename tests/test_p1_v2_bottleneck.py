import numpy as np
import pytest

from diagnose_p1_v2_bottleneck import budget_point, matched_transitions, rank_auc
from regionssod.evaluation.precision_recall import threshold_curve


def test_equal_exposure_budgets_keep_whole_ties_and_exact_fp_limits():
    curve = threshold_curve([.9, .8, .8, .7, .6], [100, 100, -1, -1, 100],
                            {"small": 3, "medium": 1, "large": 1})
    # A budget of two predictions cannot admit either half of the .8 tie.
    assert budget_point(curve, "predictions", 2)["predictions"] == 1
    assert budget_point(curve, "fp", 1)["predictions"] == 3
    assert budget_point(curve, "fp", 2)["small_tp"] == 3
    assert budget_point(curve, "predictions", 0) is None


def test_auc_is_pairwise_ranking_with_ties_and_not_calibration():
    scores = np.array([.9, .8, .8, .1])
    positive = np.array([True, True, False, False])
    assert rank_auc(scores, positive) == pytest.approx(.875)
    assert rank_auc(scores ** 3, positive) == pytest.approx(.875)
    assert rank_auc([.1, .9], [True, False]) == 0
    assert rank_auc([.9], [True]) is None
    with pytest.raises(ValueError):
        rank_auc([float('nan')], [True])


def test_transitions_preserve_lost_and_gained_gt_instead_of_only_net_gain():
    result = matched_transitions([1, 1, 0, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 1, 0])
    assert result == {"both": 1, "left_only": 1, "right_only": 1, "neither": 1}
    assert sum(result.values()) == 4
    with pytest.raises(ValueError):
        matched_transitions([1], [1, 0], [1])
