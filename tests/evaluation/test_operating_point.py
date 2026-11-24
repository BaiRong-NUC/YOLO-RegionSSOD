import numpy as np
import pytest

from regionssod.evaluation.operating_point import (
    confidence_grid, evaluate_policy, select_global, select_scale,
)


def test_grid_is_strict_and_includes_unit_confidence():
    grid = confidence_grid()
    assert len(grid) == 201 and grid[0] == pytest.approx(.001) and grid[-1] == pytest.approx(1)
    assert (np.diff(grid) > 0).all()


def test_global_requires_realizable_precision_and_keeps_small_coverage():
    scores = np.array([.9, .8, .7, .6])
    matches = np.array([10, 11, -1, 12])
    areas = np.array([100, 2000, -1, 300])
    grid = np.array([.5, .6, .7, .8, .9])
    policy = select_global(scores, matches, areas, grid, .7)
    assert policy['threshold'] == .6
    evaluated = evaluate_policy(scores, matches, areas, np.array([1, 1, 1, 1]), policy)
    assert evaluated['precision'] == .75 and evaluated['small_tp'] == 2
    with pytest.raises(ValueError):
        select_global(scores, matches, areas, grid, 1.01)


def test_scale_policy_can_trade_small_threshold_against_other_threshold():
    scores = np.array([.9, .8, .7, .6])
    matches = np.array([10, -1, 11, -1])
    areas = np.array([100, -1, 200, -1])
    predicted_small = np.array([1, 0, 1, 0])
    grid = np.array([.6, .7, .8, .9])
    policy = select_scale(scores, matches, areas, predicted_small, grid, .8)
    assert policy['small_threshold'] == .7 and policy['other_threshold'] == .9
    evaluated = evaluate_policy(scores, matches, areas, predicted_small, policy)
    assert evaluated['tp'] == 2 and evaluated['precision'] == 1 and evaluated['small_tp'] == 2


def test_unreachable_area_scale_and_cross_fitting_is_separate():
    false_small = np.array([1, 0, 1, 0])
    policy = select_global(np.array([.5, .6]), np.array([-1, -1]), np.array([-1, -1]),
                           np.array([.5, .6]), .01)
    assert policy is None
    with pytest.raises(ValueError):
        select_scale(np.arange(2), np.arange(2), np.arange(2), false_small, np.array([.5]), 0)


def test_predicted_small_does_not_change_matched_gt_size():
    scores, matches, areas = np.array([.9]), np.array([10]), np.array([2000])
    evaluated = evaluate_policy(scores, matches, areas, np.array([True]), {'threshold': .5})
    assert evaluated['small_tp'] == 0
