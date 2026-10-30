from itertools import combinations, permutations
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_ignore_recovery import maximum_pairs, residual_analysis
from regionssod.data.annotations import YoloBox


def test_augmenting_path_recovers_capacity_missed_by_greedy():
    pairs = maximum_pairs(np.array([[True, True], [True, False]]))
    assert len(pairs) == 2
    assert len({p for p, _ in pairs}) == len({g for _, g in pairs}) == 2


def test_maximum_matching_equals_exhaustive_capacity_on_all_two_by_three_graphs():
    for bits in range(64):
        edges = np.array([(bits >> i) & 1 for i in range(6)], dtype=bool).reshape(2, 3)
        expected = 0
        for count in (1, 2):
            for ps in combinations(range(2), count):
                for gs in permutations(range(3), count):
                    if all(edges[p, g] for p, g in zip(ps, gs)):
                        expected = count
        assert len(maximum_pairs(edges)) == expected


def test_locked_gt_duplicates_cannot_count_as_recovery_and_wrong_class_is_excluded():
    gt = [YoloBox(0,.2,.2,.1,.1), YoloBox(0,.8,.8,.1,.1)]
    predictions = [gt[0], gt[0], gt[1], YoloBox(1,.8,.8,.1,.1)]
    result = residual_analysis(predictions, gt, {0})
    assert result["only_overlaps_locked_gt"] == 2
    assert result["greedy_extra_matches"] == result["oracle_max_extra_matches"] == 1
    assert result["greedy_extra_gt_indices"] == [1]
    assert result["wrong_class_only"] == 1


def test_empty_residual_and_empty_predictions_have_zero_capacity():
    box = YoloBox(0,.5,.5,.1,.1)
    assert residual_analysis([box], [box], {0})["oracle_max_extra_matches"] == 0
    assert residual_analysis([], [box], set())["raw_unmatched_gt_coverage"] == 0
