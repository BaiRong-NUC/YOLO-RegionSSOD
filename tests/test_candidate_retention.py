from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_candidate_retention import newly_retained, novel_indices, qualifies, quality
from regionssod.data.annotations import Tile, YoloBox


def test_novel_candidates_require_class_aware_non_overlap():
    original = [YoloBox(0, .25, .25, .1, .1)]
    proposed = [original[0], YoloBox(1, .25, .25, .1, .1),
                YoloBox(0, .75, .75, .1, .1)]
    assert novel_indices(original, proposed) == [1, 2]
    assert novel_indices([], proposed) == [0, 1, 2]
    assert novel_indices(original, []) == []


def test_quality_does_not_count_duplicates_or_wrong_classes_as_extra_gt():
    small = YoloBox(0, .25, .25, .1, .1)
    medium = YoloBox(1, .6, .6, .4, .4)
    result = quality([small, small, YoloBox(0, .6, .6, .4, .4)],
                     [small, medium], Tile(0, 0, 100, 100))
    assert result["tp"] == result["small_tp"] == 1
    assert result["medium_tp"] == 0
    assert result["gt"] == result["small_gt"] + result["medium_gt"] == 2


def test_identity_comparison_preserves_overlapping_crowded_proposals():
    original = [YoloBox(0, .25, .25, .1, .1)]
    proposed = [original[0], YoloBox(0, .26, .25, .1, .1),
                YoloBox(1, .25, .25, .1, .1)]
    assert newly_retained(original, proposed) == [1, 2]
    assert novel_indices(original, proposed, .7) == [2]
    assert newly_retained([], proposed) == [0, 1, 2]
    assert newly_retained(original, []) == []


def test_gate_rejects_sparse_noisy_and_coverage_losing_additions():
    good = {"added": 40, "small_gt": 1000, "positives": 1000,
            "baseline_tp": 700, "added_tp": 20, "added_small_tp": 10}
    assert qualifies(good)
    assert not qualifies({**good, "added": 0})
    assert not qualifies({**good, "added_tp": 19})
    assert not qualifies({**good, "added_small_tp": 9})
    assert not qualifies({**good, "baseline_tp": 1000})
