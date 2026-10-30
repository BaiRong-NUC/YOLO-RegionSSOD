import copy
from pathlib import Path
import sys

import pytest

from regionssod.data.annotations import Tile
from regionssod.pseudo.context import map_context, rectify_categories

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_context_rectification import qualifies


def box(cls=0, x=.5, confidence=.4, scale="small"):
    return {"class_id": cls, "xywhn": [x, .5, .1, .1],
            "confidence": confidence, "scale": scale, "state": "positive"}


def test_rectification_changes_only_eligible_class_and_preserves_original_inputs():
    positives = [box(), box(x=.2, confidence=.5), box(x=.8, scale="medium")]
    context = [box(cls=1, confidence=.5), box(cls=2, x=.2, confidence=.9),
               box(cls=3, x=.8, confidence=.9)]
    snapshot = copy.deepcopy((positives, context))
    after, changes = rectify_categories(positives, context)
    expected = copy.deepcopy(positives)
    expected[0]["class_id"] = 1
    assert after == expected and len(changes) == 1
    assert changes[0]["positive_index"] == 0
    assert (positives, context) == snapshot


def test_high_confidence_positive_reserves_support_against_nearby_low_box():
    positive = [box(x=.501), box(confidence=.8)]
    after, changes = rectify_categories(positive, [box(cls=1, confidence=.9)])
    assert changes == [] and after == positive


def test_greedy_support_is_distinct_and_confidence_filter_keeps_index():
    positive = [box(), box(x=.501)]
    context = [box(cls=1, confidence=.49), box(cls=2, confidence=.5)]
    after, changes = rectify_categories(positive, context)
    assert [p["class_id"] for p in after] == [2, 0]
    assert len(changes) == 1 and changes[0]["context_index"] == 1


def test_unsupported_or_same_class_predictions_are_unchanged():
    assert rectify_categories([], []) == ([], [])
    for context in ([], [box(cls=1, x=.8, confidence=.9)], [box(confidence=.9)]):
        positive = [box()]
        assert rectify_categories(positive, context) == (positive, [])


def test_source_to_rectangular_tile_clipping_and_visibility():
    source = [{"class_id": 2, "xywhn": [.2, .2, .2, .2], "confidence": .8}]
    snapshot = copy.deepcopy(source)
    mapped = map_context(source, Tile(150, 50, 250, 150), (1000, 500))
    assert mapped[0]["xywhn"] == pytest.approx([.5, .5, 1., 1.])
    assert mapped[0]["confidence"] == .8 and mapped[0]["source_prediction_index"] == 0
    assert map_context(source, Tile(150, 50, 250, 150), (1000, 500), min_visibility=.6) == []
    assert source == snapshot


@pytest.mark.parametrize("bad", [
    {"confidence": float("nan")}, {"xywhn": [.5, .5, 0, .1]},
    {"xywhn": [.5, .5, float("inf"), .1]}, {"class_id": -1}, {"state": "ignore"}])
def test_invalid_positives_are_rejected(bad):
    with pytest.raises(ValueError):
        rectify_categories([{**box(), **bad}], [])


def test_invalid_thresholds_and_tile_are_rejected():
    with pytest.raises(ValueError):
        rectify_categories([], [], match_iou=0)
    with pytest.raises(ValueError):
        rectify_categories([], [], only_scale="unknown")
    with pytest.raises(ValueError):
        map_context([], Tile(0, 0, 11, 10), (10, 10))


def test_candidate_gate_requires_sample_quality_net_coverage_and_harm_bound():
    rule = {"min_corrected": 20, "min_corrected_quality": .7,
            "min_corrected_quality_gain": .2, "min_unique_matches_gain": 10,
            "min_small_matches_gain": 10, "max_small_matches_lost": 3}
    counts = {"corrected": 20, "corrected_correct_before": 4, "corrected_correct_after": 16,
              "matches_before": 100, "matches_after": 112,
              "small_matches_before": 70, "small_matches_after": 82, "small_matches_lost": 3}
    assert qualifies(counts, rule)["passed"]
    assert not qualifies({**counts, "small_matches_lost": 4}, rule)["passed"]
    assert not qualifies({**counts, "small_matches_after": 79}, rule)["passed"]
    empty = {k: 0 for k in counts}
    result = qualifies(empty, rule)
    assert not result["passed"] and result["after_quality"] is None
