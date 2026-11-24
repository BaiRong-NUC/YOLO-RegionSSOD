import copy

import pytest

from regionssod.pseudo.consistency import promote_consistent


def candidate(cls=0, x=.5, confidence=.25, state="ignore"):
    return {"class_id": cls, "xywhn": [x, .5, .1, .1], "confidence": confidence,
            "scale": "small", "state": state}


def test_support_is_class_aware_unique_and_keeps_original_geometry_and_weight():
    original = [candidate(), candidate(x=.501), candidate(cls=1)]
    snapshot = copy.deepcopy(original)
    result = promote_consistent(original, [candidate(confidence=.8)])
    assert len(result) == 1
    assert result[0]["xywhn"] == original[0]["xywhn"]
    assert result[0]["confidence"] == .25
    assert result[0]["support_confidence"] == .8
    assert original == snapshot


def test_existing_positive_duplicates_are_excluded_before_support_assignment():
    original = [candidate(x=.2, state="positive"), candidate(x=.2), candidate(x=.8)]
    view = [candidate(x=.2, confidence=.9), candidate(x=.8, confidence=.9)]
    result = promote_consistent(original, view)
    assert len(result) == 1 and result[0]["xywhn"][0] == .8


def test_support_bounds_scale_and_empty_inputs():
    assert promote_consistent([], [candidate(confidence=.8)]) == []
    assert promote_consistent([candidate()], []) == []
    assert promote_consistent([candidate()], [candidate(confidence=.49)]) == []
    assert promote_consistent([candidate(confidence=.14)], [candidate(confidence=.8)]) == []
    assert promote_consistent([{**candidate(), "scale": "medium"}], [candidate(confidence=.8)]) == []
    assert promote_consistent([candidate()], [candidate(x=.6, confidence=.8)]) == []
    with pytest.raises(ValueError):
        promote_consistent([], [], match_iou=0)
