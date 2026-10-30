import copy
from pathlib import Path
import sys

import pytest

from regionssod.pseudo.localization import localization_stability

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_positive_localization import controlled_gap


def box(cls=0, x=.5, confidence=.8):
    return {"class_id": cls, "xywhn": [x, .5, .1, .1], "confidence": confidence}


def test_stability_has_distinct_same_class_support_and_preserves_predictions():
    positive = [box(), box(x=.501), box(cls=1)]
    view = [box()]
    snapshot = copy.deepcopy((positive, view))
    result = localization_stability(positive, view)
    assert result[0]["group"] == "stable" and result[0]["iou"] == 1
    assert result[0]["ambiguous"]
    assert result[1]["group"] == "unsupported" and result[1]["ambiguous"]
    assert result[2]["group"] == "unsupported" and not result[2]["ambiguous"]
    assert (positive, view) == snapshot


def test_missing_support_is_unknown_and_bad_inputs_are_rejected():
    assert localization_stability([], []) == []
    assert localization_stability([box()], [box(confidence=.04)])[0]["iou"] is None
    with pytest.raises(ValueError):
        localization_stability([{**box(), "state": "ignore"}], [])
    with pytest.raises(ValueError):
        localization_stability([box()], [], stable_at=.4)
    with pytest.raises(ValueError):
        localization_stability([{**box(), "xywhn": [.5, .5, 0, .1]}], [])


def test_geometry_displacement_uses_box_units_in_rectangular_native_tiles():
    result = localization_stability([box()], [box(x=.55)])[0]
    assert result["iou"] == pytest.approx(1/3)
    assert result["center_shift"] == pytest.approx(.5)
    assert result["log_shape_change"] == 0
    assert result["group"] == "unstable"


def test_confidence_control_removes_composition_only_quality_gap():
    # Stable boxes are mostly in the easy high-confidence band.
    rows = []
    for confidence, quality, stable, unstable in ((.35, .2, 10, 90), (.85, .9, 90, 10)):
        for group, count in (("stable", stable), ("unstable", unstable)):
            rows.extend({"confidence": confidence, "class_id": 0, "group": group,
                         "gt_same_iou": quality} for _ in range(count))
    rule = {"min_per_group_per_stratum": 5, "min_comparable_per_group": 10,
            "min_strata": 2, "min_comparable_fraction": .1, "min_quality_fraction_gap": .1}
    config = {"confidence_edges": [0, .5, 1.01], "quality_iou": .75,
              "confidence_control": rule}
    result = controlled_gap(rows, config)
    assert result["weighted_quality_gap"] == 0
    assert not result["passed"]


def test_sparse_strata_cannot_pass_from_a_large_raw_quality_gap():
    rows = [{"confidence": .8, "class_id": 0, "group": g, "gt_same_iou": q}
            for g, q in (("stable", 1), ("unstable", 0))]
    rule = {"min_per_group_per_stratum": 5, "min_comparable_per_group": 10,
            "min_strata": 2, "min_comparable_fraction": .1, "min_quality_fraction_gap": .1}
    result = controlled_gap(rows, {"confidence_edges": [0, 1.01], "quality_iou": .75,
                                  "confidence_control": rule})
    assert result["weighted_quality_gap"] is None
    assert not result["passed"]
