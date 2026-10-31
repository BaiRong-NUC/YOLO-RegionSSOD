from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_regression_confidence import describe


def test_constant_weight_does_not_invent_a_quality_improvement():
    rows = [{"confidence": .6, "gt_same_iou": .2 + .1*(i % 5),
             "unique_match_050": i % 2 == 0, "class_id": i % 3} for i in range(360)]
    result = describe(rows)
    assert result["effective_samples"] == pytest.approx(360)
    assert result["delta"]["mean_iou"] == pytest.approx(0, abs=1e-12)
    assert not result["candidate"]


def test_class_control_removes_improvement_due_only_to_class_composition():
    rows = [{"confidence": .9 if c == 0 else .3, "gt_same_iou": .9 if c == 0 else .2,
             "unique_match_050": c == 0, "class_id": c} for c in range(3) for _ in range(120)]
    result = describe(rows)
    assert result["delta"]["mean_iou"] > .02
    assert result["class_controlled_iou_gap"] == pytest.approx(0, abs=1e-12)
    assert not result["candidate"]


def test_invalid_weights_and_empty_samples_fail_closed():
    assert not describe([])["candidate"]
    with pytest.raises(ValueError):
        describe([{"confidence": 0., "gt_same_iou": .9, "unique_match_050": True, "class_id": 0}])
