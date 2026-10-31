from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_regression_confidence_mass import compute


def test_mass_exposes_scale_composition_and_empty_groups_without_claiming_loss():
    rows = [
        {"class_id": 0, "scale": "small", "confidence": .3},
        {"class_id": 0, "scale": "small", "confidence": .5},
        {"class_id": 3, "scale": "medium", "confidence": .8},
    ]
    result = compute(rows)
    assert result["all"]["mean_confidence"] == pytest.approx(1.6 / 3)
    small = result["per_scale"]["small"]
    assert small["count_share"] == pytest.approx(2 / 3)
    assert small["confidence_mass_share"] == pytest.approx(.5)
    assert small["share_delta_pp"] == pytest.approx(-100 / 6)
    assert small["effective_samples"] == pytest.approx(.8**2 / (.3**2 + .5**2))
    assert result["per_scale"]["large"]["mean_confidence"] is None
    assert result["per_class"]["9"]["boxes"] == 0
    assert result["per_class_scale"]["0"]["small"] == small
    assert result["uses_hidden_or_validation_gt"] is False


@pytest.mark.parametrize("changes", [
    {"confidence": 0}, {"confidence": 1.1}, {"confidence": float("nan")},
    {"scale": "unknown"}, {"class_id": 10}, {"class_id": True},
])
def test_mass_rejects_invalid_pseudo_metadata(changes):
    row = {"class_id": 0, "scale": "small", "confidence": .5} | changes
    with pytest.raises(ValueError):
        compute([row])
