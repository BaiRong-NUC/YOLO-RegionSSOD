"""The continuation decision must retain its guardrails and fail closed."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_class_ambiguity_pair import continuation, analyze_records


def result():
    return {"ultralytics": {"metrics/mAP50-95(B)": .25},
            "coco": {"metrics": {"AP_small": .1}},
            "precision_constraints": {"0.5": {
                "0.7": {"small_recall": .3}, "0.9": {"small_recall": .1}}},
            "conf025": {"0.5": {"small_recall": .4}}}


def test_gain_at_p90_requires_all_preservation_guards():
    ref = result()
    candidate = deepcopy(ref)
    candidate["precision_constraints"]["0.5"]["0.9"]["small_recall"] += .004
    assert continuation(ref, candidate)["worth_larger_training"]
    candidate["coco"]["metrics"]["AP_small"] -= .001
    report = continuation(ref, candidate)
    assert not report["worth_larger_training"]
    assert not report["gates"]["AP_small"]
    assert report["gates"]["small_at_P0.9"]


@pytest.mark.parametrize("missing", [None, {"small_recall": float("nan")}])
def test_missing_or_nonfinite_operating_point_never_passes(missing):
    ref, candidate = result(), result()
    candidate["precision_constraints"]["0.5"]["0.9"] = missing
    assert not continuation(ref, candidate)["worth_larger_training"]


def test_partial_validation_population_rejected():
    with pytest.raises(ValueError, match="population"):
        analyze_records([], {"images": []}, {})
