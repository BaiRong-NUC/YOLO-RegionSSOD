import copy
from pathlib import Path
import sys

import pytest

from regionssod.pseudo.calibration import (
    calibrated_weights, fit_calibrator, monotone_means, reliability_metrics,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_confidence_calibration import evaluate, qualifies


def item(score, cls=0, scale="small"):
    return {"confidence": score, "class_id": cls, "scale": scale}


def test_isotonic_projection_pools_with_observation_weights():
    assert monotone_means([.8, .2, .9], [1, 3, 1]) == pytest.approx([.35, .35, .9])


def test_open_label_calibration_is_monotone_and_sparse_groups_fall_back():
    items = [item(.35), item(.35), item(.7), item(.7), item(.7, cls=1)]
    targets = [0, 0, 1, 1, 1]
    snapshot = copy.deepcopy(items)
    model = fit_calibrator(items, targets, edges=[.3, .5, 1.000001],
                           global_prior=1, group_prior=1, min_group=2)
    assert "1:small" not in model["groups"]
    conditional, supported = calibrated_weights(items, model)
    global_weights, _ = calibrated_weights(items, model, conditional=False)
    assert conditional[-1] == global_weights[-1] and not supported[-1]
    assert conditional[0] <= conditional[2]
    assert all(0 <= w <= 1 for w in conditional)
    assert items == snapshot


def test_group_calibration_uses_classes_and_scale_without_geometry_or_gt_at_inference():
    items = [item(.5, cls=0)] * 10 + [item(.5, cls=1)] * 10
    model = fit_calibrator(items, [1] * 10 + [0] * 10, edges=[.3, 1.000001],
                           global_prior=1, group_prior=1, min_group=5)
    weights, support = calibrated_weights(
        [item(.5, cls=0), item(.5, cls=1), item(.5, cls=0, scale="large")], model)
    assert weights[0] > weights[2] > weights[1]
    assert support == [True, True, False]


def test_brier_and_error_mass_are_distinct_checks():
    result = reliability_metrics([1, 0], [.8, .2])
    assert result["brier"] == pytest.approx(.04)
    assert result["error_weight_share"] == pytest.approx(.2)
    assert reliability_metrics([], [])["brier"] is None


@pytest.mark.parametrize("targets,options", [
    ([1, 0], {}), ([float("nan")], {}), ([.5], {}), ([1], {"group_prior": 0})])
def test_bad_fit_targets_or_priors_cannot_produce_a_model(targets, options):
    with pytest.raises(ValueError):
        fit_calibrator([item(.5)], targets, edges=[.3, 1.000001], **options)


def test_conditional_rule_cannot_pass_from_only_global_calibration_gain():
    model = {"edges": [.3, 1.000001], "fit_positives": 10, "global_probabilities": [.8],
             "groups": {"0:small": {"probabilities": [.8]}}}
    result, _ = evaluate([item(.4)], [1], model)
    rule = {"min_fit_positives": 1, "min_positives": 1, "min_small_positives": 1,
            "min_conditional_coverage": .8, "min_brier_gain_vs_raw": .01,
            "min_brier_gain_vs_global": .002, "min_small_brier_gain_vs_raw": .005,
            "max_error_weight_share_increase": .02}
    gate = qualifies(result, model, rule)
    assert gate["checks"]["brier_vs_raw"] and not gate["checks"]["brier_vs_global"]
    assert not gate["passed"]
    empty, _ = evaluate([], [], model)
    assert not qualifies(empty, model, rule)["passed"]


def test_nonfinite_confidence_is_rejected_before_fitting():
    with pytest.raises(ValueError):
        fit_calibrator([item(float("nan"))], [1], edges=[.3, 1.000001])
