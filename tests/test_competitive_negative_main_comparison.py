from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


@pytest.fixture
def runner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("main_comparison_test", scripts / "run_p1_main_comparison.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts))


def metrics(ap, recall):
    return {"ultralytics": {"metrics/mAP50(B)": ap + .1, "metrics/mAP50-95(B)": ap},
        "coco": {"metrics": {"AP_small": ap - .1}},
        "precision_constraints": {iou: {p: {"small_recall": recall} for p in ("0.7", "0.9")} for iou in ("0.5", "0.75")},
        "conf025": {"0.5": {"small_recall": recall, "fp": 1000}}}


def test_main_table_reports_total_and_increment_with_separate_control(runner, tmp_path, monkeypatch):
    models = {"baseline": metrics(.23, .30), "reference": metrics(.24, .32),
              "candidate": metrics(.242, .324), "strength": metrics(.241, .322)}
    actual = runner.main_comparison(models)
    assert list(actual["absolute_fraction"]) == ["baseline", "reference", "candidate"]
    assert actual["delta_pp"]["P1_minus_B2"]["AP50_95"] == pytest.approx(1.)
    assert actual["delta_pp"]["improved_minus_P1"]["AP50_95"] == pytest.approx(.2)
    assert actual["delta_pp"]["improved_minus_B2"]["AP50_95"] == pytest.approx(1.2)
    assert actual["delta_pp"]["improved_minus_B2"]["small_P90"] == pytest.approx(2.4)
    monkeypatch.setattr(runner, "OUT", tmp_path)
    runner.write_main_table(actual, {"baseline": 5000., "reference": 7000., "candidate": 7500.})
    saved = (tmp_path / "main_comparison.md").read_text(encoding="utf-8")
    assert "B2基线" in saved and "improved_minus_B2" in saved and "+1.2000" in saved
    assert "| strength |" not in saved


def test_main_success_does_not_fabricate_mechanism_confirmation(runner):
    models = {"baseline": metrics(.23, .30), "reference": metrics(.24, .32), "candidate": metrics(.242, .324)}
    result = runner.decision(models)
    assert result["performance_conditions_passed"]
    assert result["recommend_separate_control30"]
    assert result["mechanism_control"]["status"] == "NOT_TESTED"
    assert not result["recommend_second_seed30"]
    models["strength"] = deepcopy(models["candidate"])
    assert not runner.decision(models)["recommend_second_seed30"]
    models["strength"] = metrics(.241, .322)
    assert runner.decision(models)["recommend_second_seed30"]


@pytest.mark.parametrize("control", [False, True])
def test_check_only_never_calls_training_and_has_requested_arms(runner, monkeypatch, tmp_path, control):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    recorded = []
    monkeypatch.setattr(runner, "preflight", lambda: {"arms": dict(runner.NAMES)})
    monkeypatch.setattr(runner, "write", lambda p, v: recorded.append(v))
    monkeypatch.setattr(runner, "train", lambda: pytest.fail("check-only launched training"))
    monkeypatch.setattr(sys, "argv", ["run_p1_main_comparison.py", "--check-only"] + (["--with-control"] if control else []))
    runner.main()
    assert list(recorded[0]["arms"]) == ["baseline", "reference", "candidate"] + (["strength"] if control else [])


def test_baseline_is_independent30_not_formal100(runner):
    baseline = yaml.safe_load(runner.B2_CONFIG.read_text(encoding="utf-8"))
    original = yaml.safe_load((runner.ROOT / "configs/train_naive_10.yaml").read_text(encoding="utf-8"))
    original["name"] = runner.ALL_NAMES["baseline"]
    original["train"]["epochs"] = 30
    assert baseline == original


def test_unreachable_fixed_point_is_reported_and_cannot_pass(runner):
    models = {"baseline": metrics(.23, .30), "reference": metrics(.24, .32), "candidate": metrics(.242, .324)}
    models["candidate"]["conf025"]["0.5"] = None
    result = runner.decision(models)
    assert result["main_comparison"]["absolute_fraction"]["candidate"]["small_conf025"] is None
    assert result["main_comparison"]["delta_pp"]["improved_minus_B2"]["small_conf025"] is None
    assert not result["performance_conditions_passed"]
