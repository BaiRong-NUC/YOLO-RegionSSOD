"""Synthetic-only evaluator tests: 548 synthetic records, no real inference."""
import copy
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_bounded_retention as ev


def population():
    images = [dict(id=i+1, file_name=f"synthetic/{i:03}.jpg", width=128, height=128) for i in range(548)]
    gt = dict(info={"synthetic": True}, images=images,
        categories=[dict(id=i+1, name=f"class_{i}") for i in range(10)],
        annotations=[dict(id=i+1, image_id=i+1, category_id=i+1, bbox=[0., 0., 10., 10.], area=100., iscrowd=0)
                     for i in range(10)])
    rows = [dict(image=im["file_name"], shape=[128, 128], classes=[], scores=[], xyxy=[]) for im in images]
    return gt, rows


def put(rows, image, predictions):
    row = rows[image]
    row["classes"] = [int(p[0]) for p in predictions]
    row["scores"] = [float(p[1]) for p in predictions]
    row["xyxy"] = [list(p[2]) for p in predictions]


def test_all_548_and_ten_class_coco_reports_without_performance_gates():
    gt, rows = population()
    for j in range(10):
        put(rows, j, [(j, .8, [0, 0, 10, 10])])
    report = ev.compare_records({"reference": rows, "candidate": copy.deepcopy(rows)}, gt)
    assert report["performance_gates"] is None and report["scope"]["images"] == 548
    metrics = report["arms"]["candidate"]
    assert metrics["coco"]["metrics"]["AP_all"] == pytest.approx(1.)
    assert metrics["coco"]["metrics"]["AP_small"] == pytest.approx(1.)
    assert len(metrics["coco"]["per_class"]) == 10
    for value in metrics["coco"]["per_class"].values():
        assert value["AP_all"] == pytest.approx(1.) and value["AP_small"] == pytest.approx(1.)
    for iou in ("0.5", "0.75"):
        assert metrics["operating_points"]["conf025"][iou]["sizes"]["all"]["recall"] == 1.
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "bad_score", "bad_box", "bad_class", "bad_index"])
def test_invalid_or_incomplete_capture_rejected(problem):
    gt, rows = population()
    put(rows, 0, [(0, .8, [0, 0, 10, 10])])
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif problem == "bad_score":
        rows[0]["scores"][0] = float("nan")
    elif problem == "bad_box":
        rows[0]["xyxy"][0] = [1, 0, 0, 10]
    elif problem == "bad_class":
        rows[0]["classes"][0] = .5
    else:
        rows[0]["original_indices"] = [-1]
    with pytest.raises(ValueError):
        ev.analyze_records(rows, gt)


def test_same_k_breaks_score_tie_by_path_and_original_index_not_gt():
    gt, rows = population()
    put(rows, 0, [(0, .8, [0, 0, 10, 10]), (0, .8, [80, 80, 90, 90])])
    rows[0]["original_indices"] = [1, 0]
    put(rows, 1, [(1, .8, [0, 0, 10, 10])])
    prepared = ev._prepare(list(reversed(rows)), gt)
    selected, meta = ev.select_same_k(prepared, 1)
    metric, _ = ev._match(prepared, gt, selected, .5)
    assert meta["available"] and meta["selected_count"] == 1
    assert selected[rows[0]["image"]] == [0]
    assert metric["tp"] == 0 and metric["fp"] == 1
    selected2, _ = ev.select_same_k(prepared, 2)
    assert selected2[rows[0]["image"]] == [0, 1] and not selected2[rows[1]["image"]]


def test_same_fp_uses_whole_score_groups_and_reports_slack():
    gt, rows = population()
    for j, x in enumerate((20, 40), 11):
        gt["annotations"].append(dict(id=j, image_id=1, category_id=1, bbox=[x, 0, 10, 10], area=100, iscrowd=0))
    put(rows, 0, [(0, .9, [0, 0, 10, 10]), (0, .8, [80, 80, 90, 90]),
                  (0, .8, [20, 0, 30, 10]), (0, .7, [40, 0, 50, 10]), (0, .7, [80, 60, 90, 70])])
    p = ev._prepare(rows, gt)
    selection, meta = ev.select_same_fp(p, gt, 1, .5)
    metric, _ = ev._match(p, gt, selection, .5)
    assert meta["available"] and meta["threshold"] == .8 and meta["actual_fp"] == 1
    assert metric["predictions"] == 3 and metric["tp"] == 2
    zero, meta0 = ev.select_same_fp(p, gt, 0, .5)
    assert meta0["selected_count"] == 1 and meta0["threshold"] == .9
    assert ev._match(p, gt, zero, .5)[0]["fp"] == 0
    _, exhausted = ev.select_same_fp(p, gt, 100, .5)
    assert exhausted["available_within_frozen_inference"] and exhausted["actual_fp"] == 2 and exhausted["slack"] == 98
    assert not exhausted["budget_saturated"] and not exhausted["uncapped_population_certified"]


def test_zero_budget_can_choose_empty_prefix_without_choosing_tp_from_tie():
    gt, rows = population()
    put(rows, 0, [(0, .8, [80, 80, 90, 90]), (0, .8, [0, 0, 10, 10]), (0, .8, [60, 60, 70, 70])])
    p = ev._prepare(rows, gt)
    selection, meta = ev.select_same_fp(p, gt, 0, .5)
    assert meta["available"] and meta["empty_prefix"] and meta["threshold"] is None
    assert ev._match(p, gt, selection, .5)[0]["predictions"] == 0
    _, slack = ev.select_same_fp(p, gt, 1, .5)
    assert slack["available_within_frozen_inference"] and slack["slack"] == 1
    assert not slack["budget_saturated"] and not slack["exhausted_saved_predictions"]


def test_selected_images_are_rematched_not_filtered_from_full_tp_flags():
    gt, rows = population()
    put(rows, 0, [(0, .9, [0, 0, 10, 10]), (0, .8, [0, 0, 10, 10])])
    p = ev._prepare(rows, gt)
    _, full = ev._match(p, gt, ev._selection(p), .5)
    assert not full["flags"][rows[0]["image"], 1]
    selected = ev._prefix_selection(p, [], 0)
    selected[rows[0]["image"]] = [1]
    metric, _ = ev._match(p, gt, selected, .5)
    assert metric["tp"] == 1 and metric["fp"] == 0


def test_floor_and_cap_flags_do_not_invalidate_frozen_population_budget_results():
    gt, rows = population()
    put(rows, 0, [(0, .5, [80, 80, 90, 90])]*500)
    put(rows, 1, [(1, .5, [0, 0, 10, 10])])
    p = ev._prepare(rows, gt)
    assert ev.select_same_k(p, 500)[1]["available"]
    assert ev.select_same_k(p, 500)[1]["uncapped_population_certified"]
    assert ev.select_same_k(p, 501)[1]["available_within_frozen_inference"]
    assert not ev.select_same_k(p, 501)[1]["uncapped_population_certified"]
    assert not ev.select_same_k(p, 502)[1]["available"]
    assert ev.select_same_fp(p, gt, 500, .5)[1]["available_within_frozen_inference"]
    assert not ev.select_same_fp(p, gt, 500, .5)[1]["uncapped_population_certified"]
    gt, rows = population()
    put(rows, 0, [(0, .001, [0, 0, 10, 10])])
    p = ev._prepare(rows, gt)
    assert ev.select_same_k(p, 1)[1]["available_within_frozen_inference"]
    assert not ev.select_same_k(p, 1)[1]["uncapped_population_certified"]
    assert ev.select_same_k(p, 0)[1]["available"]


@pytest.mark.parametrize("box", [[0, 0, 0, 10], [0, 0, 10, 0], [4, 4, 4, 4]])
def test_zero_area_native_prediction_is_retained_as_fp_in_ap_and_budgets(box):
    gt, rows = population()
    put(rows, 0, [(0, .9, box), (0, .8, [0, 0, 10, 10])])
    summary = ev.analyze_records(rows, gt)
    assert summary["prediction_count"] == 2 and summary["zero_area_predictions"] == 1
    assert summary["coco"]["per_class"]["class_0"]["AP_all"] == pytest.approx(.5)
    point = summary["operating_points"]["conf025"]["0.5"]
    assert point["tp"] == 1 and point["fp"] == 1 and point["predictions"] == 2
    p = ev._prepare(rows, gt)
    selected, meta = ev.select_same_k(p, 1)
    assert meta["available"] and ev._match(p, gt, selected, .5)[0]["fp"] == 1
    _, fp = ev.select_same_fp(p, gt, 0, .5)
    assert fp["empty_prefix"]


def test_zero_area_gt_is_rejected_even_though_zero_area_predictions_are_allowed():
    gt, rows = population()
    gt["annotations"][0].update(bbox=[0, 0, 0, 10], area=0)
    with pytest.raises(ValueError, match="validation GT"):
        ev._prepare(rows, gt)


def test_all_three_predeclared_pairs_reuse_coco_and_control_sets_its_own_budgets(monkeypatch):
    gt, reference = population()
    candidate, control = copy.deepcopy(reference), copy.deepcopy(reference)
    put(reference, 0, [(0, .9, [0, 0, 10, 10])])  # K=1, FP=0
    put(control, 0, [(0, .9, [0, 0, 10, 10]), (0, .8, [80, 80, 90, 90])])  # K=2, FP=1
    put(candidate, 0, [(0, .9, [1, 0, 11, 10]), (0, .7, [80, 80, 90, 90])])
    real = ev._coco
    calls = []
    def counted(*args):
        calls.append(1)
        return real(*args)
    monkeypatch.setattr(ev, "_coco", counted)
    report = ev.compare_records({"control": control, "candidate": candidate, "reference": reference}, gt)
    assert len(calls) == 3
    assert list(report["pairs"]) == ["reference->candidate", "reference->control", "control->candidate"]
    pair = report["pairs"]["control->candidate"]
    assert pair["baseline_arm"] == "control" and pair["other_arm"] == "candidate"
    assert pair["same_k"]["requested_k"] == 2
    assert pair["same_fp"]["0.5"]["fp_budget"] == 1
    assert report["pairs"]["reference->candidate"]["same_k"]["requested_k"] == 1
    for value in report["pairs"].values():
        assert set(value["common_small_geometry"]) == {"conf025", "same_fp_iou0.5", "same_k"}
        assert value["common_small_geometry"]["conf025"]["transitions"]["both"] == 1


def test_error_partition_and_iou075_localization_recall():
    gt, rows = population()
    # Duplicate, wrong-class, nearby localization, and background are all explicit.
    put(rows, 0, [(0, .9, [0, 0, 10, 10]), (0, .8, [0, 0, 10, 10]),
                  (1, .7, [0, 0, 10, 10]), (0, .6, [5, 0, 15, 10]), (0, .5, [80, 80, 90, 90])])
    p = ev._prepare(rows, gt)
    result, _ = ev._match(p, gt, ev._selection(p), .5)
    assert result["errors"] == dict(matched=1, duplicate_competition=1, wrong_class=1,
        localization_near=1, near_other_class=0, unexplained_background=1)
    assert sum(result["errors"].values()) == result["predictions"]
    put(rows, 0, [(0, .8, [2, 0, 12, 10])])
    p = ev._prepare(rows, gt)
    assert ev._match(p, gt, ev._selection(p), .5)[0]["tp"] == 1
    assert ev._match(p, gt, ev._selection(p), .75)[0]["tp"] == 0


def test_common_small_geometry_discloses_intersection_and_gains_losses():
    gt, reference = population()
    candidate = copy.deepcopy(reference)
    put(reference, 0, [(0, .8, [0, 0, 10, 10])]); put(reference, 1, [(1, .8, [0, 0, 10, 10])])
    put(candidate, 0, [(0, .8, [1, 0, 11, 10])]); put(candidate, 2, [(2, .8, [0, 0, 10, 10])])
    report = ev.compare_records({"reference": reference, "candidate": candidate}, gt)
    geometry = report["comparisons"]["candidate"]["common_small_geometry"]["conf025"]
    assert geometry["diagnostic_only"]
    assert geometry["common_gt_ids"] == [1]
    assert geometry["transitions"] == dict(total_small=10, both=1, reference_only=1, other_only=1, neither=7, net_gain=0)
    assert geometry["geometry"]["normalized_center_error"]["other"]["mean"] == pytest.approx(.1)
    assert geometry["geometry"]["normalized_edge_error"]["other"]["mean"] == pytest.approx(.05)


def make_cache(out):
    gt, rows = population()
    put(rows, 0, [(0, .8, [0, 0, 10, 10])])
    manifest = dict(schema=ev.SCHEMA, population=548, inference=ev.INFERENCE,
        counts={k: v for k, v in ev._counts(gt).items() if k != "all"},
        weights={arm: dict(path="FORBIDDEN_NONEXISTENT_WEIGHT", sha256="a"*64) for arm in ("reference", "candidate")},
        capture_input_sha256={"FORBIDDEN_REAL_LABEL": "not_read"},
        source_sha256={k: ev.digest(p) for k, p in ev._source_files().items()})
    ev._write(out/"manifest.json", manifest); ev._write(out/"ground_truth.json", gt)
    for key, path in ev._source_files().items():
        (out/"source").mkdir(exist_ok=True)
        shutil.copyfile(path, out/"source"/f"{key}.py")
    for arm in manifest["weights"]:
        folder = out/arm; folder.mkdir()
        (folder/"predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n", encoding="utf-8")
        ev._write(folder/"inference.json", dict(weights_sha256="a"*64, inference_passes=1,
            predictions_sha256=ev.digest(folder/"predictions.jsonl"), ultralytics={"synthetic": True}))
    ev._write(out/"report.json", ev.compare_records({k: rows for k in manifest["weights"]}, gt))
    ev._seal(out)


def test_audit_only_reads_own_new_synthetic_cache_and_pure_sources(tmp_path, monkeypatch):
    out = tmp_path/"new_evaluation"
    make_cache(out)
    real_digest = ev.digest
    sources = set(p.resolve() for p in ev._source_files().values())
    def guarded(path):
        path = Path(path).resolve()
        assert path.is_relative_to(out) or path in sources
        return real_digest(path)
    monkeypatch.setattr(ev, "digest", guarded)
    monkeypatch.setattr(ev, "_capture_validator", lambda: pytest.fail("audit attempted inference"))
    monkeypatch.setattr(ev, "evaluate", lambda *args: pytest.fail("audit attempted evaluate"))
    result = ev.audit(out)
    assert result["audit_ok"] and result["numeric_replay"] and result["images"] == 548
    assert result["inference_calls"] == 0 and not result["external_data_or_weights_read"]
    (out/"reference"/"predictions.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        ev.audit(out)


def test_audit_detects_numeric_tampering_even_when_artifacts_are_rehashed(tmp_path):
    out = tmp_path/"new_evaluation"
    make_cache(out)
    report = ev._read(out/"report.json")
    report["arms"]["candidate"]["prediction_count"] += 1
    ev._write(out/"report.json", report); ev._seal(out)
    with pytest.raises(ValueError, match="numeric replay"):
        ev.audit(out)


def test_audit_rejects_old_schema_without_accessing_its_paths(tmp_path):
    ev._write(tmp_path/"manifest.json", dict(schema="old_class_ambiguity", weights={"reference": "never_read"}))
    with pytest.raises(ValueError, match="compatible"):
        ev.audit(tmp_path)


def test_capture_shell_uses_one_stub_pass_per_arm_then_audits_without_original_inputs(tmp_path, monkeypatch):
    """All I/O is newly generated synthetic bytes; no Ultralytics is imported."""
    gt, rows = population()
    put(rows, 0, [(0, .8, [0, 0, 10, 10])])
    synthetic_data = tmp_path/"synthetic_data.txt"
    synthetic_data.write_text("synthetic fixture, not a dataset", encoding="utf-8")
    helper = ModuleType("scripts.evaluate_paper1_small_pr")
    helper.DATA = synthetic_data
    helper.ground_truth = lambda: (copy.deepcopy(gt), {str(synthetic_data): ev.digest(synthetic_data)},
        {k: v for k, v in ev._counts(gt).items() if k != "all"})
    monkeypatch.setitem(sys.modules, "scripts.evaluate_paper1_small_pr", helper)
    calls = []
    sentinel = object()
    class StubModel:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint
        def val(self, *, validator, data, project, name, exist_ok, **kwargs):
            assert validator is sentinel and kwargs == ev.INFERENCE
            assert name == "validator" and exist_ok is False and data == str(synthetic_data)
            calls.append(self.checkpoint)
            folder = Path(project); folder.mkdir()
            (folder/"predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n", encoding="utf-8")
            return SimpleNamespace(results_dict={"synthetic_stub_metric": .123})
    fake_ultralytics = ModuleType("ultralytics"); fake_ultralytics.YOLO = StubModel
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(ev, "_capture_validator", lambda: sentinel)
    weights = {}
    for arm in ("reference", "candidate", "control"):
        path = tmp_path/arm/"best.pt"
        path.parent.mkdir(); path.write_bytes(b"synthetic bytes: deliberately not a checkpoint")
        weights[arm] = path
    out = tmp_path/"fresh_output"
    report = ev.evaluate(weights, out)
    assert len(calls) == 3 and set(report["arms"]) == set(weights)
    assert report["performance_gates"] is None
    with pytest.raises(FileExistsError):
        ev.evaluate(weights, out)
    for path in weights.values():
        path.unlink()
    synthetic_data.unlink()
    assert ev.audit(out)["numeric_replay"]
    assert len(calls) == 3
