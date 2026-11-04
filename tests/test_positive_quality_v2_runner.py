"""Synthetic files/metrics only: no real batch, YOLO forward, val or training."""
from copy import deepcopy
import hashlib
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from PIL import Image
import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("positive_quality_v2_runner", SCRIPTS / "run_positive_quality_transport_v2.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def models_and_errors():
    models, errors = {}, {}
    for arm in r.ARMS:
        candidate = arm == "candidate"
        fp, wrong = (90, 45) if candidate else (100, 50)
        models[arm] = {
            "ultralytics": {"metrics/mAP50(B)": .4, "metrics/mAP50-95(B)": .202 if candidate else .2},
            "coco": {"metrics": {"AP_small": .152 if candidate else .15}, "per_class_AP_small": {str(k): .1 for k in range(10)}},
            "precision_constraints": {iou: {p: {"small_recall": .3} for p in ("0.7", "0.9")} for iou in ("0.5", "0.75")},
            "conf025": {"0.5": {"small_recall": .35, "fp": fp}},
        }
        if candidate:
            models[arm]["precision_constraints"]["0.5"]["0.9"]["small_recall"] = .304
        counts = {name: 0 for name in r.ERROR_NAMES}
        counts.update(matched=10, wrong_class=wrong, unexplained_background=fp-wrong)
        errors[arm] = {"fp": fp, "counts": counts}
    return models, errors


def test_eleven_gates_and_all_signs_retained():
    models, errors = models_and_errors()
    comparison, decision = r.comparison_and_decision(models, errors)
    assert len(decision["gates"]) == 11
    assert sum(k.startswith("reference_") for k in decision["gates"]) == 8
    assert decision["all_eleven_passed"] is True
    assert decision["automatic_further_training"] is False
    assert decision["semantic_specificity"] == "necessary_development_controls_passed_not_causal_proof"
    assert decision["control_scope"] == comparison["control_scope"] == r.CONTROL_SCOPE
    assert "control does not preserve multiset" in decision["control_scope"]
    assert "changed-anchor count and target histogram remain confounds" in decision["control_scope"]
    assert len(comparison["per_class_AP_small_fraction"]["control"]) == 10
    models["candidate"]["coco"]["per_class_AP_small"]["0"] = .05
    comparison, _ = r.comparison_and_decision(models, errors)
    assert comparison["per_class_candidate_minus_other_pp"]["reference"]["0"] == -5
    control = models["control"]
    control["ultralytics"].update({"metrics/mAP50(B)": .42, "metrics/mAP50-95(B)": .19})
    control["coco"]["metrics"]["AP_small"] = .153
    control["conf025"]["0.5"].update(small_recall=.36, fp=103)
    control["precision_constraints"]["0.5"]["0.7"]["small_recall"] = .28
    control["precision_constraints"]["0.5"]["0.9"]["small_recall"] = .32
    errors["control"]["fp"] = 103
    errors["control"]["counts"].update(wrong_class=48, unexplained_background=55)
    comparison, _ = r.comparison_and_decision(models, errors)
    assert comparison["control_minus_reference_pp"] == pytest.approx({
        "AP50": 2., "AP50_95": -1., "AP_small": .3, "small_conf025": 1.,
        "small_iou050_P70": -2., "small_iou050_P90": 2., "small_iou075_P90": 0.})
    assert comparison["control_minus_reference_FP_counts"] == {"FP": 3, "wrong_class": -2}
    control["coco"]["metrics"]["AP_small"] = float("nan")
    comparison, _ = r.comparison_and_decision(models, errors)
    assert comparison["control_minus_reference_pp"]["AP_small"] is None
    json.dumps(comparison, allow_nan=False)


@pytest.mark.parametrize("gate", list(r.GATES))
def test_each_prespecified_gate_can_independently_stop(gate):
    models, errors = models_and_errors()
    candidate = models["candidate"]
    if gate == "reference_AP50_95_pp":
        models["reference"]["ultralytics"]["metrics/mAP50-95(B)"] = .202
    elif gate == "reference_AP_small_pp":
        models["reference"]["coco"]["metrics"]["AP_small"] = .16
    elif gate == "reference_small_conf025_pp":
        candidate["conf025"]["0.5"]["small_recall"] -= .003
    elif "small_iou" in gate:
        iou = "0.75" if "075" in gate else "0.5"
        p = "0.7" if "P70" in gate else "0.9"
        other = gate.split("_")[0]
        models[other]["precision_constraints"][iou][p]["small_recall"] = .31
    elif gate == "reference_FP_conf025_excess":
        candidate["conf025"]["0.5"]["fp"] = 101
        errors["candidate"]["fp"] = 101
        errors["candidate"]["counts"]["unexplained_background"] += 11
    elif gate == "reference_wrong_class_conf025_excess":
        errors["candidate"]["counts"]["wrong_class"] = 51
        errors["candidate"]["counts"]["unexplained_background"] -= 6
    elif gate == "control_AP50_95_pp":
        models["control"]["ultralytics"]["metrics/mAP50-95(B)"] = .202
    elif gate == "control_AP_small_pp":
        models["control"]["coco"]["metrics"]["AP_small"] = .16
    _, result = r.comparison_and_decision(models, errors)
    assert [k for k, v in result["gates"].items() if not v["passed"]] == [gate]
    assert result["outcome"] == "stop"
    assert result["automatic_retry"] is False and result["automatic_further_training"] is False
    if gate.startswith("control_"):
        assert result["semantic_specificity"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_unavailable_and_nonfinite_do_not_pass(value):
    models, errors = models_and_errors()
    models["candidate"]["precision_constraints"]["0.75"]["0.9"] = None if value is None else {"small_recall": value}
    _, result = r.comparison_and_decision(models, errors)
    assert not result["gates"]["reference_small_iou075_P90_pp"]["passed"]
    json.dumps(result, allow_nan=False)


def test_control_required_and_fp_must_agree():
    models, errors = models_and_errors()
    with pytest.raises(ValueError, match="three"):
        r.comparison_and_decision({k: v for k, v in models.items() if k != "control"}, errors)
    errors["candidate"]["fp"] += 1
    with pytest.raises(ValueError, match="disagree"):
        r.comparison_and_decision(models, errors)


def test_fp_priority_all_sizes_and_threshold_ties():
    # First two detections compete for class0 GT; third is wrong-class for class1.
    gt = {"images": [{"id": 1, "file_name": "synthetic.jpg"}], "annotations": [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
        {"image_id": 1, "category_id": 2, "bbox": [30, 0, 100, 100]},
    ]}
    records = [{"image": "synthetic.jpg", "classes": [0, 0, 2, 2, 2],
                "scores": [.9, .8, .25, .25, .249],
                "xyxy": [[0, 0, 10, 10], [0, 0, 10, 10], [30, 0, 130, 100],
                         [300, 300, 310, 310], [30, 0, 130, 100]]}]
    result = r.fp_decomposition(records, gt)
    assert result["fp"] == 3
    assert result["counts"]["wrong_class"] == 1
    assert result["counts"]["duplicate_competition"] == 1
    assert result["counts"]["unexplained_background"] == 1
    with pytest.raises(ValueError, match="population"):
        r.fp_decomposition(records * 2, gt)


def test_conflicting_hashes_and_tampering(tmp_path):
    p = tmp_path / "input.txt"
    p.write_text("original")
    mapping = r.hashes([p])
    r.verify(mapping)
    with pytest.raises(ValueError, match="Conflicting"):
        r.merge_hashes(mapping, {str(p): "0" * 64})
    p.write_text("drift")
    with pytest.raises(ValueError):
        r.verify(mapping)


def synthetic_population(tmp_path):
    images, rows = [], []
    for group in ("labeled", "pseudo"):
        image = tmp_path / group / "images/train/a.png"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(image)
        label = r.replace_images_component(image).with_suffix(".txt")
        label.parent.mkdir(parents=True)
        label.write_text("0 .5 .5 .4 .4\n")
        images.append(str(image))
        if group == "pseudo":
            rows.append({"image": str(image), "source_image": str(image),
                "positive": [{"class_id": 0, "confidence": 1., "xywhn": [.5, .5, .4, .4], "state": "positive"}], "ignore": []})
    return images, images[:1], rows


def test_population_confidence_one_is_still_pseudo(tmp_path):
    images, human, rows = synthetic_population(tmp_path)
    _, counts = r.validate_population(images, human, rows)
    assert counts == {"human_rows": 1, "pseudo_rows": 1, "positive": 1, "ignore": 0}


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "class", "box", "state", "label", "npy"])
def test_population_rejects_illegal_input(tmp_path, corruption):
    images, human, rows = synthetic_population(tmp_path)
    if corruption == "duplicate":
        rows *= 2
    elif corruption == "missing":
        rows = []
    elif corruption == "class":
        rows[0]["positive"][0]["class_id"] = .5
    elif corruption == "box":
        rows[0]["positive"][0]["xywhn"][0] = float("nan")
    elif corruption == "state":
        rows[0]["positive"][0]["state"] = "ignore"
    elif corruption == "label":
        r.replace_images_component(Path(images[-1])).with_suffix(".txt").write_text("0 .2 .5 .4 .4")
    elif corruption == "npy":
        Path(images[-1]).with_suffix(".npy").write_bytes(b"unbound")
    with pytest.raises(ValueError):
        r.validate_population(images, human, rows)


def fixture_engineering(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "BASE", tmp_path / "base.yaml")
    monkeypatch.setattr(r, "DOC", tmp_path / "doc")
    monkeypatch.setattr(r, "VALIDATION", tmp_path / "validation.json")
    monkeypatch.setattr(r, "INITIAL_CACHE", tmp_path / "cache.pt")
    monkeypatch.setattr(r, "SOURCE_CHECK", tmp_path / "source")
    monkeypatch.setattr(r, "PIPELINE", tmp_path / "trainer_pipeline.json")
    monkeypatch.setattr(r, "V1_VALIDATION", tmp_path / "v1_validation.json")
    paths = [r.BASE, r.DOC / "engineering_protocol.md", tmp_path / "scripts/validate_positive_quality_transport_v2.py",
             tmp_path / "src/regionssod/training/p1.py",
             tmp_path / "src/regionssod/training/reliable_competitor_negative.py",
             tmp_path / "src/regionssod/training/positive_quality_transport.py",
             tmp_path / "src/regionssod/training/positive_quality_pipeline_check.py",
             tmp_path / "src/regionssod/training/positive_quality_transport_v2.py"]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic")
    artifact = tmp_path / "cache.pt"
    artifact.write_bytes(b"synthetic cache - never loaded")
    trained = r.SOURCE_CHECK / "full_loss_cache.pt"
    trained.parent.mkdir()
    trained.write_bytes(b"synthetic trained cache - never loaded")
    r.write(r.PIPELINE, {"accepted": True,
        "v1_val_failure_reproduced": "super(type, obj): obj must be an instance or subtype of type",
        "actual_installed_get_dataloader": True, "new_real_batches": 0, "model_forward_calls": 0,
        "optimizer_steps": 0, "student_training_iterations": 0,
        "modes": {a: {"train_dataset": "SourceDataset", "val_dataset": "YOLODataset",
            "get_model_mode_bound": True, "real_detector_instantiated": False,
            "synthetic_train_images": 4, "synthetic_val_images": 1,
            "synthetic_train_batch_shape": [4, 3, 64, 64],
            "synthetic_val_batch_shape": [1, 3, 96, 96]} for a in r.ARMS}})
    r.write(r.V1_VALIDATION, {"accepted": True, "implementation": "positive_quality_transport_v1",
        "optimizer_steps": 0, "checks_actual_amp_path": True,
        "source_sha256": r.hashes([paths[3]]), "artifact_sha256": r.hashes([artifact, trained])})
    monkeypatch.setattr(r, "version", lambda name: {"torch": "synthetic", "ultralytics": "8.4.172"}[name])
    receipt = {"accepted": True, "optimizer_steps": 0, "parameter_hash_unchanged": True,
        "checks_actual_amp_path": True, "trainer_pipeline_checks": True,
        "new_synthetic_batches": 6, "new_dataloader_batches": 0, "new_model_forwards_this_script": 0,
        "pipeline_evidence": str(r.PIPELINE),
        "implementation": r.IMPLEMENTATION, "torch_version": "synthetic", "ultralytics_version": "8.4.172",
        "real_batch": {"images": 8, "amp": True, "device": "cuda:0"},
        "source_sha256": r.hashes(paths),
        "artifact_sha256": r.hashes([artifact, trained, r.V1_VALIDATION, r.PIPELINE])}
    r.write(r.VALIDATION, receipt)
    return receipt, paths, artifact


def test_engineering_receipt_complete_and_fresh(tmp_path, monkeypatch):
    receipt, paths, artifact = fixture_engineering(tmp_path, monkeypatch)
    assert str(artifact) in r.engineering()
    paths[-1].write_text("changed")
    with pytest.raises(ValueError):
        r.engineering()


@pytest.mark.parametrize("field,value", [("accepted", False), ("optimizer_steps", 1),
    ("parameter_hash_unchanged", False), ("checks_actual_amp_path", False),
    ("source_sha256", {}), ("artifact_sha256", {}),
    ("implementation", "old"), ("real_batch", {"images": 8, "amp": False, "device": "cpu"})])
def test_bad_engineering_refused(tmp_path, monkeypatch, field, value):
    receipt, _, _ = fixture_engineering(tmp_path, monkeypatch)
    receipt[field] = value
    r.write(r.VALIDATION, receipt)
    with pytest.raises(ValueError):
        r.engineering()


def dispatch_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUT", tmp_path / "output")
    monkeypatch.setattr(r, "RUNS", {a: tmp_path / "runs" / a for a in r.ARMS})
    configs = {a: {"mode": a} for a in r.ARMS}
    monkeypatch.setattr(r, "configurations", lambda: configs)
    monkeypatch.setattr(r, "expected_args", lambda c: c)
    p = tmp_path / "source.py"
    p.write_text("frozen source")
    receipt = {"input_sha256": r.hashes([p]), "train_arms": list(r.ARMS), "epochs_per_arm": 30,
        "arms": {a: str(p) for a, p in r.RUNS.items()}, "implementation": r.IMPLEMENTATION,
        "gates": r.canonical(r.GATES), "inference": r.INFERENCE, "configurations": configs,
        "expected_args": configs}
    r.write(r.OUT / "preflight.json", receipt)
    token = "test-only-token"
    r.write(r.OUT / "dispatch.json", {"preflight_sha256": r.sha(r.OUT / "preflight.json"),
        "train_arms": list(r.ARMS), "parent_pid": os.getppid(),
        "token_sha256": {a: hashlib.sha256(token.encode()).hexdigest() for a in r.ARMS}})
    r.write(r.OUT / "status.json", {"stage": "training", "arm": "reference"})
    monkeypatch.setenv("REGIONSSOD_PQT_V2_WORKER_TOKEN", token)
    return receipt


@pytest.mark.parametrize("corruption", ["missing_token", "parent", "arm", "source", "config", "dispatch", "claim", "existing"])
def test_worker_cannot_bypass_dispatch(tmp_path, monkeypatch, corruption):
    dispatch_fixture(tmp_path, monkeypatch)
    if corruption == "missing_token":
        monkeypatch.delenv("REGIONSSOD_PQT_V2_WORKER_TOKEN")
    elif corruption == "parent":
        d = r.read(r.OUT / "dispatch.json")
        d["parent_pid"] = -1
        r.write(r.OUT / "dispatch.json", d)
    elif corruption == "arm":
        r.write(r.OUT / "status.json", {"stage": "training", "arm": "candidate"})
    elif corruption == "source":
        (tmp_path / "source.py").write_text("changed")
    elif corruption == "config":
        monkeypatch.setattr(r, "configurations", lambda: {"reference": {"mode": "candidate"}})
    elif corruption == "dispatch":
        (r.OUT / "dispatch.json").unlink()
    elif corruption == "claim":
        r.write(r.OUT / "reference_claim.json", {"previous_attempt": True})
    elif corruption == "existing":
        r.RUNS["reference"].mkdir(parents=True)
    # Any successful import/execution would be a test failure, never a real run.
    implementation = ModuleType("regionssod.training.positive_quality_transport_v2")
    implementation.run_training = lambda *_: pytest.fail("training entered")
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    with pytest.raises((ValueError, FileNotFoundError, FileExistsError)):
        r.worker("reference")


def test_readonly_failed_terminal_and_extra_artifact(tmp_path, monkeypatch):
    dispatch_fixture(tmp_path, monkeypatch)
    r.write(r.OUT / "status.json", {"stage": "failed"})
    r.seal_terminal()
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(r, "write", lambda *_: pytest.fail("audit attempted a write"))
    result = r.audit()
    assert result["accepted"] is False and result["terminal_evidence_verified"] is True
    after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    (r.OUT / "unsealed.txt").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        r.audit()


def test_preflight_refuses_existing_output_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", Path.cwd())
    monkeypatch.setattr(r, "OUT", tmp_path)
    monkeypatch.setattr(r, "engineering", lambda: pytest.fail("preflight continued"))
    with pytest.raises(FileExistsError):
        r.preflight()


def test_actual_config_same_schedule_and_modes():
    configs = r.configurations()
    for a, c in configs.items():
        assert c["positive_quality_transport"] == {"mode": a}
        assert c["train"]["epochs"] == 30 and c["train"]["seed"] == 0
        assert c["train"]["batch"] == 8 and c["train"]["amp"] is True


@pytest.mark.parametrize("corruption", [None, "missing_pseudo", "duplicate", "box", "format"])
def test_combined_label_cache_contains_human_and_pseudo(tmp_path, corruption):
    images, _, _ = synthetic_population(tmp_path)
    cached = []
    for image in images:
        values = r.label_rows(r.replace_images_component(Path(image)).with_suffix(".txt"))
        cached.append({"im_file": image, "shape": (8, 8), "cls": values[:, :1], "bboxes": values[:, 1:],
                       "normalized": True, "bbox_format": "xywh", "segments": [], "keypoints": None})
    if corruption == "missing_pseudo":
        cached = cached[:1]
    elif corruption == "duplicate":
        cached.append(cached[0])
    elif corruption == "box":
        cached[1]["bboxes"][0, 0] = .1
    elif corruption == "format":
        cached[1]["bbox_format"] = "xyxy"
    cache = r.replace_images_component(Path(images[0])).parent.with_suffix(".cache")
    with cache.open("wb") as stream:
        np.save(stream, {"labels": cached})
    if corruption is None:
        assert set(r.validate_label_cache(images)) == {str(cache)}
    else:
        with pytest.raises(ValueError, match="cache"):
            r.validate_label_cache(images)


def training_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUT", tmp_path / "output")
    monkeypatch.setattr(r, "RUNS", {a: tmp_path / "runs" / a for a in r.ARMS})
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "CONFIGS", {a: tmp_path / f"{a}.yaml" for a in r.ARMS})
    data, model = tmp_path / "data.yaml", tmp_path / "model.pt"
    data.write_text("synthetic data")
    model.write_text("synthetic model, never loaded")
    configs = {a: {"name": a, "data": str(data), "model": str(model), "train": {"epochs": 30}}
               for a in r.ARMS}
    for a in r.ARMS:
        r.CONFIGS[a].write_text(yaml.safe_dump(configs[a]))
    inputs = r.hashes([data, model, *r.CONFIGS.values()])
    args = {a: {"name": a, "epochs": 30, "seed": 0, "resume": False, "amp": True} for a in r.ARMS}
    receipt = {"input_sha256": inputs, "configurations": configs, "expected_args": args}
    r.write(r.OUT / "preflight.json", receipt)
    for a, run in r.RUNS.items():
        (run / "weights").mkdir(parents=True)
        for n in ("best.pt", "last.pt"):
            (run / "weights" / n).write_text("synthetic " + a + n)
        (run / "args.yaml").write_text(yaml.safe_dump(args[a]))
        (run / "results.csv").write_text("epoch,time,loss\n" + "".join(f"{e},{e*10},.1\n" for e in range(1, 31)))
        r.write(run / "experiment.json", {"implementation": r.IMPLEMENTATION, "mode": a,
            "config": configs[a], "input_sha256": r.hashes([data, model, r.CONFIGS[a]])})
        r.write(r.OUT / f"{a}_exit.json", {"arm": a, "exit_code": 0, "worker_wall_seconds": 301.})
        r.write(r.OUT / f"{a}_claim.json", {"arm": a, "preflight_sha256": r.sha(r.OUT / "preflight.json")})
        r.write(r.OUT / f"{a}_seal.json", {"claim_sha256": r.sha(r.OUT / f"{a}_claim.json"),
            "artifact_sha256": r.hashes(p for p in run.rglob("*") if p.is_file())})
    return receipt


@pytest.mark.parametrize("corruption", [None, "resume", "epoch", "mode", "weight", "nonzero_exit", "missing_binding"])
def test_training_audit_requires_full_epochs_args_and_bound_weights(tmp_path, monkeypatch, corruption):
    receipt = training_fixture(tmp_path, monkeypatch)
    run = r.RUNS["candidate"]
    if corruption == "resume":
        args = yaml.safe_load((run / "args.yaml").read_text())
        args["resume"] = True
        (run / "args.yaml").write_text(yaml.safe_dump(args))
    elif corruption == "epoch":
        (run / "results.csv").write_text("epoch,time,loss\n1,10,.1\n")
    elif corruption in ("mode", "missing_binding"):
        exp = r.read(run / "experiment.json")
        if corruption == "mode":
            exp["mode"] = "reference"
        else:
            exp["input_sha256"].pop(str(r.CONFIGS["candidate"]))
        r.write(run / "experiment.json", exp)
    elif corruption == "weight":
        (run / "weights/best.pt").write_text("different checkpoint")
    elif corruption == "nonzero_exit":
        r.write(r.OUT / "candidate_exit.json", {"arm": "candidate", "exit_code": 1})
    if corruption not in (None, "weight"):
        seal = r.read(r.OUT / "candidate_seal.json")
        seal["artifact_sha256"] = r.hashes(p for p in run.rglob("*") if p.is_file())
        r.write(r.OUT / "candidate_seal.json", seal)
    if corruption is None:
        assert r.audit_arm("candidate", receipt)["epochs"] == 30
    else:
        with pytest.raises((ValueError, RuntimeError)):
            r.audit_arm("candidate", receipt)


def test_completed_audit_is_readonly_and_recomputes_decision(tmp_path, monkeypatch):
    dispatch_fixture(tmp_path, monkeypatch)
    models, errors = models_and_errors()
    comparison, decision = r.comparison_and_decision(models, errors)
    training, evaluated = {"accepted": True}, {"accepted": True, "numeric_replay": True}
    monkeypatch.setattr(r, "audit_training", lambda: training)
    import evaluate_class_ambiguity_pair
    monkeypatch.setattr(evaluate_class_ambiguity_pair, "audit", lambda _: evaluated)
    monkeypatch.setattr(r, "models_and_errors", lambda _: (models, errors))
    for filename, value in {
        "status.json": {"stage": "completed", "outcome": decision["outcome"]},
        "training_audit.json": training, "evaluation_audit.json": evaluated,
        "comparison.json": comparison, "decision.json": decision,
        "wrong_class_decomposition.json": errors,
        "evaluation/LEGACY_NOT_THIS_CANDIDATE.json": r.LEGACY_NOTICE,
    }.items():
        r.write(r.OUT / filename, value)
    r.seal_terminal()
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(r, "write", lambda *_: pytest.fail("audit wrote a file"))
    assert r.audit()["accepted"] is True
    after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    # A semantic replay mismatch fails even if stored artifact hashes are intact.
    models["control"]["coco"]["metrics"]["AP_small"] = .9
    with pytest.raises(ValueError, match="decision replay"):
        r.audit()


@pytest.mark.parametrize("fail_arm", [None, "reference", "candidate", "control"])
def test_dispatch_is_sequential_never_retries_and_evaluates_only_after_three(tmp_path, monkeypatch, fail_arm):
    monkeypatch.setattr(r, "OUT", tmp_path / "output")
    monkeypatch.setattr(r, "RUNS", {a: tmp_path / "runs" / a for a in r.ARMS})
    monkeypatch.setattr(r, "preflight", lambda: {"synthetic": True})
    monkeypatch.setattr(r, "verify_dispatch", lambda: {"synthetic": True})
    monkeypatch.setattr(r, "audit_arm", lambda *_: None)
    calls, evaluated = [], []
    def fake_run(command, **kwargs):
        arm = command[-1]
        calls.append(arm)
        assert "REGIONSSOD_PQT_V2_WORKER_TOKEN" in kwargs["env"]
        return r.subprocess.CompletedProcess(command, 1 if arm == fail_arm else 0)
    monkeypatch.setattr(r.subprocess, "run", fake_run)
    def finish():
        evaluated.append(True)
        r.write(r.OUT / "status.json", {"stage": "completed"})
    monkeypatch.setattr(r, "finish", finish)
    if fail_arm:
        with pytest.raises(r.subprocess.CalledProcessError):
            r.train()
        assert calls == list(r.ARMS[:r.ARMS.index(fail_arm)+1])
        assert not evaluated
        assert r.read(r.OUT / "status.json")["stage"] == "failed"
    else:
        r.train()
        assert calls == list(r.ARMS) and evaluated == [True]
    assert (r.OUT / "artifact_sha256.json").exists()


@pytest.mark.parametrize("arm", r.ARMS)
def test_preflight_refuses_any_existing_run(tmp_path, monkeypatch, arm):
    monkeypatch.setattr(r, "OUT", tmp_path / "absent_output")
    monkeypatch.setattr(r, "RUNS", {a: tmp_path / a for a in r.ARMS})
    r.RUNS[arm].mkdir()
    monkeypatch.setattr(r, "configurations", lambda: pytest.fail("preflight continued"))
    with pytest.raises(FileExistsError):
        r.preflight()


@pytest.mark.parametrize("missing", ["receipt", "loss_source", "initial_cache", "trained_cache", "actual_amp"])
def test_missing_engineering_evidence_rejected(tmp_path, monkeypatch, missing):
    receipt, paths, _ = fixture_engineering(tmp_path, monkeypatch)
    if missing == "receipt":
        r.VALIDATION.unlink()
    else:
        if missing == "loss_source":
            receipt["source_sha256"].pop(str(paths[-1]))
        elif missing == "actual_amp":
            receipt.pop("checks_actual_amp_path")
        else:
            path = r.INITIAL_CACHE if missing == "initial_cache" else r.SOURCE_CHECK / "full_loss_cache.pt"
            receipt["artifact_sha256"].pop(str(path))
        r.write(r.VALIDATION, receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        r.engineering()


def diagnostic_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "DIAGNOSTIC", tmp_path / "mixed")
    monkeypatch.setattr(r, "SOURCE_CHECK", tmp_path / "explicit")
    mixed = {"decision": "SIGNAL_INSUFFICIENT", "trained_explicit_human_source_available": False,
        "checks": {"assignment_geometry_20261031": True, "assignment_coverage_20261031": True}}
    explicit = {"human_signal_passed": True, "decision": "ELIGIBLE_FOR_MINIMAL_ENGINEERING",
        "groups": {"human": {"investment_checks": {k: True for k in (
            "multi_small_ge100", "inverse_small_ge10pct", "large_gap_small_ge5pct", "mass_transport_ge1pct")}}}}
    capture = {"new_model_forwards": 1, "new_dataloader_batches": 0, "cached_input_images": 8,
        "amp": "float16", "optimizer_steps": 0, "student_training_iterations": 0, "val_forwards": 0,
        "state_sha_before": "a" * 64, "state_sha_after": "a" * 64}
    for directory, script, protocol, summary in (
        (r.DIAGNOSTIC, "diagnose_positive_quality_transport.py", "protocol.md", mixed),
        (r.SOURCE_CHECK, "check_positive_quality_source.py", "source_check_protocol.md", explicit),
    ):
        sources = [tmp_path / "scripts" / script,
                   tmp_path / "docs/paper/diagnostics/positive_quality_transport" / protocol]
        for path in sources:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frozen synthetic source")
        r.write(directory / "summary.json", summary)
        r.write(directory / "objects.json", [])
        if directory == r.SOURCE_CHECK:
            r.write(directory / "capture.json", capture)
            r.write(directory / "started.json", {"source_sha256": r.hashes(sources)})
            for name in ("full_loss_cache.pt", "tensors.npz"):
                (directory / name).write_bytes(b"synthetic - never loaded")
        r.write(directory / "manifest.json", {"sources": r.hashes(sources),
            "artifacts": r.hashes(directory.iterdir())})


@pytest.mark.parametrize("corruption", [None, "manifest", "cache_coverage", "cache_bytes",
    "source_coverage", "false_human_signal", "missing_checks", "mixed_verdict", "extra_forward", "parameter_change"])
def test_bound_diagnostics_required_without_new_forward(tmp_path, monkeypatch, corruption):
    diagnostic_fixture(tmp_path, monkeypatch)
    manifest_path = r.SOURCE_CHECK / "manifest.json"
    if corruption == "manifest":
        manifest_path.unlink()
    elif corruption in ("cache_coverage", "source_coverage"):
        manifest = r.read(manifest_path)
        if corruption == "cache_coverage":
            manifest["artifacts"].pop(str(r.SOURCE_CHECK / "full_loss_cache.pt"))
        else:
            manifest["sources"].pop(str(tmp_path / "scripts/check_positive_quality_source.py"))
        r.write(manifest_path, manifest)
    elif corruption == "cache_bytes":
        (r.SOURCE_CHECK / "full_loss_cache.pt").write_bytes(b"tampered")
    elif corruption:
        directory = r.DIAGNOSTIC if corruption == "mixed_verdict" else r.SOURCE_CHECK
        path = directory / ("capture.json" if corruption in ("extra_forward", "parameter_change") else "summary.json")
        value = r.read(path)
        if corruption == "false_human_signal":
            value["human_signal_passed"] = False
        elif corruption == "missing_checks":
            value["groups"]["human"]["investment_checks"] = {}
        elif corruption == "mixed_verdict":
            value["decision"] = "ACCEPTED"
        elif corruption == "extra_forward":
            value["new_model_forwards"] = 2
        else:
            value["state_sha_after"] = "b" * 64
        r.write(path, value)
        manifest_path = directory / "manifest.json"
        manifest = r.read(manifest_path)
        manifest["artifacts"][str(path)] = r.sha(path)
        r.write(manifest_path, manifest)
    if corruption is None:
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
        assert str(r.SOURCE_CHECK / "full_loss_cache.pt") in r.diagnostic_evidence()
        assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            r.diagnostic_evidence()


@pytest.mark.parametrize("arm", r.ARMS)
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1.])
def test_nonfinite_or_unreachable_P90_never_proves_mechanism(arm, value):
    models, errors = models_and_errors()
    models[arm]["precision_constraints"]["0.5"]["0.9"] = None if value is None else {"small_recall": value}
    comparison, decision = r.comparison_and_decision(models, errors)
    assert decision["outcome"] == "stop"
    if arm != "reference":
        assert decision["semantic_specificity"] == "NOT_ESTABLISHED"
    json.dumps([comparison, decision], allow_nan=False)


def test_equal_FP_counts_allowed_and_control_is_only_necessary_evidence():
    models, errors = models_and_errors()
    models["candidate"]["conf025"]["0.5"]["fp"] = errors["reference"]["fp"]
    errors["candidate"] = deepcopy(errors["reference"])
    _, decision = r.comparison_and_decision(models, errors)
    assert decision["all_eleven_passed"]
    assert decision["semantic_specificity"] == "necessary_development_controls_passed_not_causal_proof"
    models["reference"]["ultralytics"]["metrics/mAP50-95(B)"] = .9
    _, decision = r.comparison_and_decision(models, errors)
    assert decision["outcome"] == "stop"
    assert not decision["automatic_further_training"]


def test_preflight_freezes_new_sources_independently_of_legacy_receipt(tmp_path, monkeypatch):
    configs = r.configurations()
    fixture_engineering(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(r, "OUT", tmp_path / "fresh_output")
    monkeypatch.setattr(r, "RUNS", {a: tmp_path / "runs" / a for a in r.ARMS})
    monkeypatch.setattr(r, "configurations", lambda: configs)
    for path in (tmp_path / "pyproject.toml", tmp_path / "AGENTS.md"):
        path.write_text("synthetic")
    helper = tmp_path / "src/regionssod/training/new_runtime_helper.py"
    helper.write_text("# runtime source added after old engineering\n")
    assert str(helper) not in r.read(r.VALIDATION)["source_sha256"]
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text("synthetic")
    monkeypatch.setattr(r, "diagnostic_evidence", lambda: r.hashes([diagnostic]))
    monkeypatch.setattr(r, "v1_failure_evidence", lambda: r.hashes([diagnostic]))
    monkeypatch.setattr(r, "runtime", lambda: (r.hashes([sys.executable]), {"torch": "synthetic"}))
    monkeypatch.setattr(r, "data_inputs", lambda _: (r.hashes([r.BASE]), {"synthetic": True}))
    monkeypatch.setattr(r.subprocess, "check_output", lambda *a, **k: "synthetic-commit\n")
    implementation = ModuleType("regionssod.training.positive_quality_transport_v2")
    implementation.IMPLEMENTATION = r.IMPLEMENTATION
    implementation.run_training = lambda *_: pytest.fail("training entered during preflight")
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    import ultralytics
    import torch
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **k: pytest.fail("preflight constructed a model"))
    monkeypatch.setattr(torch.utils.data, "DataLoader", lambda *a, **k: pytest.fail("preflight constructed a loader"))
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    receipt = r.preflight()
    assert receipt["input_sha256"][str(helper)] == r.sha(helper)
    assert receipt["input_sha256"][str(diagnostic)] == r.sha(diagnostic)
    assert receipt["forward_calls_this_check"] == receipt["optimizer_steps_this_check"] == 0
    assert receipt["epochs_per_arm"] == 30
    assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("corruption", [None, "missing_weight", "wrong_weight", "weight_bytes",
    "predictions", "analysis_weight", "analysis_manifest", "ground_truth", "unfrozen_input"])
def test_evaluation_binds_exact_arm_weights_and_predictions(tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(r, "OUT", tmp_path / "out")
    monkeypatch.setattr(r, "RUNS", {a: tmp_path / "runs" / a for a in r.ARMS})
    out = r.OUT / "evaluation"
    source = tmp_path / "source.py"
    source.write_text("frozen source")
    weights = {a: str(p / "weights/best.pt") for a, p in r.RUNS.items()}
    for a, p in weights.items():
        Path(p).parent.mkdir(parents=True)
        Path(p).write_text("synthetic weight " + a)
    gt = {"images": [{"id": i, "file_name": f"synthetic_{i}.jpg"} for i in range(548)], "annotations": []}
    counts = {"small": 0, "medium": 0, "large": 0}
    receipt = {"input_sha256": r.hashes([source]), "inference": r.INFERENCE,
        "population": {"gt_counts": counts,
            "ground_truth_sha256": hashlib.sha256(json.dumps(gt, sort_keys=True).encode()).hexdigest()}}
    manifest = {"weights": weights, "inference": r.INFERENCE, "counts": counts,
        "input_sha256": r.hashes([source, *weights.values()])}
    if corruption == "missing_weight":
        manifest["input_sha256"].pop(weights["candidate"])
    elif corruption == "wrong_weight":
        manifest["weights"] = weights | {"candidate": weights["reference"]}
    elif corruption == "unfrozen_input":
        receipt["input_sha256"] = {}
    r.write(out / "manifest.json", manifest)
    r.write(out / "ground_truth.json", gt)
    for a in r.ARMS:
        predictions = out / a / "predictions.jsonl"
        predictions.parent.mkdir(parents=True)
        predictions.write_text("\n".join(json.dumps({"image": im["file_name"],
            "scores": [], "classes": [], "xyxy": []}) for im in gt["images"]), encoding="utf-8")
        analysis = {"weights_sha256": r.sha(weights[a]), "predictions_sha256": r.sha(predictions),
            "manifest_sha256": r.sha(out / "manifest.json")}
        if a == "candidate":
            if corruption == "analysis_weight":
                analysis["weights_sha256"] = r.sha(weights["reference"])
            elif corruption == "analysis_manifest":
                analysis["manifest_sha256"] = "0" * 64
        r.write(out / a / "analysis.json", analysis)
    if corruption == "predictions":
        with (out / "candidate/predictions.jsonl").open("a") as f:
            f.write("\n{}")
    elif corruption == "weight_bytes":
        Path(weights["candidate"]).write_text("different checkpoint")
    elif corruption == "ground_truth":
        gt["images"].pop()
        r.write(out / "ground_truth.json", gt)
    if corruption is None:
        models, errors = r.models_and_errors(receipt)
        assert set(models) == set(r.ARMS)
        assert all(e["fp"] == 0 for e in errors.values())
    else:
        with pytest.raises(ValueError):
            r.models_and_errors(receipt)


def test_v2_paths_config_and_scientific_functions_are_unchanged():
    assert r.IMPLEMENTATION == "positive_quality_transport_trainerfix_v2"
    assert r.OUT == r.ROOT / "outputs/positive_quality_transport_v2_three30_s0"
    assert r.DOC == r.ROOT / "docs/paper/candidates/positive_quality_transport_v2"
    assert r.VALIDATION == r.ROOT / "outputs/positive_quality_transport_v2_engineering/validation.json"
    for arm, config in r.configurations().items():
        assert r.RUNS[arm] == r.ROOT / f"runs/positive_quality_transport_v2_{arm}30_s0"
        assert r.CONFIGS[arm] == r.ROOT / f"configs/candidates/positive_quality_transport_v2/{arm}30.yaml"
        legacy = r.load_experiment_config(r.ROOT / f"configs/candidates/positive_quality_transport/{arm}30.yaml")
        assert config == legacy | {"name": r.RUNS[arm].name}
        assert config["positive_quality_transport"] == {"mode": arm}
    old_tree = ast.parse((SCRIPTS / "run_positive_quality_transport.py").read_text(encoding="utf-8"))
    new_tree = ast.parse((SCRIPTS / "run_positive_quality_transport_v2.py").read_text(encoding="utf-8"))
    def declarations(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, ast.FunctionDef)}
    before, after = declarations(old_tree), declarations(new_tree)
    for name in before.keys() - {"engineering", "preflight", "worker", "train"}:
        assert before[name] == after[name], name
    for name in ("GATES", "CONTROL_SCOPE", "LEGACY_NOTICE", "INFERENCE"):
        def assignments(tree):
            return [ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert assignments(old_tree) == assignments(new_tree), name


@pytest.mark.parametrize("corruption", ["missing_flag", "false_flag", "missing_file", "missing_binding",
    "accepted", "v1_error", "modes", "val_dataset", "val_shape", "mode_binding", "detector",
    "real_batch", "forward", "optimizer", "iteration", "installed_loader", "synthetic_budget", "pipeline_path"])
def test_v2_requires_bound_actual_synthetic_pipeline(tmp_path, monkeypatch, corruption):
    receipt, _, _ = fixture_engineering(tmp_path, monkeypatch)
    pipeline = r.read(r.PIPELINE)
    if corruption == "missing_flag":
        receipt.pop("trainer_pipeline_checks")
    elif corruption == "false_flag":
        receipt["trainer_pipeline_checks"] = False
    elif corruption == "missing_file":
        r.PIPELINE.unlink()
    elif corruption == "missing_binding":
        receipt["artifact_sha256"].pop(str(r.PIPELINE))
    elif corruption == "synthetic_budget":
        receipt["new_synthetic_batches"] = 5
    elif corruption == "pipeline_path":
        receipt["pipeline_evidence"] = str(tmp_path / "another.json")
    else:
        if corruption == "accepted":
            pipeline["accepted"] = False
        elif corruption == "v1_error":
            pipeline["v1_val_failure_reproduced"] = "unrelated error"
        elif corruption == "modes":
            pipeline["modes"].pop("control")
        elif corruption == "val_dataset":
            pipeline["modes"]["candidate"]["val_dataset"] = "SourceDataset"
        elif corruption == "val_shape":
            pipeline["modes"]["control"]["synthetic_val_batch_shape"] = [1, 3, 64, 64]
        elif corruption == "mode_binding":
            pipeline["modes"]["reference"]["get_model_mode_bound"] = False
        elif corruption == "detector":
            pipeline["modes"]["control"]["real_detector_instantiated"] = True
        elif corruption == "installed_loader":
            pipeline["actual_installed_get_dataloader"] = False
        else:
            key = {"real_batch": "new_real_batches", "forward": "model_forward_calls",
                   "optimizer": "optimizer_steps", "iteration": "student_training_iterations"}[corruption]
            pipeline[key] = 1
        r.write(r.PIPELINE, pipeline)
        receipt["artifact_sha256"][str(r.PIPELINE)] = r.sha(r.PIPELINE)
    r.write(r.VALIDATION, receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        r.engineering()


def v1_failure_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "V1_OUT", tmp_path / "v1_failed")
    monkeypatch.setattr(r, "V1_RUNS", {a: tmp_path / "v1_runs" / a for a in r.ARMS})
    monkeypatch.setattr(r, "V1_AUDIT", tmp_path / "external_audit.json")
    source = tmp_path / "v1_source.py"
    source.write_text("original frozen V1 source")
    original = {"implementation": "positive_quality_transport_v1", "train_arms": list(r.ARMS),
        "epochs_per_arm": 30, "arms": {a: str(p) for a, p in r.V1_RUNS.items()},
        "gates": r.canonical(r.GATES), "inference": r.INFERENCE, "input_sha256": r.hashes([source])}
    r.write(r.V1_OUT / "preflight.json", original)
    r.write(r.V1_OUT / "dispatch.json", {"train_arms": list(r.ARMS), "parent_pid": 123,
        "preflight_sha256": r.sha(r.V1_OUT / "preflight.json")})
    r.write(r.V1_OUT / "reference_claim.json", {"arm": "reference", "parent_pid": 123,
        "preflight_sha256": r.sha(r.V1_OUT / "preflight.json")})
    r.write(r.V1_OUT / "reference_exit.json", {"arm": "reference", "exit_code": 1, "worker_wall_seconds": 30.})
    r.write(r.V1_OUT / "status.json", {"stage": "failed", "automatic_retry": False})
    (r.V1_OUT / "reference_stdout.log").write_text("TypeError: super(type, obj)")
    r.V1_RUNS["reference"].mkdir(parents=True)
    (r.V1_RUNS["reference"] / "args.yaml").write_text("synthetic: true")
    audit = {"accepted": False, "terminal_evidence_verified": True, "stage": "failed",
        "performance": "NOT_ACCEPTED", "automatic_retry": False}
    log = tmp_path / "external_audit.log"
    log.write_text(json.dumps(audit), encoding="utf-8-sig")
    external = {"stage": "failed", "audit": audit, "audit_process_exit_code": 0,
        "audit_log": str(log), "audit_log_sha256": r.sha(log),
        "reference_exit": r.read(r.V1_OUT / "reference_exit.json"), "completed_epochs": 0,
        "candidate_started": False, "control_started": False, "source_modified": False, "retry_started": False}
    def seal():
        paths = [p for p in r.V1_OUT.rglob("*") if p.is_file() and p.name != "artifact_sha256.json"]
        paths += [p for run in r.V1_RUNS.values() for p in run.rglob("*") if p.is_file()]
        r.write(r.V1_OUT / "artifact_sha256.json", r.hashes(paths))
        external["frozen_file_sha256"] = r.hashes(r.V1_OUT / n for n in (
            "artifact_sha256.json", "preflight.json", "status.json", "reference_exit.json",
            "reference_claim.json", "reference_stdout.log"))
        r.write(r.V1_AUDIT, external)
        monkeypatch.setattr(r, "V1_AUDIT_SHA", r.sha(r.V1_AUDIT))
    seal()
    return external, seal, source


@pytest.mark.parametrize("corruption", [None, "external_bytes", "missing_external", "source_bytes",
    "log_bytes", "artifact_bytes", "missing_binding", "completed", "exit0", "candidate_exit",
    "control_exit", "candidate_claim", "control_run", "results", "accepted", "epochs", "claim", "preflight"])
def test_v1_failure_is_bound_readonly_and_never_relabelled_success(tmp_path, monkeypatch, corruption):
    external, seal, source = v1_failure_fixture(tmp_path, monkeypatch)
    if corruption == "external_bytes":
        r.V1_AUDIT.write_text("{}")
    elif corruption == "missing_external":
        r.V1_AUDIT.unlink()
    elif corruption == "source_bytes":
        source.write_text("changed V1")
    elif corruption == "log_bytes":
        Path(external["audit_log"]).write_text("{}")
    elif corruption == "artifact_bytes":
        (r.V1_OUT / "reference_stdout.log").write_text("changed")
    elif corruption == "missing_binding":
        external["frozen_file_sha256"].pop(str(r.V1_OUT / "status.json"))
        r.write(r.V1_AUDIT, external)
        monkeypatch.setattr(r, "V1_AUDIT_SHA", r.sha(r.V1_AUDIT))
    elif corruption:
        if corruption == "completed":
            r.write(r.V1_OUT / "status.json", {"stage": "completed", "automatic_retry": False})
        elif corruption == "exit0":
            external["reference_exit"]["exit_code"] = 0
            r.write(r.V1_OUT / "reference_exit.json", external["reference_exit"])
        elif corruption in ("candidate_exit", "control_exit", "candidate_claim"):
            r.write(r.V1_OUT / f"{corruption}.json", {"exit_code": 1})
        elif corruption == "control_run":
            r.V1_RUNS["control"].mkdir()
        elif corruption == "results":
            (r.V1_RUNS["reference"] / "results.csv").write_text("epoch\n1\n")
        elif corruption == "accepted":
            external["audit"]["accepted"] = True
        elif corruption == "epochs":
            external["completed_epochs"] = 1
        elif corruption == "claim":
            r.write(r.V1_OUT / "reference_claim.json", {"arm": "reference", "preflight_sha256": "0" * 64})
        elif corruption == "preflight":
            p = r.V1_OUT / "preflight.json"
            original = r.read(p)
            original["implementation"] = r.IMPLEMENTATION
            r.write(p, original)
        seal()  # Semantic failures must fail even with internally consistent synthetic hashes.
    if corruption is None:
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
        mapping = r.v1_failure_evidence()
        assert set(mapping) >= {str(source), str(r.V1_AUDIT), str(r.V1_OUT / "artifact_sha256.json")}
        assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            r.v1_failure_evidence()


def test_authorized_worker_imports_v2_only(tmp_path, monkeypatch):
    dispatch_fixture(tmp_path, monkeypatch)
    new = ModuleType("regionssod.training.positive_quality_transport_v2")
    old = ModuleType("regionssod.training.positive_quality_transport")
    old.run_training = lambda *_: pytest.fail("V1 worker invoked")
    calls = []
    def fake_training(config):
        calls.append(config)
        run = r.RUNS["reference"]
        (run / "weights").mkdir(parents=True)
        for name in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "experiment.json"):
            (run / name).write_text("synthetic worker output")
        return run
    new.run_training = fake_training
    monkeypatch.setitem(sys.modules, new.__name__, new)
    monkeypatch.setitem(sys.modules, old.__name__, old)
    monkeypatch.setattr(r, "check_epochs", lambda *_: None)
    r.worker("reference")
    assert calls == [r.CONFIGS["reference"]]
    assert (r.OUT / "reference_claim.json").exists() and (r.OUT / "reference_seal.json").exists()


def test_runner_import_has_no_training_or_filesystem_side_effects(monkeypatch):
    import ultralytics
    import torch
    def forbidden(*args, **kwargs):
        pytest.fail("runner import attempted a side effect")
    monkeypatch.setattr(ultralytics, "YOLO", forbidden)
    monkeypatch.setattr(torch.utils.data, "DataLoader", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(r.subprocess, "run", forbidden)
    spec = importlib.util.spec_from_file_location("v2_import_only", SCRIPTS / "run_positive_quality_transport_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.IMPLEMENTATION == r.IMPLEMENTATION
