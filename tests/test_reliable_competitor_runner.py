"""Synthetic files/metrics only: no real batch, YOLO forward, val or training."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("reliable_runner", SCRIPTS / "run_reliable_competitor_negative.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def models_and_errors():
    models, errors = {}, {}
    for arm in r.ARMS:
        candidate = arm == "candidate"
        fp, wrong = (90, 45) if candidate else (100, 50)
        models[arm] = {
            "ultralytics": {"metrics/mAP50(B)": .4, "metrics/mAP50-95(B)": .202 if candidate else .2},
            "coco": {"metrics": {"AP_small": .15}, "per_class_AP_small": {str(k): .1 for k in range(10)}},
            "precision_constraints": {iou: {p: {"small_recall": .3} for p in ("0.7", "0.9")} for iou in ("0.5", "0.75")},
            "conf025": {"0.5": {"small_recall": .35, "fp": fp}},
        }
        counts = {name: 0 for name in r.ERROR_NAMES}
        counts.update(matched=10, wrong_class=wrong, unexplained_background=fp-wrong)
        errors[arm] = {"fp": fp, "counts": counts}
    return models, errors


def test_ten_gates_and_all_signs_retained():
    models, errors = models_and_errors()
    comparison, decision = r.comparison_and_decision(models, errors)
    assert len(decision["gates"]) == 10
    assert sum(k.startswith("reference_") for k in decision["gates"]) == 8
    assert decision["all_ten_passed"] is True
    assert decision["automatic_further_training"] is False
    assert len(comparison["per_class_AP_small_fraction"]["control"]) == 10
    models["candidate"]["coco"]["per_class_AP_small"]["0"] = .05
    comparison, _ = r.comparison_and_decision(models, errors)
    assert comparison["per_class_candidate_minus_other_pp"]["reference"]["0"] == -5


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
        candidate["precision_constraints"][iou][p]["small_recall"] -= .01
    elif gate == "reference_FP_conf025_excess":
        candidate["conf025"]["0.5"]["fp"] = 98
        errors["candidate"]["fp"] = 98
        errors["candidate"]["counts"]["unexplained_background"] += 8
    elif gate == "reference_wrong_class_conf025_excess":
        errors["candidate"]["counts"]["wrong_class"] = 48
        errors["candidate"]["counts"]["unexplained_background"] -= 3
    elif gate == "control_AP50_95_pp":
        models["control"]["ultralytics"]["metrics/mAP50-95(B)"] = .202
    elif gate == "control_AP_small_pp":
        models["control"]["coco"]["metrics"]["AP_small"] = .16
    _, result = r.comparison_and_decision(models, errors)
    assert [k for k, v in result["gates"].items() if not v["passed"]] == [gate]
    assert result["outcome"] == "stop"


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
    monkeypatch.setattr(r, "VALIDATION", tmp_path / "validation_final.json")
    paths = [r.BASE, r.DOC / "engineering_protocol.md", tmp_path / "scripts/validate_reliable_competitor_negative.py",
             tmp_path / "src/regionssod/training/reliable_competitor_negative.py"]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic")
    artifact = tmp_path / "cache.pt"
    artifact.write_bytes(b"synthetic cache - never loaded")
    monkeypatch.setattr(r, "version", lambda name: {"torch": "synthetic", "ultralytics": "8.4.172"}[name])
    receipt = {"accepted": True, "optimizer_steps": 0, "parameter_hash_unchanged": True,
        "implementation": r.IMPLEMENTATION, "torch_version": "synthetic", "ultralytics_version": "8.4.172",
        "real_batch": {"images": 8, "amp": True, "device": "cuda:0"},
        "source_sha256": r.hashes(paths), "artifact_sha256": r.hashes([artifact])}
    r.write(r.VALIDATION, receipt)
    return receipt, paths, artifact


def test_engineering_receipt_complete_and_fresh(tmp_path, monkeypatch):
    receipt, paths, artifact = fixture_engineering(tmp_path, monkeypatch)
    assert str(artifact) in r.engineering()
    paths[-1].write_text("changed")
    with pytest.raises(ValueError):
        r.engineering()


@pytest.mark.parametrize("field,value", [("accepted", False), ("optimizer_steps", 1),
    ("parameter_hash_unchanged", False), ("source_sha256", {}), ("artifact_sha256", {}),
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
    monkeypatch.setenv("REGIONSSOD_RCN_WORKER_TOKEN", token)
    return receipt


@pytest.mark.parametrize("corruption", ["missing_token", "parent", "arm", "source", "config", "dispatch", "claim", "existing"])
def test_worker_cannot_bypass_dispatch(tmp_path, monkeypatch, corruption):
    dispatch_fixture(tmp_path, monkeypatch)
    if corruption == "missing_token":
        monkeypatch.delenv("REGIONSSOD_RCN_WORKER_TOKEN")
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
    import regionssod.training.reliable_competitor_negative as implementation
    monkeypatch.setattr(implementation, "run_training", lambda *_: pytest.fail("training entered"))
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
        assert c["reliable_competitor_negative"] == {"mode": a}
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
        assert "REGIONSSOD_RCN_WORKER_TOKEN" in kwargs["env"]
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
