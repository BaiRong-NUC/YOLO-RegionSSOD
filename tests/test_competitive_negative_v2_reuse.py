import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from test_competitive_negative_main_comparison import metrics


@pytest.fixture
def runner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("repair_runner_test", scripts / "run_p1_main_comparison_repair.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts))


@pytest.fixture
def prepared(runner, monkeypatch, tmp_path):
    r = runner
    monkeypatch.setattr(r, "ROOT", tmp_path)
    for attr, path in {"OLD": "outputs/old", "OUT": "outputs/new", "FAILURE": "outputs/failure.json",
                       "CONFIG": "configs/v2.yaml", "PROTOCOL": "protocol.md", "VALIDATION": "validation.json"}.items():
        monkeypatch.setattr(r, attr, tmp_path / path)
    monkeypatch.setattr(r, "RUNS", {a: tmp_path / f"runs/{a}" for a in ("baseline", "reference", "candidate")})
    r.OLD.mkdir(parents=True)
    source = tmp_path / "source.py"
    source.write_text("frozen source")
    source_hashes = {str(source): r.sha(source)}
    snapshots, weights = {}, {}
    configs = {}
    for arm in ("baseline", "reference"):
        run = r.RUNS[arm]
        (run / "weights").mkdir(parents=True)
        config = {"name": arm, "data": str(tmp_path / f"{arm}.yaml"), "train": {"epochs": 30, "seed": 0}}
        configs[arm] = config
        r.write(r.OLD / f"{arm}_exit.json", {"exit_code": 0})
        (run / "args.yaml").write_text(yaml.safe_dump(config["train"] | {"name": arm, "save_dir": str(run),
            "data": config["data"], "resume": False}))
        r.write(run / "experiment.json", {"input_sha256": source_hashes})
        with (run / "results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "time"])
            writer.writeheader()
            writer.writerows({"epoch": i, "time": i * 2} for i in range(1, 31))
        weights[arm] = {}
        for name in ("best", "last"):
            p = run / f"weights/{name}.pt"
            p.write_bytes(f"{arm}/{name}".encode())
            weights[arm][str(p)] = {"sha256": r.sha(p)}
        for p in [run / n for n in ("args.yaml", "experiment.json", "results.csv")] + [r.OLD / f"{arm}_exit.json"]:
            snapshots[str(p)] = {"sha256": r.sha(p)}
    monkeypatch.setattr(r, "BEST", {a: r.sha(r.RUNS[a] / "weights/best.pt") for a in ("baseline", "reference")})
    r.write(r.OLD / "preflight.json", {"input_sha256": source_hashes})
    monkeypatch.setattr(r, "OLD_PREFLIGHT_SHA", r.sha(r.OLD / "preflight.json"))
    r.write(r.OLD / "dispatch.json", {"preflight_sha256": r.OLD_PREFLIGHT_SHA})
    r.write(r.OLD / "status.json", {"stage": "failed"})
    r.write(r.FAILURE, {"snapshots": snapshots, "arms": {a: {"weights": weights[a]} for a in weights}})
    for path, config in ((tmp_path / "configs/candidates/competitive_negative/baseline30.yaml", configs["baseline"]),
                         (tmp_path / "configs/train_p1_10.yaml", configs["reference"])):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config))
    candidate = configs["reference"] | {"name": "semantic"}
    (tmp_path / "configs/candidates/competitive_negative/semantic30.yaml").write_text(yaml.safe_dump(candidate))
    candidate["name"] = r.RUNS["candidate"].name
    r.CONFIG.write_text(yaml.safe_dump(candidate))
    r.PROTOCOL.write_text("protocol")
    helper = tmp_path / "scripts/validate_competitive_negative_v2.py"
    helper.parent.mkdir()
    helper.write_text("validator")
    r.write(r.VALIDATION, {"accepted": True, "optimizer_steps": 0, "backward_calls": 0,
        "torch_version": torch.__version__, "source_sha256": source_hashes, "artifact_sha256": {}})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(r.subprocess, "check_output", lambda *a, **kw: "fixture-source-commit\n")
    return r


def test_preflight_reuses_completed_arms_and_does_not_create_candidate(prepared):
    r = prepared
    receipt = r.preflight()
    assert receipt["train_arms"] == ["candidate"]
    assert receipt["reused_best_sha256"] == r.BEST
    assert not r.OUT.exists() and not r.RUNS["candidate"].exists()


@pytest.mark.parametrize("tamper", ["weight", "args", "epochs", "source"])
def test_reuse_refuses_changed_evidence(prepared, tamper):
    r = prepared
    path = {"weight": r.RUNS["baseline"] / "weights/best.pt", "args": r.RUNS["reference"] / "args.yaml",
            "epochs": r.RUNS["reference"] / "results.csv", "source": r.ROOT / "source.py"}[tamper]
    path.write_text("tampered")
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        r.reuse_bindings()


def test_existing_new_run_cannot_be_overwritten(prepared):
    prepared.RUNS["candidate"].mkdir()
    with pytest.raises(FileExistsError):
        prepared.preflight()


def test_training_dispatches_only_candidate_and_preserves_reused_inputs(prepared, monkeypatch):
    r = prepared
    receipt = r.preflight()
    old_hashes = {str(p): r.sha(p) for run in list(r.RUNS.values())[:2] for p in run.rglob("*") if p.is_file()}
    calls = []
    monkeypatch.setattr(r, "preflight", lambda: receipt)
    monkeypatch.setattr(r.subprocess, "run", lambda command, **kw: calls.append(command)
        or SimpleNamespace(returncode=0, check_returncode=lambda: None))
    monkeypatch.setattr(r, "finish_comparison", lambda: calls.append("evaluate_three"))
    r.train()
    assert len(calls) == 2 and calls[0][-1] == "--worker" and calls[1] == "evaluate_three"
    assert r.read(r.OUT / "dispatch.json")["train_arms"] == ["candidate"]
    assert old_hashes == {p: r.sha(p) for p in old_hashes}


def test_failed_candidate_keeps_error_and_never_evaluates(prepared, monkeypatch):
    r = prepared
    receipt = r.preflight()
    def fail():
        raise subprocess.CalledProcessError(1, ["candidate"])
    monkeypatch.setattr(r, "preflight", lambda: receipt)
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, check_returncode=fail))
    monkeypatch.setattr(r, "finish_comparison", lambda: pytest.fail("Incomplete candidate evaluated"))
    with pytest.raises(subprocess.CalledProcessError):
        r.train()
    assert r.read(r.OUT / "status.json")["stage"] == "failed"
    assert r.read(r.OUT / "candidate_exit.json")["exit_code"] == 1


def test_main_table_discloses_reuse_and_all_three_direct_deltas(runner):
    models = {"baseline": metrics(.23, .30), "reference": metrics(.24, .32), "candidate": metrics(.242, .324)}
    for a, value in (("baseline", .2), ("reference", .3), ("candidate", .25)):
        models[a]["coco"]["per_class_AP_small"] = {"negative_case": value}
    comparison, decision = runner.comparison_and_decision(models)
    assert comparison["delta_pp"]["improved_minus_B2"]["AP50_95"] == pytest.approx(1.2)
    assert comparison["delta_pp"]["improved_minus_P1"]["AP50_95"] == pytest.approx(.2)
    assert comparison["delta_pp"]["P1_minus_B2"]["AP50_95"] == pytest.approx(1.)
    assert decision["main_comparison"] == comparison
    assert decision["mechanism_control"]["status"] == "NOT_TESTED"
    text = runner.table_text(comparison, {a: 6000. for a in runner.RUNS})
    assert "B2（复用）" in text and "P1（复用）" in text and "+1.2000" in text
    assert "negative_case" in text and "-5.0000" in text


def test_evaluation_rejects_swapped_reused_weight_paths(runner, monkeypatch):
    weights = {a: str(p / "weights/best.pt") for a, p in runner.RUNS.items()}
    weights["baseline"], weights["reference"] = weights["reference"], weights["baseline"]
    monkeypatch.setattr(runner, "read", lambda p: {"weights": weights})
    with pytest.raises(ValueError, match="intended frozen best"):
        runner.models_and_binding()


def test_check_only_cannot_train_or_infer(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "preflight", lambda: {"train_arms": ["candidate"]})
    monkeypatch.setattr(runner, "train", lambda: pytest.fail("check-only trained"))
    monkeypatch.setattr(runner, "finish_comparison", lambda: pytest.fail("check-only inferred"))
    monkeypatch.setattr(sys, "argv", ["repair", "--check-only"])
    runner.main()
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_cli_cannot_dispatch_baseline_or_reference_worker(runner, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["repair", "--worker", "baseline"])
    with pytest.raises(SystemExit):
        runner.main()


def test_terminal_reporting_audit_and_tamper_detection_without_inference(prepared, monkeypatch):
    """Exercise report/audit plumbing using synthetic evaluator output, no models."""
    import numpy as np
    import evaluate_class_ambiguity_pair as evaluator
    from regionssod.evaluation.precision_recall import threshold_curve
    r = prepared
    r.OUT.mkdir()
    # Complete synthetic candidate files; preserve the two reused arms byte-for-byte.
    candidate = r.RUNS["candidate"]
    (candidate / "weights").mkdir(parents=True)
    for name in ("best", "last"):
        (candidate / f"weights/{name}.pt").write_bytes(name.encode())
    for name in ("results.csv", "args.yaml", "experiment.json"):
        (candidate / name).write_text("synthetic")
    (candidate / "classification_diagnostics.jsonl").write_text("synthetic")
    r.write(r.OUT / "candidate_exit.json", {"exit_code": 0})
    training = {"accepted": True, "student_seconds": {a: 60. for a in r.RUNS}}
    monkeypatch.setattr(r, "audit_training", lambda: training)
    monkeypatch.setattr(r, "verify_dispatch", lambda: {})
    def fake_evaluate(weights, out):
        r.write(out / "manifest.json", {"weights": {a: str(p) for a, p in weights.items()}})
        curve = threshold_curve([.9, .5, .3], [100, 2000, -1], {"small": 3, "medium": 2, "large": 1})
        for index, (a, weight) in enumerate(weights.items()):
            data = metrics(.2 + index*.01, .3 + index*.01)
            data["weights_sha256"] = r.sha(weight)
            data["coco"]["per_class_AP_small"] = {str(c): .1 + c*.01 + index*.001 for c in range(10)}
            r.write(out / a / "analysis.json", data)
            np.savez_compressed(out / a / "micro_iou0.5.npz", **curve)
        r.write(out / "artifact_sha256.json", {})
    monkeypatch.setattr(evaluator, "evaluate", fake_evaluate)
    monkeypatch.setattr(evaluator, "audit", lambda p: {"accepted": True, "images": 548})
    r.finish_comparison()
    assert r.audit()["accepted"]
    assert r.read(r.OUT / "status.json")["stage"] == "completed"
    text = r.OUT / "main_comparison.md"
    text.write_text(text.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen source differs"):
        r.audit()
