"""Pure synthetic files/processes: no actual data, weights, loader or inference."""
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


MODULE = Path(__file__).resolve().parents[1] / "scripts/run_bounded_retention_continuous.py"
spec = importlib.util.spec_from_file_location("continuous_runner_review", MODULE)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def stopped_geometry():
    row = {"partial_objects_checked": 1, "supported_objects": 1, "empty_anchor_events": 0}
    return {
        "status": "STOP_BEFORE_STUDENT_TRAINING", "error": {"message": "nonfinite original P1 loss/gradient"},
        "objective": "V6_native_E_same_continuous_density", "stage": "real_model_amp_forward_backward",
        "optimizer_steps": 0, "training_started": False, "validation_inference_calls": 0,
        "geometry_phase": "mosaic_closed", "geometry_batch_index": 1413,
        "command": ["checker", "--scan-all-batches"],
        "geometry_batch_checks": [{**row, "phase": phase} for phase in ("mosaic_open", "mosaic_closed")
                                  for _ in range(1414)],
        "dataloaders": {"train_images": 11310, "val_images": 548, "workers": 0,
                        "image_size": 640, "validation_inference": False},
        "model_forward_calls": 1, "model_forward_attempts": 1, "amp_initial_gradient_scale": 65536.,
        "parameters_before_forward_sha256": "a" * 64,
        "forward_batch_selection": {"rule": "first_maximum_active_dimension", "phase": "mosaic_closed",
                                     "batch_index": 1316, "active_dimension": 3},
        "source_factory_counts": {"human": 105058},
        "actual_head_numeric_audits": [{"identity": [1, i], "dimension": 1 if i < 6 else 3,
            "diagnostics": {"qualified": True, "numerator": {"qualified": True},
                            "denominator": {"qualified": True}}} for i in range(7)],
        "input_sha256": {str(MODULE): "b" * 64},
    }


def stats():
    return {"status": "observed", "finite": True, "numel": 3,
            "nonfinite_count": 0, "nan_count": 0, "inf_count": 0}


def parameter_stats():
    return {"by_name": {"weight": {"scaled": stats(), "unscaled_native_dtype": stats()}},
        "trainable_parameter_tensors": 1, "observed_parameter_tensors": 1, "unused_parameter_names": [],
        "complete_parameter_coverage": True, "observed_scaled_nonfinite_count": 0,
        "observed_unscaled_nonfinite_count": 0, "observed_scaled_finite": True, "observed_unscaled_finite": True}


def passed_engineering(geometry=None):
    g = stopped_geometry() if geometry is None else geometry
    losses = {"reference": [1., 2., 5.], "candidate": [1., 2., 3.], "control": [1., 2., 4.]}
    return {**deepcopy(g), "status": "AMP_INTEGRATION_COMPONENT_PASSED_NOT_STUDENT_PERMIT",
        "audit_protocol": "native_dynamic_scaling_v1", "parent_geometry_receipt": str(r.GEOMETRY_ENGINEERING),
        "parent_geometry_sha256": r.GEOMETRY_ENGINEERING_SHA,
        "replayed_batches": {"mosaic_open": 1414, "mosaic_closed": 1414},
        "parameters_after_backward_sha256": "a"*64, "parameter_hash_unchanged": True,
        "real_shared_ciou_classification": True, "real_control_gradient_difference_l1": .25,
        "parent_actual_head_numeric_audits_equal": True, "legacy_reference_value_equal": True,
        "legacy_reference_gradient_equal": True, "unscaled_losses": losses,
        "diagnostic_previews": {arm: {"path": str(r.preview_path(arm)), "sha256": "c"*64} for arm in r.ARMS},
        "amp_backward_audit": {
            "status": "PASSED_FIXED_GRAPH_BACKWARD_ONLY", "arms": list(r.ARMS),
            "scale_sequence": [2.**e for e in range(16, -17, -1)], "initial_scale": 65536.,
            "selected_scale": 65536., "forward_calls_by_helper": 0, "optimizer_steps": 0,
            "optimizer_created": False, "training_scaler_unchanged": True,
            "unscaled_losses": {arm: {"values": loss, "components": stats(), "sum": stats(),
                                       "sum_value": sum(loss)} for arm, loss in losses.items()},
            "unscaled_head_values": stats(),
            "attempts": [{"scale": 65536., "common_finite": True, "arms": {
                arm: {"head_scaled": stats(), "head_unscaled_fp64": stats(), "parameters": parameter_stats(),
                      "finite_observed_backward": True} for arm in r.ARMS}}],
            "candidate_populated_backward": parameter_stats(),
            "parameters_before_sha256": "d"*64, "parameters_after_sha256": "d"*64,
            "buffers_before_sha256": "e"*64, "buffers_after_sha256": "e"*64}}


def test_complete_engineering_fields_pass_without_opening_a_model():
    geometry = stopped_geometry()
    original = deepcopy(geometry)
    r.verify_engineering(passed_engineering(), geometry)
    assert geometry == original and geometry["status"] == "STOP_BEFORE_STUDENT_TRAINING"


@pytest.mark.parametrize("drift", ["order", "position", "flag_only"])
def test_geometry_requires_actual_complete_stream_not_success_only_summary(drift):
    evidence = stopped_geometry()
    assert "all_stream_geometry_checked" not in evidence
    if drift == "order":
        evidence["geometry_batch_checks"].reverse()
    elif drift == "position":
        evidence["geometry_batch_index"] = 1412
    else:
        evidence["all_stream_geometry_checked"] = True
        evidence["geometry_batch_checks"] = []
    with pytest.raises(ValueError, match="1414|ordering/final position"):
        r.verify_geometry_engineering(evidence)


@pytest.mark.parametrize("key,value", [
    ("status", "RUNNING"), ("optimizer_steps", 1), ("training_started", True),
    ("validation_inference_calls", 1), ("replayed_batches", {}),
    ("model_forward_calls", 0), ("model_forward_attempts", 2),
    ("parent_geometry_sha256", "f"*64), ("parameter_hash_unchanged", False),
    ("parameters_after_backward_sha256", "c" * 64),
    ("real_control_gradient_difference_l1", 0), ("real_control_gradient_difference_l1", float("nan")),
    ("real_shared_ciou_classification", False), ("unscaled_losses", {}),
    ("actual_head_numeric_audits", []), ("diagnostic_previews", {}),
    ("legacy_reference_value_equal", False), ("legacy_reference_gradient_equal", False),
    ("source_factory_counts", {}), ("forward_batch_selection", {}),
])
def test_engineering_cannot_pass_with_missing_or_failed_gate(key, value):
    evidence = passed_engineering()
    evidence[key] = value
    with pytest.raises(ValueError):
        r.verify_engineering(evidence, stopped_geometry())


def test_partial_second_phase_and_unqualified_numeric_receipt_fail():
    evidence = stopped_geometry()
    evidence["geometry_batch_checks"].pop()
    with pytest.raises(ValueError, match="1414"):
        r.verify_geometry_engineering(evidence)
    evidence = stopped_geometry()
    evidence["actual_head_numeric_audits"][0]["diagnostics"]["denominator"]["qualified"] = False
    with pytest.raises(ValueError, match="qualified"):
        r.verify_geometry_engineering(evidence)
    evidence = stopped_geometry()
    evidence["geometry_batch_checks"][0]["supported_objects"] = 0
    with pytest.raises(ValueError, match="geometry"):
        r.verify_geometry_engineering(evidence)


def test_common_scale_backoff_is_allowed_but_skipping_scales_or_first_success_is_not():
    evidence = passed_engineering()
    audit = evidence["amp_backward_audit"]
    first = deepcopy(audit["attempts"][0])
    first["common_finite"] = False
    ref = first["arms"]["reference"]
    ref["finite_observed_backward"] = False
    ref["head_scaled"].update(finite=False, nonfinite_count=1, inf_count=1)
    audit["attempts"][0]["scale"] = audit["selected_scale"] = 32768.
    audit["attempts"].insert(0, first)
    r.verify_engineering(evidence, stopped_geometry())
    audit["attempts"][-1]["scale"] = audit["selected_scale"] = 16384.
    with pytest.raises(ValueError, match="skipped/reordered"):
        r.verify_engineering(evidence, stopped_geometry())
    audit["attempts"][-1]["scale"] = audit["selected_scale"] = 32768.
    # Pretend the first scale was already successful; later selection is invalid.
    audit["attempts"][0] = deepcopy(audit["attempts"][-1])
    audit["attempts"][0]["scale"] = 65536.
    with pytest.raises(ValueError, match="first common"):
        r.verify_engineering(evidence, stopped_geometry())


@pytest.mark.parametrize("keys,value", [
    (("status",), "STOP"),
    (("optimizer_steps",), 1),
    (("forward_calls_by_helper",), 1),
    (("training_scaler_unchanged",), False),
    (("buffers_after_sha256",), "f"*64),
    (("unscaled_losses", "reference", "sum_value"), float("nan")),
    (("attempts", 0, "arms", "reference", "head_unscaled_fp64", "finite"), False),
    (("attempts", 0, "arms", "control", "parameters", "by_name"), {}),
    (("attempts", 0, "arms", "candidate", "parameters", "by_name", "weight", "scaled", "finite"), False),
    (("attempts", 0, "arms", "candidate", "parameters", "unused_parameter_names"), ["weight"]),
    (("candidate_populated_backward", "by_name", "weight", "unscaled_native_dtype", "nonfinite_count"), 1),
])
def test_amp_summary_pass_cannot_override_actual_gradient_or_state_evidence(keys, value):
    evidence = passed_engineering()
    target = evidence["amp_backward_audit"]
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    evidence["accepted"] = True
    with pytest.raises(ValueError):
        r.verify_engineering(evidence, stopped_geometry())


def test_explicit_unused_parameter_is_not_false_full_gradient_coverage():
    evidence = passed_engineering()
    audit = evidence["amp_backward_audit"]
    records = [a["parameters"] for a in audit["attempts"][0]["arms"].values()]
    records.append(audit["candidate_populated_backward"])
    for record in records:
        record["by_name"]["unused"] = {"scaled": {"status": "unused", "finite": None},
                                       "unscaled_native_dtype": {"status": "unused", "finite": None}}
        record.update(trainable_parameter_tensors=2, unused_parameter_names=["unused"], complete_parameter_coverage=False)
    r.verify_engineering(evidence, stopped_geometry())
    records[0]["complete_parameter_coverage"] = True
    with pytest.raises(ValueError, match="coverage"):
        r.verify_engineering(evidence, stopped_geometry())


def test_joint_gate_accepts_actual_helper_schema_on_pure_tensor_graph():
    """No model/Trainer/forward: a CPU tensor graph and parameter iterator stub."""
    import torch
    source = MODULE.parents[1] / "src/regionssod/training/bounded_retention_amp_audit.py"
    helper_spec = importlib.util.spec_from_file_location("amp_helper_runner_schema_test", source)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    weight = torch.tensor([1., 2., 3.], requires_grad=True)
    boxes = weight * 2
    model = SimpleNamespace(named_parameters=lambda: iter([("weight", weight)]),
                            parameters=lambda: iter([weight]), named_buffers=lambda: iter([]))
    losses = {arm: torch.stack((boxes[0], boxes[1], boxes[2] * (i+1)))
              for i, arm in enumerate(r.ARMS)}
    report = {}
    helper.audit_backward(losses, boxes, model, report)
    r.verify_amp_backward(report["amp_backward_audit"], {arm: v.detach().tolist() for arm, v in losses.items()})


def test_parent_head_comparison_uses_payload_not_just_equal_flag():
    evidence = passed_engineering()
    evidence["actual_head_numeric_audits"][0]["identity"] = [99, 99]
    with pytest.raises(ValueError, match="head receipts"):
        r.verify_engineering(evidence, stopped_geometry())
    geometry = stopped_geometry()
    geometry["status"] = "INTEGRATION_COMPONENT_PASSED_NOT_STUDENT_PERMIT"
    with pytest.raises(ValueError, match="STOP"):
        r.verify_engineering(passed_engineering(), geometry)


def test_preview_rehash_cannot_conceal_wrong_arm_or_training_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ENGINEERING", tmp_path / "result.json")
    report = passed_engineering()
    for arm in r.ARMS:
        p = r.preview_path(arm)
        p.parent.mkdir(parents=True)
        r.write(p, {"schema": "bounded_retention_read_only_diagnostics_v1", "mode": arm,
            "rank": -1, "epoch_index": 0, "epoch_number": 1, "batch_count": 1,
            "diagnostic_only": True, "participates_in_loss_tuning_or_stopping": False})
        report["diagnostic_previews"][arm]["sha256"] = r.digest(p)
    r.verify_previews(report)
    p = r.preview_path("control")
    original = r.read(p)
    for change in ({"mode": "candidate"}, {"epoch_index": 1, "epoch_number": 2}):
        r.write(p, original | change)
        report["diagnostic_previews"]["control"]["sha256"] = r.digest(p)
        with pytest.raises(ValueError, match="zero-step"):
            r.verify_previews(report)


def test_full_suite_is_separate_from_engineering():
    report = {"status": "PASSED", "scope": "full_repository", "exit_code": 0,
              "passed": 1862, "failed": 0, "errors": 0, "input_sha256": {str(MODULE): "a" * 64}}
    r.verify_suite(report)
    for change in ({"scope": "targeted"}, {"passed": 1861}, {"exit_code": 1}, {"errors": 1}):
        with pytest.raises(ValueError):
            r.verify_suite(report | change)


def test_hash_conflicts_and_snapshot_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    source = tmp_path / "src/a.py"
    source.parent.mkdir()
    source.write_text("original", encoding="utf-8")
    ep = tmp_path / "outputs/integration/result.json"
    snapshot = ep.parent / "source_snapshot/src/a.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    mapping = {str(source): r.digest(source)}
    r.check_subset(mapping, mapping)
    r.verify_snapshots({"input_sha256": mapping}, ep)
    with pytest.raises(ValueError, match="conflicting"):
        r.check_subset(mapping, {str(source): "a" * 64})
    snapshot.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot"):
        r.verify_snapshots({"input_sha256": mapping}, ep)


@pytest.mark.parametrize("relative", ["data/train/labels/1.txt", "data/manifest.jsonl",
                                      "outputs/old_quality_cache.npy", "outputs/old_box_quality.json"])
def test_forbidden_inputs_are_not_accepted_even_for_hashing(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    assert not r.allowed_input(tmp_path / relative, legal=set(), pointers=set(), model=tmp_path / "initial.pt")


def test_manifest_pointer_rejected_before_read(monkeypatch):
    monkeypatch.setattr(r, "digest", lambda _: pytest.fail("forbidden file opened"))
    with pytest.raises(ValueError, match="forbidden"):
        r.pointer({"path": "manifest.json", "sha256": "a" * 64})


@pytest.fixture
def configs(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "OUT", tmp_path / "outputs/three30")
    monkeypatch.setattr(r, "BASE", tmp_path / "reference30.yaml")
    baseline = {"name": "old_reference", "project": "runs", "method": "p1", "model": "initial.pt",
        "data": "dataset.yaml", "p1": {"soft_weights": True, "uncertain_ignore": True},
        "train": {"epochs": 30, "batch": 8, "seed": 0, "workers": 0, "amp": True},
        "positive_quality_transport": {"mode": "reference"}}
    r.BASE.write_text(yaml.safe_dump(baseline), encoding="utf-8")
    def ptr(p):
        return {"path": str(p), "sha256": r.digest(p)}
    spec = {"baseline": ptr(r.BASE), "configs": {}}
    for arm in r.ARMS:
        value = {k: v for k, v in baseline.items() if k != "positive_quality_transport"}
        value.update(project=str(r.OUT), name=arm, bounded_retention={"mode": arm})
        p = tmp_path / f"{arm}.yaml"
        p.write_text(yaml.safe_dump(value), encoding="utf-8")
        spec["configs"][arm] = ptr(p)
    return spec


def test_config_only_declared_changes_and_original_trainer_dispatch(configs, monkeypatch):
    parsed = r.configurations(configs)
    calls = []
    class Plain:
        def __init__(self, **kwargs):
            calls.append(("P1", kwargs))
    class Candidate:
        def __init__(self, **kwargs):
            calls.append(("V6", kwargs))
    monkeypatch.setattr(r, "expected_args", lambda c: c["train"])
    source = object()
    for arm in r.ARMS:
        r.make_trainer(arm, parsed[arm], classes=(Plain, Candidate), source=source)
    assert [name for name, _ in calls] == ["P1", "V6", "V6"]
    assert "mode" not in calls[0][1] and "source_factory" not in calls[0][1]
    assert calls[1][1]["source_factory"] is source and calls[1][1]["mode"] == "candidate"
    assert calls[2][1]["mode"] == "control"
    assert calls[2][1]["source_audit_sha256"] == r.AUDIT_SHA
    p = Path(configs["configs"]["candidate"]["path"])
    changed = yaml.safe_load(p.read_text())
    changed["train"]["epochs"] = 1
    p.write_text(yaml.safe_dump(changed))
    configs["configs"]["candidate"]["sha256"] = r.digest(p)
    with pytest.raises(ValueError, match="config change"):
        r.configurations(configs)


def test_existing_output_rejected_before_receipt_or_population_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUT", tmp_path / "used")
    r.OUT.mkdir()
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": r.SCHEMA, "permit": "FINAL_THREE_ARM_30_FROZEN",
                                "failure_policy": r.FAILURE_POLICY}))
    monkeypatch.setattr(r, "configurations", lambda _: pytest.fail("should fail before further reads"))
    with pytest.raises(FileExistsError):
        r.preflight(spec)


@pytest.mark.parametrize("epochs,bad_epoch,nonfinite", [(29, False, False), (30, True, False), (30, False, True)])
def test_epoch_integrity_rejects_short_duplicate_or_nonfinite(tmp_path, epochs, bad_epoch, nonfinite):
    rows = [f"{1 if bad_epoch and i == 30 else i},{'nan' if nonfinite and i == 30 else i * .1}" for i in range(1, epochs + 1)]
    (tmp_path / "results.csv").write_text("epoch,time\n" + "\n".join(rows))
    with pytest.raises(ValueError):
        r.check_epochs(tmp_path)


def fake_evaluation_audit(_):
    return {"audit_ok": True, "numeric_replay": True, "images": 548, "arms": list(r.ARMS),
            "inference_calls": 0, "external_data_or_weights_read": False}


def test_exact_arm_order_and_evaluation_after_three_audits(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUT", tmp_path)
    r.write(tmp_path / "preflight.json", {})
    events = []
    class Child:
        pid = 123
        def wait(self):
            return 0
    def launch(command, **kwargs):
        events.append(("launch", command[-1]))
        assert r.WORKER_TOKEN in kwargs["env"]
        return Child()
    def audit_arm(arm, receipt):
        events.append(("audit", arm))
        return {"epochs": 30}
    def evaluate(weights, out):
        assert events[:6] == [(kind, arm) for arm in r.ARMS for kind in ("launch", "audit")]
        assert weights == {a: tmp_path / a / "weights/best.pt" for a in r.ARMS}
        events.append(("evaluate", None))
    monkeypatch.setattr(r, "audit_arm", audit_arm)
    monkeypatch.setattr(r, "verify_dispatch", lambda: None)
    r.execute_arms({}, popen=launch, evaluator=evaluate, evaluation_audit=fake_evaluation_audit)
    assert events.count(("evaluate", None)) == 1
    assert r.read(tmp_path / "status.json")["stage"] == "completed"


@pytest.mark.parametrize("poll_raises", [False, True])
def test_parent_interrupt_keeps_live_worker_and_never_seals_or_dispatches(tmp_path, monkeypatch, poll_raises):
    monkeypatch.setattr(r, "OUT", tmp_path / "run")
    monkeypatch.setattr(r, "preflight", lambda _: {})
    dispatched = []
    class LiveChild:
        pid = 4321
        def wait(self):
            raise KeyboardInterrupt("synthetic wrapper interruption")
        def poll(self):
            if poll_raises:
                raise OSError("liveness unavailable")
            return None
        def terminate(self):
            pytest.fail("must not terminate a user process")
        def kill(self):
            pytest.fail("must not kill a user process")
    def launch(command, **kwargs):
        dispatched.append(command[-1])
        return LiveChild()
    original = r.execute_arms
    monkeypatch.setattr(r, "execute_arms", lambda receipt: original(receipt, popen=launch))
    monkeypatch.setattr(r, "artifact_hashes", lambda *a, **kw: pytest.fail("live output must not be sealed"))
    with pytest.raises(r.WorkerStillLive):
        r.train(tmp_path / "unopened_spec.json")
    assert dispatched == ["reference"]
    state = r.read(r.OUT / "status.json")
    assert state["stage"] == "parent_interrupted_worker_live" and state["pid"] == 4321
    assert state["terminal"] is False and state["command"][-2:] == ["--worker", "reference"]
    assert not (r.OUT / "artifact_sha256.json").exists()
    assert not (r.OUT / "reference_exit.json").exists()
    with pytest.raises(ValueError, match="not terminal"):
        r.audit()


def test_natural_nonzero_exit_is_failed_sealed_and_never_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUT", tmp_path / "run")
    monkeypatch.setattr(r, "preflight", lambda _: {})
    dispatched = []
    class Child:
        pid = 987
        def wait(self):
            return 2
    def launch(command, **kwargs):
        dispatched.append(command[-1])
        return Child()
    original = r.execute_arms
    monkeypatch.setattr(r, "execute_arms", lambda receipt: original(receipt, popen=launch))
    with pytest.raises(ValueError, match="failed"):
        r.train(tmp_path / "unopened_spec.json")
    assert dispatched == ["reference"]
    assert r.read(r.OUT / "status.json")["stage"] == "failed"
    seal = r.OUT / "artifact_sha256.json"
    assert r.read(seal) == r.artifact_hashes(r.OUT, exclude=(seal,))


def test_no_authenticated_worker_without_parent_and_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUT", tmp_path)
    r.write(tmp_path / "status.json", {"stage": "training", "arm": "reference"})
    monkeypatch.delenv(r.WORKER_TOKEN, raising=False)
    monkeypatch.setattr(r, "verify_dispatch", lambda: ({}, {"parent_pid": -999, "token_sha256": {}}))
    monkeypatch.setattr(r, "make_trainer", lambda *a, **kw: pytest.fail("unauthenticated model construction"))
    with pytest.raises(ValueError, match="one-shot"):
        r.worker("reference")


def test_arm_seal_and_resealed_args_or_short_run_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUT", tmp_path)
    arm = "reference"
    run = tmp_path / arm
    (run / "weights").mkdir(parents=True)
    for name in ("best.pt", "last.pt"):
        (run / "weights" / name).write_bytes(b"synthetic bytes, never torch-loaded")
    (run / "results.csv").write_text("epoch,time\n" + "\n".join(f"{i},{i}.5" for i in range(1, 31)))
    args = {"epochs": 30, "resume": False, "name": arm}
    (run / "args.yaml").write_text(yaml.safe_dump(args))
    receipt = {"expected_args": {arm: args}, "configs": {arm: {"method": "p1"}},
               "input_sha256": {"synthetic_input": "a" * 64}}
    r.write(tmp_path / "preflight.json", receipt)
    claim = {"pid": 123, "parent_pid": 12, "arm": arm,
             "preflight_sha256": r.digest(tmp_path / "preflight.json")}
    r.write(tmp_path / f"{arm}_claim.json", claim)
    r.write(tmp_path / f"{arm}_process.json", {"pid": 123, "parent_pid": 12, "command": ["python", "--worker", arm]})
    r.write(tmp_path / f"{arm}_exit.json", {"pid": 123, "exit_code": 0})
    r.write(run / "experiment.json", {"arm": arm, "trainer": "P1Trainer", "config": receipt["configs"][arm],
        "input_sha256": receipt["input_sha256"], "preflight_sha256": claim["preflight_sha256"]})
    def seal():
        r.write(tmp_path / f"{arm}_seal.json", {"claim_sha256": r.digest(tmp_path / f"{arm}_claim.json"),
                "artifact_sha256": r.artifact_hashes(run)})
    seal()
    assert r.audit_arm(arm, receipt)["epochs"] == 30
    (run / "weights/best.pt").write_bytes(b"changed synthetic bytes")
    with pytest.raises(ValueError, match="seal"):
        r.audit_arm(arm, receipt)
    (run / "args.yaml").write_text(yaml.safe_dump(args | {"resume": True}))
    seal()
    with pytest.raises(ValueError, match="args"):
        r.audit_arm(arm, receipt)
    (run / "args.yaml").write_text(yaml.safe_dump(args))
    (run / "results.csv").write_text("epoch,time\n1,1.5\n")
    seal()
    with pytest.raises(ValueError, match="30 complete"):
        r.audit_arm(arm, receipt)


def test_complete_frozen_spec_preflight_and_snapshot_binding(configs, monkeypatch):
    """Exercise the complete validator using only temporary fake artifacts."""
    root = r.ROOT
    monkeypatch.setattr(r, "GEOMETRY_ENGINEERING", root / "outputs/native_e_integration_01/result.json")
    monkeypatch.setattr(r, "ENGINEERING", root / "outputs/native_amp_audit_01/result.json")
    monkeypatch.setattr(r, "AUDIT", root / "outputs/source/source_audit.json")
    def put(relative, text="synthetic"):
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p
    code = {put(p) for p in (
        "scripts/run_bounded_retention_continuous.py", "scripts/evaluate_bounded_retention.py",
        "scripts/check_bounded_retention_native_e.py", "scripts/check_bounded_retention_amp.py",
        "src/regionssod/training/bounded_retention_amp_audit.py", "src/scaler_probe.py",
        "src/regionssod/data/annotations.py",
        "src/regionssod/data/partial.py", "src/regionssod/training/p1.py",
        "src/regionssod/training/bounded_retention_domain_v6.py",
        "src/regionssod/training/bounded_retention_run_diagnostics.py",
        "tests/test_bounded_retention_continuous_runner.py", "src/native_trainer_probe.py")}
    monkeypatch.setattr(r, "installed_trainer_path", lambda: root / "src/native_trainer_probe.py")
    amp_contract = put("docs/amp_contract.md")
    amp_code = {amp_contract, root / "scripts/check_bounded_retention_amp.py",
        root / "src/regionssod/training/bounded_retention_amp_audit.py",
        root / "src/regionssod/training/bounded_retention_run_diagnostics.py", root / "src/scaler_probe.py"}
    monkeypatch.setattr(r, "required_amp_code", lambda: amp_code)
    reports = {put(p, "{}") for p in (
        "outputs/source/source_audit.json",
        "outputs/reliable_region_ranking_feasibility_20261105/results.json",
        "outputs/reliable_region_ranking_feasibility_20261105/receipt.json",
        "outputs/human_multiscale_observation_review_20261105/geometry.json",
        "outputs/human_multiscale_observation_review_20261105/geometry_receipt.json")}
    monkeypatch.setattr(r, "AUDIT_SHA", r.digest(r.AUDIT))
    model, dataset = put("initial.pt"), put("dataset.yaml", "{}")
    monkeypatch.setattr(r, "legal_population", lambda config, frozen: {dataset})
    monkeypatch.setattr(r, "expected_args", lambda c: c["train"] | {"name": c["name"]})
    monkeypatch.setattr(r, "version", lambda key: "8.4.172" if key == "ultralytics" else "synthetic")
    monkeypatch.setattr(r.sys, "prefix", str(root / "yolo"))
    monkeypatch.setattr(r, "evaluator_api", lambda: (None, None))
    monkeypatch.setattr(r.subprocess, "check_output", lambda *a, **kw: "fake-commit\n")
    geometry = stopped_geometry()
    geometry["runtime"] = {"ultralytics": "8.4.172", "torch": "synthetic"}
    geometry["input_sha256"] = {str(p): r.digest(p) for p in r.required_engineering_code() | reports | {model, r.BASE}}
    r.GEOMETRY_ENGINEERING.parent.mkdir(parents=True)
    r.write(r.GEOMETRY_ENGINEERING, geometry)
    monkeypatch.setattr(r, "GEOMETRY_ENGINEERING_SHA", r.digest(r.GEOMETRY_ENGINEERING))
    eng = passed_engineering(geometry)
    eng["input_sha256"].update({str(p): r.digest(p) for p in amp_code | {r.GEOMETRY_ENGINEERING}})
    previews = set()
    for arm in r.ARMS:
        p = r.preview_path(arm)
        p.parent.mkdir(parents=True)
        r.write(p, {"schema": "bounded_retention_read_only_diagnostics_v1", "mode": arm,
            "rank": -1, "epoch_index": 0, "epoch_number": 1, "batch_count": 1,
            "diagnostic_only": True, "participates_in_loss_tuning_or_stopping": False})
        eng["diagnostic_previews"][arm]["sha256"] = r.digest(p)
        previews.add(p)
    r.ENGINEERING.parent.mkdir(parents=True, exist_ok=True)
    r.write(r.ENGINEERING, eng)
    for receipt_path, report in ((r.GEOMETRY_ENGINEERING, geometry), (r.ENGINEERING, eng)):
        for name in report["input_sha256"]:
            p = Path(name)
            if p.suffix in {".py", ".json", ".yaml", ".md"}:
                snapshot = receipt_path.parent / "source_snapshot" / p.relative_to(root)
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(p.read_bytes())
    suite = put("outputs/full_suite.json", "{}")
    r.write(suite, {"status": "PASSED", "scope": "full_repository", "exit_code": 0,
        "passed": 1900, "failed": 0, "errors": 0, "input_sha256": {str(p): r.digest(p) for p in code}})
    contract = put("docs/final.md")
    all_inputs = code | reports | previews | amp_code | {model, dataset, r.BASE, r.GEOMETRY_ENGINEERING, r.ENGINEERING, suite, contract}
    all_inputs.update(Path(x["path"]) for x in configs["configs"].values())
    spec = configs | {"schema": r.SCHEMA, "permit": "FINAL_THREE_ARM_30_FROZEN",
        "failure_policy": r.FAILURE_POLICY,
        "geometry_engineering": {"path": str(r.GEOMETRY_ENGINEERING), "sha256": r.GEOMETRY_ENGINEERING_SHA},
        "engineering": {"path": str(r.ENGINEERING), "sha256": r.digest(r.ENGINEERING)},
        "full_suite": {"path": str(suite), "sha256": r.digest(suite)},
        "contract": {"path": str(contract), "sha256": r.digest(contract)},
        "input_sha256": {str(p): r.digest(p) for p in all_inputs}}
    spec_path = put("outputs/final_spec.json")
    r.write(spec_path, spec)
    result = r.preflight(spec_path)
    assert result["arms"] == list(r.ARMS) and result["epochs_per_arm"] == 30
    assert result["evaluation"] == {"images": 548, "imgsz": 960}
    assert not r.OUT.exists()  # check creates no run or permit
    for receipt_path in (r.ENGINEERING, r.GEOMETRY_ENGINEERING):
        snapshot = receipt_path.parent / "source_snapshot/src/regionssod/training/p1.py"
        original = snapshot.read_bytes()
        snapshot.write_text("changed")
        with pytest.raises(ValueError, match="snapshot"):
            r.preflight(spec_path)
        snapshot.write_bytes(original)
    # Rehashing the altered G0 pointer does not change the independent pin.
    r.write(r.GEOMETRY_ENGINEERING, geometry | {"status": "PASSED"})
    spec["geometry_engineering"]["sha256"] = r.digest(r.GEOMETRY_ENGINEERING)
    r.write(spec_path, spec)
    with pytest.raises(ValueError, match="immutable STOP pin"):
        r.preflight(spec_path)


def guard_stub():
    import torch
    weight = torch.tensor([1., -2.], requires_grad=True)
    # A normal GradScaler skip may leave scaled gradients nonfinite. These
    # guards must never inspect them or touch the scaler/optimizer.
    weight.grad = torch.tensor([float("inf"), float("nan")])
    model_state = {"weight": weight, "buffer": torch.tensor([.5]), "counter": torch.tensor(3)}
    ema_state = {"weight": torch.tensor([.9, -1.9]), "buffer": torch.tensor([.4])}
    callbacks = {}
    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint/optimizer/rebuild action is forbidden")
    trainer = SimpleNamespace(
        loss=torch.tensor(1.), loss_items={"box": torch.tensor(.25), "dfl": torch.tensor(.75)},
        fitness=.5, start_epoch=0, batch_size=8, args=SimpleNamespace(batch=8),
        model=SimpleNamespace(state_dict=lambda: model_state),
        ema=SimpleNamespace(ema=SimpleNamespace(state_dict=lambda: ema_state)),
        optimizer=SimpleNamespace(state={"momentum": torch.tensor([7.])}, zero_grad=forbidden,
                                  load_state_dict=forbidden, step=forbidden),
        _handle_nan_recovery=forbidden, _load_checkpoint_state=forbidden,
        _build_train_pipeline=forbidden,
        add_callback=lambda event, fn: callbacks.setdefault(event, []).append(fn),
    )
    return trainer, callbacks, model_state, ema_state


@pytest.mark.parametrize("arm", r.ARMS)
def test_common_failure_guards_preserve_normal_state_and_allow_scaled_grad_overflow(arm):
    import torch
    trainer, callbacks, state, ema = guard_stub()
    trainer.mode = arm  # guard does not branch on arm or replace its loss
    tensors = [*state.values(), *ema.values(), state["weight"].grad, trainer.optimizer.state["momentum"]]
    before = [v.detach().clone() for v in tensors]
    rng_before = torch.random.get_rng_state().clone()
    loss_identity = trainer.loss
    r.install_failure_guards(trainer)
    for event in ("on_train_epoch_start", "on_train_batch_end", "on_train_epoch_end"):
        for fn in callbacks[event]:
            fn(trainer)
    assert trainer._oom_retries == 3 and trainer.batch_size == trainer.args.batch == 8
    assert trainer._handle_nan_recovery(0) is False
    assert trainer._handle_nan_recovery(9) is False
    assert trainer.loss is loss_identity
    for a, b in zip(tensors, before):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize("field,value", [("loss", float("nan")), ("fitness", float("inf"))])
@pytest.mark.parametrize("epoch", [0, 20])
def test_nan_recovery_never_calls_original_or_loads_checkpoint(field, value, epoch):
    import torch
    trainer, callbacks, state, ema = guard_stub()
    r.install_failure_guards(trainer)
    setattr(trainer, field, torch.tensor(value) if field == "loss" else value)
    before = [v.detach().clone() for v in (*state.values(), *ema.values())]
    with pytest.raises(FloatingPointError):
        trainer._handle_nan_recovery(epoch)
    if field == "loss":
        with pytest.raises(FloatingPointError, match="unscaled loss"):
            callbacks["on_train_batch_end"][0](trainer)
    for a, b in zip((*state.values(), *ema.values()), before):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert trainer.optimizer.state["momentum"].item() == 7


def test_nonfinite_loss_item_and_post_validation_ema_are_checked():
    import torch
    trainer, callbacks, _, ema = guard_stub()
    r.install_failure_guards(trainer)
    trainer.loss_items["dfl"] = torch.tensor(float("inf"))
    with pytest.raises(FloatingPointError, match="loss_items.dfl"):
        callbacks["on_train_batch_end"][0](trainer)
    trainer.loss_items["dfl"] = torch.tensor(.75)
    callbacks["on_train_epoch_end"][0](trainer)
    # Validation runs after that callback: the replacement recovery hook must
    # recheck actual EMA state before native save_model can sanitize it.
    ema["buffer"][0] = float("inf")
    with pytest.raises(FloatingPointError, match="EMA.buffer"):
        trainer._handle_nan_recovery(1)
    assert torch.isinf(ema["buffer"]).all()


@pytest.mark.parametrize("which,key", [("model", "weight"), ("model", "buffer"),
                                       ("EMA", "weight"), ("EMA", "buffer")])
def test_actual_poisoned_state_stops_before_native_sanitizing_save(which, key):
    import torch
    trainer, callbacks, state, ema = guard_stub()
    target = (state if which == "model" else ema)[key]
    with torch.no_grad():
        target.flatten()[0] = float("nan")
    before = target.detach().clone()
    r.install_failure_guards(trainer)
    with pytest.raises(FloatingPointError, match=which):
        callbacks["on_train_epoch_end"][0](trainer)
    with pytest.raises(FloatingPointError, match=which):
        trainer._handle_nan_recovery(10)
    torch.testing.assert_close(target, before, rtol=0, atol=0, equal_nan=True)


def test_guard_disables_actual_installed_oom_handler_without_batch_or_optimizer_changes():
    """Execute only the installed exception branch with a stub, never a Trainer."""
    tree = ast.parse(r.installed_trainer_path().read_text(encoding="utf-8"))
    train = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_do_train")
    handler = next(n for n in ast.walk(train) if isinstance(n, ast.ExceptHandler)
                   and "self._oom_retries" in ast.unparse(n))
    probe = ast.parse('''def probe(self):
    epoch = self.start_epoch
    for _ in (0,):
        try:
            raise RuntimeError("CUDA out of memory")
        except RuntimeError as e:
            pass
''')
    native_try = next(n for n in ast.walk(probe) if isinstance(n, ast.Try))
    native_try.handlers = [deepcopy(handler)]
    namespace = {"RANK": -1}
    exec(compile(ast.fix_missing_locations(probe), "<native-oom-branch-only>", "exec"), namespace)
    trainer, callbacks, _, _ = guard_stub()
    r.install_failure_guards(trainer)
    callbacks["on_train_epoch_start"][0](trainer)
    with pytest.raises(RuntimeError, match="out of memory"):
        namespace["probe"](trainer)
    assert trainer.batch_size == trainer.args.batch == 8
    assert trainer.optimizer.state["momentum"].item() == 7
    # Execute the real successful-for/else reset, then prove the next epoch
    # re-arms the guard. No simulated batches/model/forward calls are involved.
    reset = next(n for n in ast.walk(train) if isinstance(n, ast.For)
                 and n.orelse and "self._oom_retries = 0" in ast.unparse(n.orelse))
    exec(compile(ast.fix_missing_locations(ast.Module(body=deepcopy(reset.orelse), type_ignores=[])),
                 "<native-success-reset-only>", "exec"), {"self": trainer})
    assert trainer._oom_retries == 0
    callbacks["on_train_epoch_start"][0](trainer)
    assert trainer._oom_retries == 3
