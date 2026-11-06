"""Synthetic launcher guards only; no real dataset, model, evidence or training."""
import ast
import builtins
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/run_bounded_retention_cached.py"
OLD = ROOT / "scripts/run_bounded_retention_continuous_v2.py"


@pytest.fixture
def r():
    spec = importlib.util.spec_from_file_location("cached_runner_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(source):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_inherited_guards_evaluation_seals_dispatch_and_factory_have_only_declared_changes():
    old_source = OLD.read_text(encoding="utf-8")
    new_source = MODULE.read_text(encoding="utf-8")
    # Exact AST inheritance after the declared mechanical adaptations. Only the
    # two gate implementations are intentionally replaced/extended below.
    adapted = old_source.replace(
        "from regionssod.training.bounded_retention_training_v6 import ContinuousNativeETrainer",
        "from regionssod.training.bounded_retention_training_cached import CachedNativeETrainer")
    adapted = adapted.replace("ContinuousNativeETrainer", "CachedNativeETrainer")
    adapted = adapted.replace('report["passed"] >= 1862', 'report["passed"] >= 2235')
    adapted = adapted.replace('                code = child.wait()',
                              '                code = wait_with_progress(child, arm, start)')
    adapted = adapted.replace('    configs = configurations(spec)\n',
        '    require(path(spec.get("contract", {}).get("path", "")) == CACHE_CONTRACT.resolve(),\n'
        '            "cached training contract required")\n    configs = configurations(spec)\n', 1)
    adapted = adapted.replace('    legal = legal_population(configs["reference"], frozen)\n', '', 1)
    adapted = adapted.replace('    check_subset(suite_hashes, frozen)\n',
        '    check_subset(suite_hashes, frozen)\n'
        '    legal = legal_population(configs["reference"], frozen)\n'
        '    cache_inputs, cache_pointers = verify_cache_evidence(spec, frozen, suite_hashes, legal=legal)\n', 1)
    adapted = adapted.replace('    pointers = reports | {cp, BASE.resolve(),',
                              '    pointers = reports | cache_pointers | {cp, BASE.resolve(),', 1)
    adapted = adapted.replace('require(allowed_input(Path(name), legal=legal, pointers=pointers, model=model),',
        'require(allowed_input(Path(name), legal=legal, pointers=pointers, model=model)\n'
        '                or cached_evidence_input(Path(name), cache_inputs),', 1)
    expected, current = functions(adapted), functions(new_source)
    for name, node in expected.items():
        if name != "required_engineering_code":
            assert current[name] == node, name
    # No diagnostic constructor flags, hidden old-runner imports or class wrappers.
    assert 'from regionssod.training.p1 import P1Trainer' in new_source
    assert 'classes = (P1Trainer, CachedNativeETrainer)' in new_source


def test_import_is_inert(monkeypatch):
    code = compile(MODULE.read_text(encoding="utf-8"), str(MODULE), "exec")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(("regionssod", "torch", "ultralytics", "evaluate_bounded", "run_bounded"))
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("import attempted IO/process/model execution")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    exec(code, {"__name__": "inert_cached_runner", "__file__": str(MODULE)})


def test_fixed_destination_schema_and_new_token(r):
    assert r.OUT == ROOT / "outputs/bounded_retention_cached_three30_s0"
    assert r.SCHEMA == "bounded_retention_cached_three30_v1"
    assert r.WORKER_TOKEN == "REGIONSSOD_CACHED_WORKER_TOKEN"
    assert r.CACHE_WIRING == r.CACHE_ROOT / "wiring_01/result.json"
    assert r.MONITOR_SECONDS == 1800
    parent_freeze = ROOT / "outputs/bounded_retention_continuous_freeze_20261105_v2/freeze.json"
    assert r.CACHE_PINS[parent_freeze] == "a04ec4f117472fc2e11f988a61534d3de5898c9c9bff5d7f7a592928a6a57901"
    assert {p.name for p in r.required_cached_code()} == {
        "bounded_retention_point_search_v2.py", "bounded_retention_point_geometry_cache.py",
        "bounded_retention_objective_cached.py", "bounded_retention_training_cached.py"}


def test_g0_source_requirement_uses_only_immutable_executed_inventory(r, tmp_path, monkeypatch):
    receipt = tmp_path / "g0.json"
    executed = tmp_path / "old.py"
    later = tmp_path / "bounded_retention_later.py"
    later.write_text("# not historically executed", encoding="utf-8")
    receipt.write_text(json.dumps({"input_sha256": {str(executed): "a" * 64,
                                                     str(tmp_path / "report.json"): "b" * 64}}), encoding="utf-8")
    monkeypatch.setattr(r, "GEOMETRY_ENGINEERING", receipt)
    monkeypatch.setattr(r, "GEOMETRY_ENGINEERING_SHA", r.digest(receipt))
    assert r.required_engineering_code() == {executed}
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory changed"):
        r.required_engineering_code()


def test_hash_merge_rejects_conflicts_instead_of_overwriting(r, tmp_path):
    p = tmp_path / "input.py"
    assert r.merge_hashes({str(p): "a" * 64}, {str(p): "a" * 64}) == {str(p): "a" * 64}
    with pytest.raises(ValueError, match="conflicting"):
        r.merge_hashes({str(p): "a" * 64}, {str(p): "b" * 64})
    with pytest.raises(ValueError, match="conflicting input aliases"):
        r.hashes({str(p): "a" * 64, str(tmp_path / "sub" / ".." / "input.py"): "b" * 64})


def valid_status(r):
    trace = {"status": "DIAGNOSTIC_ONLY", "training_started": False,
             "point_equivalence_passed": True, "all_bound_sources_unchanged": True,
             "arms": {a: {"status": "COMPLETED", "returncode": 0, "training_started": False,
                           "verified_loss_seconds": 120.} for a in ("p1", "candidate", "control")}}
    pressure = {"status": "QUALIFIED_SAVED_PRESSURE_EVIDENCE", "accepted": True,
                "training_permit": False, "run": str(r.CACHE_ROOT / "audit_01"),
                "checks": {"checked_cases": 34, "checked_arms": 102, "trainable_parameter_tensors": 255,
                           "positive_3u_exposure": {s: [{"tal_weight": .1}] for s in ("initial", "epoch9")}}}
    wiring = {"status": "PASSED_CACHED_PRODUCTION_WIRING", "training_started": False,
              "optimizer_steps": 0, "validation_inference_calls": 0, "case_count": 6,
              "all_sources_unchanged": True, "all_state_restored": True}
    equiv = {"accepted": True, "error": None,
             "comparisons": {a: {"count": 2590, "equal": True} for a in ("candidate", "control")}}
    return trace, pressure, wiring, equiv


def test_complete_zero_training_status_contract_passes(r):
    r.verify_cache_status(*valid_status(r))


@pytest.mark.parametrize("field,value", [
    ("status", "RUNNING"), ("training_started", True), ("optimizer_steps", 1),
    ("optimizer_steps", False), ("validation_inference_calls", 1), ("case_count", 5),
    ("all_sources_unchanged", False), ("all_state_restored", False)])
def test_incomplete_or_training_wiring_is_rejected(r, field, value):
    rows = valid_status(r)
    rows[2][field] = value
    with pytest.raises(ValueError, match="wiring qualification"):
        r.verify_cache_status(*rows)


@pytest.mark.parametrize("seconds", [120.05, -1., float("nan"), float("inf")])
def test_trace_terminal_does_not_hide_expired_loss(r, seconds):
    rows = valid_status(r)
    rows[0]["arms"]["candidate"]["verified_loss_seconds"] = seconds
    with pytest.raises(ValueError, match="trace arm"):
        r.verify_cache_status(*rows)


@pytest.mark.parametrize("change", ["cases", "arms", "parameters", "positive", "equivalence", "bound_sources"])
def test_pressure_counts_positive_3u_and_point_equivalence_are_required(r, change):
    trace, pressure, wiring, equiv = valid_status(r)
    if change in {"cases", "arms", "parameters"}:
        field = {"cases": "checked_cases", "arms": "checked_arms", "parameters": "trainable_parameter_tensors"}[change]
        pressure["checks"][field] -= 1
    elif change == "positive":
        pressure["checks"]["positive_3u_exposure"]["epoch9"][0]["tal_weight"] = 0
    elif change == "equivalence":
        equiv["comparisons"]["candidate"]["count"] = 2589
    else:
        trace["all_bound_sources_unchanged"] = False
    with pytest.raises(ValueError):
        r.verify_cache_status(trace, pressure, wiring, equiv)


def test_wrong_evidence_pointer_rejected_before_any_hash_read(r, monkeypatch):
    def forbidden(*args):
        raise AssertionError("opened unclassified pointer")
    monkeypatch.setattr(r, "digest", forbidden)
    with pytest.raises(ValueError, match="wrong fixed"):
        r.fixed_cache_pointer({"cache_wiring": {"path": "data/hidden.json", "sha256": "a" * 64}},
                              "cache_wiring", r.CACHE_WIRING)


def test_consumed_evidence_exception_does_not_open_arbitrary_outputs_or_hidden_data(r):
    known = (r.CACHE_ROOT / "audit_01/01_initial_pressure/request.pt").resolve()
    assert r.cached_evidence_input(known, {str(known): "a" * 64})
    assert not r.cached_evidence_input(known, {})
    for p in (r.CACHE_ROOT / "audit_99/request.pt", r.ROOT / "data/hidden/labels.txt",
              r.CACHE_ROOT / "audit_01/hidden.npy", r.CACHE_ROOT / "wiring_01/anything.pt"):
        assert not r.cached_evidence_input(p, {str(p.resolve()): "a" * 64})
    assert not r.allowed_input(known, legal=set(), pointers=set(), model=r.INITIAL_MODEL)
    manifest = (r.CACHE_ROOT / "audit_01/manifest.json").resolve()
    assert not r.allowed_input(manifest, legal=set(), pointers={manifest}, model=r.INITIAL_MODEL)
    assert r.cached_evidence_input(manifest, {str(manifest): "a" * 64})


@pytest.fixture
def cache_receipts(r, tmp_path, monkeypatch):
    """All reads/digests are synthetic dictionaries; no saved .pt is opened."""
    monkeypatch.setattr(r, "ROOT", tmp_path)
    launcher = tmp_path / "scripts/run_bounded_retention_cached.py"
    monkeypatch.setattr(r, "__file__", str(launcher))
    cache = tmp_path / "outputs/bounded_retention_cache_20261106"
    monkeypatch.setattr(r, "CACHE_ROOT", cache)
    for name, p in {
        "CACHE_TRACE": cache / "trace_01/terminal.json",
        "CACHE_PRESSURE": cache / "qualification_02/report.json",
        "CACHE_WIRING": cache / "wiring_01/result.json",
        "CACHE_CONTRACT": tmp_path / "docs/cached_training_contract.md",
        "CACHE_AUDITOR": tmp_path / "scripts/audit_bounded_retention_cached_pressure_v2.py",
        "BASE": tmp_path / "configs/reference30.yaml",
    }.items():
        monkeypatch.setattr(r, name, p)
    manifest = cache / "trace_01/manifest.json"
    equivalence = cache / "trace_01/point_equivalence.json"
    parent_freeze = tmp_path / "outputs/bounded_retention_continuous_freeze_20261105_v2/freeze.json"
    pins = {p: "b" * 64 for p in (r.CACHE_TRACE, r.CACHE_PRESSURE, manifest, equivalence, parent_freeze)}
    monkeypatch.setattr(r, "CACHE_PINS", pins)
    trace, pressure, wiring, equiv = valid_status(r)
    sources = {str(p): "a" * 64 for p in r.required_cached_code()}
    artifact = cache / "audit_01/01_initial_pressure/request.pt"
    pressure_inputs = {str(p): "a" * 64 for p in r.required_cached_code()
                       if p.name in {"bounded_retention_point_search_v2.py", "bounded_retention_point_geometry_cache.py"}}
    pressure_inputs[str(artifact)] = "d" * 64
    pressure.update(evidence_sha256=pressure_inputs, auditor_sha256="a" * 64)
    wiring_inputs = {**pressure_inputs, **sources, str(r.CACHE_CONTRACT): "a" * 64,
                     str(r.CACHE_PRESSURE): pins[r.CACHE_PRESSURE], str(r.BASE): "a" * 64,
                     str(parent_freeze): pins[parent_freeze],
                     str(launcher): "a" * 64, str(tmp_path / "scripts/check_bounded_retention_cached_wiring.py"): "a" * 64}
    wiring["input_sha256"] = wiring_inputs
    records = {r.CACHE_TRACE: trace, r.CACHE_PRESSURE: pressure, r.CACHE_WIRING: wiring,
               manifest: {"source_sha256": pressure_inputs.copy()}, equivalence: equiv,
               r.CACHE_WIRING.parent / "terminal.json": {"status": "PASSED", "returncode": 0,
                   "source_verified_after": True, "training_started": False, "worker_result_sha256": "c" * 64}}
    spec = {key: {"path": str(p), "sha256": pins.get(p, "c" * 64)} for key, p in (
        ("cache_trace", r.CACHE_TRACE), ("cache_pressure", r.CACHE_PRESSURE), ("cache_wiring", r.CACHE_WIRING))}
    suite = {**sources, str(launcher): "a" * 64,
             str(tmp_path / "tests/test_bounded_retention_cached_runner.py"): "a" * 64}
    frozen = {**wiring_inputs, **suite, **{str(p): sha for p, sha in pins.items()},
              str(r.CACHE_AUDITOR): "a" * 64, str(r.CACHE_WIRING): "c" * 64,
              str(r.CACHE_WIRING.parent / "terminal.json"): "e" * 64}
    opened = []
    def fake_digest(p):
        opened.append(Path(p))
        return frozen[str(p)]
    monkeypatch.setattr(r, "digest", fake_digest)
    monkeypatch.setattr(r, "read", lambda p: deepcopy(records[Path(p)]))
    return r, spec, frozen, suite, records, opened


def test_all_cache_evidence_bound_without_loading_tensor_artifacts(cache_receipts):
    r, spec, frozen, suite, records, opened = cache_receipts
    consumed, pointers = r.verify_cache_evidence(spec, frozen, suite, legal=set())
    assert r.CACHE_WIRING in pointers
    manifest = r.CACHE_ROOT / "trace_01/manifest.json"
    assert r.cached_evidence_input(manifest, consumed)
    assert all(p.suffix != ".pt" for p in opened)
    parent_freeze = r.ROOT / "outputs/bounded_retention_continuous_freeze_20261105_v2/freeze.json"
    assert parent_freeze in pointers and parent_freeze in opened
    # There is deliberately no JSON record for this opaque receipt in the read
    # stub: digest is allowed, traversing its nested inventory would fail.
    assert parent_freeze not in records


@pytest.mark.parametrize("change", [None, "unlisted_data", "missing_legal", "conflicting_sha", "unlisted_freeze"])
def test_wiring_accepts_only_complete_enumerated_loader_inputs(cache_receipts, change):
    r, spec, frozen, suite, records, opened = cache_receipts
    legal = {r.ROOT / name for name in (
        "data/legal/dataset.yaml", "data/legal/train.txt", "data/legal/metadata.jsonl",
        "data/legal/images/train/human.jpg", "data/legal/labels/train/human.txt",
        "data/pseudo/images/train/pseudo.jpg", "data/pseudo/labels/train/pseudo.txt",
        "data/legal/images/val/full.jpg", "data/legal/labels/val/full.txt")}
    wiring = records[r.CACHE_WIRING]["input_sha256"]
    for p in legal:
        wiring[str(p)] = frozen[str(p)] = "f" * 64
    selected = r.ROOT / "data/legal/labels/train/human.txt"
    denied = None
    if change == "unlisted_data":
        # Same directory and file type does not grant membership.
        denied = selected.with_name("unlisted.txt")
        wiring[str(denied)] = frozen[str(denied)] = "f" * 64
    elif change == "missing_legal":
        wiring.pop(str(selected))
    elif change == "conflicting_sha":
        wiring[str(selected)] = "e" * 64
    elif change == "unlisted_freeze":
        denied = r.ROOT / "outputs/bounded_retention_continuous_freeze_20261105_v2/other.json"
        wiring[str(denied)] = frozen[str(denied)] = "f" * 64
    if change is None:
        r.verify_cache_evidence(spec, frozen, suite, legal=legal)
    else:
        with pytest.raises(ValueError):
            r.verify_cache_evidence(spec, frozen, suite, legal=legal)
    # Classification and inventory consistency do not load image/label contents;
    # preflight's unchanged final hash loop verifies their bytes after approval.
    assert not legal.intersection(opened)
    if denied is not None:
        assert denied not in opened


def test_legal_population_is_enumerated_once_before_cache_evidence():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    preflight = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preflight")
    calls = [n for n in ast.walk(preflight) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    populations = [n for n in calls if n.func.id == "legal_population"]
    evidence = next(n for n in calls if n.func.id == "verify_cache_evidence")
    assert len(populations) == 1 and populations[0].lineno < evidence.lineno
    assert any(k.arg == "legal" and isinstance(k.value, ast.Name) and k.value.id == "legal"
               for k in evidence.keywords)


@pytest.mark.parametrize("change", ["wiring_conflict", "frozen_conflict", "missing_source",
                                    "suite_missing", "hidden_input", "unconsumed_case", "missing_pressure",
                                    "failed_terminal", "missing_terminal", "wrong_result_sha"])
def test_final_cache_gate_fails_closed_on_missing_or_conflicting_evidence(cache_receipts, change):
    r, spec, frozen, suite, records, opened = cache_receipts
    wiring = records[r.CACHE_WIRING]["input_sha256"]
    source = str(r.ROOT / "src/regionssod/training/bounded_retention_point_geometry_cache.py")
    denied = None
    if change == "wiring_conflict":
        wiring[source] = "f" * 64
    elif change == "frozen_conflict":
        frozen[source] = "f" * 64
    elif change == "missing_source":
        wiring.pop(str(r.ROOT / "src/regionssod/training/bounded_retention_training_cached.py"))
    elif change == "suite_missing":
        suite.pop(source)
    elif change in {"hidden_input", "unconsumed_case"}:
        denied = (r.ROOT / "data/hidden/manifest.jsonl" if change == "hidden_input"
                  else r.CACHE_ROOT / "audit_01/arbitrary_request.pt")
        wiring[str(denied)] = "e" * 64
        frozen[str(denied)] = "e" * 64
    elif change == "failed_terminal":
        records[r.CACHE_WIRING.parent / "terminal.json"]["status"] = "TIMEOUT"
    elif change == "missing_terminal":
        frozen.pop(str(r.CACHE_WIRING.parent / "terminal.json"))
    elif change == "wrong_result_sha":
        records[r.CACHE_WIRING.parent / "terminal.json"]["worker_result_sha256"] = "f" * 64
    else:
        wiring.pop(str(r.CACHE_ROOT / "audit_01/01_initial_pressure/request.pt"))
    with pytest.raises(ValueError):
        r.verify_cache_evidence(spec, frozen, suite, legal=set())
    if denied is not None:
        assert denied not in opened


def test_make_trainer_keeps_original_reference_and_cached_class_without_diagnostic_switch(r, monkeypatch):
    class P1:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    class Cached(P1):
        pass
    monkeypatch.setattr(r, "expected_args", lambda config: {"save_dir": "unused", "epochs": 30})
    config = {"p1": {"enabled": True}}
    source = object()
    reference = r.make_trainer("reference", config, classes=(P1, Cached))
    assert type(reference) is P1 and reference.kwargs == {"overrides": {"epochs": 30}, "p1_options": config["p1"]}
    for arm in ("candidate", "control"):
        trainer = r.make_trainer(arm, config, classes=(P1, Cached), source=source)
        assert type(trainer) is Cached and trainer.kwargs["mode"] == arm
        assert trainer.kwargs["source_factory"] is source
        assert trainer.kwargs["source_audit_sha256"] == r.AUDIT_SHA


def test_parent_monitor_records_progress_then_stops_without_controlling_worker(r, tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUT", tmp_path)
    now = [0.]
    monkeypatch.setattr(r.time, "monotonic", lambda: now[0])
    class Child:
        pid = 123
        waits = 0
        def wait(self, timeout):
            assert timeout == 60
            self.waits += 1
            now[0] += 1800
            if self.waits == 1:
                raise subprocess.TimeoutExpired("synthetic", timeout)
            return 0
        def terminate(self):
            raise AssertionError("monitor must not terminate")
        kill = terminate
    child = Child()
    assert r.wait_with_progress(child, "reference", 0.) == 0
    rows = [json.loads(s) for s in (tmp_path / "reference_monitor.jsonl").read_text().splitlines()]
    assert [row["stage"] for row in rows] == ["monitor_started", "worker_running", "worker_exited"]
    assert child.waits == 2 and rows[-1]["exit_code"] == 0
    assert all(row["worker_termination_requested"] is False and row["automatic_retry"] is False for row in rows)


def test_monitor_interruption_preserves_exception_and_does_not_kill(r, tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUT", tmp_path)
    class Child:
        pid = 456
        def wait(self, timeout):
            raise KeyboardInterrupt()
        def terminate(self):
            raise AssertionError("worker retained")
        kill = terminate
    with pytest.raises(KeyboardInterrupt):
        r.wait_with_progress(Child(), "candidate", r.time.monotonic())
    rows = (tmp_path / "candidate_monitor.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["stage"] == "monitor_started"
