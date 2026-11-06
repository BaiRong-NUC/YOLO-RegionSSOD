"""Static inheritance and artifact/path guards; no model, dataset or solver execution.

The copied numerical runner and evidence auditor may differ from their protected
parents only at the explicitly declared wiring/source-binding sites. AST parity
checks protect every original guard, including ones not exercised by tiny fixtures.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OLD_RUNNER = SCRIPTS / "check_bounded_retention_point_pressure.py"
NEW_RUNNER = SCRIPTS / "check_bounded_retention_cached_pressure.py"
OLD_AUDITOR = SCRIPTS / "audit_bounded_retention_point_pressure.py"
NEW_AUDITOR = SCRIPTS / "audit_bounded_retention_cached_pressure.py"
CACHE_IMPORT = "regionssod.training.bounded_retention_point_geometry_cache"
OLD_IMPORT = "regionssod.training.bounded_retention_point_search_v2"
EXTRA_NAMES = {
    "LEGACY_PRESSURE", "LEGACY_AUDITOR", "LEGACY_PROTOCOL", "REPLAY_ADDENDUM",
    "ACCEPTANCE_ADDENDUM", "CACHE_SOLVER", "CACHED_PRESSURE", "CACHED_AUDITOR",
    "BINDING_SOURCES",
}


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def dump(node):
    return ast.dump(node, include_attributes=False)


def assignment_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def load_auditor(path):
    # Neither auditor imports torch or the runner at module scope.
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def auditor():
    return load_auditor(NEW_AUDITOR)


class WiringNormalizer(ast.NodeTransformer):
    """Undo only authorized wiring differences, preserving all executable guards."""

    def __init__(self, runner):
        self.runner = runner

    def visit_Constant(self, node):
        replacements = {
            "outputs/bounded_retention_cache_20261106": "outputs/bounded_retention_pressure_20261106",
            "docs/paper/candidates/bounded_retention/point_geometry_cache_protocol.md":
                "docs/paper/candidates/bounded_retention/point_pressure_protocol.md",
            "scripts/check_bounded_retention_cached_pressure.py": "scripts/check_bounded_retention_point_pressure.py",
        }
        if isinstance(node.value, str) and node.value in replacements:
            return ast.Constant(replacements[node.value])
        return node

    def visit_ImportFrom(self, node):
        if self.runner and node.module == CACHE_IMPORT:
            node.module = OLD_IMPORT
        return node

    def visit_List(self, node):
        if node.elts and isinstance(node.elts[0], ast.Starred) and dump(node.elts[0].value) == dump(ast.Name("BINDING_SOURCES", ast.Load())):
            names = "[Path(__file__).resolve(), PROTOCOL]" if self.runner else "[PRESSURE, PROTOCOL]"
            node.elts = ast.parse(names, mode="eval").body.elts + node.elts[1:]
        return self.generic_visit(node)

    def visit_For(self, node):
        if isinstance(node.iter, ast.Name) and node.iter.id == "BINDING_SOURCES":
            expected_target = "p" if self.runner else "path"
            assert isinstance(node.target, ast.Name) and node.target.id == expected_target
            expression = "(Path(__file__).resolve(), PROTOCOL)" if self.runner else "(PRESSURE, PROTOCOL)"
            node.iter = ast.parse(expression, mode="eval").body
        return self.generic_visit(node)


def normalized_copy(path, original, *, runner):
    copied, frozen = tree(path), tree(original)
    # The module docstring documents the new entry; function docstrings stay bound.
    copied.body.pop(0)
    frozen.body.pop(0)
    copied.body = [n for n in copied.body if assignment_name(n) not in EXTRA_NAMES]
    if not runner:
        # Test pins separately so the new bindings cannot replace historical pins.
        original_pins = next(n for n in frozen.body if assignment_name(n) == "PINS")
        copied.body = [deepcopy(original_pins) if assignment_name(n) == "PINS" else n for n in copied.body]
        history = next(n for n in copied.body if isinstance(n, ast.FunctionDef) and n.name == "verify_history")
        overlap_guard = ast.parse(
            "for path in BINDING_SOURCES:\n"
            "    if path.resolve() in expected:\n"
            "        evidence.pin(path, expected[path.resolve()])\n"
        ).body[0]
        matches = [n for n in history.body if dump(n) == dump(overlap_guard)]
        assert len(matches) == 1, "historical source overlap must be checked before extending the map"
        guard_index = history.body.index(matches[0])
        assert isinstance(history.body[guard_index + 1], ast.Expr)
        update = history.body[guard_index + 1].value
        assert isinstance(update, ast.Call) and isinstance(update.func, ast.Attribute)
        assert dump(update.func.value) == dump(ast.Name("expected", ast.Load())) and update.func.attr == "update"
        history.body.remove(matches[0])
    return WiringNormalizer(runner).visit(copied), frozen


def test_runner_inherits_all_numerical_and_process_guards_except_solver_wiring():
    actual, expected = normalized_copy(NEW_RUNNER, OLD_RUNNER, runner=True)
    assert dump(actual) == dump(expected)
    imports = [n for n in ast.walk(tree(NEW_RUNNER)) if isinstance(n, ast.ImportFrom)
               and any(a.name == "optimize_point" and a.asname == "new" for a in n.names)]
    assert len(imports) == 1 and imports[0].module == CACHE_IMPORT


def test_checker_inherits_every_original_evidence_rule_without_relaxation():
    actual, expected = normalized_copy(NEW_AUDITOR, OLD_AUDITOR, runner=False)
    assert dump(actual) == dump(expected)


def test_new_bindings_match_in_runner_and_checker_and_are_explicit(auditor):
    runner_defs = {assignment_name(n): n for n in tree(NEW_RUNNER).body if assignment_name(n) in EXTRA_NAMES}
    audit_defs = {assignment_name(n): n for n in tree(NEW_AUDITOR).body if assignment_name(n) in EXTRA_NAMES}
    assert set(runner_defs) == set(audit_defs) == EXTRA_NAMES
    assert all(dump(runner_defs[n]) == dump(audit_defs[n]) for n in EXTRA_NAMES)
    assert set(auditor.BINDING_SOURCES) == {
        NEW_RUNNER, NEW_AUDITOR, OLD_RUNNER, OLD_AUDITOR,
        ROOT / "src/regionssod/training/bounded_retention_point_geometry_cache.py",
        *(ROOT / "docs/paper/candidates/bounded_retention" / name for name in (
            "point_pressure_protocol.md", "point_pressure_replay_addendum.md",
            "point_pressure_acceptance_addendum.md", "point_geometry_cache_protocol.md")),
    }
    assert len(auditor.BINDING_SOURCES) == 9
    assert auditor.PRESSURE == NEW_RUNNER
    assert auditor.OUT_ROOT == ROOT / "outputs/bounded_retention_cache_20261106"
    assert auditor.PROTOCOL == ROOT / "docs/paper/candidates/bounded_retention/point_geometry_cache_protocol.md"


def test_original_evidence_pins_survive_and_new_protocol_is_independently_pinned(auditor):
    original = load_auditor(OLD_AUDITOR)
    assert original.PINS.items() <= auditor.PINS.items()
    additions = {OLD_RUNNER, OLD_AUDITOR, auditor.REPLAY_ADDENDUM,
                 auditor.ACCEPTANCE_ADDENDUM, auditor.PROTOCOL}
    assert set(auditor.PINS) - set(original.PINS) == additions
    for path in additions:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == auditor.PINS[path]


def test_fixed_population_remains_34_cases_and_102_arms(auditor):
    expected = [("initial", "pressure"), ("epoch9", "pressure")] + [
        ("epoch9", f"{phase}_{i:02d}") for phase in ("open", "closed") for i in range(16)]
    assert auditor.fixed_cases() == expected
    assert len(set(expected)) == 34 and len(expected) * len(auditor.ARMS) == 102


@pytest.mark.parametrize("name", ["audit_1", "audit_001", "qualification_01", "other"])
def test_invalid_run_names_are_rejected(auditor, tmp_path, monkeypatch, name):
    monkeypatch.setattr(auditor, "OUT_ROOT", tmp_path.resolve())
    with pytest.raises(auditor.QualificationStop):
        auditor.validate_paths(tmp_path / name, tmp_path / "qualification_01")


def test_old_output_root_nested_or_existing_output_cannot_be_used(auditor, tmp_path, monkeypatch):
    current = tmp_path / "cache"
    current.mkdir()
    monkeypatch.setattr(auditor, "OUT_ROOT", current.resolve())
    run, output = current / "audit_01", current / "qualification_01"
    assert auditor.validate_paths(run, output) == (run, output)
    for bad_run, bad_output in ((tmp_path / "old" / "audit_01", output),
                                (run, run / "qualification_01"), (run, run)):
        with pytest.raises(auditor.QualificationStop):
            auditor.validate_paths(bad_run, bad_output)
    output.mkdir()
    with pytest.raises(auditor.QualificationStop, match="overwrite"):
        auditor.validate_paths(run, output)


@pytest.mark.parametrize("status", [None, "TIMEOUT", "WORKER_FAILED", "PARENT_FAILED"])
def test_failed_or_unfinished_run_cannot_load_tensors_or_qualify(auditor, tmp_path, monkeypatch, status):
    if status is not None:
        (tmp_path / "terminal.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    monkeypatch.setattr(auditor.Evidence, "tensor", lambda *a: pytest.fail("must not load tensor artifacts"))
    monkeypatch.setattr(auditor, "audit_completed", lambda *a: pytest.fail("must not inspect failed evidence"))
    report = auditor.audit_run(tmp_path)
    assert report["accepted"] is False and report["training_permit"] is False
    assert report["status"] == "STOP"
