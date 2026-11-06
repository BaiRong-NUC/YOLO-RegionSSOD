"""Synthetic metadata and auditor inheritance checks; no real evidence execution."""
import ast
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "scripts/audit_bounded_retention_cached_pressure.py"
NEW = ROOT / "scripts/audit_bounded_retention_cached_pressure_v2.py"
REVISION = ROOT / "docs/paper/candidates/bounded_retention/cache_metadata_audit_revision.md"
OBJECT_ADDENDUM = ROOT / "docs/paper/candidates/bounded_retention/cache_metadata_object_addendum.md"
STOP_REPORT = ROOT / "outputs/bounded_retention_cache_20261106/qualification_01/report.json"
REVISION_SHA = "76243a6491af266e8d77282031399eb08a445e262ee3649b7201ad96e1404afe"
STOP_SHA = "2a72884d41d1c19275a16d6756b218538d2c31edc0b8aeebeb06139bd4b54ea4"
OBJECT_SHA = "e46fd2b04281f8118bbc462b8679c2a9dd2400f6f3b9846bb4b04a40391d0167"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def audit():
    return load(NEW)


def node_dump(node):
    return ast.dump(node, include_attributes=False)


def pins_node(module):
    return next(n.value for n in module.body if isinstance(n, ast.Assign)
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "PINS")


def numpy_branch(module):
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "saved_tree_equal")
    expected_test = ast.parse("isinstance(expected, np.ndarray)", mode="eval").body
    return next(n for n in function.body if isinstance(n, ast.If)
                and node_dump(n.test) == node_dump(expected_test))


def test_entire_auditor_diff_is_only_numpy_comparison_three_audit_pins_and_docstring():
    old = ast.parse(OLD.read_text(encoding="utf-8"))
    new = ast.parse(NEW.read_text(encoding="utf-8"))
    old.body.pop(0)
    new.body.pop(0)
    old_pins, new_pins = pins_node(old), pins_node(new)
    count = len(old_pins.keys)
    assert len(new_pins.keys) == count + 3
    assert [node_dump(n) for n in new_pins.keys[:count]] == [node_dump(n) for n in old_pins.keys]
    assert [node_dump(n) for n in new_pins.values[:count]] == [node_dump(n) for n in old_pins.values]
    expected_keys = [ast.parse(s, mode="eval").body for s in (
        'ROOT / "docs/paper/candidates/bounded_retention/cache_metadata_audit_revision.md"',
        'OUT_ROOT / "qualification_01/report.json"',
        'ROOT / "docs/paper/candidates/bounded_retention/cache_metadata_object_addendum.md"')]
    assert [node_dump(n) for n in new_pins.keys[count:]] == [node_dump(n) for n in expected_keys]
    assert [ast.literal_eval(n) for n in new_pins.values[count:]] == [REVISION_SHA, STOP_SHA, OBJECT_SHA]
    new_pins.keys = new_pins.keys[:count]
    new_pins.values = new_pins.values[:count]
    replacement = numpy_branch(new)
    expected_body = ast.parse(
        'if actual.dtype != expected.dtype or actual.shape != expected.shape:\n'
        '    return False\n'
        'if expected.dtype.hasobject:\n'
        '    return np.array_equal(actual, expected)\n'
        'return actual.tobytes(order="C") == expected.tobytes(order="C")'
    ).body
    assert [node_dump(n) for n in replacement.body] == [node_dump(n) for n in expected_body]
    replacement.body = deepcopy(numpy_branch(old).body)
    # Includes main self-SHA binding, source manifest equality, deadlines, tensor
    # comparison, 102 arms and positive 3U checks: no other executable change.
    assert node_dump(new) == node_dump(old)


def test_old_pins_and_numerical_manifest_binding_remain_unchanged(audit):
    old = load(OLD)
    assert old.PINS.items() <= audit.PINS.items()
    assert {p: h for p, h in audit.PINS.items() if p not in old.PINS} == {
        REVISION: REVISION_SHA, STOP_REPORT: STOP_SHA, OBJECT_ADDENDUM: OBJECT_SHA}
    assert audit.BINDING_SOURCES == old.BINDING_SOURCES
    assert audit.CACHED_AUDITOR == OLD
    assert all(p not in audit.BINDING_SOURCES for p in (NEW, REVISION, STOP_REPORT, OBJECT_ADDENDUM))


def test_identical_nan_placeholders_are_equal_recursively(audit):
    values = np.full(63, np.nan, dtype=np.float64)
    a = {"obs": {"original_area_cap": values}, "sequence": [values, (values,)]}
    b = {"obs": {"original_area_cap": values.copy()}, "sequence": [values.copy(), (values.copy(),)]}
    assert audit.saved_tree_equal(a, b)


def test_nan_payload_and_position_are_not_interchangeable(audit):
    a = np.array([0x7ff8000000000001, 0x3ff0000000000000], dtype=np.uint64).view(np.float64)
    assert audit.saved_tree_equal(a, a.copy())
    changed_payload = a.view(np.uint64).copy()
    changed_payload[0] = 0x7ff8000000000002
    assert not audit.saved_tree_equal(a, changed_payload.view(np.float64))
    assert not audit.saved_tree_equal(a, a[::-1].copy())


@pytest.mark.parametrize("kind", ["finite", "signed_zero", "shape", "dtype", "byte_order"])
def test_metadata_changes_are_rejected(audit, kind):
    a = np.array([0., 1., np.nan, 2.], dtype=np.float64)
    b = a.copy()
    if kind == "finite":
        b[1] = 1.5
    elif kind == "signed_zero":
        b[0] = -0.
    elif kind == "shape":
        b = b.reshape(2, 2)
    elif kind == "dtype":
        b = b.astype(np.float32)
    else:
        b = b.astype(b.dtype.newbyteorder("S"))
    assert not audit.saved_tree_equal(a, b)


@dataclass(frozen=True)
class SyntheticBoundary:
    coordinates: tuple


def object_metadata():
    return np.array([SyntheticBoundary((1, 2, 3, 4)), None], dtype=object)


def test_independent_equal_object_contents_and_none_are_preserved(audit):
    a, b = object_metadata(), object_metadata()
    assert a[0] is not b[0]
    assert audit.saved_tree_equal({"obs": {"physical": a}}, {"obs": {"physical": b}})
    assert audit.saved_tree_equal(np.array([None, None], dtype=object),
                                  np.array([None, None], dtype=object))
    assert audit.saved_tree_equal(np.empty(0, dtype=object), np.empty(0, dtype=object))


@pytest.mark.parametrize("kind", ["content", "shape", "none_position", "none_replacement"])
def test_object_metadata_changes_are_rejected(audit, kind):
    a, b = object_metadata(), object_metadata()
    if kind == "content":
        b[0] = SyntheticBoundary((1, 2, 3, 5))
    elif kind == "shape":
        b = b.reshape(1, 2)
    elif kind == "none_position":
        b = b[::-1].copy()
    else:
        b[0] = None
    assert not audit.saved_tree_equal(a, b)


def test_object_nan_keeps_original_rejection(audit):
    a = np.array([float("nan"), None], dtype=object)
    b = np.array([float("nan"), None], dtype=object)
    assert not audit.saved_tree_equal(a, b)


def test_structured_object_arrays_use_contents_and_require_exact_dtype(audit):
    dtype = [("number", "i4"), ("value", object)]
    a = np.array([(1, SyntheticBoundary((1, 2))), (2, None)], dtype=dtype)
    b = np.array([(1, SyntheticBoundary((1, 2))), (2, None)], dtype=dtype)
    assert a.dtype.hasobject and a["value"][0] is not b["value"][0]
    assert audit.saved_tree_equal(a, b)
    assert not audit.saved_tree_equal(a, b.astype([("number", "i8"), ("value", object)]))
    b["value"][0] = SyntheticBoundary((1, 3))
    assert not audit.saved_tree_equal(a, b)


def test_c_order_logical_bytes_ignore_storage_layout(audit):
    a = np.array([[np.nan, 1.], [2., 3.]], dtype=np.float64)
    assert audit.saved_tree_equal(a, np.asfortranarray(a))
    storage = np.zeros((2, 4), dtype=np.float64)
    storage[:, ::2] = a
    assert audit.saved_tree_equal(a, storage[:, ::2])
    assert audit.saved_tree_equal(np.empty((0, 2)), np.empty((0, 2)))
    assert not audit.saved_tree_equal(np.empty((0, 2)), np.empty((0, 3)))


def test_equal_nan_does_not_relax_other_metadata_types(audit):
    assert audit.saved_tree_equal({"items": [1, 2]}, {"items": [1, 2]})
    assert not audit.saved_tree_equal({"items": [1, 2]}, {"items": (1, 2)})
    assert not audit.saved_tree_equal({"items": [1, 2]}, {"other": [1, 2]})
    assert not audit.saved_tree_equal(float("nan"), float("nan"))
