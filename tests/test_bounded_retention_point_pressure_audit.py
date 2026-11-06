"""Synthetic saved-artifact qualification tests; no real data, model or forward."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    "pressure_qualification", Path(__file__).resolve().parents[1] /
    "scripts/audit_bounded_retention_point_pressure.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("status", [None, "STARTING", "TIMEOUT", "WORKER_FAILED", "PARENT_FAILED"])
def test_nonpassing_terminal_stops_before_any_followup_read(tmp_path, monkeypatch, status):
    if status is not None:
        dump(tmp_path / "terminal.json", {"status": status})
    monkeypatch.setattr(audit, "audit_completed", lambda *a: pytest.fail("must not inspect unfinished artifacts"))
    monkeypatch.setattr(audit.Evidence, "tensor", lambda *a: pytest.fail("must not load tensors"))
    result = audit.audit_run(tmp_path)
    assert result["status"] == "STOP" and result["accepted"] is False
    assert result["training_permit"] is False
    assert "no tensor artifacts loaded" in result["reason"]


def test_passing_label_without_complete_worker_receipt_cannot_qualify(tmp_path, monkeypatch):
    dump(tmp_path / "terminal.json", {"status": "PASSED"})
    monkeypatch.setattr(audit.Evidence, "tensor", lambda *a: pytest.fail("missing worker receipt must stop first"))
    result = audit.audit_run(tmp_path)
    assert result["status"] == "STOP" and not result["accepted"]


def test_evidence_rejects_mutation_after_read(tmp_path):
    path = tmp_path / "receipt.json"
    dump(path, {"value": 1})
    evidence = audit.Evidence()
    evidence.json(path)
    dump(path, {"value": 2})
    with pytest.raises(audit.QualificationStop, match="changed during audit"):
        evidence.unchanged()


def test_pointer_rejects_same_bytes_at_substituted_path(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"same bytes")
    b.write_bytes(a.read_bytes())
    with pytest.raises(audit.QualificationStop, match="substituted"):
        audit.pointer(audit.Evidence(), {"path": str(b), "sha256": audit.file_sha(b)}, a)


def test_historical_pair_root_detects_coordinated_batch_and_receipt_replacement():
    pairs = [[f"{phase}_{i:02d}", "a" * 64, "b" * 64]
             for phase in ("open", "closed") for i in range(16)]
    original = audit.prefix_pairs_digest(pairs)
    replaced = deepcopy(pairs)
    replaced[0][1:] = ["c" * 64, "d" * 64]
    assert audit.prefix_pairs_digest(replaced) != original
    assert audit.prefix_pairs_digest(list(reversed(pairs))) != original


def exposure_payload():
    # The correct global anchor is 2, not the local anchor_row 0.
    return {
        "geometry": {"0:7/mask": torch.tensor([True, False, True, True]),
                     "0:7/indices": torch.tensor([2, 0])},
        "certificates": {"0:7": {"identity": (101, 202)}},
        "assignment": {"foreground": torch.tensor([[False, False, True]]),
                       "gt_index": torch.tensor([[0, 0, 7]]),
                       "scores": torch.tensor([[[0., 0.], [0., 0.], [.2, .3]]])},
        "criterion_diagnostics": {"posterior_event_details": [
            {"identity": (101, 202), "anchor_row": 0}]},
    }


def test_3u_consumption_joins_identity_local_row_global_anchor_and_weight():
    result = audit.positive_3u_exposure(exposure_payload())
    assert result == [{"support": "0:7", "identity": [101, 202], "anchor_row": 0,
                       "anchor_index": 2, "tal_weight": .5}]


@pytest.mark.parametrize("kind", ["zero", "nan", "inf", "negative", "not_foreground", "wrong_gt",
                                  "unknown_identity", "bad_row", "no_event", "not_3u", "duplicate_event",
                                  "ambiguous_identity"])
def test_3u_observation_or_computation_alone_never_qualifies(kind):
    p = exposure_payload()
    if kind in {"zero", "nan", "inf", "negative"}:
        p["assignment"]["scores"][0, 2] = {"zero": 0., "nan": float("nan"),
                                            "inf": float("inf"), "negative": -1.}[kind]
    elif kind == "not_foreground":
        p["assignment"]["foreground"][0, 2] = False
    elif kind == "wrong_gt":
        p["assignment"]["gt_index"][0, 2] = 8
    elif kind == "unknown_identity":
        p["criterion_diagnostics"]["posterior_event_details"][0]["identity"] = (101, 999)
    elif kind == "bad_row":
        p["criterion_diagnostics"]["posterior_event_details"][0]["anchor_row"] = 2
    elif kind == "no_event":
        p["criterion_diagnostics"]["posterior_event_details"] = []
    elif kind == "not_3u":
        p["geometry"]["0:7/mask"][0] = False
    elif kind == "duplicate_event":
        p["criterion_diagnostics"]["posterior_event_details"] *= 2
    elif kind == "ambiguous_identity":
        p["certificates"]["0:8"] = {"identity": (101, 202)}
    with pytest.raises(audit.QualificationStop):
        audit.positive_3u_exposure(p)


def test_each_pressure_state_must_qualify_independently():
    assert audit.positive_3u_exposure(exposure_payload())
    epoch9 = exposure_payload()
    epoch9["assignment"]["scores"].zero_()
    with pytest.raises(audit.QualificationStop, match="positive TAL weight"):
        audit.positive_3u_exposure(epoch9)


def test_certificate_must_bind_to_the_actual_batch_target():
    batch = {"batch_idx": torch.zeros(8), "retention_observations": [None] * 7 + [
        {"source_object_id": 101, "occurrence_id": 202}]}
    assert audit.positive_3u_exposure(exposure_payload(), batch)
    batch["retention_observations"][7]["occurrence_id"] = 999
    with pytest.raises(audit.QualificationStop, match="saved target"):
        audit.positive_3u_exposure(exposure_payload(), batch)


def state_fixture():
    return {"model.0.conv.weight": torch.tensor([1., 2.]),
            "model.0.bn.running_mean": torch.tensor([0.]),
            "model.0.bn.running_var": torch.tensor([1.]),
            "model.0.bn.num_batches_tracked": torch.tensor(0),
            audit.DFL_PARAMETER: torch.arange(16.).reshape(1, 16, 1, 1)}


def test_legacy_digest_uses_parameter_order_and_frozen_dfl_not_bn_buffers():
    state = state_fixture()
    names = ["model.0.conv.weight"]
    h = hashlib.sha256()
    for name in names + [audit.DFL_PARAMETER]:
        h.update(name.encode())
        h.update(state[name].numpy().tobytes())
    assert audit.legacy_digest(state, names) == h.hexdigest()
    changed = deepcopy(state)
    changed["model.0.bn.running_mean"] += 1
    assert audit.legacy_digest(changed, names) == h.hexdigest()
    assert audit.state_hashes(changed, names)["buffers"] != audit.state_hashes(state, names)["buffers"]
    changed[audit.DFL_PARAMETER][0, 0, 0, 0] += 1
    assert audit.legacy_digest(changed, names) != h.hexdigest()


@pytest.mark.parametrize("kind", ["value", "dtype", "missing", "order", "bn"])
def test_full_state_comparison_rejects_partial_or_silent_cast_loads(kind):
    expected = state_fixture()
    actual = deepcopy(expected)
    key = "model.0.conv.weight"
    if kind == "value":
        actual[key][0] += 1
    elif kind == "dtype":
        actual[key] = actual[key].half()
    elif kind == "missing":
        del actual[key]
    elif kind == "order":
        actual = dict(reversed(list(actual.items())))
    else:
        actual["model.0.bn.num_batches_tracked"] += 1
    with pytest.raises(audit.QualificationStop, match="state"):
        audit.verify_state_equal(actual, expected)


def receipt_fixture():
    shapes = {"model.weight": [2]}
    all_shapes = {"head/boxes": [8, 64, 8400], "head/scores": [8, 10, 8400], "parameter/model.weight": [2]}
    receipt = {"status": "COMPLETED", "arm": "new_candidate", "gradient_scale": 1024.,
               "posterior_call_number": 1, "unused_parameters": [], "gradient_finite": True,
               "model_grad_storage_untouched": True, "full_loss_stats": {"finite": True},
               "full_loss": [1., 2., 3.], "loss_seconds": 1., "backward_seconds": 2.,
               "gradients": {name: {"status": "OBSERVED", "finite": True, "nonfinite": 0,
                                     "shape": shape, "numel": math.prod(shape), "dtype": "torch.float64",
                                     "sha256": "a" * 64} for name, shape in all_shapes.items()}}
    return receipt, shapes


def test_complete_scaled_receipt_is_accepted_without_executing_backward():
    receipt, shapes = receipt_fixture()
    audit.verify_arm_receipt(receipt, "new_candidate", 1, shapes)


@pytest.mark.parametrize("kind", ["unused", "missing", "extra", "nonfinite", "shape", "scale", "time", "loss"])
def test_full_model_guard_rejects_incomplete_or_invalid_gradient_receipts(kind):
    receipt, shapes = receipt_fixture()
    name = "parameter/model.weight"
    if kind == "unused":
        receipt["unused_parameters"] = [name]
    elif kind == "missing":
        del receipt["gradients"][name]
    elif kind == "extra":
        receipt["gradients"]["parameter/unknown"] = receipt["gradients"][name]
    elif kind == "nonfinite":
        receipt["gradients"][name]["finite"] = False
    elif kind == "shape":
        receipt["gradients"][name]["shape"] = [1, 2]
    elif kind == "scale":
        receipt["gradient_scale"] = 512.
    elif kind == "time":
        receipt["backward_seconds"] = 120.1
    else:
        receipt["full_loss"][0] = float("nan")
    with pytest.raises(audit.QualificationStop):
        audit.verify_arm_receipt(receipt, "new_candidate", 1, shapes)


def test_request_batch_rejects_changed_pixels_and_identity():
    raw = {"img": torch.tensor([0, 127, 255], dtype=torch.uint8),
           "cls": torch.tensor([[1.]]),
           "retention_observations": [{"source_object_id": 1, "occurrence_id": 2}]}
    actual = deepcopy(raw)
    actual["img"] = actual["img"].float() / 255
    audit.verify_request_batch(actual, raw)
    actual["img"][0] = .01
    with pytest.raises(audit.QualificationStop, match="image"):
        audit.verify_request_batch(actual, raw)
    actual["img"][0] = 0.
    actual["retention_observations"][0]["occurrence_id"] = 3
    with pytest.raises(audit.QualificationStop, match="identities"):
        audit.verify_request_batch(actual, raw)


def test_request_rejects_changed_geometry_even_when_identity_is_unchanged():
    raw = {"img": torch.tensor([255], dtype=torch.uint8),
           "retention_observations": [{"source_object_id": 1, "occurrence_id": 2,
                                       "affine": [1., 0., 0.]}]}
    actual = deepcopy(raw)
    actual["img"] = actual["img"].float() / 255
    actual["retention_observations"][0]["affine"][0] = 2.
    with pytest.raises(audit.QualificationStop, match="metadata"):
        audit.verify_request_batch(actual, raw)


def test_output_path_cannot_overwrite_run_or_existing_report(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "OUT_ROOT", tmp_path.resolve())
    run = tmp_path / "audit_01"
    run.mkdir()
    fresh = tmp_path / "qualification_01"
    assert audit.validate_paths(run, fresh) == (run, fresh)
    with pytest.raises(audit.QualificationStop):
        audit.validate_paths(run, run)
    with pytest.raises(audit.QualificationStop):
        audit.validate_paths(run, run / "qualification_01")
    fresh.mkdir()
    with pytest.raises(audit.QualificationStop, match="overwrite"):
        audit.validate_paths(run, fresh)


def test_stop_report_is_written_only_to_fresh_sibling(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "OUT_ROOT", tmp_path.resolve())
    run, output = tmp_path / "audit_01", tmp_path / "qualification_01"
    run.mkdir()
    dump(run / "terminal.json", {"status": "WORKER_FAILED"})
    before = (run / "terminal.json").read_bytes()
    monkeypatch.setattr(audit.Evidence, "tensor", lambda *a: pytest.fail("no tensor loading for STOP"))
    assert audit.main(["--run", str(run), "--output", str(output)]) == 2
    assert (run / "terminal.json").read_bytes() == before
    assert set(p.name for p in run.iterdir()) == {"terminal.json"}
    assert json.loads((output / "report.json").read_text())["status"] == "STOP"
