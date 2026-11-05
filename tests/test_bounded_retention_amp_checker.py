"""Audit replay/source-binding regressions; tensors only, no real dataset/model."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    return importlib.import_module("check_bounded_retention_amp")


def parent_rows():
    return [{"phase": phase, "active_dimensions": {str(3 if phase == "mosaic_closed" and i == 1316 else 1): 1},
             "partial_objects_checked": 1, "supported_objects": 1, "empty_anchor_events": 0}
            for phase in ("mosaic_open", "mosaic_closed") for i in range(1414)]


def selection():
    return dict(rule="first_maximum_active_dimension", phase="mosaic_closed", batch_index=1316, active_dimension=3)


def fake_stream():
    class Loader:
        closed = False
        resets = 0

        def reset(self):
            self.resets += 1

        def __iter__(self):
            for i in range(1414):
                n = 3 if self.closed and i == 1316 else 1
                yield {"img": torch.zeros(1, 3, 2, 2, dtype=torch.uint8), "cls": torch.zeros(1, 1),
                       "retention_observations": [dict(n=n, source_object_id=i,
                                                       occurrence_id=i+2000*int(self.closed))]}
    return Loader()


def test_replay_retains_fixed_pressure_batch_and_checks_both_full_streams(checker, monkeypatch):
    loader = fake_stream()
    trainer = SimpleNamespace(_close_dataloader_mosaic=lambda: setattr(loader, "closed", True))
    monkeypatch.setattr(checker, "current_unknown", lambda r, w, h: (np.arange(4) < r["n"], None))
    report = {}
    batch = checker.replay_streams(loader, trainer, dict(geometry_batch_checks=parent_rows(),
                                                       forward_batch_selection=selection()), report)
    assert report["replayed_batches"] == {"mosaic_open": 1414, "mosaic_closed": 1414}
    assert loader.resets == 1 and trainer.train_loader is loader
    assert report["forward_batch_selection"] == selection()
    assert report["batch_identity"] == [[1316, 3316]]
    assert batch["retention_observations"][0]["n"] == 3


@pytest.mark.parametrize("drift", ["activity", "selection", "population"])
def test_transport_replay_refuses_to_silently_choose_a_different_batch(checker, monkeypatch, drift):
    loader = fake_stream()
    trainer = SimpleNamespace(_close_dataloader_mosaic=lambda: setattr(loader, "closed", True))
    monkeypatch.setattr(checker, "current_unknown", lambda r, w, h: (np.arange(4) < r["n"], None))
    parent = dict(geometry_batch_checks=parent_rows(), forward_batch_selection=selection())
    if drift == "activity":
        parent["geometry_batch_checks"][1500]["active_dimensions"] = {"2": 1}
    elif drift == "selection":
        parent["forward_batch_selection"]["batch_index"] = 4
    else:
        parent["geometry_batch_checks"].append(parent["geometry_batch_checks"][-1])
    with pytest.raises(ValueError, match="replay differs|pressure batch changed|replay incomplete"):
        checker.replay_streams(loader, trainer, parent, {})


@pytest.mark.parametrize("drift", [None, "source", "snapshot", "receipt"])
def test_geometry_inheritance_requires_immutable_parent_and_original_sources(checker, monkeypatch, tmp_path, drift):
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    source = tmp_path/"source.py"
    source.write_text("frozen_source = True\n")
    folder = tmp_path/"old_check"
    (folder/"source_snapshot").mkdir(parents=True)
    snapshot = folder/"source_snapshot/source.py"
    snapshot.write_bytes(source.read_bytes())
    receipt = folder/"result.json"
    data = dict(status="STOP_BEFORE_STUDENT_TRAINING", error=dict(message="nonfinite original P1 loss/gradient"),
                stage="real_model_amp_forward_backward", optimizer_steps=0, model_forward_calls=1,
                validation_inference_calls=0, geometry_batch_checks=parent_rows(),
                input_sha256={str(source): checker.sha(source)})
    receipt.write_text(json.dumps(data))
    monkeypatch.setattr(checker, "PARENT_SHA", checker.sha(receipt))
    if drift == "source":
        source.write_text("changed")
    elif drift == "snapshot":
        snapshot.write_text("changed")
    elif drift == "receipt":
        receipt.write_text(receipt.read_text()+" ")
    if drift is None:
        assert checker.verify_parent(receipt) == data
    else:
        with pytest.raises(ValueError, match="changed"):
            checker.verify_parent(receipt)


def test_tensor_fingerprints_bind_shape_dtype_value_without_mutation(checker):
    value = torch.arange(4., requires_grad=True)
    before = value.detach().clone()
    digest = checker.tensor_fingerprints({"value": value})
    assert digest != checker.tensor_fingerprints({"value": value.reshape(2, 2)})
    assert digest != checker.tensor_fingerprints({"value": value.double()})
    assert torch.equal(value, before) and value.grad is None


def test_receipt_comparison_normalizes_json_containers_without_relaxing_values(checker):
    actual = [{"identity": (123, 456), "diagnostics": {"gap": 1e-8}}]
    saved = json.loads(json.dumps(actual))
    assert checker.json_safe(actual) == saved
    saved[0]["diagnostics"]["gap"] = 2e-8
    assert checker.json_safe(actual) != saved
