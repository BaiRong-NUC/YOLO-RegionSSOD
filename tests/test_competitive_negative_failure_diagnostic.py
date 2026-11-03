"""Regression evidence for the real AMP boundary, without local data or training."""
import importlib.util
from pathlib import Path

import torch


def diagnostic_module():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_competitive_negative_failure.py"
    spec = importlib.util.spec_from_file_location("compneg_failure_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(mass=8.708838983295664e-9):
    raw = torch.zeros(1, 10)
    raw[0, 9] = mass
    teacher = torch.zeros(1, 10)
    teacher[0, 9], teacher[0, 1] = .6, .2
    return {"logits": torch.zeros(1, 10, dtype=torch.float16),
            "targets": raw.half(), "teacher": teacher, "flags": torch.tensor([True]),
            "labels": torch.tensor([9]), "raw": raw}


def test_real_failure_mass_underflows_and_is_not_misreported_as_class_mismatch():
    module = diagnostic_module()
    data = row()
    stats = module.summarize(**data)
    assert data["raw"].max() > 0 and data["targets"].max() == 0
    assert stats["teacher_vs_assigned_class_mismatch"] == 0
    assert stats["raw_zero_mass_pseudo"] == 0
    assert stats["cast_only_zero_mass_pseudo"] == 1
    assert stats["positive_mass_wrong_class"] == 0
    replay = module.replay_foreground([data])
    for mode in ("semantic", "strength"):
        assert replay[mode]["targets"] == [{"image": 0, "error": "original hard pseudo-positive TAL target required"}]
        assert replay[mode]["raw"] == []


def test_diagnostic_keeps_real_binding_errors_separate():
    data = row(.5)
    data["labels"][0] = 1
    stats = diagnostic_module().summarize(**data)
    assert stats["teacher_vs_assigned_class_mismatch"] == 1
    assert stats["cast_only_zero_mass_pseudo"] == 0


def test_diagnostic_distinguishes_raw_zero_mass_from_cast_underflow():
    stats = diagnostic_module().summarize(**row(0.))
    assert stats["raw_zero_mass_pseudo"] == 1
    assert stats["cast_only_zero_mass_pseudo"] == 0
