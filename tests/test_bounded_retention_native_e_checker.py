"""V6 checker/Trainer dispatch regressions with tensor stubs, no real data."""
from pathlib import Path
import importlib
import pytest
import test_bounded_retention_sampling_independent as prior


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    module = importlib.import_module("check_bounded_retention_native_e")
    monkeypatch.setattr(prior, "load_engineering", lambda _: module)
    return module


@pytest.mark.parametrize("case", ["two_phases", "closed_failure", "hash_drift", "wrong_population", "selection"])
def test_new_entry_preserves_real_check_control_flow(checker, monkeypatch, tmp_path, case):
    tests = {
        "two_phases": prior.test_engineering_native_close_reset_two_complete_streams_before_only_forward,
        "closed_failure": prior.test_engineering_closed_mosaic_failure_stops_before_any_forward,
        "hash_drift": prior.test_engineering_terminal_hash_drift_cannot_receive_pass,
        "wrong_population": prior.test_engineering_rejects_wrong_train_population_before_batch_forward,
        "selection": prior.test_engineering_first_maximum_dimension_across_phases_selects_actual_batch,
    }
    tests[case](monkeypatch, tmp_path)


def test_v6_val_dispatch_still_uses_legal_lexical_super(monkeypatch):
    from regionssod.training.bounded_retention_training_v6 import ContinuousNativeETrainer
    from regionssod.training.p1 import P1Trainer
    sentinel = object()
    monkeypatch.setattr(P1Trainer, "build_dataset", lambda self, path, mode, batch: sentinel)
    trainer = object.__new__(ContinuousNativeETrainer)
    assert trainer.build_dataset("synthetic:val", mode="val", batch=8) is sentinel
