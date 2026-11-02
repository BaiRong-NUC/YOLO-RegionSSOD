import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from regionssod.training.p1 import P1Trainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import resume_paper1_p1_seed1 as continuation


def test_continuation_preserves_prior_time_and_exact_csv_prefix(tmp_path):
    archive = tmp_path / "archive"
    (archive / "original_run").mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    old = archive / "original_run/results.csv"
    with old.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "time"])
        writer.writerows((i, i * 247) for i in range(1, 56))
    current = run / "results.csv"
    current.write_bytes(old.read_bytes())
    with current.open("a", newline="") as handle:
        csv.writer(handle).writerows((i, (i - 55) * 250) for i in range(56, 101))
    assert continuation.student_time(run, archive) == 55 * 247 + 45 * 250
    current.write_bytes(current.read_bytes().replace(b"1,247", b"1,248", 1))
    with pytest.raises(ValueError, match="exact original CSV prefix"):
        continuation.student_time(run, archive)


def test_incomplete_continuation_is_not_a_complete_formal_result(tmp_path):
    archive = tmp_path / "archive"
    (archive / "original_run").mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    old = archive / "original_run/results.csv"
    old.write_text("epoch,time\n" + "".join(f"{i},{i*247}\n" for i in range(1, 56)))
    (run / "results.csv").write_bytes(old.read_bytes() + b"56,250\n")
    with pytest.raises(ValueError, match="exactly epochs56-100"):
        continuation.student_time(run, archive)


@pytest.mark.parametrize("exit_code,stage", [(None, "completed"), (1, "completed"), (0, "training")])
def test_audit_requires_real_continuation_exit(tmp_path, monkeypatch, exit_code, stage):
    monkeypatch.setattr(continuation, "LOC", {"out": tmp_path})
    continuation.write(tmp_path / "resume_launch_receipt.json", {"exit_code": exit_code})
    continuation.write(tmp_path / "resume_status.json", {"stage": stage})
    monkeypatch.setattr(continuation, "measure", lambda: pytest.fail("must reject before measurement"))
    with pytest.raises(ValueError, match="real continuation exit0"):
        continuation.audit()


def test_capture_rejects_another_run_and_missing_original(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "ARCHIVE", tmp_path)
    saved = tmp_path / "saved.pt"
    saved.write_bytes(b"saved original")
    capture = {"arm": "p1", "student_seed": 1, "completed_epochs": 55,
               "files": {str(saved): {"original": str(saved),
                                     "sha256": continuation.digest(saved)}}}
    continuation.write(tmp_path / "capture.json", capture)
    assert continuation.validate_capture(originals=True) == capture
    continuation.write(tmp_path / "capture.json", capture | {"student_seed": 0})
    with pytest.raises(ValueError, match="registered P1"):
        continuation.validate_capture()
    continuation.write(tmp_path / "capture.json", capture)
    saved.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed interruption archive"):
        continuation.validate_capture()


@pytest.mark.parametrize("change", ["none", "schedule", "p1", "optimizer", "ema", "scaler"])
def test_setup_requires_original_p1_and_exact_restored_state(tmp_path, monkeypatch, change):
    archive = tmp_path / "archive"
    (archive / "original_run").mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "last.pt"
    original = {"model": "initial.pt", "resume": False, "epochs": 100, "seed": 1}
    (archive / "original_run/args.yaml").write_text(yaml.safe_dump(original))
    (run / "args.yaml").write_text(yaml.safe_dump(
        original | {"model": str(checkpoint), "resume": str(checkpoint)}))
    monkeypatch.setattr(continuation, "ARCHIVE", archive)
    model = torch.nn.Linear(1, 1)
    optimizer = {"state": {0: {"momentum_buffer": torch.tensor([2.0])}}, "param_groups": []}
    saved = {"optimizer": optimizer, "ema": model, "updates": 10496, "scaler": {"scale": 4096}}
    monkeypatch.setattr(torch, "load", lambda *a, **kw: saved)
    trainer = object.__new__(P1Trainer)
    trainer.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    trainer.start_epoch, trainer.epochs = 55, 100
    trainer.scheduler = SimpleNamespace(last_epoch=54)
    trainer.save_dir = run
    restored = {"state": {0: {"momentum_buffer": torch.tensor([2.0])}}, "param_groups": []}
    trainer.optimizer = SimpleNamespace(state_dict=lambda: restored)
    trainer.ema = SimpleNamespace(ema=torch.nn.Linear(1, 1), updates=10496)
    trainer.ema.ema.load_state_dict(model.state_dict())
    scaler = {"scale": 4096}
    trainer.scaler = SimpleNamespace(state_dict=lambda: scaler)
    if change == "schedule":
        trainer.epochs = 155
    elif change == "p1":
        trainer.p1_options["uncertain_ignore"] = False
    elif change == "optimizer":
        restored["state"][0]["momentum_buffer"].add_(1)
    elif change == "ema":
        with torch.no_grad():
            trainer.ema.ema.weight.add_(1)
    elif change == "scaler":
        scaler["scale"] = 1
    if change == "none":
        assert continuation.verify_setup(trainer, checkpoint, run)["p1_trainer"]
    else:
        with pytest.raises(ValueError):
            continuation.verify_setup(trainer, checkpoint, run)
