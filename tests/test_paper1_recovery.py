import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper1_recovery as recovery


def test_resume_bookkeeping_cannot_hide_changed_optimization(tmp_path):
    checkpoint = tmp_path / "last.pt"
    original = {"model": "coco.pt", "resume": False, "epochs": 100, "batch": 8}
    actual = original | {"model": str(checkpoint), "resume": str(checkpoint)}
    recovery.verify_resume_arguments(original, actual, checkpoint)
    with pytest.raises(ValueError, match="settings"):
        recovery.verify_resume_arguments(original, actual | {"batch": 16}, checkpoint)
    with pytest.raises(ValueError, match="captured last"):
        recovery.verify_resume_arguments(original, actual | {"resume": True}, checkpoint)


def test_segment_time_preserves_prefix_and_adds_prior_training(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    (archive / "original_run").mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    old = archive / "original_run/results.csv"
    with old.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "time"])
        writer.writerows((i, i * 200) for i in range(1, 64))
    csv_path = run / "results.csv"
    csv_path.write_bytes(old.read_bytes())
    with csv_path.open("a", newline="") as handle:
        csv.writer(handle).writerows((i, (i - 63) * 180) for i in range(64, 101))
    monkeypatch.setattr(recovery, "ARCHIVE", archive)
    assert recovery.student_time(run, {"completed_epochs": 63}) == 63 * 200 + 37 * 180
    csv_path.write_bytes(csv_path.read_bytes().replace(b"1,200", b"1,201", 1))
    with pytest.raises(ValueError, match="exact original prefix"):
        recovery.student_time(run, {"completed_epochs": 63})


def test_source_exception_only_allows_archived_entry(tmp_path, monkeypatch):
    entry = tmp_path / "entry.py"
    entry.write_text("new recovery entry")
    archived = tmp_path / "original_source/scripts/run_paper1_full100.py"
    archived.parent.mkdir(parents=True)
    archived.write_text("old entry")
    other = tmp_path / "loss.py"
    other.write_text("original loss")
    record = {"recovery_sources": {str(entry): recovery.digest(entry)}}
    monkeypatch.setattr(recovery, "ENTRY", entry)
    monkeypatch.setattr(recovery, "ARCHIVE", tmp_path)
    monkeypatch.setattr(recovery, "recovery", lambda out: record)
    hashes = {str(entry): recovery.digest(archived), str(other): recovery.digest(other)}
    assert recovery.verify_sources(tmp_path, hashes) == record
    other.write_text("altered loss")
    with pytest.raises(ValueError, match="changed formal source"):
        recovery.verify_sources(tmp_path, hashes)


def test_archive_tampering_fails(tmp_path, monkeypatch):
    source = tmp_path / "saved.pt"
    source.write_bytes(b"checkpoint evidence")
    (tmp_path / "capture.json").write_text(json.dumps({
        "files": {str(source): {"sha256": recovery.digest(source)}},
    }))
    monkeypatch.setattr(recovery, "ARCHIVE", tmp_path)
    recovery.validate_capture()
    source.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="changed interruption archive"):
        recovery.validate_capture()
