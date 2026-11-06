"""Regression for the actual Windows heartbeat replace/open-reader failure."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "point_real_audit_io", Path(__file__).resolve().parents[1]
    / "scripts/check_bounded_retention_point_real.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_publish_heartbeat_with_previous_event_held_open(tmp_path):
    beat = AUDIT.Heartbeat(tmp_path)
    beat("loss", arm="old_candidate")
    first = tmp_path / "heartbeats/00000001.json"
    # Python's Windows file handle can prevent replacing this file. The next
    # publication must not need to replace it, even while a slow reader holds it.
    with first.open(encoding="utf-8") as reader:
        original = reader.read()
        beat("save_loss")
        assert AUDIT.latest_heartbeat(tmp_path)["stage"] == "save_loss"
        reader.seek(0)
        assert reader.read() == original
    assert first.read_text(encoding="utf-8") == original
    assert AUDIT.latest_heartbeat(tmp_path)["arm"] == "old_candidate"


def test_unpublished_event_is_not_a_progress_signal(tmp_path):
    assert AUDIT.latest_heartbeat(tmp_path) is None
    beat = AUDIT.Heartbeat(tmp_path)
    beat("loss")
    (tmp_path / "heartbeats/00000002.json.tmp").write_text('{"stage":', encoding="utf-8")
    assert AUDIT.latest_heartbeat(tmp_path)["sequence"] == 1
    beat("save_loss")
    assert AUDIT.latest_heartbeat(tmp_path)["sequence"] == 2


def test_corrupt_published_event_does_not_silently_continue(tmp_path):
    beat = AUDIT.Heartbeat(tmp_path)
    beat("loss")
    (tmp_path / "heartbeats/00000002.json").write_text("invalid", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        AUDIT.latest_heartbeat(tmp_path)
