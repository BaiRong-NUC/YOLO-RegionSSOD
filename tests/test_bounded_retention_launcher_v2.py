"""Windows receipt spelling is not a change of frozen input identity."""
import importlib.util
from pathlib import Path

import pytest

module = Path(__file__).resolve().parents[1] / "scripts/run_bounded_retention_continuous_v2.py"
spec = importlib.util.spec_from_file_location("bounded_launcher_v2", module)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def test_mixed_native_receipt_paths_match_without_opening_or_changing_files(tmp_path):
    p = tmp_path / "initial.pt"
    forward, native = p.as_posix(), str(p)
    child, parent = {forward: "a"*64}, {native: "a"*64}
    r.check_subset(child, parent)
    r.check_subset(parent, child)
    assert child == {forward: "a"*64} and parent == {native: "a"*64}


@pytest.mark.parametrize("parent", [{}, {"initial.pt": "b"*64}, {"other.pt": "a"*64}])
def test_path_normalization_never_waives_missing_or_changed_input(tmp_path, parent):
    child = {str(tmp_path / "initial.pt"): "a"*64}
    parent = {str(tmp_path / p): h for p, h in parent.items()}
    with pytest.raises(ValueError):
        r.check_subset(child, parent)


def test_conflicting_alias_hashes_remain_rejected(tmp_path):
    p = tmp_path / "initial.pt"
    parent = {str(p): "a"*64, str(tmp_path / "sub" / ".." / "initial.pt"): "b"*64}
    with pytest.raises(ValueError, match="conflicting input aliases"):
        r.check_subset({str(p): "a"*64}, parent)
