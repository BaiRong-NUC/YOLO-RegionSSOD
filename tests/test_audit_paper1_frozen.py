import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "audit_paper1_frozen", Path(__file__).resolve().parents[1] / "scripts/audit_paper1_frozen.py"
)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def setup(tmp_path):
    source = tmp_path / "old.json"
    source.write_bytes(b'{"accepted":true,"at":"old"}\n')
    target = tmp_path / "new/report.json"
    module = SimpleNamespace(write=lambda *_: pytest.fail("Old writer must not run"))
    return source, target, module


def test_redirect_keeps_original_and_restores_writer(tmp_path):
    source, target, module = setup(tmp_path)
    before = source.read_bytes()
    old_write = module.write
    with frozen.redirect_receipts(module, {source: target}) as emitted:
        module.write(source, {"accepted": True, "at": "new"})
    assert source.read_bytes() == before
    assert json.loads(target.read_text())["at"] == "new"
    assert emitted == [str(target)]
    assert module.write is old_write


def test_unexpected_write_rejected_even_on_failure(tmp_path):
    source, target, module = setup(tmp_path)
    old_write = module.write
    with pytest.raises(ValueError, match="Unexpected write"):
        with frozen.redirect_receipts(module, {source: target}):
            module.write(tmp_path / "predictions.json", {})
    assert module.write is old_write
    assert not target.exists()
    assert not (tmp_path / "predictions.json").exists()


def test_existing_receipt_never_overwritten(tmp_path):
    source, target, module = setup(tmp_path)
    target.parent.mkdir()
    target.write_text("previous")
    with pytest.raises(FileExistsError):
        with frozen.redirect_receipts(module, {source: target}):
            pass
    assert target.read_text() == "previous"
