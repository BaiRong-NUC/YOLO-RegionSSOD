import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "preserve_paper1", Path(__file__).resolve().parents[1] / "scripts/preserve_paper1.py"
)
preserve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preserve)


def make_archive(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    original = root / "runs/weights.pt"
    original.parent.mkdir()
    original.write_bytes(b"saved-weights")
    saved = root / preserve.STORE / "test"
    saved.mkdir(parents=True)
    records = preserve.pack_files(root, [original], saved / "evidence.zip")
    with zipfile.ZipFile(saved / "source.zip", "x") as archive:
        archive.writestr("source.py", "# baseline\n")
    manifest = {
        "source_commit": "test",
        "files": records,
        "archives": {name: preserve.digest(saved / name) for name in ("source.zip", "evidence.zip")},
    }
    (saved / "manifest.json").write_text(json.dumps(manifest))
    return root, saved, original


def test_archive_survives_original_change(tmp_path):
    root, saved, original = make_archive(tmp_path)
    assert preserve.audit(saved, root)["originals_compared"]
    original.write_bytes(b"new-work")
    assert preserve.audit(saved)["accepted"]
    with pytest.raises(ValueError, match="Original has changed"):
        preserve.audit(saved, root)


def test_detects_corrupt_archive(tmp_path):
    _, saved, _ = make_archive(tmp_path)
    with (saved / "evidence.zip").open("ab") as stream:
        stream.write(b"unexpected")
    with pytest.raises(ValueError, match="Archive digest mismatch"):
        preserve.audit(saved)


def test_refuses_overwrite(tmp_path):
    root, saved, original = make_archive(tmp_path)
    before = preserve.digest(saved / "evidence.zip")
    with pytest.raises(FileExistsError):
        preserve.pack_files(root, [original], saved / "evidence.zip")
    with pytest.raises(FileExistsError):
        preserve.create(root, saved)
    assert preserve.digest(saved / "evidence.zip") == before


def test_inventory_excludes_preservation_store(tmp_path):
    root, _, original = make_archive(tmp_path)
    for tree in preserve.TREES:
        (root / tree).mkdir(exist_ok=True)
    assert preserve.inventory(root) == [original]
