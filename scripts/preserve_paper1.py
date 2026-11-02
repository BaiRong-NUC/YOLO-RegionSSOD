"""Create and verify an append-only local copy of existing paper evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = Path("outputs/paper1_preservation")
DEFAULT = STORE / "20261102_baseline"
TREES = ("runs", "outputs", "weights", "data")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path) -> list[Path]:
    files = []
    for tree in TREES:
        directory = root / tree
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in directory.rglob("*"):
            if path.is_relative_to(root / STORE) or not path.is_file():
                continue
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"External file link: {path}")
            files.append(path)
    return sorted(files)


def pack_files(root: Path, files: list[Path], archive: Path) -> list[dict]:
    records = []
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as target:
        for path in files:
            name = path.relative_to(root).as_posix()
            before = path.stat()
            sha = hashlib.sha256()
            with path.open("rb") as source, target.open(name, "w", force_zip64=True) as out:
                while block := source.read(1024 * 1024):
                    sha.update(block)
                    out.write(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed during copy: {path}")
            records.append({"path": name, "bytes": after.st_size, "sha256": sha.hexdigest()})
    return records


def audit(directory: Path, root: Path | None = None) -> dict:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("evidence.zip", "source.zip"):
        if digest(directory / name) != manifest["archives"][name]:
            raise ValueError(f"Archive digest mismatch: {name}")
    records = manifest["files"]
    expected = {entry["path"] for entry in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate manifest entries")
    with zipfile.ZipFile(directory / "evidence.zip") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Archive member set mismatch")
        for record in records:
            path = Path(record["path"])
            if path.is_absolute() or ".." in path.parts or ":" in str(path):
                raise ValueError("Unsafe archive member")
            if archive.getinfo(record["path"]).file_size != record["bytes"]:
                raise ValueError(f"Size mismatch: {path}")
            with archive.open(record["path"]) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                    raise ValueError(f"Member digest mismatch: {path}")
            if root is not None and digest(root / path) != record["sha256"]:
                raise ValueError(f"Original has changed: {path}")
    return {
        "accepted": True,
        "files": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "source_commit": manifest["source_commit"],
        "manifest_sha256": digest(manifest_path),
        "archives": manifest["archives"],
        "originals_compared": root is not None,
    }


def create(root: Path, directory: Path) -> dict:
    if not directory.resolve().is_relative_to((root / STORE).resolve()):
        raise ValueError(f"Destination must be inside {root / STORE}")
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite preservation directory: {directory}")
    files = inventory(root)
    for arm in ("b2", "p1"):
        for seed in (0, 1):
            for filename in ("best.pt", "last.pt"):
                required = root / f"runs/partial10_{arm}_formal100_s{seed}/weights/{filename}"
                if required not in files:
                    raise FileNotFoundError(required)
    size = sum(path.stat().st_size for path in files)
    if shutil.disk_usage(root).free < size + 1024**3:
        raise OSError("Insufficient free space for evidence plus 1 GiB reserve")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    directory.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={directory / 'source.zip'}", commit],
        cwd=root, check=True,
    )
    records = pack_files(root, files, directory / "evidence.zip")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "source_scope": "Committed source only; uncommitted changes are not in source.zip.",
        "evidence_scope": list(TREES),
        "excluded": [STORE.as_posix(), "External original datasets and environment binaries"],
        "archives": {name: digest(directory / name) for name in ("evidence.zip", "source.zip")},
        "files": records,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    result = audit(directory, root)
    (directory / "creation_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--audit", action="store_true")
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--compare-originals", action="store_true")
    args = parser.parse_args()
    directory = (ROOT / args.directory).resolve()
    result = create(ROOT, directory) if args.create else audit(
        directory, ROOT if args.compare_originals else None
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
