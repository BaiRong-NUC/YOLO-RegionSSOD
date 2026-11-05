"""Adversarial provenance checks using temporary synthetic reports only.

The 6471 label identities are metadata, not label files. The factory consumes
row text supplied by its audited caller; this suite does not construct a
dataset or read any real source/audit/geometry/label artifact.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from regionssod.training import bounded_retention_source as source_module


def encode(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def synthetic_chain(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    legal = root / "data/visdrone_area10/labeled/labels/train"
    audit_path = root / "outputs/bounded_retention_engineering_20261105/source_audit.json"
    previous = root / "outputs/reliable_region_ranking_feasibility_20261105"
    geometry_dir = root / "outputs/human_multiscale_observation_review_20261105"
    geometry_path = geometry_dir / "geometry.json"
    geometry_receipt_path = geometry_dir / "geometry_receipt.json"
    manifest_key = str(root / "data/visdrone_area10/manifest.jsonl")
    parameters = dict(area_ratio=.1, seed=0, min_box_visibility=.35, jitter_fraction=.2)
    row = "0 .15 .4 .3 .4"
    label_digest = digest((row + "\n").encode())
    identities = {str(legal / f"{i:07d}.txt"): label_digest for i in range(6471)}
    code_bytes = {
        root / "src/regionssod/data/annotations.py": b"# synthetic exporter identity\n",
        root / "src/regionssod/data/partial.py": b"# synthetic construction identity\n",
    }
    source = {
        "parameters": parameters,
        "gates": {"G1_human_identity_and_export_consistency": "PASS"},
        "inventory": {"actual_area_ratio_min": .0993113425925926},
        "input_sha256": {
            manifest_key: digest(b"synthetic population identity; no manifest exists"),
            **{str(path): digest(raw) for path, raw in code_bytes.items()},
        },
    }
    source_raw = encode(source)
    audit_raw = encode({
        "parameters": parameters,
        "input_sha256": {str(previous / "results.json"): digest(source_raw), **identities},
    })
    geometry = {
        "human_tiles": 6471, "non_square_tiles": 0,
        "tile_long_side_min_max": [131, 548],
        "generator_version": "reviewed-synthetic-v1",
        "input_sha256": {
            str(previous / "results.json"): digest(source_raw),
            manifest_key: source["input_sha256"][manifest_key],
        },
    }
    geometry_raw = encode(geometry)
    receipt = {"geometry_sha256": digest(geometry_raw), "exact_replay": True}
    files = {
        **code_bytes,
        audit_path: audit_raw,
        previous / "results.json": source_raw,
        previous / "receipt.json": encode({"results_sha256": digest(source_raw)}),
        geometry_path: geometry_raw,
        geometry_receipt_path: encode(receipt),
    }
    for path, raw in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for name, value in {
        "ROOT": root, "LEGAL": legal, "AUDIT": audit_path,
        "AUDIT_SHA": digest(audit_raw), "GEOMETRY_SHA": digest(geometry_raw),
    }.items():
        monkeypatch.setattr(source_module, name, value)

    reads = []
    original_read = Path.read_bytes

    def guarded_read(path):
        resolved = path.resolve()
        if resolved not in files:
            raise AssertionError(f"unexpected non-fixture read: {resolved}")
        reads.append(resolved)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    return SimpleNamespace(
        legal=legal, label=legal / "0000000.txt", row=row,
        label_digest=label_digest, geometry=geometry, receipt=receipt,
        geometry_path=geometry_path, geometry_receipt_path=geometry_receipt_path,
        geometry_sha=digest(geometry_raw), audit_sha=digest(audit_raw),
        allowed_reads=set(files), reads=reads,
    )


def test_pinned_synthetic_chain_builds_and_binds_an_audited_row(synthetic_chain):
    chain = synthetic_chain
    factory = source_module.BoundSourceFactory()
    packet = factory(
        label_path=chain.label, label_sha256=chain.label_digest,
        audit_sha256=chain.audit_sha, line_number=1, text=chain.row,
        source_object_id=17,
    )
    assert factory.reads[str(chain.geometry_path)] == chain.geometry_sha
    assert packet["physical"].certificate["kind"] == "SOURCE_OUTER"
    assert packet["physical"].certificate["historical_execution_verified"] is False
    assert packet["original_unknown"].tolist() == [True, False, False, False]
    assert factory.bindings[17]["label_sha256"] == chain.label_digest
    assert factory.counts["objects"] == 1
    assert set(chain.reads) == chain.allowed_reads
    assert not chain.label.exists()  # factory never opens caller-bound labels


def test_replacing_b_and_rehashing_receipt_cannot_bypass_fixed_pin(synthetic_chain):
    chain = synthetic_chain
    changed = {**chain.geometry, "generator_version": "unreviewed-synthetic-v2"}
    changed_raw = encode(changed)
    changed_sha = digest(changed_raw)
    chain.geometry_path.write_bytes(changed_raw)
    chain.geometry_receipt_path.write_bytes(encode({
        **chain.receipt, "geometry_sha256": changed_sha,
    }))
    # A linkage, population identity, replay flag, counts and size domain are
    # unchanged. The replacement and its receipt are mutually consistent.
    assert changed_sha != chain.geometry_sha
    assert source_module.GEOMETRY_SHA == chain.geometry_sha
    with pytest.raises(ValueError, match="public geometry report binding changed"):
        source_module.BoundSourceFactory()


def test_non_allowlisted_paths_fail_before_observation_or_label_access(synthetic_chain, monkeypatch):
    chain = synthetic_chain
    factory = source_module.BoundSourceFactory()
    initial_reads = list(chain.reads)

    def must_not_build(*args, **kwargs):
        raise AssertionError("unbound row reached physical observer")

    monkeypatch.setattr(source_module, "build_observation", must_not_build)
    # Reuse a VALID label digest and audit SHA: neither authorizes another path.
    for path in (
        chain.legal / "unlisted.txt",
        chain.legal.with_name("train_extra") / chain.label.name,
        chain.legal / ".." / "outside.txt",
        chain.legal / "nested" / chain.label.name,
    ):
        with pytest.raises(ValueError, match="audited human-row binding"):
            factory(label_path=path, label_sha256=chain.label_digest,
                    audit_sha256=chain.audit_sha, line_number=1, text=chain.row,
                    source_object_id=17)
    assert chain.reads == initial_reads
    assert not factory.bindings
    assert not factory.counts
