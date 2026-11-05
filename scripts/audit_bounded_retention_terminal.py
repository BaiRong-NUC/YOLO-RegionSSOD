"""Replay the terminal exact-edge obstruction; no images/model/hidden labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from ultralytics.utils.tal import make_anchors, TaskAlignedAssigner
from regionssod.training.bounded_retention_observation import build_observation
from regionssod.training.bounded_retention_objective_v2 import current_unknown, support_geometry

ATTEMPT = ROOT / "outputs/bounded_retention_engineering_20261105/native_event_integration_01"
OUTPUT = ROOT / "outputs/bounded_retention_engineering_20261105/terminal_audit.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay():
    report_path = ATTEMPT / "result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "STOP_BEFORE_STUDENT_TRAINING"
    assert report["stage"] == "real_augmented_geometry_preflight"
    assert not report["training_started"] and report["optimizer_steps"] == 0
    assert report["model_forward_calls"] == report["validation_inference_calls"] == 0
    snapshots = 0
    for name, expected in report["input_sha256"].items():
        path = Path(name)
        assert digest(path) == expected, f"current input changed: {path}"
        if path.is_relative_to(ROOT) and path.suffix in {".py", ".md", ".yaml", ".json"}:
            assert digest(ATTEMPT / "source_snapshot" / path.relative_to(ROOT)) == expected
            snapshots += 1
    witness = report["blocking_witness"]
    source = witness["source_binding"]
    label = Path(source["label_path"]).resolve()
    legal = (ROOT / "data/visdrone_area10/labeled/labels/train").resolve()
    assert label.parent == legal and label.suffix == ".txt"
    assert digest(label) == source["label_sha256"]
    row = label.read_text().splitlines()[source["line_number"] - 1]
    assert row.split()[1:] == witness["source_xywh_tokens"]
    physical = build_observation(witness["source_xywh_tokens"], **report["source_premises"])
    record = {key: np.asarray(witness[key]) for key in (
        "original_unknown", "affine", "affine_bounds", "augmentation_clip_window_bounds")}
    record["physical"] = physical
    assert np.array_equal(physical.unknown, record["original_unknown"])
    assert np.array_equal(physical.visible_edge_bounds, witness["physical_visible_bounds"])
    effective, _ = current_unknown(record, 640, 640)
    assert effective.tolist() == [False, False, True, False]
    box = torch.tensor(witness["nominal_output_xyxy"])
    features = [torch.empty(1, 1, 640 // s, 640 // s) for s in (8, 16, 32)]
    anchors, strides = make_anchors(features, [8, 16, 32], .5)
    anchors *= strides
    native = TaskAlignedAssigner(stride=[8, 16, 32]).select_candidates_in_gts(
        anchors, box[None, None], torch.ones(1, 1, 1)).reshape(-1).bool()
    assert torch.equal(anchors[native], torch.tensor(witness["anchors"]))
    assert torch.equal(strides[native], torch.tensor(witness["strides"]))
    support = support_geometry(record, box, anchors[native], strides[native], (640, 640))
    assert torch.equal(support["target"], torch.tensor(witness["unclamped_nominal_distances"]))
    assert support["valid"].sum(1).tolist() == witness["legal_atom_counts"] == [0] * 7
    assert (support["target"][:, 1] < 0).all()
    # Stronger than the native-pool counterexample: no grid center can satisfy
    # this E coordinate with a nonnegative top distance, even outside TAL.
    assert anchors[:, 1].max() < box[1]
    return {"status": "EXACT_EDGE_SUPPORT_BLOCK_REPLAYED", "student_training_started": False,
        "report_sha256": digest(report_path), "input_count": len(report["input_sha256"]),
        "source_snapshots_verified": snapshots, "legal_label_sha256": digest(label),
        "source_object_id": witness["source_object_id"], "occurrence_id": witness["occurrence_id"],
        "effective_unknown": effective.tolist(), "native_anchor_count": int(native.sum()),
        "native_legal_atom_counts": support["valid"].sum(1).tolist(),
        "grid_anchor_count": len(anchors), "grid_max_y": float(anchors[:, 1].max()),
        "fixed_exact_top": float(box[1]), "unrepresentable_gap_pixels": float(box[1] - anchors[:, 1].max()),
        "new_model_or_validation_inference": False, "optimizer_steps": 0,
        "hidden_training_gt_read": False,
        "scope": "current fixed-E nonnegative DFL support contract; not all bounded-distribution methods"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    result = replay()
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.audit:
        assert OUTPUT.read_text(encoding="utf-8") == encoded
        print("terminal evidence exact replay PASS")
    else:
        with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
