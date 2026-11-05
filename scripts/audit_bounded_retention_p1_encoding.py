"""Replay V3's literal-U obstruction and distinguish encoding from geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from ultralytics.utils.loss import DFLoss
from ultralytics.utils.tal import make_anchors, TaskAlignedAssigner
from regionssod.training.bounded_retention_observation import build_observation
from regionssod.training.bounded_retention_objective_v3 import (
    support_geometry, current_unknown, inverse_interval, _multiply_interval,
)

OUT = ROOT / "outputs/bounded_retention_p1_encoding_review_20261105"
ATTEMPT = OUT / "p1_encoding_integration_01"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_from(report):
    witness = report["blocking_witness"]
    record = {k: np.asarray(witness[k]) for k in (
        "original_unknown", "affine", "affine_bounds", "augmentation_clip_window_bounds")}
    record["physical"] = build_observation(witness["source_xywh_tokens"], **report["source_premises"])
    return record


def replay():
    path = ATTEMPT / "result.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "STOP_BEFORE_STUDENT_TRAINING"
    assert report["stage"] == "real_augmented_geometry_preflight"
    assert not report["training_started"]
    assert all(report[k] == 0 for k in (
        "optimizer_steps", "model_forward_calls", "model_forward_attempts", "validation_inference_calls"))
    snapshots = 0
    for name, expected in report["input_sha256"].items():
        source = Path(name)
        assert digest(source) == expected, f"changed input: {name}"
        if source.is_relative_to(ROOT) and source.suffix in {".py", ".md", ".yaml", ".json"}:
            assert digest(ATTEMPT / "source_snapshot" / source.relative_to(ROOT)) == expected
            snapshots += 1
    witness = report["blocking_witness"]
    binding = witness["source_binding"]
    label = Path(binding["label_path"]).resolve()
    assert label.parent == (ROOT / "data/visdrone_area10/labeled/labels/train").resolve()
    assert digest(label) == binding["label_sha256"]
    assert label.read_text().splitlines()[binding["line_number"] - 1].split()[1:] == witness["source_xywh_tokens"]
    record = record_from(report)
    assert current_unknown(record, 640, 640)[0].tolist() == [False, False, False, True]
    box = torch.tensor(witness["nominal_output_xyxy"])
    anchors, strides = make_anchors([torch.empty(1, 1, 640 // s, 640 // s) for s in (8, 16, 32)],
                                    [8, 16, 32], .5)
    anchors *= strides
    native = TaskAlignedAssigner(stride=[8, 16, 32]).select_candidates_in_gts(
        anchors, box[None, None], torch.ones(1, 1, 1)).reshape(-1).bool()
    assert native.sum() == 32
    assert anchors[native].tolist() == witness["anchors"]
    assert strides[native].tolist() == witness["strides"]
    support = support_geometry(record, box, anchors[native], strides[native], (640, 640))
    assert support["target"].tolist() == witness["unclamped_nominal_distances"]
    assert support["valid"].sum(1).tolist() == witness["legal_atom_counts"] == [0] * 32
    known = support["target"][:, ~support["mask"]]
    assert ((known >= 0) & (known <= 14.99)).all()
    physical = record["physical"]
    boundary = _multiply_interval(physical.visible_edge_bounds[3], record["affine_bounds"][1])
    boundary += record["affine_bounds"][3]
    boundary = np.array([np.nextafter(boundary[0], -np.inf), np.nextafter(boundary[1], np.inf)])
    atom_coordinates = anchors[:, 1, None] + strides * torch.arange(16)[None]
    max_in_canvas = float(atom_coordinates[atom_coordinates <= 640].max())
    assert max_in_canvas == 636 and boundary[0] > max_in_canvas
    queries = np.array([638., 639., 640.])
    ranges = np.empty((3, 4, 2))
    ranges[:, :, 0], ranges[:, :, 1] = -np.inf, np.inf
    ranges[:, 3] = inverse_interval(np.repeat(queries[:, None], 2, 1),
        record["affine_bounds"][1], record["affine_bounds"][3])
    certificate = physical.certify_ranges(ranges)
    assert certificate.mask.all()
    # A valid decoded mean is possible although no literal categorical atom
    # lies in C. The model is not incapable of producing this continuous edge.
    assert float(636 + 8 * torch.tensor([.625, .375]).dot(torch.tensor([0., 1.]))) == 639.
    # The old E counterexample is now feasible with fixed E, not by relabeling.
    old_path = ROOT / "outputs/bounded_retention_engineering_20261105/native_event_integration_01/result.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_record = record_from(old)
    old_w = old["blocking_witness"]
    repaired = support_geometry(old_record, torch.tensor(old_w["nominal_output_xyxy"]),
        torch.tensor(old_w["anchors"]), torch.tensor(old_w["strides"]), (640, 640))
    assert repaired["valid"].sum(1).tolist() == [9, 9, 9, 9, 4, 4, 4]
    assert (repaired["geometry"][:, :, 1] == 636.80810546875).all()
    raw = repaired["target"][0, 1:2].clone()
    saved = raw.clone()
    pred = torch.arange(16, dtype=torch.float32).reshape(1, 16).requires_grad_()
    native_loss = DFLoss(16)(pred, raw.clone()[None])
    encoded_loss = DFLoss(16)(pred, raw.clamp(0, 14.99)[None])
    assert torch.equal(native_loss, encoded_loss) and torch.equal(raw, saved)
    # Support-only kernel projection loses fractional constraints. This is
    # a counterexample to calling that proposed change a lossless repair.
    p = np.array([.882, .098])
    point_a = min(-(1-u)*math.log(p[0])-u*math.log(p[1]) for u in (.2, .4))
    point_b = min(-(1-u)*math.log(p[0])-u*math.log(p[1]) for u in (.6, .8))
    assert point_a != point_b
    return {"status": "V3_LITERAL_U_EMPTY_CONTINUOUS_SET_NONEMPTY", "report_sha256": digest(path),
        "audit_source_sha256": digest(Path(__file__)), "bound_inputs": len(report["input_sha256"]),
        "verified_snapshots": snapshots, "legal_label_sha256": digest(label),
        "original_E_counterexample_sha256": digest(old_path),
        "old_E_repaired_atom_counts": repaired["valid"].sum(1).tolist(),
        "old_E_coordinate_unchanged": True, "source_object_id": witness["source_object_id"],
        "occurrence_id": witness["occurrence_id"], "full_batches_passed": len(report["geometry_batch_checks"]),
        "supported_object_occurrences": sum(x["supported_objects"] for x in report["geometry_batch_checks"])
                                        + report["geometry_preflight"]["supported_objects"],
        "native_anchors": int(native.sum()), "native_atom_counts": support["valid"].sum(1).tolist(),
        "known_distance_min_max": [float(known.min()), float(known.max())],
        "transported_U_boundary_interval": boundary.tolist(), "canvas_upper": 640,
        "all_grid_max_literal_U_atom_in_canvas": max_in_canvas,
        "continuous_query_coordinates": queries.tolist(), "continuous_outer_compatible": certificate.mask.tolist(),
        "query_retention_ratios": [float(w.ratio) for w in certificate.witnesses],
        "decoded_mean_counterexample": {"anchor_y": 636, "stride": 8, "p_bins_0_1": [.625, .375], "mean_y": 639},
        "support_projection_information_loss": {"C_a": [.2, .4], "C_b": [.6, .8], "same_codes": [0, 1],
            "same_event_loss": -math.log(p.sum()), "same_code_MAP": -math.log(p.max()),
            "different_best_point_DFL": [point_a, point_b]},
        "student_training_started": False, "new_model_forward": False, "optimizer_steps": 0,
        "hidden_training_GT_read": False,
        "scope": "literal-atom implementation fails; retained-event information and continuous localization are not refuted"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    result = replay()
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    target = OUT / "terminal_audit.json"
    if args.audit:
        assert target.read_text(encoding="utf-8") == encoded
        print("V3 terminal exact replay PASS")
    else:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
