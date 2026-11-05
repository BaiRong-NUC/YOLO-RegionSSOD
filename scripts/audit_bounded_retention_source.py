"""New, scoped source audit; never imports/runs a previous audit or trainer.

Only the already-bound legal human label text and named code/protocol files are
read. No manifest, summary container, pixels, hidden labels or quality cache.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from regionssod.training.bounded_retention import edge_intervals, serialized_interval
from regionssod.data.annotations import Tile, YoloBox, clip_box_to_tile

OUT = ROOT / "outputs/bounded_retention_engineering_20261105"
OLD = ROOT / "outputs/reliable_region_ranking_feasibility_20261105"
LEGAL = (ROOT / "data/visdrone_area10/labeled/labels/train").resolve()
PROTOCOL = ROOT / "docs/paper/candidates/bounded_retention/engineering_protocol.md"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized_path(path):
    path = Path(path).resolve()
    if path.parent != LEGAL or path.suffix != ".txt":
        raise ValueError("outside explicitly allowed human label directory")
    return path


def export_synthetic(box):
    result = clip_box_to_tile(box, Tile(0, 0, 100, 100), 100, 100, .35)
    if result is None:
        raise AssertionError("synthetic retained object unexpectedly filtered")
    return result, " ".join([str(result.class_id), *(format(x, ".10g") for x in (result.x, result.y, result.w, result.h))])


def synthetic_rounding_collision():
    # Exact same legal five-column export, different uncut known right edge.
    # Both accepted inputs have a genuinely clipped left edge; no real GT used.
    a = YoloBox(0, .1, .5, .4, .2)
    b = YoloBox(0, .100000000001, .5, .4, .2)
    clipped_a, text_a = export_synthetic(a)
    clipped_b, text_b = export_synthetic(b)
    right_a, right_b = a.x + a.w / 2, b.x + b.w / 2
    assert text_a == text_b and right_a != right_b
    # A natural touching / a just-inside edge can also have identical text.
    c = YoloBox(0, .2, .5, .4, .2)
    d = YoloBox(0, .200000000001, .5, .4, .2)
    _, text_c = export_synthetic(c)
    _, text_d = export_synthetic(d)
    assert text_c == text_d
    return {
        "synthetic_only": True,
        "same_export_with_different_known_edge": text_a,
        "known_right_edge_a": right_a, "known_right_edge_b": right_b,
        "visible_width_a": clipped_a.w, "visible_width_b": clipped_b.w,
        "same_export_for_touching_and_inside": text_c,
        "meaning": "text certifies a rounding interval, not unique pre-export continuous edge or actual cut",
    }


def analyze():
    reads = {}
    def read(path):
        path = Path(path).resolve()
        data = path.read_bytes()
        reads[str(path)] = sha(data)
        return data
    receipt = json.loads(read(OLD / "receipt.json"))
    prior_raw = read(OLD / "results.json")
    if sha(prior_raw) != receipt["results_sha256"]:
        raise ValueError("prior legal provenance receipt differs")
    prior = json.loads(prior_raw)
    if prior["parameters"] != {"area_ratio": .1, "seed": 0, "min_box_visibility": .35, "jitter_fraction": .2}:
        raise ValueError("retention parameters differ")
    inputs = prior["input_sha256"]
    # Restrict path before any file open: previous mapping contains manifest,
    # summary and other containers that this audit MUST NOT read.
    labels = {Path(p).resolve(): digest for p, digest in inputs.items()
              if Path(p).resolve().parent == LEGAL and Path(p).suffix == ".txt"}
    if len(labels) != prior["inventory"]["human_label_files"] or set(LEGAL.glob("*.txt")) != set(labels):
        raise ValueError("legal human inventory differs")
    for relative in ("src/regionssod/data/annotations.py", "src/regionssod/data/partial.py"):
        path = (ROOT / relative).resolve()
        content = read(path)
        if sha(content) != inputs[str(path)]:
            raise ValueError("export implementation differs from audited source")
    for path in (PROTOCOL, Path(__file__), ROOT / "src/regionssod/training/bounded_retention.py"):
        read(path)
    counts = Counter()
    examples = []
    with localcontext() as context:
        context.prec = 80
        for path in sorted(labels):
            raw = read(normalized_path(path))
            if sha(raw) != labels[path]:
                raise ValueError(f"human label hash differs: {path.name}")
            for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5 or not 0 <= int(fields[0]) < 10:
                    raise ValueError("invalid legal label")
                intervals = edge_intervals(fields[1:])
                possible = [v.contains(boundary) for v, boundary in zip(intervals, (0, 0, 1, 1))]
                counts["boxes"] += 1
                counts[f"possible_boundary_sides_{sum(possible)}"] += 1
                counts["positive_width_edge_intervals"] += sum(v.lo < v.hi for v in intervals)
                nominal = np.asarray(fields[1:], dtype=np.float32)
                point_edges = np.concatenate((nominal[:2] - nominal[2:] / 2, nominal[:2] + nominal[2:] / 2))
                outside = [not v.contains(Decimal.from_float(float(p))) for v, p in zip(intervals, point_edges)]
                counts["float32_edges_outside_serialization_envelope"] += sum(outside)
                if any(possible):
                    counts["possible_boundary_boxes"] += 1
                    counts["possible_boundary_known_nominals_outside_envelope"] += sum(
                        out and not unknown for out, unknown in zip(outside, possible))
                    if len(examples) < 3:
                        examples.append({"file": path.name, "line": line_number, "text": line,
                            "possible_boundary_sides": possible,
                            "edge_intervals": [[str(v.lo), str(v.hi)] for v in intervals],
                            "p1_float32_edges": point_edges.tolist(), "outside_decimal_envelope": outside})
    if counts["boxes"] != prior["inventory"]["human_exported_boxes"]:
        raise ValueError("legal box inventory differs")
    return {"parameters": prior["parameters"], "human_files": len(labels), "counts": dict(counts),
        "examples": examples, "synthetic": synthetic_rounding_collision(),
        "interpretation": {
            "decimal_envelope_only": True,
            "binary_clip_arithmetic_error_certified": False,
            "preexport_exact_coordinates_recovered": False,
            "float32_outside_is_not_bad_annotation": True,
            "boundary_counts_are_not_true_cuts_or_effective_training_counts": True,
            "training_authorized_by_this_audit": False,
        }, "access": {"manifest": False, "summary": False, "hidden_labels": False,
            "hidden_counts": False, "pixels": False, "quality_caches": False,
            "model_forward": False, "optimizer_steps": 0}, "input_sha256": reads}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", help="replay only this new source audit")
    args = parser.parse_args()
    path = OUT / "source_audit.json"
    if path.exists() and not args.audit:
        raise FileExistsError(path)
    result = analyze()
    encoded = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if args.audit:
        if path.read_bytes() != encoded:
            raise ValueError("new source audit does not replay byte-for-byte")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    print(json.dumps({"replay": args.audit, "sha256": sha(encoded), "human_files": result["human_files"],
                      "counts": result["counts"], "training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
