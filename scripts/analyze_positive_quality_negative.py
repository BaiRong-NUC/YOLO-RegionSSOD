"""Post-hoc analysis of sealed predictions; no model imports or inference entrypoint.

See docs/paper/diagnostics/positive_quality_negative_review/protocol.md.
Existing output is never overwritten. --audit replays this analysis only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from diagnose_competitive_negative import read, sha, verify
from diagnose_p1_v2_bottleneck import capture_arrays, describe, budget_point, matched_transitions, rank_auc
from regionssod.evaluation.error_diagnosis import ERROR_NAMES, json_ready
from regionssod.evaluation.precision_recall import threshold_curve, fixed_threshold_point, curve_point

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/positive_quality_transport_v2_three30_s0"
OUT = ROOT / "outputs/positive_quality_negative_review_20261105"
PROTOCOL = ROOT / "docs/paper/diagnostics/positive_quality_negative_review/protocol.md"
ARMS = ("reference", "candidate", "control")


def bindings():
    if read(SOURCE / "status.json")["stage"] != "completed":
        raise ValueError("sealed completed source required")
    if not all(read(SOURCE / f"{name}_audit.json")["accepted"] for name in ("training", "evaluation")):
        raise ValueError("accepted upstream audits required")
    mapping = {}
    def merge(other):
        for name, digest in other.items():
            if name in mapping and mapping[name] != digest:
                raise ValueError(f"conflicting binding: {name}")
            mapping[name] = digest
    for manifest in (SOURCE / "artifact_sha256.json", SOURCE / "evaluation/artifact_sha256.json"):
        merge(read(manifest))
        merge({str(manifest): sha(manifest)})
    for arm in ARMS:
        merge(read(SOURCE / f"{arm}_seal.json")["artifact_sha256"])
        analysis = read(SOURCE / "evaluation" / arm / "analysis.json")
        weight = read(SOURCE / "evaluation/manifest.json")["weights"][arm]
        merge({weight: analysis["weights_sha256"],
               str(SOURCE / "evaluation" / arm / "predictions.jsonl"): analysis["predictions_sha256"],
               str(SOURCE / "evaluation/manifest.json"): analysis["manifest_sha256"]})
    for path in (PROTOCOL, Path(__file__).resolve(), ROOT / "scripts/diagnose_competitive_negative.py",
                 ROOT / "scripts/diagnose_p1_v2_bottleneck.py",
                 ROOT / "src/regionssod/evaluation/error_diagnosis.py",
                 ROOT / "src/regionssod/evaluation/precision_recall.py"):
        merge({str(path): sha(path)})
    verify(mapping)
    return mapping


def calculate():
    mapping = bindings()
    gt = read(SOURCE / "evaluation/ground_truth.json")
    areas = np.asarray([a["area"] for a in gt["annotations"]])
    small = areas < 1024
    counts = {"small": int(small.sum()), "medium": int(((areas >= 1024) & (areas < 9216)).sum()),
              "large": int((areas >= 9216).sum())}
    if len(gt["images"]) != 548 or counts != {"small": 26577, "medium": 11115, "large": 1067}:
        raise ValueError("GT population differs")
    result = {"scope": "post-hoc fixed-weight development-val analysis; not seed replication or causal attribution",
              "input_sha256": mapping, "counts": counts, "arms": {}, "pairs": {},
              "comparison": read(SOURCE / "comparison.json"), "decision": read(SOURCE / "decision.json")}
    arrays, curves = {}, {}
    for arm in ARMS:
        folder = SOURCE / "evaluation" / arm
        arrays[arm] = a = capture_arrays(folder, gt)
        stored = read(folder / "analysis.json")
        with (ROOT / f"runs/positive_quality_transport_v2_{arm}30_s0/results.csv").open() as handle:
            rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(handle)]
        if [int(row["epoch"]) for row in rows] != list(range(1, 31)):
            raise ValueError("complete 30 epochs required")
        cap_tails = [min(r["scores"]) for r in
                     (json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines())
                     if len(r["scores"]) == 500]
        result["arms"][arm] = info = {"student_seconds": float(rows[-1]["time"]),
            "max_cap_tail_score": max(cap_tails, default=None), "capped_images": len(cap_tails), "iou": {}}
        curves[arm] = {}
        for iou in (.5, .75):
            key = str(iou)
            curve = threshold_curve(a["scores"], a[f"areas_{key}"], counts)
            curves[arm][key] = curve
            with np.load(folder / f"micro_iou{iou}.npz") as original:
                for name, value in curve.items():
                    np.testing.assert_array_equal(original[name], value)
            point = fixed_threshold_point(curve, .25)
            if point != stored["conf025"][key]:
                raise ValueError("fixed-point numeric replay differs")
            errors = a[f"errors_{key}"]
            fp = {name: int(((a["scores"] >= .25) & (errors == j)).sum())
                  for j, name in enumerate(ERROR_NAMES) if j}
            if sum(fp.values()) != point["fp"]:
                raise ValueError("FP partition differs")
            baseline_point = read(SOURCE / "evaluation/reference/analysis.json")["conf025"][key]
            info["iou"][key] = {"conf025": point, "fp_conf025": fp,
                "same_reference_FP": budget_point(curve, "fp", baseline_point["fp"]),
                "same_reference_K": budget_point(curve, "predictions", baseline_point["predictions"]),
                "saved_floor": curve_point(curve, -1),
                "TP_FP_rank_AUROC": rank_auc(a["scores"], errors == 0)}
    for left, right in (("reference", "candidate"), ("reference", "control"), ("control", "candidate")):
        l, r = arrays[left], arrays[right]
        pair = {"transitions": {}, "common_small_geometry": {}}
        result["pairs"][f"{right}_minus_{left}"] = pair
        for iou in (.5, .75):
            key = str(iou)
            points = {"conf025": [.25, .25], "saved_floor": [0., 0.]}
            for p in (.7, .9):
                values = [read(SOURCE / "evaluation" / a / "analysis.json")["precision_constraints"][key][str(p)]
                          for a in (left, right)]
                points[f"P{int(p*100)}"] = [v["threshold"] for v in values] if all(v is not None for v in values) else None
            pair["transitions"][key] = {label: matched_transitions(l[f"gt_score_{key}"] >= t[0],
                r[f"gt_score_{key}"] >= t[1], small) if t else None for label, t in points.items()}
        common = small & (l["gt_score_0.5"] >= 0) & (r["gt_score_0.5"] >= 0)
        for name in ("iou", "normalized_center_error", "normalized_edge_error", "score"):
            lv, rv = l[f"geometry_{name}"][common], r[f"geometry_{name}"][common]
            pair["common_small_geometry"][name] = {"left": describe(lv), "right": describe(rv), "delta": describe(rv-lv)}
    differences = result["comparison"]["per_class_candidate_minus_other_pp"]
    result["leave_one_class_out_APsmall_pp"] = {other: {name: float(np.mean([v for c, v in values.items() if c != name]))
        for name in values} for other, values in differences.items()}
    # Detect any concurrent mutation of the sealed inputs before saving results.
    verify(mapping)
    return json_ready(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="read-only replay of this new cache analysis")
    args = parser.parse_args()
    start = time.monotonic()
    if args.audit:
        receipt = read(OUT / "receipt.json")
        verify(receipt["output_sha256"])
        if calculate() != read(OUT / "results.json"):
            raise ValueError("analysis replay differs")
        print(json.dumps({"accepted": True, "numeric_replay": True, "inference": False,
                          "seconds": time.monotonic() - start}))
    else:
        if OUT.exists():
            raise FileExistsError("never overwrite an earlier analysis")
        result = calculate()
        OUT.mkdir()
        output = OUT / "results.json"
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        (OUT / "receipt.json").write_text(json.dumps({"output_sha256": {str(output): sha(output)},
            "seconds": time.monotonic() - start, "training": False, "inference": False,
            "terminal_audit_rerun": False}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"saved": str(output), "seconds": time.monotonic() - start}))


if __name__ == "__main__":
    main()
