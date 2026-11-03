"""Exploratory cache-only audit; no model, training, hidden GT, or fitted thresholds."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from diagnose_competitive_negative import lines, model_summary, read, sha, verify
from regionssod.evaluation.error_diagnosis import (
    ERROR_NAMES, box_iou_matrix, classify_false_predictions, json_ready, paired_geometry,
)
from regionssod.evaluation.precision_recall import (
    assign_predictions, curve_point, precision_constrained_point, threshold_curve,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/p1_competitive_negative_main30_v2_s0"
TEACHER = ROOT / "outputs/p1_class_ambiguity_diagnostic_20261102"
OUT = ROOT / "outputs/p1_v2_bottleneck_20261103"
PROTOCOL = ROOT / "docs/paper/diagnostics/competitive_negative_v2_bottleneck/protocol.md"
ARMS = ("baseline", "reference", "candidate")
IOUS = (.5, .75)


def describe(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return {"n": len(values), "mean": float(values.mean()) if len(values) else None,
            "q10_q50_q90": np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None}


def rank_auc(scores, positive):
    """P(score of a TP > score of an FP), with half credit for ties."""
    scores, positive = np.asarray(scores), np.asarray(positive, bool)
    if scores.shape != positive.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("finite parallel one-dimensional scores/labels required")
    npos, nneg = positive.sum(), (~positive).sum()
    if not npos or not nneg:
        return None
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    positives = np.add.reduceat(positive[order].astype(np.int64), starts)
    negatives = np.diff(np.r_[starts, len(scores)]) - positives
    negatives_below = np.cumsum(negatives) - negatives
    return float((positives * (negatives_below + .5 * negatives)).sum() / (npos * nneg))


def budget_point(curve, key, limit):
    """Longest whole score group within a predeclared budget; never split ties."""
    if key not in {"predictions", "fp"} or limit < 0:
        raise ValueError("prediction or FP budget required")
    eligible = np.flatnonzero(curve[key] <= limit)
    return curve_point(curve, eligible[-1]) if len(eligible) else None


def matched_transitions(left, right, mask):
    left, right, mask = (np.asarray(v, bool) for v in (left, right, mask))
    if left.shape != right.shape or left.shape != mask.shape:
        raise ValueError("shared GT universe required")
    return {"both": int((left & right & mask).sum()),
            "left_only": int((left & ~right & mask).sum()),
            "right_only": int((~left & right & mask).sum()),
            "neither": int((~left & ~right & mask).sum())}


def inputs():
    if read(SOURCE / "status.json")["stage"] != "completed":
        raise ValueError("completed audited source required")
    if not read(SOURCE / "evaluation_audit.json")["accepted"]:
        raise ValueError("accepted source evaluation required")
    # Verify current immutable output artifacts and intended weight/prediction bindings.
    mapping = read(SOURCE / "artifact_sha256.json")
    mapping.update(read(SOURCE / "evaluation/artifact_sha256.json"))
    manifest = read(SOURCE / "evaluation/manifest.json")
    for arm, weight in manifest["weights"].items():
        result = read(SOURCE / "evaluation" / arm / "analysis.json")
        if (result["weights_sha256"] != sha(weight)
                or result["predictions_sha256"] != sha(SOURCE / "evaluation" / arm / "predictions.jsonl")
                or result["manifest_sha256"] != sha(SOURCE / "evaluation/manifest.json")):
            raise ValueError("weight/prediction binding differs")
        mapping[weight] = result["weights_sha256"]
    capture = read(TEACHER / "capture_receipt.json")
    teacher_outputs = capture["output_sha256"]
    selected = [TEACHER / n for n in ("events.jsonl", "labeled_predictions.jsonl", "plan.json", "capture_receipt.json")]
    for path in selected:
        mapping[str(path)] = teacher_outputs.get(str(path), sha(path))
    # events.jsonl is generated after capture, also bind it against its diagnostic artifact manifest.
    teacher_manifest = TEACHER / "artifact_sha256.json"
    if teacher_manifest.exists():
        mapping.update(read(teacher_manifest))
    sources = [PROTOCOL, Path(__file__).resolve(), ROOT / "scripts/diagnose_competitive_negative.py",
        ROOT / "src/regionssod/evaluation/error_diagnosis.py",
        ROOT / "src/regionssod/evaluation/precision_recall.py", SOURCE / "artifact_sha256.json",
        SOURCE / "evaluation/artifact_sha256.json", SOURCE / "status.json"]
    mapping.update({str(p): sha(p) for p in sources})
    verify(mapping)
    return mapping


def teacher_risk():
    rows = lines(TEACHER / "events.jsonl")
    result = {}
    for group in ("design", "confirmation"):
        result[group] = {}
        for tag, flag in (("ambiguous", True), ("unambiguous", False)):
            part = [r for r in rows if r["group"] == group and (r["ratio"] >= .25) == flag]
            usable = [r for r in part if r["localizable"] and not r["geometry_ambiguous"]]
            errors = sum(int(r["class_error"]) for r in usable)
            result[group][tag] = {"all": len(part), "localizable_single_class": len(usable),
                "class_errors": errors, "class_error_rate": errors / len(usable) if usable else None,
                "not_localizable": sum(not r["localizable"] for r in part),
                "geometry_ambiguous": sum(r["geometry_ambiguous"] for r in part),
                "true_class_top2": sum(r["true_class_rank"] == 2 for r in usable)}
    return result


def capture_arrays(folder, gt):
    rows = {r["image"]: r for r in lines(folder / "predictions.jsonl")}
    annotations = defaultdict(list)
    for a in gt["annotations"]:
        annotations[a["image_id"]].append(a)
    n_gt = len(gt["annotations"])
    if [a["id"] for a in gt["annotations"]] != list(range(1, n_gt + 1)):
        raise ValueError("consecutive canonical GT identifiers required")
    parts = defaultdict(list)
    hit_scores = {str(i): np.full(n_gt, -1.) for i in IOUS}
    geometry = {k: np.full(n_gt, np.nan) for k in ("iou", "normalized_center_error", "normalized_edge_error", "score")}
    for image in gt["images"]:
        row, truth = rows[image["file_name"]], annotations[image["id"]]
        ids = np.asarray([a["id"] - 1 for a in truth], int)
        truth_boxes = np.asarray([a["bbox"] for a in truth], float).reshape(-1, 4)
        truth_boxes[:, 2:] += truth_boxes[:, :2]
        gt_classes = np.asarray([a["category_id"] - 1 for a in truth], int)
        areas = np.asarray([a["area"] for a in truth])
        cls, scores, boxes = np.asarray(row["classes"], int), np.asarray(row["scores"]), np.asarray(row["xyxy"]).reshape(-1, 4)
        overlaps = box_iou_matrix(boxes, truth_boxes)
        parts["scores"].extend(scores)
        parts["classes"].extend(cls)
        for iou in IOUS:
            key = str(iou)
            local = assign_predictions(cls, scores, boxes, gt_classes, truth_boxes, iou)
            errors = classify_false_predictions(local, overlaps, cls, gt_classes, iou)
            valid = local >= 0
            matched = np.full(len(local), -1, int)
            matched[valid] = ids[local[valid]]
            matched_areas = np.full(len(local), -1.)
            matched_areas[valid] = areas[local[valid]]
            hit_scores[key][matched[valid]] = scores[valid]
            for label, values in (("match", matched), ("errors", errors), ("areas", matched_areas)):
                parts[f"{label}_{key}"].extend(values)
            if iou == .5:
                paired = paired_geometry(boxes[valid], truth_boxes[local[valid]])
                for label, values in paired.items():
                    geometry[label][matched[valid]] = values
                geometry["score"][matched[valid]] = scores[valid]
    result = {k: np.asarray(v) for k, v in parts.items()}
    result.update({f"gt_score_{k}": v for k, v in hit_scores.items()})
    result.update({f"geometry_{k}": v for k, v in geometry.items()})
    return result


def additional_summary(arm, arrays, gt, summary):
    scores = arrays["scores"]
    result = {"scores": {}, "budgets": {}, "per_class_conf025": {}, "oracle_remove_fp": {}}
    categories = gt["categories"]
    counts = summary["gt_counts"]
    for iou in IOUS:
        key = str(iou)
        errors, areas = arrays[f"errors_{key}"], arrays[f"areas_{key}"]
        curve = threshold_curve(scores, areas, counts)
        with np.load(SOURCE / "evaluation" / arm / f"micro_iou{iou}.npz") as stored:
            for k in curve:
                np.testing.assert_array_equal(stored[k], curve[k])
        result["scores"][key] = {"TP": describe(scores[errors == 0]), "FP": describe(scores[errors > 0]),
            "TP_FP_rank_AUROC": rank_auc(scores, errors == 0),
            "by_error": {name: describe(scores[errors == index]) for index, name in enumerate(ERROR_NAMES)}}
        result["budgets"][key] = {"K27237": budget_point(curve, "predictions", 27237),
                                     "FP7889": budget_point(curve, "fp", 7889),
                                     "saved_floor": curve_point(curve, -1)}
        active = scores >= .25
        result["per_class_conf025"][key] = {c["name"]: {
            "predictions": int((active & (arrays["classes"] == c["id"] - 1)).sum()),
            "errors": {name: int((active & (arrays["classes"] == c["id"] - 1) & (errors == j)).sum())
                       for j, name in enumerate(ERROR_NAMES)}} for c in categories}
        if arm == "reference":
            result["oracle_remove_fp"][key] = {}
            for j, name in enumerate(ERROR_NAMES[1:], 1):
                keep = errors != j
                oracle = threshold_curve(scores[keep], areas[keep], counts)
                result["oracle_remove_fp"][key][name] = {
                    "P70": precision_constrained_point(oracle, .7),
                    "P90": precision_constrained_point(oracle, .9)}
    return result


def pair_summary(left, right, summaries, arrays, truth_areas):
    result = {"transitions": {}, "common_small_geometry": {}}
    l, r = arrays[left], arrays[right]
    small = truth_areas < 1024
    for iou in IOUS:
        key = str(iou)
        result["transitions"][key] = {}
        for point in ("conf025", "P70", "P90"):
            thresholds = [summaries[a]["points"][key][point]["operating_point"]["threshold"] for a in (left, right)]
            hits = [v[f"gt_score_{key}"] >= t for v, t in zip((l, r), thresholds)]
            result["transitions"][key][point] = matched_transitions(*hits, small)
    common = small & (l["gt_score_0.5"] >= 0) & (r["gt_score_0.5"] >= 0)
    for name in ("iou", "normalized_center_error", "normalized_edge_error", "score"):
        lv, rv = l[f"geometry_{name}"][common], r[f"geometry_{name}"][common]
        result["common_small_geometry"][name] = {"left": describe(lv), "right": describe(rv), "right_minus_left": describe(rv-lv)}
    return result


def analyze(save):
    gt = read(SOURCE / "evaluation/ground_truth.json")
    summaries, arrays = {}, {}
    result = {"scope": "exploratory; previously inspected development val and open training labels",
        "training_steps": 0, "inference_calls": 0, "hidden_gt_read": False, "models": {},
        "teacher_open_region": teacher_risk()}
    for arm in ARMS:
        folder = SOURCE / "evaluation" / arm
        summary = model_summary(folder, gt)
        summaries[arm] = summary
        array = capture_arrays(folder, gt)
        arrays[arm] = array
        extra = additional_summary(arm, array, gt, summary)
        stored = read(folder / "analysis.json")
        result["models"][arm] = {"errors": summary, "extra": extra,
                                   "coco": stored["coco"], "ultralytics": stored["ultralytics"]}
        cache = OUT / f"{arm}_arrays.npz"
        if save:
            np.savez_compressed(cache, **array)
        else:
            with np.load(cache) as original:
                if set(original.files) != set(array):
                    raise ValueError("array key replay differs")
                for k, value in array.items():
                    np.testing.assert_array_equal(original[k], value)
        print(f"{arm}: replayed both IoUs; fixed points, budgets, errors, scores", flush=True)
    areas = np.asarray([a["area"] for a in gt["annotations"]])
    result["pairs"] = {f"{r}_minus_{l}": pair_summary(l, r, summaries, arrays, areas)
        for l, r in (("baseline", "reference"), ("reference", "candidate"), ("baseline", "candidate"))}
    return json_ready(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    if args.audit:
        verify(read(OUT / "input_sha256.json"))
        verify(read(OUT / "artifact_sha256.json"))
        actual = analyze(False)
        if actual != read(OUT / "analysis.json"):
            raise ValueError("diagnostic JSON replay differs")
        print("Exact JSON/array replay passed; no inference/training.")
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    mapping = inputs()
    OUT.mkdir(parents=True)
    result = analyze(True)
    verify(mapping)
    payloads = {"input_sha256.json": mapping, "analysis.json": result,
        "receipt.json": {"accepted": True, "at": datetime.now().astimezone().isoformat(),
            "input_bindings": len(mapping), "seconds": time.monotonic()-start,
            "training_steps": 0, "inference_calls": 0, "hidden_gt_read": False}}
    for name, value in payloads.items():
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    artifacts = {str(p): sha(p) for p in OUT.iterdir() if p.is_file()}
    (OUT / "artifact_sha256.json").write_text(json.dumps(artifacts, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(payloads["receipt.json"], ensure_ascii=False))


if __name__ == "__main__":
    main()
