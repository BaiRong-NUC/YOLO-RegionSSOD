"""Read frozen scores/predictions only; no model, optimizer, inference or hidden GT."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from regionssod.evaluation.error_diagnosis import (
    ERROR_NAMES, LOST_NAMES, box_iou_matrix, classify_false_predictions,
    gt_candidate_summary, json_ready, lost_gt_reason,
)
from regionssod.evaluation.precision_recall import (
    assign_predictions, fixed_threshold_point, precision_constrained_point, threshold_curve,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/p1_competitive_negative_diagnostic_20261103"
DOC = ROOT / "docs/paper/candidates/competitive_negative/diagnostic_protocol.md"
TEACHER = ROOT / "outputs/p1_class_ambiguity_diagnostic_20261102"
SCORES = ROOT / "outputs/p1_class_scores_20261102"
PAIR = ROOT / "outputs/p1_class_ambiguity_pair30_s0"
FORMAL = ROOT / "outputs/paper1_small_pr_20261102/bound_v2"


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def lines(p):
    with Path(p).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sha(p):
    with Path(p).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(mapping):
    for p, expected in mapping.items():
        if sha(p) != expected:
            raise ValueError(f"Frozen source differs: {p}")


def inputs():
    mapping = {}
    def merge(values):
        for p, v in values.items():
            if p in mapping and mapping[p] != v:
                raise ValueError(f"Conflicting source bindings: {p}")
            mapping[p] = v
    for p in (TEACHER / "plan.json", SCORES / "plan.json", PAIR / "preflight.json",
              PAIR / "evaluation/manifest.json", FORMAL / "manifest.json"):
        merge(read(p)["input_sha256"])
    for p in (PAIR / "artifact_sha256.json", PAIR / "evaluation/artifact_sha256.json",
              FORMAL / "artifact_sha256.json"):
        merge(read(p))
    merge(read(TEACHER / "capture_receipt.json")["output_sha256"])
    receipt = read(SCORES / "receipt.json")
    merge({str(SCORES / "class_scores.jsonl"): receipt["sidecar_sha256"],
           str(SCORES / "plan.json"): receipt["plan_sha256"]})
    if not (read(PAIR / "training_audit.json")["accepted"]
            and read(PAIR / "evaluation_audit.json")["accepted"]
            and read(FORMAL / "independent_audit.json")["cached_replay_exact"]):
        raise ValueError("Missing accepted upstream audit")
    for p in [DOC, Path(__file__), TEACHER / "events.jsonl", TEACHER / "plan.json",
              SCORES / "receipt.json", PAIR / "training_audit.json", PAIR / "evaluation_audit.json",
              ROOT / "src/regionssod/evaluation/error_diagnosis.py",
              ROOT / "src/regionssod/evaluation/precision_recall.py"]:
        merge({str(p): sha(p)})
    verify(mapping)
    return mapping


def teacher_summary():
    from diagnose_p1_class_ambiguity import events
    plan = read(TEACHER / "plan.json")
    rows = events(plan)
    if rows != lines(TEACHER / "events.jsonl"):
        raise ValueError("Teacher event replay mismatch")
    # Bind the full vectors to the same positive order used to create the events.
    positives = [p for image in lines(TEACHER / "labeled_predictions.jsonl")
                 for p in image["candidates"] if p["state"] == "positive"]
    output = {}
    for group in ("design", "confirmation"):
        group_rows = [(r, p) for r, p in zip(rows, positives, strict=True) if r["group"] == group]
        triggered = [(r, p) for r, p in group_rows if r["ratio"] >= .25]
        eligible = [(r, p) for r, p in triggered if r["localizable"] and not r["geometry_ambiguous"]]
        useful = 0
        for r, p in eligible:
            s = np.asarray(p["class_scores"])
            if p["class_id"] != r["class_id"] or p["confidence"] != r["confidence"]:
                raise ValueError("Teacher vector/event identity differs")
            # Stable class-id tie break, same for the proposed action.
            top2 = np.argsort(-s, kind="stable")[1]
            useful += int(top2 == r["target_class"])
        errors = sum(r["class_error"] for r, _ in eligible)
        output[group] = {
            "all_positive": len(group_rows), "all_triggered": len(triggered),
            "triggered_not_localizable": sum(not r["localizable"] for r, _ in triggered),
            "triggered_multi_class_overlap": sum(r["geometry_ambiguous"] for r, _ in triggered),
            "eligible_triggered": len(eligible), "wrong_top1": errors,
            "true_top2": useful, "false_top2": len(eligible) - useful,
            "true_top2_fraction_of_errors": useful / errors if errors else None,
            "true_top2_fraction_of_actions": useful / len(eligible) if eligible else None,
            "random_competitor_expected_true": errors / 9,
            "information_gate": errors >= 30 and useful / errors > .5 if errors else False,
            "by_scale": {scale: {"n": sum(r["scale"] == scale for r, _ in eligible),
                                  "wrong_top1": sum(r["class_error"] for r, _ in eligible if r["scale"] == scale)}
                         for scale in ("small", "medium", "large")},
        }
    selected, scales, pairs = 0, Counter(), Counter()
    all_rows = lines(SCORES / "class_scores.jsonl")
    total, images = 0, 0
    for row in all_rows:
        active = False
        for p in row["positive"]:
            s = np.asarray(p["teacher_class_scores"])
            order = np.argsort(-s, kind="stable")
            if int(order[0]) != p["class_id"] or s[order[0]] != p["confidence"]:
                raise ValueError("Full score top1 identity differs")
            total += 1
            if s[order[1]] / s[order[0]] >= .25:
                selected += 1
                active = True
                scales[p["scale"]] += 1
                pairs[f"{order[0]}->{order[1]}"] += 1
        images += active
    if total != 35067 or len(all_rows) != 4839:
        raise ValueError("Incomplete original pseudo-positive population")
    output["unlabeled_scores_only"] = {"all_positive": total, "triggered": selected,
        "triggered_images": images, "by_scale": dict(scales), "class_pairs": dict(pairs),
        "hidden_gt_read": False}
    return output


def model_summary(folder, gt):
    stored = read(folder / "analysis.json")
    records = lines(folder / "predictions.jsonl")
    by_name = {r["image"]: r for r in records}
    if len(records) != 548 or len(by_name) != 548 or set(by_name) != {i["file_name"] for i in gt["images"]}:
        raise ValueError("Incomplete/duplicated full-image predictions")
    annotations = defaultdict(list)
    for a in gt["annotations"]:
        annotations[a["image_id"]].append(a)
    all_scores, all_errors, all_areas, all_matches = [], {i: [] for i in (.5, .75)}, {i: [] for i in (.5, .75)}, {i: [] for i in (.5, .75)}
    candidates, gt_areas = [], []
    cap_scores = []
    for image in gt["images"]:
        row = by_name[image["file_name"]]
        if row["shape"] != [image["height"], image["width"]] or len(row["scores"]) > 500:
            raise ValueError("Shape or detection cap mismatch")
        truth = annotations[image["id"]]
        boxes = np.array([a["bbox"] for a in truth], float).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        classes = np.array([a["category_id"] - 1 for a in truth], int)
        areas = np.array([a["area"] for a in truth])
        overlap = box_iou_matrix(row["xyxy"], boxes)
        summary = gt_candidate_summary(overlap, row["classes"], row["scores"], classes)
        candidates.extend([{k: v[j] for k, v in summary.items()} for j in range(len(truth))])
        if len(row["scores"]) == 500:
            cap_scores.append(min(row["scores"]))
        all_scores.extend(row["scores"])
        for iou in (.5, .75):
            local = assign_predictions(row["classes"], row["scores"], row["xyxy"], classes, boxes, iou)
            errors = classify_false_predictions(local, overlap, row["classes"], classes, iou)
            matched = local >= 0
            matched_area = np.full(len(local), -1.)
            matched_area[matched] = areas[local[matched]]
            global_ids = np.where(matched, local + len(gt_areas), -1)
            all_matches[iou].extend(global_ids)
            all_errors[iou].extend(errors)
            all_areas[iou].extend(matched_area)
        gt_areas.extend(areas)
    gt_areas, scores = np.array(gt_areas), np.array(all_scores)
    counts = {"small": int((gt_areas < 1024).sum()), "medium": int(((gt_areas >= 1024) & (gt_areas < 9216)).sum()), "large": int((gt_areas >= 9216).sum())}
    result = {"images": len(records), "gt_counts": counts,
              "capped_images": len(cap_scores), "max_cap_tail_score": max(cap_scores, default=None), "points": {}}
    for iou in (.5, .75):
        matches, errors = np.array(all_matches[iou]), np.array(all_errors[iou])
        curve = threshold_curve(scores, all_areas[iou], counts)
        saved_curve = np.load(folder / f"micro_iou{iou}.npz")
        for k in curve:
            np.testing.assert_array_equal(curve[k], saved_curve[k])
        points = {"P70": precision_constrained_point(curve, .7), "P90": precision_constrained_point(curve, .9), "conf025": fixed_threshold_point(curve, .25)}
        result["points"][str(iou)] = {}
        for label, point in points.items():
            old = stored["conf025"][str(iou)] if label == "conf025" else stored["precision_constraints"][str(iou)]["0.7" if label == "P70" else "0.9"]
            if point != old:
                raise ValueError("Saved operating point replay differs")
            if point is None:
                result["points"][str(iou)][label] = None
                continue
            active = scores >= point["threshold"]
            hit = np.zeros(len(gt_areas), bool)
            hit[matches[active & (matches >= 0)]] = True
            reasons = Counter(lost_gt_reason(s, 0, iou, point["threshold"])
                              for j, s0 in enumerate(candidates) if gt_areas[j] < 1024 and not hit[j]
                              for s in [{k: np.array([v]) for k, v in s0.items()}])
            fp = {name: int((active & (errors == j)).sum()) for j, name in enumerate(ERROR_NAMES) if j}
            if sum(fp.values()) != point["fp"] or sum(reasons.values()) != counts["small"] - point["small_tp"]:
                raise ValueError("Nonexhaustive error partition")
            result["points"][str(iou)][label] = {"operating_point": point, "fp": fp,
                "small_fn": {name: reasons[name] for name in LOST_NAMES},
                "cap_tail_ge_threshold": sum(s >= point["threshold"] for s in cap_scores)}
    return result


def analyze():
    gt = read(PAIR / "evaluation/ground_truth.json")
    if gt != read(FORMAL / "ground_truth.json"):
        raise ValueError("Formal and new-pair GT differ")
    result = {"scope": "exploratory; all data previously inspected", "training_steps": 0,
              "inference_calls": 0, "hidden_gt_read": False, "teacher": teacher_summary(), "students": {}}
    for name, folder in [("reference30", PAIR / "evaluation/reference"), ("class_scores30", PAIR / "evaluation/candidate")] + [(k + "_100", FORMAL / k) for k in ("b2_s0", "p1_s0", "b2_s1", "p1_s1")]:
        result["students"][name] = model_summary(folder, gt)
        print(f"Replayed {name}: 548 images, both IoUs, all fixed points", flush=True)
    return json_ready(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        verify(read(OUT / "input_sha256.json"))
        verify(read(OUT / "artifact_sha256.json"))
        actual = analyze()
        if actual != read(OUT / "analysis.json"):
            raise ValueError("New diagnostic replay differs")
        print("Exact replay passed; historical sources and outputs unchanged.")
    else:
        if OUT.exists():
            raise FileExistsError(OUT)
        mapping = inputs()
        OUT.mkdir(parents=True)
        (OUT / "input_sha256.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        result = analyze()
        verify(mapping)
        (OUT / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        receipt = {"accepted": True, "created_at": datetime.now().astimezone().isoformat(),
                   "input_hashes": len(mapping), "training_steps": 0, "inference_calls": 0}
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        paths = [OUT / n for n in ("input_sha256.json", "analysis.json", "receipt.json")]
        (OUT / "artifact_sha256.json").write_text(json.dumps({str(p): sha(p) for p in paths}, indent=2) + "\n", encoding="utf-8")
        print(receipt)


if __name__ == "__main__":
    main()
