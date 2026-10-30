"""Post-hoc NWD tradeoff diagnosis from saved full-val predictions; CPU only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from regionssod.data.annotations import iter_images, label_path_for_image, read_label_file
from regionssod.data.dataset import load_source_dataset
from regionssod.evaluation.tradeoff import curve_point, operating_curve, prediction_matches
from run_p1_scale_seed_pair import ROOT, digest, read, write
from screen_p1_small_nwd import OUT, RECALL, REF_RECALL, REF_VAL, VAL

DESTINATION = ROOT / "outputs/screens/p1_small_nwd_tradeoff_20261030_v3"
IOUS = (.5, .75, .9)
THRESHOLDS = (.25, .3, .4, .5, .6, .7, .8, .9)


def collect_predictions(directory, source, hashes):
    summary = read(directory / "metrics.json")
    if (summary["images"] != 548 or summary["conf"] != .25 or summary["imgsz"] != 960
            or summary["match_iou"] != .5 or summary["nms_iou"] != .7
            or summary["max_det"] != 500):
        raise ValueError("outside cached full-val recall protocol")
    names = (directory / "images.txt").read_text(encoding="utf-8").splitlines()
    expected = [str(p.resolve()) for p in iter_images(source.images_val)]
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if names != expected or [r["image"] for r in rows] != names or len(names) != 548:
        raise ValueError("incomplete or reordered full-val predictions")
    scores, matches = [], [[] for _ in IOUS]
    gt_counts = np.zeros(3, dtype=int)
    for row in rows:
        image = Path(row["image"])
        label = label_path_for_image(image, source.images_val, source.labels_val)
        if not label.is_file():
            raise FileNotFoundError(label)
        gt = read_label_file(label)
        gt_boxes = np.array([[p.x, p.y, p.w, p.h] for p in gt]).reshape(-1, 4)
        with Image.open(image) as pixels:
            width, height = pixels.size
        areas = gt_boxes[:, 2] * width * gt_boxes[:, 3] * height
        sizes = np.where(areas < 1024, 0, np.where(areas < 9216, 1, 2))
        gt_counts += np.bincount(sizes, minlength=3)
        assignment = prediction_matches(
            row["classes"], row["confidences"], row["xywhn"],
            [p.class_id for p in gt], gt_boxes, IOUS)
        if len(row["classes"]) > 500:
            raise ValueError("cache exceeds max_det")
        for index, match in enumerate(assignment):
            outcome = np.full(len(match), -1, dtype=int)
            valid = match >= 0
            outcome[valid] = sizes[match[valid]]
            matches[index].append(outcome)
        scores.extend(row["confidences"])
        for path in (image, label):
            if str(path) not in hashes:
                hashes[str(path)] = digest(path)
    curves = [operating_curve(scores, np.concatenate(match), gt_counts, confidence_floor=.25)
              for match in matches]
    if gt_counts.tolist() != [26577, 11115, 1067]:
        raise ValueError("full-val GT counts changed")
    point = curve_point(curves[0], threshold=.25)
    if point is None or point["matched"] != [summary["per_size"][s]["matched"]
                                            for s in ("small", "medium", "large")]:
        raise ValueError("new matcher differs from accepted cached recall")
    for filename in ("metrics.json", "images.txt", "predictions.jsonl"):
        path = directory / filename
        hashes[str(path)] = digest(path)
    return curves


def build():
    hashes = {}
    source_path = Path(read(REF_RECALL / "metrics.json")["data"])
    source = load_source_dataset(source_path)
    if read(RECALL / "metrics.json")["data"] != str(source_path):
        raise ValueError("different source data")
    audit = read(OUT / "independent_audit.json")
    if not audit["accepted"] or audit["worth_full30_confirmation"]:
        raise ValueError("expected accepted but continuation-failed original screen")
    print("Replaying cached reference predictions", flush=True)
    reference = collect_predictions(REF_RECALL, source, hashes)
    print("Replaying cached NWD predictions", flush=True)
    candidate = collect_predictions(RECALL, source, hashes)
    fixed, budgets = [], []
    for iou, old, new in zip(IOUS, reference, candidate):
        fixed.extend({"match_iou": iou, "threshold": threshold,
                      "reference": curve_point(old, threshold=threshold),
                      "candidate": curve_point(new, threshold=threshold)}
                     for threshold in THRESHOLDS)
        for threshold in (.25, .5):
            anchor = curve_point(old, threshold=threshold)
            if anchor is None:
                raise ValueError("reference operating point is not attainable")
            selected = curve_point(new, fp_budget=anchor["fp"])
            budgets.append({
                "match_iou": iou, "reference_threshold": threshold,
                "fp_budget": anchor["fp"], "reference": anchor, "candidate": selected,
                "small_recall_delta_pp": (
                    100 * (selected["recall"][0] - anchor["recall"][0])
                    if selected else None),
                "selection_scope": "post-hoc val-GT operating-point diagnostic; not a deployment threshold",
            })
    before, after = read(REF_VAL / "metrics.json"), read(VAL / "metrics.json")
    if before["class_indices"] != list(range(10)) or after["class_indices"] != list(range(10)):
        raise ValueError("expected all ten AP classes")
    class_ap = [{"class_id": index, "name": before["names"][str(index)],
                 "reference_ap50_95": old, "candidate_ap50_95": new,
                 "delta_pp": 100 * (new - old)}
                for index, old, new in zip(before["class_indices"],
                                          before["class_maps"], after["class_maps"])]
    map_delta = 100 * (after["results"]["metrics/mAP50-95(B)"]
                       - before["results"]["metrics/mAP50-95(B)"])
    if not np.isclose(np.mean([r["delta_pp"] for r in class_ap]), map_delta, rtol=0, atol=1e-12):
        raise ValueError("class AP deltas do not reconstruct aggregate mAP")
    for path in (source_path, REF_VAL / "metrics.json", VAL / "metrics.json",
                 OUT / "comparison.json", OUT / "independent_audit.json",
                 Path(__file__).resolve(), ROOT / "src/regionssod/evaluation/tradeoff.py",
                 ROOT / "src/regionssod/evaluation/size_recall.py",
                 ROOT / "docs/paper/small_nwd_tradeoff_protocol.md"):
        hashes[str(path)] = digest(path)
    return {
        "scope": "post-hoc single-seed five-epoch subset diagnostic; no inference or training",
        "new_training_epochs": 0, "new_inference_images": 0,
        "confidence_floor": .25, "full_val_images": 548,
        "matching": "confidence-descending class-aware greedy, all GT sizes jointly",
        "fp_scope": "all unmatched cached detections against repository YOLO GT, all sizes jointly",
        "limitations": [
            "Caches omit scores below .25; full PR or AP-by-IoU cannot be reconstructed.",
            "FP-constrained thresholds use validation GT post hoc; not confirmatory evidence or deployment settings.",
            "NMS .7/max_det500 are inherited; no alternative NMS or max_det is evaluated.",
            "Strict-IoU recall is not official AP_small or AR500 and cannot identify the causal source of AP loss.",
            "Original continuation failure stays unchanged; no new training gate is introduced.",
        ],
        "class_ap50_95": class_ap, "map_delta_pp": map_delta,
        "fixed_thresholds": fixed, "equal_fp_diagnostics": budgets,
        "sha256": hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = DESTINATION / "analysis.json"
    if args.check:
        expected = read(path)
        for filename, checksum in expected["sha256"].items():
            if digest(filename) != checksum:
                raise ValueError(f"changed diagnostic input: {filename}")
        if build() != expected:
            raise ValueError("tradeoff replay differs")
        print(f"Verified {len(expected['sha256'])} input hashes and exact diagnostic replay")
    else:
        if DESTINATION.exists():
            raise FileExistsError("existing tradeoff output; use --check instead")
        result = build()
        DESTINATION.mkdir(parents=True)
        write(path, result)
        for point in result["equal_fp_diagnostics"]:
            print({key: point[key] for key in (
                "match_iou", "reference_threshold", "fp_budget", "small_recall_delta_pp")})


if __name__ == "__main__":
    main()
