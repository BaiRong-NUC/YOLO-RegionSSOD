"""Read-only inventory of complete pilots and cached full-image predictions."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from regionssod.data.annotations import label_path_for_image, read_label_file
from regionssod.data.dataset import load_source_dataset
from regionssod.evaluation.size_recall import match_by_confidence
from run_p1_scale_seed_pair import ROOT, digest, read, write

OUT = ROOT / "outputs/training_investment_20261031"
KEYS = (
    "b1_pilot30_s0", "b2_pilot30_s0", "b2_pilot30_s1",
    "p1_pilot30_s0", "p1_pilot30_s1",
    "p1_global_t050_pilot30_s0", "p1_global_t050_pilot30_s1",
    "p1_no_ignore_pilot30_s0", "p1_no_soft_weights_pilot30_s0",
    "p1_small_nwd_pilot30_s0", "p1_regression_confidence_valfix_pilot30_s0",
    "p1_coverage_guard_pilot30_s0",
)
SIZES = ("small", "medium", "large")


def paired_counts(before, after, masks):
    result = {}
    for size, mask in masks.items():
        gained = [int((~a & b & m).sum()) for a, b, m in zip(before, after, mask)]
        lost = [int((a & ~b & m).sum()) for a, b, m in zip(before, after, mask)]
        delta = np.asarray(gained) - lost
        result[size] = {
            "gained": sum(gained), "lost": sum(lost), "net": int(delta.sum()),
            "images_helped": int((delta > 0).sum()),
            "images_harmed": int((delta < 0).sum()),
            "images_tied": int((delta == 0).sum()),
        }
    return result


def main():
    reference = ROOT / "runs/size_recall/p1_pilot30_s0"
    protocol = read(reference / "metrics.json")
    images = (reference / "images.txt").read_text(encoding="utf-8").splitlines()
    source = load_source_dataset(Path(protocol["data"]))
    gt_classes, gt_boxes = [], []
    masks = {size: [] for size in SIZES}
    sources = {str(Path(__file__).resolve()): digest(__file__)}
    for name in images:
        image = Path(name)
        label = label_path_for_image(image, source.images_val, source.labels_val)
        gt = read_label_file(label)
        boxes = np.asarray([[g.x, g.y, g.w, g.h] for g in gt]).reshape(-1, 4)
        gt_boxes.append(boxes)
        gt_classes.append([g.class_id for g in gt])
        with Image.open(image) as pixels:
            width, height = pixels.size
        area = boxes[:, 2] * width * boxes[:, 3] * height
        for size, mask in zip(SIZES, (area < 1024, (area >= 1024) & (area < 9216), area >= 9216)):
            masks[size].append(mask)
        sources[str(label)] = digest(label)
    totals = {size: sum(int(mask.sum()) for mask in group) for size, group in masks.items()}
    if len(images) != 548 or list(totals.values()) != [26577, 11115, 1067]:
        raise ValueError("unexpected validation population")
    records, matches = {}, {}
    for key in KEYS:
        directory = ROOT / "runs/size_recall" / key
        metrics = read(directory / "metrics.json")
        for field in ("data", "imgsz", "conf", "match_iou", "nms_iou", "max_det", "matching"):
            if metrics[field] != protocol[field]:
                raise ValueError(f"different recall protocol: {key}/{field}")
        if (directory / "images.txt").read_text(encoding="utf-8").splitlines() != images:
            raise ValueError("different image order")
        weight = Path(metrics["weights"])
        if digest(weight) != metrics["weights_sha256"]:
            raise ValueError(f"changed evaluated checkpoint: {key}")
        run = weight.parents[1]
        with (run / "results.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if ([int(float(row["epoch"])) for row in rows] != list(range(1, 31))
                or not all(math.isfinite(float(v)) for row in rows for v in row.values())):
            raise ValueError(f"incomplete pilot: {key}")
        val = ROOT / "runs/val" / f"{run.name}_val960/metrics.json"
        if not val.exists():
            val = run / "validation/metrics.json"
        validation = read(val)
        if Path(validation["weights"]) != weight or validation["imgsz"] != 960:
            raise ValueError("validation weight or resolution mismatch")
        prediction_path = directory / "predictions.jsonl"
        predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
        if [item["image"] for item in predictions] != images:
            raise ValueError("incomplete prediction cache")
        by_iou = {}
        for iou in (.5, .75):
            by_iou[str(iou)] = [
                match_by_confidence(p["classes"], p["confidences"], p["xywhn"], cls, box, iou)
                for p, cls, box in zip(predictions, gt_classes, gt_boxes)
            ]
        matched = {
            size: sum(int((m & mask).sum()) for m, mask in zip(by_iou["0.5"], group))
            for size, group in masks.items()
        }
        if any(matched[size] != metrics["per_size"][size]["matched"] for size in SIZES):
            raise ValueError(f"recall replay mismatch: {key}")
        count = sum(len(p["classes"]) for p in predictions)
        tp = sum(matched.values())
        records[key] = {
            "run": str(run), "epochs": 30, "student_seconds": float(rows[-1]["time"]),
            "weights_sha256": metrics["weights_sha256"],
            "mAP50": validation["results"]["metrics/mAP50(B)"],
            "mAP50_95": validation["results"]["metrics/mAP50-95(B)"],
            "class_maps": validation["class_maps"],
            "matched": matched,
            "fixed_point": {
                "predictions": count, "tp": tp, "unmatched_predictions": count - tp,
                "precision": tp / count, "recall": tp / sum(totals.values()),
                "f1": 2 * tp / (count + sum(totals.values())),
            },
            "matched_iou75": {
                size: sum(int((m & mask).sum()) for m, mask in zip(by_iou["0.75"], group))
                for size, group in masks.items()
            },
        }
        matches[key] = by_iou["0.5"]
        for path in (directory / "metrics.json", directory / "images.txt", prediction_path,
                     run / "results.csv", run / "args.yaml", val):
            sources[str(path)] = digest(path)
        print(f"Replayed {key}: {tp} matches / {count} predictions", flush=True)
    pairs = {}
    for key in KEYS:
        seed = key.rsplit("_s", 1)[1]
        before = f"p1_pilot30_s{seed}"
        if key == before:
            before = f"b2_pilot30_s{seed}"
        if key == before:
            continue
        pairs[key] = {
            "reference": before, "gt_turnover": paired_counts(matches[before], matches[key], masks),
            "delta_mAP50_95_pp": 100 * (records[key]["mAP50_95"] - records[before]["mAP50_95"]),
            "delta_unmatched_predictions": (
                records[key]["fixed_point"]["unmatched_predictions"]
                - records[before]["fixed_point"]["unmatched_predictions"]),
        }
    coverage = "p1_coverage_guard_pilot30_s0"
    before = matches["p1_pilot30_s0"]
    after = matches[coverage]
    delta = np.asarray([int(((b.astype(int) - a) * mask).sum())
                        for a, b, mask in zip(before, after, masks["small"])])
    denominators = np.asarray([int(mask.sum()) for mask in masks["small"]])
    samples = np.random.default_rng(20261031).integers(0, len(images), (2000, len(images)))
    bootstrap = 100 * delta[samples].sum(axis=1) / denominators[samples].sum(axis=1)
    pairs[coverage]["small_image_bootstrap_percentile95_pp"] = np.quantile(bootstrap, [.025, .975]).tolist()
    summary = {
        "scope": "retrospective development evidence; cached conf>=0.25; no new training/inference",
        "limits": [
            "unmatched predictions are diagnostic FP under this matcher, not official VisDrone FP",
            "IoU75 recall is not AP75; cached predictions cannot reconstruct full AP or low-conf PR",
            "image bootstrap is conditional on fitted weights; not seed or split uncertainty",
            "multiple exploratory candidates and shared val; no independent confirmation claim",
        ],
        "gt": totals, "records": records, "pairs": pairs, "sha256": sources,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "inventory.json", summary)
    print(f"Saved {OUT / 'inventory.json'}", flush=True)


if __name__ == "__main__":
    main()
