"""Evaluate four accepted frozen students; never train or change checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

from regionssod.data.annotations import iter_images, label_path_for_image, read_label_file
from regionssod.data.dataset import load_source_dataset
from regionssod.evaluation.precision_recall import (
    assign_predictions, coco_ap, fixed_threshold_point, precision_constrained_point, threshold_curve,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper1_small_pr_20261102"
DATA = ROOT / "data/visdrone_area10/dataset.yaml"
SOURCE = Path("D:/Python/YOLO/data/visdrone/dataset.yaml")
PROTOCOL = ROOT / "docs/paper/small_pr_protocol.md"
LEVELS = (.5, .6, .7, .8, .9)
KEYS = tuple(f"{arm}_s{seed}" for seed in (0, 1) for arm in ("b2", "p1"))
INFERENCE = dict(imgsz=960, batch=8, workers=0, device="0", conf=.001, iou=.7,
                 max_det=500, rect=True, quantize=None, augment=False, plots=False,
                 save_json=False, verbose=False, split="val")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def locations(key):
    arm, seed = key.split("_s")
    return {"weights": ROOT / f"runs/partial10_{arm}_formal100_s{seed}/weights/best.pt",
            "original": ROOT / f"outputs/paper1_{arm}_formal100_s{seed}",
            "cache": OUT / key}


def ground_truth():
    # The full-source loader expects image directories for BOTH splits. The
    # partial YAML has a training-list file and is only passed to Ultralytics.
    source = load_source_dataset(SOURCE)
    images = iter_images(source.images_val)
    dataset = {"info": {"scope": "current YOLO validation labels; not official VisDrone"},
               "images": [], "annotations": [],
               "categories": [{"id": k + 1, "name": v} for k, v in source.names.items()]}
    hashes = {str(DATA): digest(DATA), str(SOURCE): digest(SOURCE)}
    counts = Counter(small=0, medium=0, large=0)
    for image_id, image in enumerate(images, 1):
        label = label_path_for_image(image, source.images_val, source.labels_val)
        hashes[str(image)] = digest(image)
        hashes[str(label)] = digest(label)
        with Image.open(image) as pixels:
            width, height = pixels.size
        dataset["images"].append({"id": image_id, "file_name": str(image), "width": width, "height": height})
        for box in read_label_file(label):
            w, h = box.w * width, box.h * height
            area = w * h
            counts["small" if area < 1024 else "medium" if area < 9216 else "large"] += 1
            dataset["annotations"].append({"id": len(dataset["annotations"]) + 1, "image_id": image_id,
                "category_id": box.class_id + 1, "bbox": [(box.x - box.w / 2) * width,
                (box.y - box.h / 2) * height, w, h], "area": area, "iscrowd": 0})
    if len(images) != 548 or dict(counts) != {"small": 26577, "medium": 11115, "large": 1067}:
        raise ValueError(f"validation population changed: {len(images)}, {counts}")
    return dataset, hashes, dict(counts)


def verify_hashes(hashes):
    for filename, expected in hashes.items():
        if digest(filename) != expected:
            raise ValueError(f"frozen source changed: {filename}")


def make_manifest():
    gt, hashes, counts = ground_truth()
    for path in (Path(__file__).resolve(), PROTOCOL,
                 ROOT / "src/regionssod/evaluation/precision_recall.py",
                 ROOT / "src/regionssod/data/annotations.py", ROOT / "src/regionssod/data/dataset.py"):
        hashes[str(path)] = digest(path)
    for key in KEYS:
        loc = locations(key)
        audit = read(loc["original"] / "independent_audit.json")
        if audit.get("accepted") is not True or audit.get("epochs") != 100:
            raise ValueError(f"unaccepted frozen model: {key}")
        artifact = read(loc["original"] / "artifact_sha256.json")
        if artifact[str(loc["weights"])] != digest(loc["weights"]):
            raise ValueError(f"weight differs from accepted artifact: {key}")
        for path in (loc["weights"], loc["original"] / "metrics.json"):
            hashes[str(path)] = digest(path)
    return {"created_at": now(), "inference": INFERENCE, "models": KEYS,
            "versions": {name: version(name) for name in ("ultralytics", "torch", "numpy", "pycocotools", "matplotlib")},
            "gt_counts_strict": counts, "input_sha256": hashes}, gt


def predict():
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator

    if version("ultralytics") != "8.4.172":
        raise ValueError("use pinned yolo environment")
    if (OUT / "manifest.json").exists() or any((locations(key)["cache"] / "predictions.jsonl").exists() for key in KEYS):
        raise FileExistsError("existing evaluation: analyze or audit; do not repeat inference")
    manifest, gt = make_manifest()
    write(OUT / "manifest.json", manifest)
    write(OUT / "ground_truth.json", gt)

    class CaptureValidator(DetectionValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.captured = []

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                native = self.scale_preds(self._prepare_pred(pred), pbatch)
                self.captured.append({"image": str(Path(pbatch["im_file"]).resolve()),
                    "shape": list(pbatch["ori_shape"]), "classes": native["cls"].cpu().tolist(),
                    "scores": native["conf"].cpu().tolist(), "xyxy": native["bboxes"].cpu().tolist()})

        def finalize_metrics(self):
            super().finalize_metrics()
            path = self.save_dir.parent / "predictions.jsonl"
            with path.open("x", encoding="utf-8") as handle:
                for item in self.captured:
                    handle.write(json.dumps(item, allow_nan=False) + "\n")

    for key in KEYS:
        loc = locations(key)
        started = time.monotonic()
        write(OUT / "status.json", {"stage": "inference", "model": key, "at": now()})
        print(f"BEGIN {key} {now()}", flush=True)
        model = YOLO(str(loc["weights"]))
        result = model.val(validator=CaptureValidator, data=str(DATA), project=str(loc["cache"]),
                           name="validator", exist_ok=False, **INFERENCE)
        original = read(loc["original"] / "metrics.json")["metrics"]
        deltas = {"mAP50": float(result.box.map50) - original["mAP50"],
                  "mAP50_95": float(result.box.map) - original["mAP50_95"]}
        values = {"model": key, "completed_at": now(), "elapsed_seconds": time.monotonic() - started,
                  "ultralytics": result.results_dict, "replay_delta_absolute": deltas,
                  "replay_within_1e_6": all(abs(value) <= 1e-6 for value in deltas.values()),
                  "weights_sha256": digest(loc["weights"])}
        write(loc["cache"] / "inference.json", values)
        print(f"END {key}: {values}", flush=True)
        if not values["replay_within_1e_6"]:
            raise ValueError(f"original validation metrics did not replay for {key}; inspect before continuing")
        del model
    verify_hashes(manifest["input_sha256"])
    write(OUT / "status.json", {"stage": "inference_complete", "at": now(), "models": KEYS})


def load_predictions(key, gt):
    loc = locations(key)
    records = [json.loads(line) for line in (loc["cache"] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 548 or len({row["image"] for row in records}) != 548:
        raise ValueError(f"incomplete/duplicated image predictions: {key}")
    by_name = {row["image"]: row for row in records}
    if set(by_name) != {image["file_name"] for image in gt["images"]}:
        raise ValueError("prediction population differs from frozen GT")
    ordered, detections = [], []
    for image in gt["images"]:
        row = by_name[image["file_name"]]
        if row["shape"] != [image["height"], image["width"]] or len(row["scores"]) > 500:
            raise ValueError("prediction shape or max_det differs")
        if not len(row["scores"]) == len(row["xyxy"]) == len(row["classes"]):
            raise ValueError("prediction array lengths differ")
        ordered.append(row)
        for cls, score, box in zip(row["classes"], row["scores"], row["xyxy"]):
            if not np.isfinite(box).all() or not .001 <= score <= 1 or not 0 <= cls < 10:
                raise ValueError("invalid native prediction")
            x1, y1, x2, y2 = box
            detections.append({"image_id": image["id"], "category_id": int(cls) + 1,
                               "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    return ordered, detections


def analyze_model(key, gt, counts, save):
    records, detections = load_predictions(key, gt)
    coco, coco_curves = coco_ap(gt, detections)
    annotations = {image["id"]: [] for image in gt["images"]}
    for annotation in gt["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    curves, points, fixed = {}, {}, {}
    for iou in (.5, .75):
        scores, matched_areas = [], []
        for image, row in zip(gt["images"], records):
            truth = annotations[image["id"]]
            boxes = np.asarray([item["bbox"] for item in truth]).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            match = assign_predictions(row["classes"], row["scores"], row["xyxy"],
                                       [item["category_id"] - 1 for item in truth], boxes, iou)
            areas = np.full(len(match), -1.)
            valid = match >= 0
            areas[valid] = np.asarray([item["area"] for item in truth])[match[valid]]
            scores.extend(row["scores"])
            matched_areas.extend(areas)
        curve = threshold_curve(scores, matched_areas, counts)
        curves[str(iou)] = curve
        points[str(iou)] = {str(p): precision_constrained_point(curve, p) for p in LEVELS}
        fixed[str(iou)] = fixed_threshold_point(curve, .25)
        if save:
            np.savez_compressed(locations(key)["cache"] / f"micro_iou{iou}.npz", **curve)
    old = read(ROOT / f"runs/size_recall/{key.replace('_s', '_formal100_s')}/metrics.json")
    compatibility = {size: fixed["0.5"][f"{size}_tp"] - old["per_size"][size]["matched"]
                     for size in counts}
    result = {"coco": coco, "precision_constraints": points, "conf025": fixed,
              "old_predict_recall_count_deltas": compatibility,
              "prediction_count": len(detections),
              "images_capped_at_500": sum(len(row["scores"]) == 500 for row in records)}
    if save:
        np.savez_compressed(locations(key)["cache"] / "coco_pr.npz", **coco_curves)
        write(locations(key)["cache"] / "analysis.json", result)
        path = locations(key)["cache"] / "precision_constraints.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["iou", "precision_constraint", "achieved_precision", "small_recall", "threshold", "tp", "unmatched_predictions"])
            for iou, group in points.items():
                for p, point in group.items():
                    writer.writerow([iou, p] + ([point[k] for k in ("precision", "small_recall", "threshold", "tp", "fp")] if point else [None] * 5))
    return result, curves, coco_curves


def aggregate(models, curves):
    pairs = {}
    for seed in (0, 1):
        b2, p1 = models[f"b2_s{seed}"], models[f"p1_s{seed}"]
        delta_ap = {key: 100 * (p1["coco"]["metrics"][key] - b2["coco"]["metrics"][key])
                    for key in b2["coco"]["metrics"] if b2["coco"]["metrics"][key] is not None
                    and p1["coco"]["metrics"][key] is not None}
        constrained = {}
        for iou in ("0.5", "0.75"):
            constrained[iou] = {}
            for p in map(str, LEVELS):
                a, b = b2["precision_constraints"][iou][p], p1["precision_constraints"][iou][p]
                constrained[iou][p] = None if a is None or b is None else 100 * (b["small_recall"] - a["small_recall"])
        anchor = b2["conf025"]["0.5"]
        anchored_p1 = precision_constrained_point(curves[f"p1_s{seed}"]["0.5"], anchor["precision"])
        pairs[str(seed)] = {"delta_coco_pp": delta_ap, "delta_small_recall_at_precision_pp": constrained,
            "b2_conf025_precision_anchor": {"b2": anchor, "p1": anchored_p1,
                "delta_small_recall_pp": None if anchored_p1 is None else 100 * (anchored_p1["small_recall"] - anchor["small_recall"])},
            "delta_per_class_AP_small_pp": {name: None if a is None or p1["coco"]["per_class_AP_small"][name] is None else
                100 * (p1["coco"]["per_class_AP_small"][name] - a) for name, a in b2["coco"]["per_class_AP_small"].items()}}
    summary = {name: {"mean_pp": float(np.mean([pairs[str(seed)]["delta_coco_pp"][name] for seed in (0, 1)])),
                      "sample_sd_pp": float(np.std([pairs[str(seed)]["delta_coco_pp"][name] for seed in (0, 1)], ddof=1))}
               for name in pairs["0"]["delta_coco_pp"]}
    return {"models": models, "pairs": pairs, "paired_coco_summary": summary,
            "scope": "retrospective fixed-val evaluation; no training; COCO-style on current YOLO GT, maxDet500",
            "limits": ["not official VisDrone AP; different matching/interpolation from Ultralytics",
                "confidence>=0.001, NMS IoU0.7 and at most500 total detections/image",
                "precision constraints/thresholds estimated on this validation set, not an independent deployment policy",
                "same split0 and teacher; two student seeds do not prove significance or external generalization",
                "seed0 B2 and seed1 P1 resumed; no bitwise uninterrupted equivalence", "historical Gate3/4 unchanged"]}


def plot_results(curves, coco_curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = {"b2": "#0072B2", "p1": "#D55E00"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for seed in (0, 1):
        for arm in ("b2", "p1"):
            key = f"{arm}_s{seed}"
            curve = curves[key]["0.5"]
            # Plot every threshold for the micro curve. The COCO panel is the
            # standard101-point macro interpolation, labelled separately.
            axes[seed, 0].plot(curve["recall"], curve["precision"], color=colors[arm], lw=1.5, label=arm.upper())
            axes[seed, 1].plot(np.linspace(0, 1, 101), coco_curves[key]["small_iou50"], color=colors[arm], lw=1.5)
            grid = np.linspace(.5, .95, 91)
            values = [precision_constrained_point(curve, p) for p in grid]
            axes[seed, 2].plot(grid, [v["small_recall"] if v else np.nan for v in values], color=colors[arm], lw=1.5)
        titles = ("All-size micro PR, IoU0.50", "Small COCO-style macro PR, IoU0.50", "Small recall at global precision constraint")
        for col, title in enumerate(titles):
            ax = axes[seed, col]
            ax.set_title(f"Student seed{seed}: {title}", fontsize=10)
            ax.grid(alpha=.25)
            ax.set_xlabel("Recall" if col < 2 else "Required all-size precision")
            ax.set_ylabel("Precision" if col < 2 else "Small recall")
            ax.set_xlim((0, 1) if col < 2 else (.5, .95))
            ax.set_ylim(0, 1)
        axes[seed, 0].legend()
    fig.suptitle("Frozen B2/P1,100 epochs |548 full validation images |conf>=0.001,maxDet500\n"
                 "COCO-style on YOLO labels; empirical precision constraints; no new training", fontsize=12)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"small_pr_comparison.{ext}", dpi=180)
    plt.close(fig)


def analyze():
    manifest = read(OUT / "manifest.json")
    verify_hashes(manifest["input_sha256"])
    gt = read(OUT / "ground_truth.json")
    models, curves, coco_curves = {}, {}, {}
    for key in KEYS:
        print(f"ANALYZE {key} {now()}", flush=True)
        models[key], curves[key], coco_curves[key] = analyze_model(key, gt, manifest["gt_counts_strict"], True)
    result = aggregate(models, curves)
    areas = np.asarray([item["area"] for item in gt["annotations"]])
    result["gt_counts_coco_inclusive"] = {"small": int((areas <= 1024).sum()),
        "medium": int(((areas >= 1024) & (areas <= 9216)).sum()), "large": int((areas >= 9216).sum())}
    result["gt_counts_diagnostic_strict"] = manifest["gt_counts_strict"]
    write(OUT / "comparison.json", result)
    plot_results(curves, coco_curves)
    artifacts = [OUT / "ground_truth.json", OUT / "comparison.json"]
    for key in KEYS:
        artifacts.extend(locations(key)["cache"] / name for name in
                         ("predictions.jsonl", "inference.json", "analysis.json", "coco_pr.npz",
                          "micro_iou0.5.npz", "micro_iou0.75.npz", "precision_constraints.csv"))
    artifacts.extend(OUT / f"small_pr_comparison.{ext}" for ext in ("png", "pdf", "svg"))
    write(OUT / "artifact_sha256.json", {str(path): digest(path) for path in artifacts})
    write(OUT / "status.json", {"stage": "analysis_complete", "at": now()})
    print(json.dumps(result["paired_coco_summary"], ensure_ascii=False), flush=True)


def audit():
    manifest = read(OUT / "manifest.json")
    verify_hashes(manifest["input_sha256"])
    verify_hashes(read(OUT / "artifact_sha256.json"))
    gt, _, counts = ground_truth()
    if gt != read(OUT / "ground_truth.json"):
        raise ValueError("COCO GT conversion did not replay")
    models, curves = {}, {}
    for key in KEYS:
        models[key], curves[key], _ = analyze_model(key, gt, counts, False)
    result = aggregate(models, curves)
    areas = np.asarray([item["area"] for item in gt["annotations"]])
    result["gt_counts_coco_inclusive"] = {"small": int((areas <= 1024).sum()),
        "medium": int(((areas >= 1024) & (areas <= 9216)).sum()), "large": int((areas >= 9216).sum())}
    result["gt_counts_diagnostic_strict"] = counts
    if result != read(OUT / "comparison.json"):
        raise ValueError("cached COCO/PR analysis did not replay exactly")
    report = {"accepted": True, "at": now(), "models": KEYS, "images_per_model": 548,
              "input_hashes": len(manifest["input_sha256"]), "cached_replay_exact": True,
              "training_steps": 0, "inference_repeated_by_audit": False}
    write(OUT / "independent_audit.json", report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predict", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    mode.add_argument("--audit", action="store_true")
    options = parser.parse_args()
    started = time.monotonic()
    try:
        if options.predict:
            predict()
        elif options.analyze:
            analyze()
        else:
            audit()
    except Exception:
        write(OUT / f"failure_{'predict' if options.predict else 'analyze' if options.analyze else 'audit'}.json",
              {"at": now(), "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
