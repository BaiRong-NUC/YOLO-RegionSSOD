"""Bound new paired30 checkpoints to full-image AP_small and empirical PR evidence."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from evaluate_paper1_small_pr import ground_truth, INFERENCE, DATA, LEVELS, digest, read, write, verify_hashes
from regionssod.evaluation.precision_recall import (
    assign_predictions, coco_ap, fixed_threshold_point, precision_constrained_point, threshold_curve,
)


def analyze_records(records, gt, counts):
    by_name = {r["image"]: r for r in records}
    if len(records) != 548 or len(by_name) != 548 or set(by_name) != {im["file_name"] for im in gt["images"]}:
        raise ValueError("Incomplete or duplicated validation population")
    detections, ordered = [], []
    annotations = {im["id"]: [] for im in gt["images"]}
    for annotation in gt["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    for im in gt["images"]:
        r = by_name[im["file_name"]]
        if r["shape"] != [im["height"], im["width"]] or len(r["scores"]) > 500:
            raise ValueError("Invalid shape or detection cap")
        ordered.append(r)
        for cls, score, box in zip(r["classes"], r["scores"], r["xyxy"], strict=True):
            if not np.isfinite(box).all() or not .001 <= score <= 1 or not 0 <= cls < 10 or int(cls) != cls:
                raise ValueError("Invalid prediction")
            x1, y1, x2, y2 = box
            detections.append({"image_id": im["id"], "category_id": int(cls) + 1,
                               "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
    coco, coco_curves = coco_ap(gt, detections)
    curves, points, fixed = {}, {}, {}
    for iou in (.5, .75):
        scores, areas = [], []
        for im, r in zip(gt["images"], ordered, strict=True):
            truth = annotations[im["id"]]
            boxes = np.asarray([a["bbox"] for a in truth]).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            matches = assign_predictions(r["classes"], r["scores"], r["xyxy"],
                                         [a["category_id"]-1 for a in truth], boxes, iou)
            matched_area = np.full(len(matches), -1.)
            selected = matches >= 0
            matched_area[selected] = np.asarray([a["area"] for a in truth])[matches[selected]]
            scores.extend(r["scores"])
            areas.extend(matched_area)
        curve = threshold_curve(scores, areas, counts)
        curves[str(iou)] = curve
        points[str(iou)] = {str(p): precision_constrained_point(curve, p) for p in LEVELS}
        fixed[str(iou)] = fixed_threshold_point(curve, .25)
    return {"coco": coco, "precision_constraints": points, "conf025": fixed,
            "prediction_count": len(detections),
            "images_capped_at_500": sum(len(r["scores"]) == 500 for r in records)}, curves, coco_curves


def continuation(reference, candidate):
    def difference(a, b):
        return None if a is None or b is None else 100 * (b-a)

    delta = {
        "mAP50_95": difference(reference["ultralytics"]["metrics/mAP50-95(B)"],
                              candidate["ultralytics"]["metrics/mAP50-95(B)"]),
        "AP_small": difference(reference["coco"]["metrics"]["AP_small"],
                              candidate["coco"]["metrics"]["AP_small"]),
    }
    for p in ("0.7", "0.9"):
        a, b = [r["precision_constraints"]["0.5"][p] for r in (reference, candidate)]
        delta[f"small_at_P{p}"] = difference(a["small_recall"] if a else None,
                                             b["small_recall"] if b else None)
    a, b = [r["conf025"]["0.5"] for r in (reference, candidate)]
    delta["small_conf025"] = difference(a["small_recall"] if a else None, b["small_recall"] if b else None)
    limits = {"mAP50_95": -.1, "AP_small": 0., "small_at_P0.9": .3,
              "small_at_P0.7": 0., "small_conf025": -.2}
    gates = {k: v is not None and bool(np.isfinite(v)) and v >= limits[k] for k, v in delta.items()}
    return {"delta_pp": delta, "gates": gates, "worth_larger_training": all(gates.values()),
            "scope": "single paired student seed,30epochs; development decision, not statistical significance"}


def evaluate(weights, out):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    import ultralytics

    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if ultralytics.__version__ != "8.4.172":
        raise ValueError("Pinned version required")
    gt, hashes, counts = ground_truth()
    hashes.update({str(Path(p).resolve()): digest(p) for p in weights.values()})
    hashes[str(Path(__file__).resolve())] = digest(__file__)
    root = Path(__file__).resolve().parents[1]
    for p in (root / "src/regionssod/evaluation/precision_recall.py",
              root / "scripts/evaluate_paper1_small_pr.py"):
        hashes[str(p)] = digest(p)
    write(out / "manifest.json", {"input_sha256": hashes, "inference": INFERENCE,
                                  "weights": {k: str(v) for k, v in weights.items()}, "counts": counts})
    write(out / "ground_truth.json", gt)

    class CaptureValidator(DetectionValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.captured = []

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                prepared = self._prepare_batch(si, batch)
                native = self.scale_preds(self._prepare_pred(pred), prepared)
                self.captured.append({"image": str(Path(prepared["im_file"]).resolve()),
                    "shape": list(prepared["ori_shape"]), "classes": native["cls"].cpu().tolist(),
                    "scores": native["conf"].cpu().tolist(), "xyxy": native["bboxes"].cpu().tolist()})

        def finalize_metrics(self):
            super().finalize_metrics()
            with (self.save_dir.parent / "predictions.jsonl").open("x", encoding="utf-8") as stream:
                for r in self.captured:
                    stream.write(json.dumps(r, allow_nan=False) + "\n")

    models, plots = {}, {}
    for arm, checkpoint in weights.items():
        start = time.monotonic()
        result = YOLO(str(checkpoint)).val(validator=CaptureValidator, data=str(DATA),
                    project=str(out / arm), name="validator", exist_ok=False, **INFERENCE)
        path = out / arm / "predictions.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        analysis, curves, coco_curves = analyze_records(records, gt, counts)
        analysis.update(ultralytics=result.results_dict, seconds=time.monotonic()-start,
                        weights_sha256=digest(checkpoint), predictions_sha256=digest(path),
                        manifest_sha256=digest(out / "manifest.json"))
        write(out / arm / "analysis.json", analysis)
        for iou, curve in curves.items():
            np.savez_compressed(out / arm / f"micro_iou{iou}.npz", **curve)
        np.savez_compressed(out / arm / "coco_pr.npz", **coco_curves)
        models[arm], plots[arm] = analysis, curves["0.5"]
    verify_hashes(hashes)
    comparison = continuation(models["reference"], models["candidate"])
    write(out / "comparison.json", comparison)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for arm, curve in plots.items():
        axes[0].plot(curve["recall"], curve["precision"], label=arm)
        grid = np.linspace(.5, .95, 91)
        points = [precision_constrained_point(curve, p) for p in grid]
        axes[1].plot(grid, [p["small_recall"] if p else np.nan for p in points], label=arm)
    axes[0].set(xlabel="All-size recall", ylabel="All-size precision")
    axes[1].set(xlabel="Required all-size precision", ylabel="Small recall")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    fig.suptitle("Original P1 + class-score allocation | paired30 | IoU0.50")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"paired_pr.{ext}", dpi=180)
    plt.close(fig)
    artifacts = [p for p in out.rglob("*") if p.is_file()]
    write(out / "artifact_sha256.json", {str(p): digest(p) for p in artifacts})
    return comparison


def audit(out):
    out = Path(out)
    manifest = read(out / "manifest.json")
    verify_hashes(manifest["input_sha256"])
    verify_hashes(read(out / "artifact_sha256.json"))
    gt = read(out / "ground_truth.json")
    models = {}
    for arm, weight in manifest["weights"].items():
        stored = read(out / arm / "analysis.json")
        path = out / arm / "predictions.jsonl"
        if (stored["weights_sha256"] != digest(weight)
                or stored["predictions_sha256"] != digest(path)
                or stored["manifest_sha256"] != digest(out / "manifest.json")):
            raise ValueError("Arm/weight/prediction binding changed")
        replayed, curves, coco_curves = analyze_records(
            [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()], gt, manifest["counts"])
        if any(stored[k] != v for k, v in replayed.items()):
            raise ValueError("AP/PR numeric replay differs")
        for filename, expected in [(f"micro_iou{k}.npz", v) for k, v in curves.items()] + [("coco_pr.npz", coco_curves)]:
            with np.load(out / arm / filename) as cache:
                if set(cache.files) != set(expected) or any(not np.array_equal(cache[k], v) for k, v in expected.items()):
                    raise ValueError("PR curve replay differs")
        models[arm] = stored
    if continuation(models["reference"], models["candidate"]) != read(out / "comparison.json"):
        raise ValueError("Continuation replay differs")
    return {"accepted": True, "images": 548, "models": list(models), "numeric_replay": True}
