"""Replay the formal seed0 pair to assess benefits, errors and training cost."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_training_investment import paired_counts
from regionssod.data.annotations import label_path_for_image, read_label_file
from regionssod.data.dataset import load_source_dataset
from regionssod.evaluation.size_recall import match_by_confidence
from run_paper1_full100 import ARMS, paths
from run_p1_scale_seed_pair import ROOT, digest, read, write

OUT = ROOT / "outputs/paper1_full100_effects_s0"


def main():
    sources = {str(Path(__file__).resolve()): digest(__file__)}
    summaries, predictions = {}, {}
    names = None
    protocol = None
    for arm in ARMS:
        loc = paths(arm)
        audit = read(loc["out"] / "independent_audit.json")
        if audit.get("accepted") is not True or audit.get("epochs") != 100:
            raise ValueError("both formal arms must pass independent audit")
        for path, expected in read(loc["out"] / "artifact_sha256.json").items():
            if digest(path) != expected:
                raise ValueError(f"changed formal artifact: {path}")
        metrics = read(loc["recall"] / "metrics.json")
        images = (loc["recall"] / "images.txt").read_text(encoding="utf-8").splitlines()
        current_protocol = {key: metrics[key] for key in
                            ("data", "imgsz", "conf", "match_iou", "nms_iou", "max_det", "matching")}
        if names is None:
            names, protocol = images, current_protocol
        if names != images or protocol != current_protocol or len(images) != 548:
            raise ValueError("paired full-image evaluation protocols differ")
        pred_path = loc["recall"] / "predictions.jsonl"
        saved = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines()]
        if [item["image"] for item in saved] != images:
            raise ValueError("saved prediction population differs")
        summaries[arm], predictions[arm] = audit, saved
        for file in (loc["out"] / "independent_audit.json", loc["recall"] / "metrics.json",
                     loc["recall"] / "images.txt", pred_path):
            sources[str(file)] = digest(file)
    source = load_source_dataset(Path(protocol["data"]))
    gt_classes, gt_boxes = [], []
    masks = {size: [] for size in ("small", "medium", "large")}
    for filename in names:
        image = Path(filename)
        label = label_path_for_image(image, source.images_val, source.labels_val)
        gt = read_label_file(label)
        boxes = np.asarray([[box.x, box.y, box.w, box.h] for box in gt]).reshape(-1, 4)
        gt_boxes.append(boxes)
        gt_classes.append(np.asarray([box.class_id for box in gt]))
        with Image.open(image) as pixels:
            width, height = pixels.size
        area = boxes[:, 2] * width * boxes[:, 3] * height
        for size, mask in zip(masks, (area < 1024, (area >= 1024) & (area < 9216), area >= 9216)):
            masks[size].append(mask)
        sources[str(label)] = digest(label)
    gt_counts = {size: sum(int(mask.sum()) for mask in group) for size, group in masks.items()}
    records, matches = {}, {}
    for arm, predictions_for_arm in predictions.items():
        records[arm] = {}
        for conf, iou in ((.25, .5), (.25, .75), (.35, .5), (.5, .5)):
            matched = []
            count = 0
            for pred, classes, boxes in zip(predictions_for_arm, gt_classes, gt_boxes):
                selected = np.asarray(pred["confidences"]) >= conf
                count += int(selected.sum())
                matched.append(match_by_confidence(
                    np.asarray(pred["classes"])[selected], np.asarray(pred["confidences"])[selected],
                    np.asarray(pred["xywhn"]).reshape(-1, 4)[selected], classes, boxes, iou))
            counts = {
                size: sum(int((match & mask).sum()) for match, mask in zip(matched, group))
                for size, group in masks.items()
            }
            tp = sum(counts.values())
            records[arm][f"conf{conf}_iou{iou}"] = {
                "predictions": count, "matched": counts, "tp": tp, "unmatched_predictions": count - tp,
                "precision": tp / count, "recall": tp / sum(gt_counts.values()),
                "f1": 2 * tp / (count + sum(gt_counts.values())),
            }
            if conf == .25 and iou == .5:
                reported = read(paths(arm)["recall"] / "metrics.json")["per_size"]
                if any(counts[size] != reported[size]["matched"] for size in counts):
                    raise ValueError("saved predictions do not reconstruct accepted recall")
                matches[arm] = matched
    turnover = paired_counts(matches["b2"], matches["p1"], masks)
    class_turnover = {}
    for class_id, name in summaries["p1"]["names"].items():
        group = [small & (cls == int(class_id)) for small, cls in zip(masks["small"], gt_classes)]
        class_turnover[name] = paired_counts(matches["b2"], matches["p1"], {"small": group})["small"]
    delta = np.asarray([
        int(((after.astype(int) - before) * small).sum())
        for before, after, small in zip(matches["b2"], matches["p1"], masks["small"])
    ])
    denominators = np.asarray([int(mask.sum()) for mask in masks["small"]])
    samples = np.random.default_rng(20261101).integers(0, len(names), (2000, len(names)))
    bootstrap = 100 * delta[samples].sum(axis=1) / denominators[samples].sum(axis=1)
    class_delta = {
        name: 100 * (summaries["p1"]["class_maps"][int(key)] - summaries["b2"]["class_maps"][int(key)])
        for key, name in summaries["p1"]["names"].items()
    }
    result = {
        "scope": "retrospective cached full-val analysis; fixed trained weights; no new inference/training",
        "gt": gt_counts, "records": records, "gt_turnover": turnover,
        "small_turnover_by_class": class_turnover,
        "delta_class_AP50_95_pp": class_delta,
        "small_image_bootstrap_percentile95_pp": np.quantile(bootstrap, [.025, .975]).tolist(),
        "student_seconds": {arm: record["student_seconds"] for arm, record in summaries.items()},
        "limits": [
            "not official VisDrone FP or AP_small; IoU75 recall is not AP75",
            "threshold replays only cover conf>=0.25; fixed thresholds are diagnostic, not selected policies",
            "image bootstrap is conditional on weights; not training-seed/split uncertainty or independent confirmation",
            "B2 resumed after epoch63; P1 uninterrupted; equal epochs have unequal data and compute",
        ],
        "sha256": sources,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "analysis.json", result)
    print({key: value for key, value in result.items() if key != "sha256"}, flush=True)


if __name__ == "__main__":
    main()
