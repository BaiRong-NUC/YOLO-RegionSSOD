"""Diagnose frozen P1 pseudo labels without training, inference, or label export."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from regionssod.data.annotations import (
    Tile, YoloBox, clip_box_to_tile, label_path_for_image, read_label_file,
    replace_images_component,
)
from run_p1_scale_seed_pair import ROOT, digest, read

DATA = ROOT / "data/visdrone_area10"
PSEUDO = DATA / "pseudo/p1_s030_m050_l050"
OUT = ROOT / "outputs/paper1_gap_audit_20261030"
SIZES = ("small", "medium", "large")
REASONS = (
    "positive_assignment_competition", "ignore_same_class_iou_ge_050",
    "other_class_iou_ge_050", "same_class_iou_010_to_050",
    "other_class_iou_010_to_050", "no_saved_candidate_iou_ge_010",
)


def overlaps(predictions: list[YoloBox], gt: list[YoloBox]) -> np.ndarray:
    def corners(boxes: list[YoloBox]) -> np.ndarray:
        return np.asarray([(b.x-b.w/2, b.y-b.h/2, b.x+b.w/2, b.y+b.h/2)
                           for b in boxes], dtype=float).reshape(-1, 4)
    pred, target = corners(predictions), corners(gt)
    intersection = np.maximum(0, np.minimum(pred[:, None, 2:], target[None, :, 2:])
                              - np.maximum(pred[:, None, :2], target[None, :, :2])).prod(axis=2)
    union = ((pred[:, 2:]-pred[:, :2]).prod(axis=1)[:, None]
             + (target[:, 2:]-target[:, :2]).prod(axis=1)[None, :] - intersection)
    return intersection / np.maximum(union, 1e-12)


def match_iou(predictions: list[YoloBox], gt: list[YoloBox]) -> list[tuple[int, int]]:
    """Legacy pseudo-quality protocol: stable descending-IoU, all GT sizes jointly."""
    iou = overlaps(predictions, gt)
    same = np.asarray([p.class_id for p in predictions])[:, None] == np.asarray([g.class_id for g in gt])[None, :]
    pi, gi = np.where(same & (iou >= .5))
    used_p, used_g, pairs = set(), set(), []
    for k in np.argsort(-iou[pi, gi], kind="stable"):
        p, g = int(pi[k]), int(gi[k])
        if p not in used_p and g not in used_g:
            pairs.append((p, g))
            used_p.add(p)
            used_g.add(g)
    return pairs


def miss_evidence(gt_class: int, candidate_classes: np.ndarray,
                  candidate_positive: np.ndarray, iou: np.ndarray) -> str:
    """Mutually exclusive evidence categories, not proven causal failure stages."""
    same = candidate_classes == gt_class
    if np.any(same & candidate_positive & (iou >= .5)):
        return REASONS[0]
    if np.any(same & ~candidate_positive & (iou >= .5)):
        return REASONS[1]
    if np.any(~same & (iou >= .5)):
        return REASONS[2]
    if np.any(same & (iou >= .1)):
        return REASONS[3]
    if np.any(~same & (iou >= .1)):
        return REASONS[4]
    return REASONS[5]


def size(box: YoloBox, tile: Tile) -> str:
    area = box.w * box.h * tile.width * tile.height
    return "small" if area < 1024 else "medium" if area < 9216 else "large"


def analyse() -> tuple[dict, dict[str, str]]:
    hashes: dict[str, str] = {}

    def frozen(path: Path) -> None:
        hashes[str(path.resolve())] = digest(path)

    for path in (DATA / "manifest.jsonl", DATA / "source.json", DATA / "summary.json",
                 PSEUDO / "candidates.jsonl", PSEUDO / "metadata.jsonl", PSEUDO / "summary.json",
                 PSEUDO / "quality_audit.json", Path(__file__).resolve()):
        frozen(path)
    source, split, summary = (read(DATA / name) for name in ("source.json", "summary.json", "pseudo/p1_s030_m050_l050/summary.json"))
    teacher = Path(summary["weights"])
    frozen(teacher)
    if digest(teacher) != summary["teacher_sha256"]:
        raise ValueError("teacher changed")
    records = [json.loads(line) for line in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    candidates = [json.loads(line) for line in (PSEUDO / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = [json.loads(line) for line in (PSEUDO / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    by_image = {row["source_image"]: row for row in candidates}
    accepted = {row["source_image"]: row for row in metadata}
    if len(by_image) != len(candidates) or len(accepted) != len(metadata):
        raise ValueError("duplicate tile records")
    counts: dict[str, Counter] = defaultdict(Counter)
    bands: dict[str, Counter] = defaultdict(Counter)
    seen: set[str] = set()
    total_pred = total_tp = 0
    for record in records:
        label = label_path_for_image(Path(record["source_image"]), Path(source["images_train"]), Path(source["labels_train"]))
        frozen(label)
        boxes = read_label_file(label)
        width, height = record["source_size"]
        for entry in record["unlabeled_tiles"]:
            image = entry["image"]
            if image in seen:
                raise ValueError("duplicate split tile")
            seen.add(image)
            tile = Tile(*entry["tile"])
            row = by_image[image]
            if row["shape"] != [tile.height, tile.width]:
                raise ValueError("native tile shape differs")
            gt = [g for b in boxes if (g := clip_box_to_tile(b, tile, width, height, split["min_box_visibility"])) is not None]
            if len(gt) != entry["visible_gt_boxes_hidden_from_training"]:
                raise ValueError("hidden GT reconstruction differs")
            items = row["candidates"]
            candidate_boxes = [YoloBox(item["class_id"], *item["xywhn"]) for item in items]
            positive = np.asarray([item["state"] == "positive" for item in items], dtype=bool)
            classes = np.asarray([item["class_id"] for item in items], dtype=int)
            accepted_row = accepted.get(image)
            predictions = []
            if accepted_row is not None:
                pseudo_label = replace_images_component(Path(accepted_row["image"])).with_suffix(".txt")
                frozen(pseudo_label)
                predictions = read_label_file(pseudo_label)
            if len(predictions) != int(positive.sum()):
                raise ValueError("saved positives and training labels differ")
            pairs = match_iou(predictions, gt)
            hit_g = {g for _, g in pairs}
            hit_p = {p for p, _ in pairs}
            total_pred += len(predictions)
            total_tp += len(pairs)
            iou = overlaps(candidate_boxes, gt)
            tile_group = "kept_tiles" if accepted_row is not None else "excluded_tiles"
            for j, target in enumerate(gt):
                groups = ("all", size(target, tile), f"class_{target.class_id}", tile_group)
                for group in groups:
                    counts[group]["gt"] += 1
                    counts[group]["matched"] += int(j in hit_g)
                if j not in hit_g:
                    reason = miss_evidence(target.class_id, classes, positive, iou[:, j])
                    for group in groups:
                        counts[group][reason] += 1
            positives = [item for item in items if item["state"] == "positive"]
            for j, (item, pred) in enumerate(zip(positives, predictions, strict=True)):
                if item["class_id"] != pred.class_id or not np.allclose(item["xywhn"], [pred.x,pred.y,pred.w,pred.h], atol=1e-9, rtol=0):
                    raise ValueError("positive order or coordinates differ")
                band = "conf_030_to_050" if item["confidence"] < .5 else "conf_ge_050"
                for group in (band, f"{size(pred, tile)}_{band}"):
                    bands[group]["predictions"] += 1
                    bands[group]["matched_predictions"] += int(j in hit_p)
    reference = read(PSEUDO / "quality_audit.json")
    if seen != set(by_image) or len(seen) != summary["processed_images"] or len(accepted) != summary["accepted_images"]:
        raise ValueError("incomplete tile scope")
    if (counts["all"]["gt"], total_pred, total_tp) != (38245, 35067, 25669):
        raise ValueError("legacy overall counts differ")
    if total_tp != reference["overall"]["true_positives"] or total_pred != summary["accepted_boxes"]:
        raise ValueError("pseudo-quality source differs")
    groups = {}
    for name, counter in sorted(counts.items()):
        missed = counter["gt"] - counter["matched"]
        evidence = {key: counter[key] for key in REASONS}
        if sum(evidence.values()) != missed:
            raise ValueError("miss categories do not partition GT")
        if name in SIZES and (counter["gt"] != reference["groups"][name]["gt"] or counter["matched"] != reference["groups"][name]["matched_gt"]):
            raise ValueError("legacy size counts differ")
        groups[name] = {"gt": counter["gt"], "matched": counter["matched"], "missed": missed, "miss_evidence": evidence}
    confidence = {name: {**dict(c), "precision": c["matched_predictions"] / c["predictions"]}
                  for name, c in sorted(bands.items())}
    result = {"tiles": len(seen), "accepted_tiles": len(accepted), "teacher_sha256": summary["teacher_sha256"],
              "positive_predictions": total_pred, "true_positive_predictions": total_tp,
              "groups": groups, "positive_confidence_bands": confidence,
              "protocol": "same-class descending-IoU greedy matching at IoU>=0.50; GT clipped at visibility>=0.35 in native tile pixels",
              "candidate_scope": "saved post-NMS candidates, confidence>=0.05, NMS IoU=0.70, max_det=500",
              "limitations": ["hidden training GT used only for offline diagnosis; no training-label export",
                              "evidence categories are ordered and descriptive, not proven causal attribution",
                              "ignore overlap does not prove a recoverable extra one-to-one detection",
                              "no saved overlap cannot distinguish absent, below-floor, NMS-suppressed, or truncated predictions",
                              "low-confidence precision is conditional on full P1 matching; not an independent B2/P1 delta",
                              "not student validation AP_small, official AP, or proof that a new training method will help"]}
    return result, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    receipt: dict = {}
    if not args.check and OUT.exists():
        raise FileExistsError("refusing overwrite of frozen diagnosis")
    if args.check:
        receipt = read(OUT / "manifest.json")
        for p, expected in receipt["input_sha256"].items():
            if digest(p) != expected:
                raise ValueError(f"diagnostic input changed: {p}")
        if digest(OUT / "analysis.json") != receipt["analysis_sha256"]:
            raise ValueError("diagnostic output changed")
    result, hashes = analyse()
    if args.check:
        if result != read(OUT / "analysis.json") or hashes != receipt["input_sha256"]:
            raise ValueError("diagnosis is not reproducible")
        print(f"Verified {result['tiles']} tiles, {len(hashes)} source hashes and exact count replay")
        return
    OUT.mkdir(parents=True)
    (OUT / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "input_sha256": hashes, "analysis_sha256": digest(OUT / "analysis.json")}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"groups": {k: result['groups'][k] for k in ('all', 'small', 'kept_tiles', 'excluded_tiles')},
                      "confidence": result['positive_confidence_bands']}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
