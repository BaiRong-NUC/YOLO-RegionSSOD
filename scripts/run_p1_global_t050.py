"""Prepare and run the full global-threshold P1 confirmation.

Data are materialized from the frozen teacher candidates; the teacher is not
rerun. `--prepare-only` performs all non-GPU checks and writes the full dataset.
The 30-epoch training/evaluation begins only without that flag.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import traceback
from zoneinfo import ZoneInfo

import yaml

from regionssod.evaluation.runner import run_validation
from regionssod.evaluation.size_recall import run_size_recall
from regionssod.pseudo.generator import _link_or_copy, combine_image_lists
from regionssod.training.runner import load_experiment_config, run_training
from run_p1_no_soft_weights import preflight

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/visdrone_area10"
BASE = SOURCE / "pseudo/p1_s030_m050_l050"
PSEUDO = SOURCE / "pseudo/p1_global_t050_from_candidates_20261027"
COMBINED = SOURCE / "combined_p1_global_t050.txt"
DATASET = SOURCE / "dataset_p1_global_t050.yaml"
CONFIG = ROOT / "configs/train_p1_global_t050_10.yaml"
POLICY = ROOT / "configs/pseudo_p1_global_t050.yaml"
FULL = ROOT / "runs/partial10_p1_pilot30_s0"
NAME = "partial10_p1_global_t050_pilot30_s0"
RUN = ROOT / "runs" / NAME
VAL = ROOT / "runs/val" / f"{NAME}_val960"
RECALL = ROOT / "runs/size_recall/p1_global_t050_pilot30_s0"
OUT = ROOT / "outputs/p1_global_t050_pilot30_s0"
EXECUTION_STARTED = False


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stage(name, **extra):
    write(OUT / "status.json", {"stage": name,
          "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra})
    print(f"[{name}] {extra}", flush=True)


def prepare_full():
    """Materialize uniform-0.50 positives and ignores from frozen candidates."""
    expected = {PSEUDO / "train.txt", PSEUDO / "metadata.jsonl", PSEUDO / "summary.json",
                COMBINED, DATASET}
    if PSEUDO.exists() or COMBINED.exists() or DATASET.exists():
        if not all(path.exists() for path in expected):
            raise RuntimeError("partial prepared dataset exists; inspect before retrying")
        receipt_path = OUT / "dataset_audit.json"
        receipt = read(receipt_path)
        if any(digest(path) != expected_hash for path, expected_hash in receipt["sha256"].items()):
            raise RuntimeError("prepared global-threshold dataset changed")
        validate_prepared(receipt)
        return receipt
    PSEUDO.mkdir(parents=True)
    for directory in (PSEUDO / "images/train", PSEUDO / "labels/train"):
        directory.mkdir(parents=True)
    base_rows = [json.loads(line) for line in (BASE / "metadata.jsonl").read_text().splitlines()]
    image_for_source = {row["source_image"]: row["image"] for row in base_rows}
    candidates = [json.loads(line) for line in (BASE / "candidates.jsonl").read_text().splitlines()]
    metadata, pseudo_images = [], []
    for index, row in enumerate(candidates):
        positive = [box for box in row["candidates"] if box["confidence"] >= .5]
        if not positive:
            continue
        ignore = [box for box in row["candidates"] if box["confidence"] < .5]
        source_image = Path(image_for_source[row["source_image"]])
        image = PSEUDO / "images/train" / f"{index:07d}_{source_image.name}"
        _link_or_copy(source_image, image)
        labels = [f"{box['class_id']} " + " ".join(f"{v:.10g}" for v in box["xywhn"]) for box in positive]
        (PSEUDO / "labels/train" / f"{image.stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        metadata.append({"image": str(image), "source_image": row["source_image"],
                         "positive": positive, "ignore": ignore})
        pseudo_images.append(str(image))
    (PSEUDO / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8")
    (PSEUDO / "train.txt").write_text("\n".join(pseudo_images) + "\n", encoding="utf-8")
    summary = read(BASE / "summary.json")
    summary.update(policy=yaml.safe_load(POLICY.read_text()), metadata=str(PSEUDO / "metadata.jsonl"),
                   requested_images=len(candidates), processed_images=len(candidates),
                   accepted_images=len(metadata), accepted_boxes=sum(len(row["positive"]) for row in metadata),
                   ignore_boxes=sum(len(row["ignore"]) for row in metadata),
                   excluded_images_without_positives=len(candidates) - len(metadata), per_class={},
                   per_scale={s: {"positive": 0, "ignore": 0} for s in ("small", "medium", "large")},
                   source_candidates=str(BASE / "candidates.jsonl"), teacher_not_rerun=True)
    for row in metadata:
        for kind in ("positive", "ignore"):
            for box in row[kind]:
                summary["per_scale"][box["scale"]][kind] += 1
                if kind == "positive":
                    key = str(box["class_id"])
                    summary["per_class"][key] = summary["per_class"].get(key, 0) + 1
    write(PSEUDO / "summary.json", summary)
    combine_image_lists(inputs=[SOURCE / "splits/labeled_train.txt", PSEUDO / "train.txt"],
                        output=COMBINED, dataset_yaml=DATASET,
                        source_data=Path("D:/Python/YOLO/data/visdrone/dataset.yaml"),
                        p1_metadata=PSEUDO / "metadata.jsonl")
    generated = [PSEUDO / "train.txt", PSEUDO / "metadata.jsonl", PSEUDO / "summary.json",
                 COMBINED, DATASET]
    receipt = {"prepared_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "train_tiles": len((COMBINED).read_text().splitlines()),
               "human_tiles": 6471, "pseudo_tiles": len(metadata),
               "accepted_boxes": summary["accepted_boxes"], "ignore_boxes": summary["ignore_boxes"],
               "policy": summary["policy"], "teacher_not_rerun": True,
               "sha256": {str(path): digest(path) for path in generated}}
    validate_prepared(receipt)
    write(OUT / "dataset_audit.json", receipt)
    return receipt


def validate_prepared(receipt):
    metadata = [json.loads(line) for line in (PSEUDO / "metadata.jsonl").read_text().splitlines()]
    candidates = {row["source_image"]: row for row in
                  (json.loads(line) for line in (BASE / "candidates.jsonl").read_text().splitlines())}
    if len(metadata) != receipt["pseudo_tiles"] or len(metadata) != 4624:
        raise RuntimeError("prepared pseudo-tile count differs from expected 4624")
    for row in metadata:
        source = candidates[row["source_image"]]
        positive = [box for box in source["candidates"] if box["confidence"] >= .5]
        ignore = [box for box in source["candidates"] if box["confidence"] < .5]
        if row["positive"] != positive or row["ignore"] != ignore:
            raise RuntimeError(f"uniform threshold metadata differs: {row['source_image']}")
        expected = [f"{box['class_id']} " + " ".join(f"{v:.10g}" for v in box["xywhn"])
                    for box in positive]
        actual = (PSEUDO / "labels/train" / f"{Path(row['image']).stem}.txt").read_text().splitlines()
        if actual != expected or not Path(row["image"]).is_file():
            raise RuntimeError(f"prepared label/image differs: {row['image']}")
    if len(COMBINED.read_text().splitlines()) != receipt["train_tiles"] or receipt["train_tiles"] != 11095:
        raise RuntimeError("combined training list differs")
    dataset = yaml.safe_load(DATASET.read_text())
    if Path(dataset["train"]) != COMBINED or Path(dataset["p1_metadata"]) != PSEUDO / "metadata.jsonl":
        raise RuntimeError("dataset metadata wiring differs")
    summary = read(PSEUDO / "summary.json")
    if (summary["accepted_images"], summary["accepted_boxes"], summary["ignore_boxes"]) != (4624, 27667, 57219):
        raise RuntimeError("prepared pseudo summary differs")


def compare(prepared):
    full_metrics = read(ROOT / "runs/val/partial10_p1_pilot30_s0_val960/metrics.json")
    full_recall = read(ROOT / "runs/size_recall/p1_pilot30_s0/metrics.json")
    metrics = read(VAL / "metrics.json")
    recall = read(RECALL / "metrics.json")
    if recall["weights_sha256"] != digest(RUN / "weights/best.pt"):
        raise RuntimeError("evaluation weight mismatch")
    protocol = ("images", "imgsz", "conf", "match_iou", "nms_iou", "max_det", "ultralytics_version")
    if any(recall[key] != full_recall[key] for key in protocol):
        raise RuntimeError("evaluation protocol mismatch")
    if recall["images"] != 548 or metrics["imgsz"] != 960:
        raise RuntimeError("full val548 at 960 required")
    if (RECALL / "images.txt").read_bytes() != (ROOT / "runs/size_recall/p1_pilot30_s0/images.txt").read_bytes():
        raise RuntimeError("validation order differs")
    before, after = {}, {}
    for label, key in (("mAP50", "metrics/mAP50(B)"), ("mAP50_95", "metrics/mAP50-95(B)")):
        before[label], after[label] = full_metrics["results"][key], metrics["results"][key]
    for size in ("small", "medium", "large"):
        if recall["per_size"][size]["gt"] != full_recall["per_size"][size]["gt"]:
            raise RuntimeError("GT counts differ")
        before[f"{size}_recall"] = full_recall["per_size"][size]["recall"]
        after[f"{size}_recall"] = recall["per_size"][size]["recall"]
    delta = {key: 100 * (after[key] - before[key]) for key in before}
    payload = {"reference": {"run": str(FULL), "weights_sha256": digest(FULL / "weights/best.pt"), "metrics": before},
               "global_t050": {"run": str(RUN), "weights_sha256": digest(RUN / "weights/best.pt"), "metrics": after},
               "delta_global_minus_scale_aware_pp": delta, "prepared_data": prepared,
               "limitations": ["single split/seed 30-epoch development confirmation",
                               "Ultralytics mAP and fixed-point recall; not official AP_small/AR500"]}
    write(OUT / "comparison.json", payload)
    lines = ["# 全量尺度阈值确认：统一 0.50 对尺度分层", "",
             "10% 标注面积、seed0、独立 30 epoch 学生，完整原图 val960。差值为“统一 0.50 − 完整 P1”。", "",
             "| 指标 | 完整 P1 / % | 统一 0.50 / % | 差值 / pp |", "|---|---:|---:|---:|"]
    lines += [f"| {key} | {100*before[key]:.4f} | {100*after[key]:.4f} | {delta[key]:+.4f} |" for key in before]
    lines += ["", "小目标 recall 为 conf=0.25、IoU=0.50、max_det=500 的开发诊断，不是 AP_small。"]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main(prepare_only=False):
    global EXECUTION_STARTED
    preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = prepare_full()
    stage("prepared", train_tiles=prepared["train_tiles"], pseudo_tiles=prepared["pseudo_tiles"])
    if prepare_only:
        return
    if RUN.exists() or VAL.exists() or RECALL.exists():
        raise FileExistsError("refusing duplicate full threshold confirmation")
    stage("training", run=str(RUN), epochs=30, train_tiles=prepared["train_tiles"])
    EXECUTION_STARTED = True
    run = run_training(CONFIG)
    with (run / "results.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if [int(float(row["epoch"])) for row in rows] != list(range(1, 31)) or any(
            not math.isfinite(float(v)) for row in rows for v in row.values()):
        raise RuntimeError("30 finite epochs required")
    actual, reference = yaml.safe_load((run / "args.yaml").read_text()), yaml.safe_load((FULL / "args.yaml").read_text())
    changes = {k: [reference.get(k), actual.get(k)] for k in actual.keys() | reference.keys()
               if k not in {"data", "name", "save_dir"} and actual.get(k) != reference.get(k)}
    if changes:
        write(OUT / "parameter_differences.json", changes)
        raise RuntimeError("actual training hyperparameters differ")
    stage("validation960", run=str(run))
    run_validation(run / "weights/best.pt", DATASET, imgsz=960, name=f"{NAME}_val960")
    stage("size_recall", run=str(run))
    run_size_recall(run / "weights/best.pt", Path("D:/Python/YOLO/data/visdrone/dataset.yaml"),
                    RECALL, imgsz=960, conf=.25, match_iou=.5)
    compare(prepared)
    stage("completed", comparison=str(OUT / "comparison.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        main(args.prepare_only)
    except Exception:
        if EXECUTION_STARTED and OUT.exists():
            stage("failed", error=traceback.format_exc())
        raise
