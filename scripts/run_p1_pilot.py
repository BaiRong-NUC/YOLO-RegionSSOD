"""Run the prepared P1 30-epoch pilot and complete the paired Gate 3 evaluation.

Execute from the repository root in the yolo environment. No data generation,
threshold tuning, overwrite, or expansion to the formal matrix happens here.
"""
from __future__ import annotations

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
from regionssod.training.runner import run_training


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "outputs/gate3_20261027"
RUN_NAME = "partial10_p1_pilot30_s0"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage(name, **extra):
    payload = {"stage": name, "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra}
    write_json(EVIDENCE / "status.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from {ROOT}")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if (ROOT / "runs" / RUN_NAME).exists():
        raise FileExistsError(f"refusing a second pilot or automatic run suffix: {RUN_NAME}")
    gate2 = json.loads((ROOT / "outputs/gate2_20261027/comparison.json").read_text(encoding="utf-8"))
    if not gate2["gate"]["passed"]:
        raise RuntimeError("Gate 2 must pass first")
    baseline = next(row for row in gate2["runs"] if row["method"] == "B2")
    if digest(baseline["best_weights"]) != baseline["best_sha256"]:
        raise RuntimeError("frozen B2 weight hash changed")
    summary = json.loads((ROOT / "data/visdrone_area10/pseudo/p1_s030_m050_l050/summary.json").read_text(encoding="utf-8"))
    if summary["teacher_sha256"] != gate2["protocol"]["teacher_sha256"] or digest(summary["weights"]) != summary["teacher_sha256"]:
        raise RuntimeError("P1 teacher differs from B2 teacher")
    # Audit created before dispatch records every pseudo image/label and source.
    preflight = json.loads((EVIDENCE / "preflight.json").read_text(encoding="utf-8"))
    for path, expected in preflight["sha256"].items():
        if digest(path) != expected:
            raise RuntimeError(f"frozen input changed: {path}")
    stage("training", run=RUN_NAME, epochs=30, train_tiles=preflight["train_tiles"])
    run = run_training(ROOT / "configs/train_p1_10.yaml", name=RUN_NAME)
    rows = list(csv.DictReader((run / "results.csv").open(encoding="utf-8-sig")))
    if len(rows) != 30 or any(not math.isfinite(float(v)) for row in rows for v in row.values()):
        raise RuntimeError("P1 must finish 30 epochs with finite CSV values")
    actual = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    b2_args = yaml.safe_load((ROOT / "runs/partial10_naive_t050_pilot30_s0/args.yaml").read_text(encoding="utf-8"))
    differences = {key: [b2_args[key], actual.get(key)] for key in b2_args
                   if key not in {"data", "name", "save_dir"} and actual.get(key) != b2_args[key]}
    if differences:
        write_json(EVIDENCE / "parameter_differences.json", differences)
        raise RuntimeError("P1/B2 actual student hyperparameters differ")
    weights = run / "weights/best.pt"
    stage("validation960", run=str(run))
    validation = run_validation(weights, ROOT / "data/visdrone_area10/dataset_p1.yaml", imgsz=960,
                                name=f"{RUN_NAME}_val960")
    stage("size_recall", run=str(run))
    b2_recall = json.loads((ROOT / "runs/size_recall/b2_pilot30_s0/metrics.json").read_text(encoding="utf-8"))
    if (b2_recall["weights_sha256"] != baseline["best_sha256"] or b2_recall["images"] != 548 or
            b2_recall["conf"] != .25 or b2_recall["match_iou"] != .5 or b2_recall["imgsz"] != 960):
        raise RuntimeError("B2 recall operating point differs from frozen protocol")
    recall = run_size_recall(weights, Path("D:/Python/YOLO/data/visdrone/dataset.yaml"),
                             ROOT / "runs/size_recall/p1_pilot30_s0", imgsz=960, conf=.25, match_iou=.5)
    if recall["images"] != 548 or any(recall["per_size"][size]["gt"] != b2_recall["per_size"][size]["gt"] for size in recall["per_size"]):
        raise RuntimeError("P1 and B2 validation ground-truth counts differ")
    metrics = json.loads((validation / "metrics.json").read_text(encoding="utf-8"))
    ap = metrics["results"]["metrics/mAP50-95(B)"]
    ap_delta = ap - baseline["mAP50_95"]
    recall_delta = recall["per_size"]["small"]["recall"] - b2_recall["per_size"]["small"]["recall"]
    payload = {"method": "P1", "run": str(run), "epochs": 30, "validation_images": 548,
               "train_tiles": preflight["train_tiles"], "training_time_seconds": float(rows[-1]["time"]),
               "best_sha256": digest(weights), "last_sha256": digest(run / "weights/last.pt"),
               "baseline": baseline, "p1_validation": str(validation / "metrics.json"),
               "mAP50_95": ap, "delta_mAP50_95": ap_delta,
               "small_recall": recall["per_size"]["small"]["recall"],
               "b2_small_recall": b2_recall["per_size"]["small"]["recall"],
               "delta_small_recall": recall_delta,
               "gate": {"required_delta_mAP50_95": .005, "require_small_recall_increase": True,
                        "passed": ap_delta >= .005 and recall_delta > 0},
               "limitations": ["Single-seed development pilot; no statistical significance claim",
                               "Ultralytics mAP and fixed-operating-point recall; not official AP_small/AR500",
                               "Equal student epochs; tile counts, compute and teacher cost differ"]}
    write_json(EVIDENCE / "comparison.json", payload)
    files = [run / "weights/best.pt", run / "weights/last.pt", run / "args.yaml", run / "results.csv",
             run / "experiment.json", validation / "metrics.json",
             ROOT / "runs/size_recall/p1_pilot30_s0/metrics.json", EVIDENCE / "comparison.json"]
    (run / "SHA256SUMS.txt").write_text("\n".join(f"{digest(p)}  {p}" for p in files) + "\n", encoding="utf-8")
    stage("completed", gate3_passed=payload["gate"]["passed"], comparison=str(EVIDENCE / "comparison.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if EVIDENCE.exists():
            stage("failed", error=traceback.format_exc())
        raise
