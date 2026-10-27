"""Train P1 without confidence soft weights, evaluate, and compare with frozen full P1.

Run from the repository root in the existing yolo environment. --check-only
verifies the frozen inputs without starting training or GPU evaluation.
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
from regionssod.training.runner import load_experiment_config, run_training


ROOT = Path(__file__).resolve().parents[1]
NAME = "partial10_p1_no_soft_weights_pilot30_s0"
CONFIG = ROOT / "configs/train_p1_no_soft_weights_10.yaml"
FULL = ROOT / "runs/partial10_p1_pilot30_s0"
DATA = ROOT / "data/visdrone_area10/dataset_p1.yaml"
SOURCE_DATA = Path("D:/Python/YOLO/data/visdrone/dataset.yaml")
EVIDENCE = ROOT / "outputs/p1_no_soft_weights_pilot30_s0"
RUN = ROOT / "runs" / NAME
VAL = ROOT / "runs/val" / f"{NAME}_val960"
RECALL = ROOT / "runs/size_recall/p1_no_soft_weights_pilot30_s0"
EXECUTION_STARTED = False


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage(name, **extra):
    write_json(EVIDENCE / "status.json", {
        "stage": name, "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra})
    print(f"[{name}] {extra}", flush=True)


def preflight():
    import ultralytics

    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from {ROOT}")
    if ultralytics.__version__ != "8.4.172":
        raise RuntimeError("use the frozen ultralytics==8.4.172 environment")
    for target in (RUN, VAL, RECALL):
        if target.exists():
            raise FileExistsError(f"refusing overwrite or automatic run suffix: {target}")
    if any((EVIDENCE / name).exists() for name in ("status.json", "dispatch.json")):
        raise FileExistsError(f"existing pipeline state requires inspection: {EVIDENCE}")
    config = load_experiment_config(CONFIG)
    full = read_json(FULL / "experiment.json")
    expected = {"name": NAME, "method": "p1", "model": full["model"],
                "data": str(DATA.relative_to(ROOT)).replace("\\", "/"), "project": "runs",
                "p1": {"soft_weights": False, "uncertain_ignore": True},
                "train": full["train_args"]}
    if config != expected or full["p1"] != {"soft_weights": True, "uncertain_ignore": True}:
        raise RuntimeError("only run name and soft_weights may differ from frozen full P1")
    print("Checking frozen P1 inputs and reference artifacts...", flush=True)
    frozen = read_json(ROOT / "outputs/gate3_20261027/preflight.json")
    hashes = dict(frozen["sha256"])
    for line in (FULL / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            checksum, filename = line.split("  ", 1)
            if filename in hashes and hashes[filename] != checksum:
                raise RuntimeError(f"conflicting frozen hashes: {filename}")
            hashes[filename] = checksum
    for index, (filename, expected_hash) in enumerate(hashes.items(), 1):
        if digest(filename) != expected_hash:
            raise RuntimeError(f"frozen input or reference artifact changed: {filename}")
        if index % 5000 == 0:
            print(f"Verified {index}/{len(hashes)} frozen files", flush=True)
    summary = read_json(ROOT / "data/visdrone_area10/pseudo/p1_s030_m050_l050/summary.json")
    if summary != full["pseudo_summary"] or digest(summary["weights"]) != summary["teacher_sha256"]:
        raise RuntimeError("pseudo labels or teacher differ from frozen full P1")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    receipt = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "only_method_change": "p1.soft_weights: true -> false",
               "train_tiles": frozen["train_tiles"], "epochs": 30,
               "reference_run": str(FULL), "sha256": hashes | {
                   str(CONFIG): digest(CONFIG), str(Path(__file__).resolve()): digest(__file__)}}
    write_json(EVIDENCE / "preflight.json", receipt)
    print(f"Preflight passed: {len(hashes)} frozen files; {frozen['train_tiles']} training tiles", flush=True)
    return receipt


def compare(run, validation, recall, receipt):
    full_metrics = read_json(ROOT / "runs/val/partial10_p1_pilot30_s0_val960/metrics.json")
    full_recall = read_json(ROOT / "runs/size_recall/p1_pilot30_s0/metrics.json")
    metrics = read_json(validation / "metrics.json")
    reference_hash = digest(FULL / "weights/best.pt")
    if full_recall["weights_sha256"] != reference_hash or recall["weights_sha256"] != digest(run / "weights/best.pt"):
        raise RuntimeError("evaluation weight hashes do not match")
    for key in ("images", "imgsz", "conf", "match_iou", "nms_iou", "max_det",
                "ultralytics_version", "size_pixel_space", "matching"):
        if recall[key] != full_recall[key]:
            raise RuntimeError(f"recall evaluation protocol differs: {key}")
    if recall["images"] != 548 or metrics["imgsz"] != 960:
        raise RuntimeError("full-image val548 at 960 is required")
    if (RECALL / "images.txt").read_text() != (ROOT / "runs/size_recall/p1_pilot30_s0/images.txt").read_text():
        raise RuntimeError("full-image validation lists/order differ")
    before, after = {}, {}
    for label, key in (("mAP50", "metrics/mAP50(B)"), ("mAP50_95", "metrics/mAP50-95(B)"),
                       ("precision", "metrics/precision(B)"), ("recall", "metrics/recall(B)")):
        before[label], after[label] = full_metrics["results"][key], metrics["results"][key]
    for size in ("small", "medium", "large"):
        if recall["per_size"][size]["gt"] != full_recall["per_size"][size]["gt"]:
            raise RuntimeError(f"GT counts differ: {size}")
        before[f"{size}_recall"] = full_recall["per_size"][size]["recall"]
        after[f"{size}_recall"] = recall["per_size"][size]["recall"]
    if any(not math.isfinite(value) for value in [*before.values(), *after.values()]):
        raise RuntimeError("non-finite evaluation metrics")
    delta = {key: 100 * (after[key] - before[key]) for key in before}
    rows = list(csv.DictReader((run / "results.csv").open(encoding="utf-8-sig")))
    full_rows = list(csv.DictReader((FULL / "results.csv").open(encoding="utf-8-sig")))
    payload = {"reference": {"run": str(FULL), "weights_sha256": reference_hash, "metrics": before},
               "ablation": {"run": str(run), "weights_sha256": recall["weights_sha256"], "metrics": after},
               "delta_no_soft_weights_minus_full_p1_percentage_points": delta,
               "full_p1_minus_no_soft_weights_percentage_points": {k: -v for k, v in delta.items()},
               "epochs": 30, "train_tiles": receipt["train_tiles"],
               "student_time_seconds": {"full_p1": float(full_rows[-1]["time"]), "no_soft_weights": float(rows[-1]["time"])},
               "limitations": ["Single split/seed development ablation; no significance claim",
                               "Ultralytics mAP and diagnostic recall; not official AP_small/AR500"]}
    write_json(EVIDENCE / "comparison.json", payload)
    lines = ["# P1 no-soft-weights ablation versus full P1", "",
             "Same 10% area split, seed0, 30-epoch students and full-image val960. Delta = no-soft-weights minus full P1.", "",
             "| Metric | Full P1 / % | No soft weights / % | Delta / pp |", "|---|---:|---:|---:|"]
    lines += [f"| {key} | {100*before[key]:.4f} | {100*after[key]:.4f} | {delta[key]:+.4f} |" for key in before]
    lines += ["", "Size recall: conf=0.25, match IoU=0.50, max_det=500; not AP_small or official AR500."]
    text = "\n".join(lines) + "\n"
    (EVIDENCE / "comparison.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main(check_only=False):
    global EXECUTION_STARTED
    receipt = preflight()
    if check_only:
        return
    with (EVIDENCE / "dispatch.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, handle)
    EXECUTION_STARTED = True
    stage("training", run=str(RUN), epochs=30)
    run = run_training(CONFIG)
    if run != RUN:
        raise RuntimeError(f"unexpected run directory: {run}")
    with (run / "results.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if ([int(float(row["epoch"])) for row in rows] != list(range(1, 31)) or
            any(not math.isfinite(float(v)) for row in rows for v in row.values())):
        raise RuntimeError("30 contiguous epochs and finite training CSV values are required")
    actual = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    reference = yaml.safe_load((FULL / "args.yaml").read_text(encoding="utf-8"))
    differences = {key: [reference.get(key), actual.get(key)] for key in reference.keys() | actual.keys()
                   if key not in {"name", "save_dir"} and reference.get(key) != actual.get(key)}
    if differences:
        write_json(EVIDENCE / "parameter_differences.json", differences)
        raise RuntimeError("actual training hyperparameters differ from full P1")
    experiment = read_json(run / "experiment.json")
    if experiment["p1"] != {"soft_weights": False, "uncertain_ignore": True}:
        raise RuntimeError("ablation loss switches differ from requested settings")
    weights = run / "weights/best.pt"
    if not weights.is_file() or not (run / "weights/last.pt").is_file():
        raise RuntimeError("best/last checkpoints are required")
    stage("validation960", run=str(run))
    validation = run_validation(weights, DATA, imgsz=960, batch=8, device="0", name=f"{NAME}_val960")
    if validation != VAL:
        raise RuntimeError(f"unexpected validation directory: {validation}")
    stage("size_recall", run=str(run))
    recall = run_size_recall(weights, SOURCE_DATA, RECALL, imgsz=960, batch=8,
                             device="0", conf=0.25, match_iou=0.5)
    stage("comparing", run=str(run))
    compare(run, validation, recall, receipt)
    write_json(run / "audit.json", {"training_and_evaluation_completed": True,
               "only_method_change": receipt["only_method_change"], "actual_parameter_differences": differences,
               "comparison": str(EVIDENCE / "comparison.json"), "preflight": str(EVIDENCE / "preflight.json")})
    files = [weights, run / "weights/last.pt", run / "args.yaml", run / "results.csv",
             run / "experiment.json", run / "audit.json", validation / "metrics.json",
             RECALL / "metrics.json", RECALL / "images.txt", EVIDENCE / "comparison.json",
             EVIDENCE / "comparison.md", EVIDENCE / "preflight.json"]
    (run / "SHA256SUMS.txt").write_text("\n".join(f"{digest(p)}  {p}" for p in files) + "\n", encoding="utf-8")
    stage("completed", comparison=str(EVIDENCE / "comparison.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="verify inputs without training/evaluation")
    args = parser.parse_args()
    try:
        main(check_only=args.check_only)
    except Exception:
        # A rejected duplicate invocation must not overwrite an existing run's status.
        if EXECUTION_STARTED:
            stage("failed", error=traceback.format_exc())
        raise
