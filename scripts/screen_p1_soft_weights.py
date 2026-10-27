"""Bounded paired screening: 1024 source images, 5 epochs per arm, full val960.

Screening values must not be added to paper result tables. No 30-epoch job or
further sweep is started automatically. This script never accesses hidden GT.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from regionssod.evaluation.runner import run_validation
from regionssod.evaluation.size_recall import run_size_recall
from regionssod.training.runner import load_experiment_config, run_training
from run_p1_no_soft_weights import preflight

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/screens/p1_soft_weights_1024"
OUT = ROOT / "outputs/screens/p1_soft_weights_1024_e5"
SOURCE = ROOT / "data/visdrone_area10"
CONFIGS = [ROOT / "configs/screen_p1_soft_weights_5ep.yaml", ROOT / "configs/screen_p1_no_soft_weights_5ep.yaml"]
OWNED = False


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stage(name, **extra):
    value = {"stage": name, "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra}
    write(OUT / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def prepare():
    if DATA.exists():
        raise FileExistsError(DATA)
    DATA.mkdir(parents=True)
    records = [json.loads(line) for line in (SOURCE / "manifest.jsonl").read_text().splitlines()]
    indices = np.sort(np.random.default_rng(20261027).choice(len(records), 1024, replace=False))
    selected = [records[i] for i in indices]
    labeled = [row["labeled_tile"]["image"] for row in selected]
    unlabeled = {item["image"] for row in selected for item in row["unlabeled_tiles"]}
    original = SOURCE / "pseudo/p1_s030_m050_l050"
    metadata = [json.loads(line) for line in (original / "metadata.jsonl").read_text().splitlines()]
    metadata = [row for row in metadata if row["source_image"] in unlabeled]
    (DATA / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8")
    images = labeled + [row["image"] for row in metadata]
    (DATA / "train.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    full = yaml.safe_load((SOURCE / "dataset_p1.yaml").read_text())
    full.update(path=str(DATA), train=str(DATA / "train.txt"), p1_metadata=str(DATA / "metadata.jsonl"))
    (DATA / "dataset.yaml").write_text(yaml.safe_dump(full, sort_keys=False), encoding="utf-8")
    summary = read(original / "summary.json")
    summary.update(metadata=str(DATA / "metadata.jsonl"), requested_images=1024, processed_images=1024,
                   accepted_images=len(metadata), excluded_images_without_positives=1024 - len(metadata),
                   accepted_boxes=sum(len(row["positive"]) for row in metadata),
                   ignore_boxes=sum(len(row["ignore"]) for row in metadata), per_class={},
                   per_scale={s: {"positive": 0, "ignore": 0} for s in ("small", "medium", "large")})
    for row in metadata:
        for kind in ("positive", "ignore"):
            for box in row[kind]:
                summary["per_scale"][box["scale"]][kind] += 1
                if kind == "positive":
                    key = str(box["class_id"])
                    summary["per_class"][key] = summary["per_class"].get(key, 0) + 1
    summary["screening_subset"] = {"source_images": 1024, "selection_seed": 20261027,
                                    "source_indices": indices.tolist(), "train_tiles": len(images),
                                    "scope": "bounded screening only; not a paper result"}
    write(DATA / "summary.json", summary)
    hashes = {str(path): digest(path) for path in [*CONFIGS, Path(__file__), DATA / "train.txt",
              DATA / "dataset.yaml", DATA / "metadata.jsonl", DATA / "summary.json"]}
    (OUT / "source_snapshot").mkdir()
    for path in [*CONFIGS, Path(__file__), ROOT / "scripts/run_p1_no_soft_weights.py"]:
        (OUT / "source_snapshot" / path.name).write_bytes(path.read_bytes())
    write(OUT / "preflight.json", {"sha256": hashes, **summary["screening_subset"],
          "epochs_per_arm": 5, "arms": 2, "close_mosaic": 0,
          "candidate_rule": "abs(delta mAP50-95) >= 0.003 and same-direction small recall; exploratory only"})
    return summary


def main():
    global OWNED
    if OUT.exists():
        raise FileExistsError(f"refusing repeat screen: {OUT}")
    preflight()
    OUT.mkdir(parents=True)
    OWNED = True
    stage("preparing")
    summary = prepare()
    configs = [load_experiment_config(path) for path in CONFIGS]
    a, b = configs
    if ({k: v for k, v in a.items() if k not in {"name", "p1"}} !=
            {k: v for k, v in b.items() if k not in {"name", "p1"}} or
            a["p1"] != {"soft_weights": True, "uncertain_ignore": True} or
            b["p1"] != {"soft_weights": False, "uncertain_ignore": True}):
        raise RuntimeError("screen arms must differ only in soft weights")
    output = []
    for config_path, config in zip(CONFIGS, configs):
        if (ROOT / config["project"] / config["name"]).exists():
            raise FileExistsError(f"refusing a duplicate screen arm: {config['name']}")
        stage("training", arm=config["name"], epochs=5, train_tiles=summary["screening_subset"]["train_tiles"])
        run = run_training(config_path)
        with (run / "results.csv").open(encoding="utf-8-sig") as handle:
            training = list(csv.DictReader(handle))
        if [int(float(row["epoch"])) for row in training] != [1, 2, 3, 4, 5]:
            raise RuntimeError("incomplete screen training")
        stage("validation960", arm=config["name"])
        weights = run / "weights/best.pt"
        validation = run_validation(weights, DATA / "dataset.yaml", imgsz=960,
                                    project=str(ROOT / "runs/screen/val"), name=config["name"])
        stage("size_recall", arm=config["name"])
        recall = run_size_recall(weights, Path("D:/Python/YOLO/data/visdrone/dataset.yaml"),
                                 OUT / config["name"], imgsz=960, batch=8, conf=.25, match_iou=.5)
        metrics = read(validation / "metrics.json")
        output.append({"name": config["name"], "run": str(run), "validation": str(validation),
                       "weight_sha256": digest(weights), "student_seconds": float(training[-1]["time"]),
                       "mAP50_95": metrics["results"]["metrics/mAP50-95(B)"],
                       "mAP50": metrics["results"]["metrics/mAP50(B)"],
                       "small_recall": recall["per_size"]["small"]["recall"],
                       "recall": recall, "p1": config["p1"]})
        write(OUT / "arms.json", output)
    if any(output[0]["recall"]["per_size"][s]["gt"] != output[1]["recall"]["per_size"][s]["gt"] for s in ("small", "medium", "large")):
        raise RuntimeError("validation GT mismatch")
    delta_ap = output[1]["mAP50_95"] - output[0]["mAP50_95"]
    delta_recall = output[1]["small_recall"] - output[0]["small_recall"]
    result = {"scope": "exploratory paired subset screening; NOT paper results or 30-epoch confirmation",
              "runs": output, "delta_no_soft_minus_full_pp": {"mAP50_95": 100 * delta_ap, "small_recall": 100 * delta_recall},
              "promising_direction": "no_soft_weights" if delta_ap >= .003 and delta_recall >= 0 else
                  "keep_soft_weights" if delta_ap <= -.003 and delta_recall <= 0 else None,
              "training_budget": {"source_images": 1024, "epochs_per_arm": 5, "arms": 2,
                                  "student_seconds": sum(row["student_seconds"] for row in output)},
              "limitations": ["single subset/seed", "short schedule with default warmup and close_mosaic=0",
                              "full-val used for development; no claim of statistical significance"]}
    write(OUT / "comparison.json", result)
    stage("completed", promising_direction=result["promising_direction"], comparison=str(OUT / "comparison.json"))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if OWNED:
            stage("failed", error=traceback.format_exc())
        raise
