"""Bounded global-threshold screen from frozen teacher candidates, one 5-epoch arm."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import traceback
from zoneinfo import ZoneInfo

import yaml

from regionssod.evaluation.runner import run_validation
from regionssod.evaluation.size_recall import run_size_recall
from regionssod.pseudo.generator import _link_or_copy
from regionssod.training.runner import load_experiment_config, run_training
from run_p1_no_soft_weights import preflight

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/screen_p1_global_t050_5ep.yaml"
BASE = ROOT / "outputs/screens/p1_soft_weights_1024_e5/comparison.json"
DATA = ROOT / "data/screens/p1_global_t050_1024_v2"
OUT = ROOT / "outputs/screens/p1_global_t050_1024_v2_e5"
SOURCE = ROOT / "data/visdrone_area10"
RUN = ROOT / "runs/screen/p1_global_t050_area10_subset1024_e5_s0"
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


def prepare(selection):
    DATA.mkdir(parents=True)
    selected_sources = set(selection["source_images_paths"])
    base_metadata = [json.loads(line) for line in (SOURCE / "pseudo/p1_s030_m050_l050/metadata.jsonl").read_text().splitlines()]
    image_for_source = {row["source_image"]: row["image"] for row in base_metadata}
    rows = [json.loads(line) for line in (SOURCE / "pseudo/p1_s030_m050_l050/candidates.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["source_image"] in selected_sources]
    if len(rows) != 1024:
        raise RuntimeError("candidate subset must contain 1024 source tiles")
    for directory in (DATA / "images/train", DATA / "labels/train"):
        directory.mkdir(parents=True, exist_ok=True)
    metadata = []
    pseudo_images = []
    for index, row in enumerate(rows):
        positive = [box for box in row["candidates"] if box["confidence"] >= .5]
        if not positive:
            continue
        ignore = [box for box in row["candidates"] if box["confidence"] < .5]
        source_image = Path(image_for_source[row["source_image"]])
        image = DATA / "images/train" / f"{index:07d}_{source_image.name}"
        _link_or_copy(source_image, image)
        labels = [f"{box['class_id']} " + " ".join(f"{v:.10g}" for v in box["xywhn"]) for box in positive]
        (DATA / "labels/train" / f"{image.stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        pseudo_images.append(str(image))
        metadata.append({"image": str(image), "source_image": row["source_image"],
                         "positive": positive, "ignore": ignore})
    (DATA / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8")
    original_train = (SOURCE / "combined_p1.txt").read_text().splitlines()
    selected_labeled = [line for line in original_train if line in selection["labeled_tiles"]]
    if len(selected_labeled) != 1024:
        raise RuntimeError("expected exactly 1024 labeled tiles from the fixed source subset")
    images = selected_labeled + pseudo_images
    (DATA / "train.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    full = yaml.safe_load((SOURCE / "dataset_p1.yaml").read_text())
    full.update(path=str(DATA), train=str(DATA / "train.txt"), p1_metadata=str(DATA / "metadata.jsonl"))
    (DATA / "dataset.yaml").write_text(yaml.safe_dump(full, sort_keys=False), encoding="utf-8")
    summary = read(SOURCE / "pseudo/p1_s030_m050_l050/summary.json")
    summary.update(policy={"small_threshold": .5, "medium_threshold": .5, "large_threshold": .5,
                           "conf_floor": .05, "small_area": 1024.0, "large_area": 9216.0},
                   metadata=str(DATA / "metadata.jsonl"), requested_images=1024, processed_images=1024,
                   accepted_images=len(metadata), accepted_boxes=sum(len(row["positive"]) for row in metadata),
                   ignore_boxes=sum(len(row["ignore"]) for row in metadata),
                   excluded_images_without_positives=1024 - len(metadata), per_class={},
                   per_scale={s: {"positive": 0, "ignore": 0} for s in ("small", "medium", "large")})
    for row in metadata:
        for kind in ("positive", "ignore"):
            for box in row[kind]:
                summary["per_scale"][box["scale"]][kind] += 1
                if kind == "positive":
                    key = str(box["class_id"])
                    summary["per_class"][key] = summary["per_class"].get(key, 0) + 1
    summary["screening_subset"] = {**selection, "train_tiles": len(images),
                                    "scope": "bounded global-threshold screening; not a paper result"}
    write(DATA / "summary.json", summary)
    files = [CONFIG, Path(__file__), BASE, SOURCE / "pseudo/p1_s030_m050_l050/candidates.jsonl",
             SOURCE / "pseudo/p1_s030_m050_l050/metadata.jsonl", DATA / "train.txt",
             DATA / "dataset.yaml", DATA / "metadata.jsonl", DATA / "summary.json"]
    write(OUT / "preflight.json", {"sha256": {str(path): digest(path) for path in files},
          "source_images": 1024, "train_tiles": len(images), "epochs": 5,
          "only_arm": "global_threshold_0.50", "baseline_arm": "frozen scale-aware 5-epoch arm",
          "candidate_rule": "abs(delta mAP50-95) >= 0.003 and same-direction small recall"})
    return summary


def main():
    global OWNED
    preflight()
    if OUT.exists() or RUN.exists():
        raise FileExistsError("refusing duplicate scale-threshold screen")
    base = read(BASE)
    if base.get("scope") is None or len(base["runs"]) != 2:
        raise RuntimeError("completed paired subset baseline is required")
    baseline = base["runs"][0]
    selection = {"source_images_paths": None, "labeled_tiles": None}
    screen = read(ROOT / "outputs/screens/p1_soft_weights_1024_e5/preflight.json")
    manifest = [json.loads(line) for line in (SOURCE / "manifest.jsonl").read_text().splitlines()]
    selection["source_images_paths"] = [tile["image"] for i in screen["source_indices"]
                                         for tile in manifest[i]["unlabeled_tiles"]]
    selection["labeled_tiles"] = [manifest[i]["labeled_tile"]["image"] for i in screen["source_indices"]]
    selection.update({"selection_seed": screen["selection_seed"], "source_indices": screen["source_indices"]})
    OUT.mkdir(parents=True)
    OWNED = True
    stage("preparing")
    summary = prepare(selection)
    config = load_experiment_config(CONFIG)
    if config["p1"] != {"soft_weights": True, "uncertain_ignore": True}:
        raise RuntimeError("scale screen must retain both loss mechanisms")
    stage("training", arm=config["name"], epochs=5, train_tiles=summary["screening_subset"]["train_tiles"])
    run = run_training(CONFIG)
    with (run / "results.csv").open(encoding="utf-8-sig") as handle:
        training = list(csv.DictReader(handle))
    if [int(float(row["epoch"])) for row in training] != [1, 2, 3, 4, 5]:
        raise RuntimeError("incomplete threshold screen")
    stage("validation960", arm=config["name"])
    validation = run_validation(run / "weights/best.pt", DATA / "dataset.yaml", imgsz=960,
                                project=str(ROOT / "runs/screen/val"), name=config["name"])
    stage("size_recall", arm=config["name"])
    recall = run_size_recall(run / "weights/best.pt", Path("D:/Python/YOLO/data/visdrone/dataset.yaml"),
                             OUT / config["name"], imgsz=960, batch=8, conf=.25, match_iou=.5)
    metrics = read(validation / "metrics.json")
    arm = {"name": config["name"], "run": str(run), "validation": str(validation),
           "weight_sha256": digest(run / "weights/best.pt"), "student_seconds": float(training[-1]["time"]),
           "mAP50_95": metrics["results"]["metrics/mAP50-95(B)"],
           "mAP50": metrics["results"]["metrics/mAP50(B)"],
           "small_recall": recall["per_size"]["small"]["recall"], "recall": recall, "p1": config["p1"]}
    delta_ap = arm["mAP50_95"] - baseline["mAP50_95"]
    delta_rec = arm["small_recall"] - baseline["small_recall"]
    direction = ("scale_aware" if delta_ap <= -.003 and delta_rec <= 0 else
                 "global_t050" if delta_ap >= .003 and delta_rec >= 0 else None)
    result = {"scope": "exploratory one-arm threshold screen; NOT paper results or 30-epoch confirmation",
              "scale_aware_baseline": baseline, "global_t050": arm,
              "delta_global_minus_scale_aware_pp": {"mAP50_95": 100 * delta_ap, "small_recall": 100 * delta_rec},
              "promising_direction": direction,
              "training_budget": {"source_images": 1024, "global_arm_epochs": 5, "new_arms": 1,
                                  "new_student_seconds": arm["student_seconds"]},
              "limitations": ["single subset/seed", "one 5-epoch arm reuses prior paired scale-aware subset",
                              "both arms cap confidence at 0.05; threshold policy only changes positive/ignore state"]}
    write(OUT / "comparison.json", result)
    stage("completed", promising_direction=direction, comparison=str(OUT / "comparison.json"))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if OWNED:
            stage("failed", error=traceback.format_exc())
        raise
