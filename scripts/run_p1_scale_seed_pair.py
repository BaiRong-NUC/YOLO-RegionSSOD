"""Repeat the frozen scale-threshold ablation with student seed 1.

Keep split seed 0, teacher, pseudo data, loss, and 30-epoch schedule fixed.
Run both arms sequentially in fresh processes, then audit saved predictions.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import yaml

from regionssod.data.annotations import label_path_for_image, read_label_file
from regionssod.data.dataset import load_source_dataset
from regionssod.evaluation.size_recall import match_by_confidence
from regionssod.training.runner import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/p1_scale_threshold_pair_s1"
SOURCE = Path("D:/Python/YOLO/data/visdrone/dataset.yaml")
ARMS = {"scale_aware": "partial10_p1_pilot30_s1",
        "global_t050": "partial10_p1_global_t050_pilot30_s1"}
CONFIGS = {"scale_aware": ROOT / "configs/train_p1_10_seed1.yaml",
           "global_t050": ROOT / "configs/train_p1_global_t050_10_seed1.yaml"}
BASE_CONFIGS = {"scale_aware": ROOT / "configs/train_p1_10.yaml",
                "global_t050": ROOT / "configs/train_p1_global_t050_10.yaml"}
METRICS = ("mAP50", "mAP50_95", "small_recall", "medium_recall", "large_recall")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage(name, **extra):
    write(OUT / "status.json", {"stage": name, "updated_at": datetime.now(
        ZoneInfo("Asia/Shanghai")).isoformat(), **extra})
    print(name, extra, flush=True)


def paths(arm):
    name = ARMS[arm]
    return (ROOT / "runs" / name, ROOT / "runs/val" / f"{name}_val960",
            ROOT / "runs/size_recall" / name.removeprefix("partial10_"))


def assert_config(arm):
    expected = load_experiment_config(BASE_CONFIGS[arm])
    expected["name"] = ARMS[arm]
    expected["train"]["seed"] = 1
    if load_experiment_config(CONFIGS[arm]) != expected:
        raise RuntimeError(f"{arm}: only student seed and name may change")


def check_epochs(run):
    with (run / "results.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if [int(float(row["epoch"])) for row in rows] != list(range(1, 31)):
        raise RuntimeError(f"{run}: expected continuous epochs 1..30")
    if not all(math.isfinite(float(v)) for row in rows for v in row.values()):
        raise RuntimeError(f"{run}: non-finite training values")
    return rows


def parameter_differences(left, right, excluded):
    return {key: [left.get(key), right.get(key)] for key in left.keys() | right.keys()
            if key not in excluded and left.get(key) != right.get(key)}


def preflight():
    import ultralytics
    from run_p1_global_t050 import validate_prepared

    if Path.cwd().resolve() != ROOT or ultralytics.__version__ != "8.4.172":
        raise RuntimeError("run at repository root in frozen yolo environment")
    if any(path.exists() for arm in ARMS for path in paths(arm)):
        raise FileExistsError("refusing overwrite or duplicate paired training")
    if (OUT / "dispatch.json").exists():
        raise FileExistsError("existing dispatch requires inspection")
    for arm in ARMS:
        assert_config(arm)
    frozen = read(ROOT / "outputs/gate3_20261027/preflight.json")["sha256"]
    hashes = dict(frozen)
    for name in ("partial10_p1_pilot30_s0", "partial10_p1_global_t050_pilot30_s0"):
        for line in (ROOT / "runs" / name / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            checksum, filename = line.split("  ", 1)
            if filename in hashes and hashes[filename] != checksum:
                raise RuntimeError(f"conflicting hash: {filename}")
            hashes[filename] = checksum
    dataset = read(ROOT / "outputs/p1_global_t050_pilot30_s0/dataset_audit.json")
    hashes.update(dataset["sha256"])
    # Include every label and image used by the second arm, not only its manifest.
    for arm in ARMS:
        config = load_experiment_config(CONFIGS[arm])
        data = ROOT / config["data"]
        spec = yaml.safe_load(data.read_text(encoding="utf-8"))
        for image in Path(spec["train"]).read_text(encoding="utf-8").splitlines():
            path = Path(image)
            label = Path(str(path).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt")
            for item in (path, label):
                if str(item) not in hashes:
                    hashes[str(item)] = digest(item)
        for item in (data, Path(spec["train"]), Path(spec["p1_metadata"]), CONFIGS[arm]):
            hashes[str(item)] = digest(item)
    for index, (filename, expected) in enumerate(hashes.items(), 1):
        if digest(filename) != expected:
            raise RuntimeError(f"frozen input changed: {filename}")
        if index % 10000 == 0:
            print(f"Verified {index}/{len(hashes)} frozen files", flush=True)
    validate_prepared(dataset)
    receipt = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "split_seed": 0, "student_seed": 1, "epochs": 30,
               "sha256": hashes, "script_sha256": digest(__file__),
               "scope": "fixed split/teacher/pseudo data; repeat student randomness only"}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "preflight.json", receipt)
    return receipt


def worker(arm):
    from regionssod.evaluation.runner import run_validation
    from regionssod.evaluation.size_recall import run_size_recall
    from regionssod.training.runner import run_training

    assert_config(arm)
    run, val, recall = paths(arm)
    if any(p.exists() for p in (run, val, recall)):
        raise FileExistsError(f"refusing duplicate arm: {arm}")
    stage("training", arm=arm, run=str(run), epochs=30)
    if run_training(CONFIGS[arm]) != run:
        raise RuntimeError("unexpected training directory")
    check_epochs(run)
    stage("validation960", arm=arm)
    data = ROOT / load_experiment_config(CONFIGS[arm])["data"]
    if run_validation(run / "weights/best.pt", data, name=f"{ARMS[arm]}_val960") != val:
        raise RuntimeError("unexpected validation directory")
    stage("size_recall", arm=arm)
    run_size_recall(run / "weights/best.pt", SOURCE, recall, imgsz=960,
                    batch=8, device="0", conf=.25, match_iou=.5)
    stage("arm_completed", arm=arm)


def replay(directory):
    summary = read(directory / "metrics.json")
    source = load_source_dataset(Path(summary["data"]))
    names = (directory / "images.txt").read_text(encoding="utf-8").splitlines()
    counts = {size: {"gt": 0, "matched": 0} for size in ("small", "medium", "large")}
    with (directory / "predictions.jsonl").open(encoding="utf-8") as handle:
        predictions = [json.loads(line) for line in handle]
    if len(predictions) != len(names) or len(names) != 548:
        raise RuntimeError("expected 548 saved predictions")
    for prediction, name in zip(predictions, names):
        image = Path(prediction["image"])
        if str(image) != name:
            raise RuntimeError("prediction order mismatch")
        gt = read_label_file(label_path_for_image(image, source.images_val, source.labels_val))
        boxes = np.array([[b.x, b.y, b.w, b.h] for b in gt]).reshape(-1, 4)
        matched = match_by_confidence(prediction["classes"], prediction["confidences"],
                                     prediction["xywhn"], [b.class_id for b in gt], boxes, .5)
        with Image.open(image) as pixels:
            width, height = pixels.size
        area = boxes[:, 2] * width * boxes[:, 3] * height
        for size, mask in (("small", area < 1024), ("medium", (area >= 1024) & (area < 9216)),
                           ("large", area >= 9216)):
            counts[size]["gt"] += int(mask.sum())
            counts[size]["matched"] += int((mask & matched).sum())
    for values in counts.values():
        values["recall"] = values["matched"] / values["gt"]
    if counts != summary["per_size"]:
        raise RuntimeError("saved prediction replay differs from reported recall")
    return counts


def collect():
    receipt = read(OUT / "preflight.json")
    if digest(__file__) != receipt["script_sha256"]:
        raise RuntimeError("runner changed after preflight")
    for filename, checksum in receipt["sha256"].items():
        if digest(filename) != checksum:
            raise RuntimeError(f"frozen input changed during training: {filename}")
    values, records, artifacts, actuals = {}, {}, [], {}
    baseline_recall = read(ROOT / "runs/size_recall/p1_pilot30_s0/metrics.json")
    for arm in ARMS:
        assert_config(arm)
        run, val_dir, recall_dir = paths(arm)
        rows = check_epochs(run)
        actual = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        actuals[arm] = actual
        base_run = ROOT / "runs" / ARMS[arm].replace("_s1", "_s0")
        base_args = yaml.safe_load((base_run / "args.yaml").read_text(encoding="utf-8"))
        if actual["seed"] != 1 or parameter_differences(base_args, actual, {"name", "save_dir", "seed"}):
            raise RuntimeError("seed1 hyperparameters differ from frozen seed0 arm")
        exp = read(run / "experiment.json")
        base_exp = read(base_run / "experiment.json")
        if (exp["p1"] != {"soft_weights": True, "uncertain_ignore": True}
                or exp["pseudo_summary"] != base_exp["pseudo_summary"]
                or exp["train_args"] != base_exp["train_args"] | {"seed": 1}):
            raise RuntimeError("loss, pseudo data or schedule mismatch")
        if any(digest(p) != h for p, h in exp["input_sha256"].items()):
            raise RuntimeError("run source/config input changed")
        val, recall = read(val_dir / "metrics.json"), read(recall_dir / "metrics.json")
        best_hash = digest(run / "weights/best.pt")
        protocol = ("data", "images", "imgsz", "conf", "match_iou", "nms_iou", "max_det",
                    "ultralytics_version", "size_pixel_space", "matching")
        if (any(recall[k] != baseline_recall[k] for k in protocol)
                or recall["weights_sha256"] != best_hash
                or Path(val["weights"]) != run / "weights/best.pt" or val["imgsz"] != 960
                or Path(val["data"]) != Path(exp["data"])):
            raise RuntimeError("evaluation protocol or weights mismatch")
        if (recall_dir / "images.txt").read_bytes() != (ROOT / "runs/size_recall/p1_pilot30_s0/images.txt").read_bytes():
            raise RuntimeError("validation image list differs")
        dataset = yaml.safe_load(Path(exp["data"]).read_text(encoding="utf-8"))
        base_dataset = yaml.safe_load(Path(base_exp["data"]).read_text(encoding="utf-8"))
        if dataset["val"] != base_dataset["val"]:
            raise RuntimeError("full validation split differs")
        replayed = replay(recall_dir)
        values[arm] = {"mAP50": val["results"]["metrics/mAP50(B)"],
                       "mAP50_95": val["results"]["metrics/mAP50-95(B)"],
                       **{f"{s}_recall": replayed[s]["recall"] for s in replayed}}
        if not all(math.isfinite(v) for v in values[arm].values()):
            raise RuntimeError("non-finite metrics")
        records[arm] = {"run": str(run), "weights_sha256": best_hash,
                        "last_sha256": digest(run / "weights/last.pt"), "metrics": values[arm],
                        "student_time_seconds": float(rows[-1]["time"]),
                        "train_tiles": len(Path(dataset["train"]).read_text().splitlines()),
                        "replayed_recall": replayed, "class_maps": val["class_maps"]}
        artifacts += [run / name for name in ("weights/best.pt", "weights/last.pt", "args.yaml",
                      "experiment.json", "results.csv")]
        artifacts += [val_dir / "metrics.json", recall_dir / "metrics.json",
                      recall_dir / "images.txt", recall_dir / "predictions.jsonl"]
    differences = parameter_differences(actuals["scale_aware"], actuals["global_t050"],
                                       {"data", "name", "save_dir"})
    if differences:
        raise RuntimeError("paired seed1 actual parameters differ")
    delta = {key: 100 * (values["global_t050"][key] - values["scale_aware"][key]) for key in METRICS}
    old = read(ROOT / "outputs/p1_global_t050_pilot30_s0/comparison.json")
    per_seed = {"0": {"scale_aware": old["reference"]["metrics"],
                      "global_t050": old["global_t050"]["metrics"],
                      "delta_global_minus_scale_aware_pp": old["delta_global_minus_scale_aware_pp"]},
                "1": {**values, "delta_global_minus_scale_aware_pp": delta}}
    aggregate = {arm: {key: {"mean_percent": statistics.mean(per_seed[s][arm][key] * 100 for s in per_seed),
                             "sample_std_pp": statistics.stdev(per_seed[s][arm][key] * 100 for s in per_seed)}
                       for key in METRICS} for arm in ARMS}
    paired = {key: {"mean_pp": statistics.mean(per_seed[s]["delta_global_minus_scale_aware_pp"][key] for s in per_seed),
                    "sample_std_pp": statistics.stdev(per_seed[s]["delta_global_minus_scale_aware_pp"][key] for s in per_seed)}
              for key in METRICS}
    comparison = {"split_seed": 0, "student_seeds": [0, 1], "epochs": 30,
                  "seed1": records, "per_seed": per_seed, "aggregate": aggregate,
                  "paired_delta_global_minus_scale_aware_pp": paired,
                  "limitations": ["one split, one fixed teacher, two student seeds; no significance claim",
                                  "Ultralytics development mAP and fixed-point recall; not AP_small",
                                  "equal epochs, different tile counts and compute; original Gate 3 unchanged"]}
    write(OUT / "comparison.json", comparison)
    artifacts.append(OUT / "comparison.json")
    audit = {"audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
             "acceptance_checks_passed": True, "parameter_differences": differences,
             "frozen_files_checked": len(receipt["sha256"]), "replayed_images_per_arm": 548,
             "sha256": {str(p): digest(p) for p in artifacts}}
    write(OUT / "acceptance_audit.json", audit)
    for arm in ARMS:
        run = paths(arm)[0]
        write(run / "audit.json", audit)
        (run / "SHA256SUMS.txt").write_text("".join(f"{h}  {p}\n" for p, h in audit["sha256"].items()), encoding="utf-8")
    lines = ["# 尺度阈值配对 seed 复验", "", "固定 split seed0、教师与数据，仅补学生 seed1；差值为统一 0.50 − 尺度分层。", "",
             "| 学生 seed | 指标 | 尺度分层 / % | 统一 0.50 / % | 差值 / pp |", "|---|---|---:|---:|---:|"]
    for seed, pair in per_seed.items():
        lines += [f"| {seed} | {key} | {pair['scale_aware'][key]*100:.4f} | {pair['global_t050'][key]*100:.4f} | {pair['delta_global_minus_scale_aware_pp'][key]:+.4f} |" for key in METRICS]
    lines += ["", "均值与样本标准差（ddof=1）见 comparison.json；两 seed 不用于显著性结论。"]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stage("completed", comparison=str(OUT / "comparison.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--worker", choices=ARMS)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.collect_only:
        collect()
    else:
        preflight()
        if args.check_only:
            return
        with (OUT / "dispatch.json").open("x", encoding="utf-8") as handle:
            json.dump({"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                       "python": sys.executable, "arms": ARMS}, handle)
        for arm in ARMS:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", arm],
                           cwd=ROOT, check=True)
        stage("acceptance")
        collect()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if (OUT / "dispatch.json").exists():
            stage("failed", error=traceback.format_exc())
        raise
