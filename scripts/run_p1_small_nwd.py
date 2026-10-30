"""One full-data 30-epoch NWD student against the cached full P1 pilot."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import traceback
from zoneinfo import ZoneInfo

import yaml

from regionssod.training.nwd import estimate_normalizer, run_nwd_training
from regionssod.training.runner import load_experiment_config
from run_p1_scale_seed_pair import (
    ROOT, check_epochs, digest, parameter_differences, read, replay, write,
)

NAME = "partial10_p1_small_nwd_pilot30_s0"
CONFIG = ROOT / "configs/train_p1_small_nwd_10.yaml"
OUT = ROOT / "outputs/p1_small_nwd_pilot30_s0"
RUN = ROOT / "runs" / NAME
VAL = ROOT / "runs/val" / f"{NAME}_val960"
RECALL = ROOT / "runs/size_recall/p1_small_nwd_pilot30_s0"
REF = ROOT / "runs/partial10_p1_pilot30_s0"
REF_VAL = ROOT / "runs/val/partial10_p1_pilot30_s0_val960"
REF_RECALL = ROOT / "runs/size_recall/p1_pilot30_s0"
SCREEN = ROOT / "outputs/screens/p1_small_nwd_1024_e5"
SOURCE = Path("D:/Python/YOLO/data/visdrone/dataset.yaml")
STARTED = False


def qualifies(delta):
    """Two pre-recorded routes: overall AP gain, or recall gain with AP guardrail."""
    ap, recall = delta["mAP50_95"], delta["small_recall"]
    return (math.isfinite(ap) and math.isfinite(recall)
            and ((ap >= .3 and recall >= 0) or (recall >= 1. and ap >= -.1)))


def stage(name, **extra):
    value = {"stage": name, "updated_at": datetime.now(
        ZoneInfo("Asia/Shanghai")).isoformat(), **extra}
    write(OUT / "status.json", value)
    print(json.dumps(value), flush=True)


def preflight():
    import ultralytics

    if Path.cwd().resolve() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("run at repository root in pinned yolo environment")
    if (any(p.exists() for p in (RUN, VAL, RECALL))
            or any((OUT / f).exists() for f in ("status.json", "dispatch.json"))):
        raise FileExistsError("existing full30 state; inspect instead of restarting")
    config = load_experiment_config(CONFIG)
    base = load_experiment_config(ROOT / "configs/train_p1_10.yaml")
    if ({k: v for k, v in config.items() if k != "nwd"} != base | {"name": NAME}
            or config["nwd"] != {"normalizer": "open_small_mean", "mix": .5, "small_side": 32.}):
        raise ValueError("only name and the fixed NWD extension may differ from full P1")
    own = [CONFIG, Path(__file__).resolve(),
           ROOT / "scripts/audit_p1_small_nwd.py", ROOT / "scripts/run_small_nwd30_background.py",
           ROOT / "docs/paper/small_nwd30_protocol.md", ROOT / "src/regionssod/training/nwd.py"]
    paths = [p.relative_to(ROOT).as_posix() for p in own]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *paths],
                   cwd=ROOT, capture_output=True, check=True)
    subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *paths], cwd=ROOT, check=True)
    frozen = read(ROOT / "outputs/gate3_20261027/preflight.json")
    hashes = dict(frozen["sha256"])
    for line in (REF / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        checksum, filename = line.split("  ", 1)
        if filename in hashes and hashes[filename] != checksum:
            raise ValueError(f"conflicting frozen reference hash: {filename}")
        hashes[filename] = checksum
    screen = read(SCREEN / "preflight.json")
    nwd_source = str(ROOT / "src/regionssod/training/nwd.py")
    if digest(nwd_source) != screen["sha256"][nwd_source]:
        raise ValueError("NWD implementation changed since the completed screen")
    print(f"Verifying {len(hashes)} frozen inputs/reference artifacts", flush=True)
    for path, checksum in hashes.items():
        if digest(path) != checksum:
            raise ValueError(f"changed frozen input: {path}")
    reference = read(REF / "experiment.json")
    if reference["train_args"] != config["train"] or reference["p1"] != config["p1"]:
        raise ValueError("cached full P1 schedule or loss switches differ")
    check_epochs(REF)
    # Reuse the original open-label C, rather than tuning C after viewing val.
    subset = ROOT / "data/screens/p1_soft_weights_1024/train.txt"
    opened = [p for p in subset.read_text(encoding="utf-8").splitlines()
              if "labeled" in Path(p).parts]
    dataset = yaml.safe_load((ROOT / base["data"]).read_text(encoding="utf-8"))
    train_list = Path(dataset["train"])
    all_images = train_list.read_text(encoding="utf-8").splitlines()
    if len(opened) != 1024 or not set(opened).issubset(all_images) or len(all_images) != 11310:
        raise ValueError("expected fixed full data and its original open-label subset")
    normalizer = estimate_normalizer(opened)
    if normalizer != screen["normalizer"]:
        raise ValueError("original open-label C cannot be reproduced")
    summary = Path(dataset["p1_metadata"]).parent / "summary.json"
    pseudo = read(summary)
    if (pseudo != reference["pseudo_summary"]
            or digest(pseudo["weights"]) != pseudo["teacher_sha256"]
            or frozen["train_tiles"] != 11310):
        raise ValueError("pseudo data/teacher do not match full P1")
    for path in [*own, subset, train_list, summary, REF / "experiment.json",
                 REF_VAL / "metrics.json", REF_RECALL / "metrics.json",
                 REF_RECALL / "images.txt", REF_RECALL / "predictions.jsonl",
                 ROOT / "scripts/run_p1_scale_seed_pair.py",
                 SCREEN / "preflight.json", SCREEN / "geometry_gpu_check.json"]:
        hashes[str(path)] = digest(path)
    receipt = {
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=ROOT, text=True).strip(),
        "epochs": 30, "train_tiles": 11310, "reference_run": str(REF),
        "only_method_change": "small-target box loss: .5 CIoU + .5 NWD; unchanged DFL/TAL/classification",
        "normalizer": normalizer, "sha256": hashes,
        "continuation": "(mAP50-95 >= +.30pp and small recall >= 0) OR "
                        "(small recall >= +1.00pp and mAP50-95 >= -.10pp)",
        "authorization": "2026-10-30 user requires at least 30 epochs for performance decisions",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "preflight.json", receipt)
    print(f"Preflight passed: {len(hashes)} hashes, 11310 tiles, 30 epochs; "
          f"C={normalizer['normalizer']:.8f}", flush=True)
    return receipt


def collect(*, check=False):
    receipt = read(OUT / "preflight.json")
    for path, checksum in receipt["sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"changed frozen input: {path}")
    before_exp, after_exp = read(REF / "experiment.json"), read(RUN / "experiment.json")
    expected_nwd = {"normalizer": receipt["normalizer"]["normalizer"], "mix": .5, "small_side": 32.}
    if (after_exp["nwd"] != expected_nwd
            or any(before_exp[k] != after_exp[k]
                   for k in ("model", "data", "p1", "train_args", "pseudo_summary"))):
        raise ValueError("paired experiment metadata differs beyond the NWD extension")
    if any(digest(p) != h for p, h in after_exp["input_sha256"].items()):
        raise ValueError("run source/config provenance changed")
    actual = yaml.safe_load((RUN / "args.yaml").read_text(encoding="utf-8"))
    old = yaml.safe_load((REF / "args.yaml").read_text(encoding="utf-8"))
    if parameter_differences(old, actual, {"name", "save_dir"}):
        raise ValueError("actual training hyperparameters differ")
    rows = check_epochs(RUN)
    ref_rows = check_epochs(REF)
    before_val, after_val = read(REF_VAL / "metrics.json"), read(VAL / "metrics.json")
    before_rec, after_rec = read(REF_RECALL / "metrics.json"), read(RECALL / "metrics.json")
    for val, rec, run in ((before_val, before_rec, REF), (after_val, after_rec, RUN)):
        if (val["imgsz"] != 960 or rec["images"] != 548 or rec["imgsz"] != 960
                or rec["conf"] != .25 or rec["match_iou"] != .5
                or rec["nms_iou"] != .7 or rec["max_det"] != 500
                or Path(val["weights"]) != run / "weights/best.pt"
                or rec["weights_sha256"] != digest(run / "weights/best.pt")):
            raise ValueError("full-val weights or operating point differ")
        if (val["class_indices"] != list(range(10))
                or len(val["class_maps"]) != 10
                or not all(math.isfinite(v) for v in val["class_maps"])
                or not math.isclose(sum(val["class_maps"]) / 10,
                                    val["results"]["metrics/mAP50-95(B)"], abs_tol=1e-12)):
            raise ValueError("class AP values do not reconstruct full-val mAP")
    if (after_val["data"] != before_val["data"]
            or any(before_rec[k] != after_rec[k] for k in (
                "data", "ultralytics_version", "size_pixel_space", "matching"))
            or (RECALL / "images.txt").read_bytes() != (REF_RECALL / "images.txt").read_bytes()):
        raise ValueError("full-val inputs or recall protocol differ")
    replay(REF_RECALL)
    replay(RECALL)
    if [after_rec["per_size"][s]["gt"] for s in ("small", "medium", "large")] != [26577, 11115, 1067]:
        raise ValueError("full-val GT counts differ")
    before, after = {}, {}
    for name, key in (("mAP50", "metrics/mAP50(B)"), ("mAP50_95", "metrics/mAP50-95(B)")):
        before[name], after[name] = before_val["results"][key], after_val["results"][key]
    for size in ("small", "medium", "large"):
        before[f"{size}_recall"] = before_rec["per_size"][size]["recall"]
        after[f"{size}_recall"] = after_rec["per_size"][size]["recall"]
    if not all(math.isfinite(v) for v in [*before.values(), *after.values()]):
        raise ValueError("non-finite evaluation values")
    delta = {key: 100 * (after[key] - before[key]) for key in before}
    result = {
        "reference": before, "candidate": after, "delta_nwd_minus_p1_pp": delta,
        "worth_larger_training": qualifies(delta),
        "new_epochs": 30, "train_tiles": 11310, "normalizer": expected_nwd["normalizer"],
        "student_time_seconds": {"reference": float(ref_rows[-1]["time"]),
                                 "candidate": float(rows[-1]["time"])},
        "scope": "single fixed split/teacher/student seed full-data 30-epoch pilot; "
                 "not significance or official AP_small",
    }
    if check:
        if read(OUT / "comparison.json") != result:
            raise ValueError("comparison differs from independent replay")
        for line in (RUN / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            checksum, path = line.split("  ", 1)
            if digest(path) != checksum:
                raise ValueError(f"changed result artifact: {path}")
    else:
        write(OUT / "comparison.json", result)
        artifacts = [RUN / f for f in ("weights/best.pt", "weights/last.pt", "args.yaml",
                                       "results.csv", "experiment.json")]
        artifacts += [VAL / "metrics.json", RECALL / "metrics.json", RECALL / "images.txt",
                      RECALL / "predictions.jsonl", OUT / "comparison.json", OUT / "preflight.json"]
        (RUN / "SHA256SUMS.txt").write_text(
            "".join(f"{digest(p)}  {p}\n" for p in artifacts), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    global STARTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    receipt = preflight()
    if args.check_only:
        return
    with (OUT / "dispatch.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, handle)
    STARTED = True
    stage("training", epochs=30, train_tiles=11310)
    if run_nwd_training(CONFIG, normalizer=receipt["normalizer"]["normalizer"]) != RUN:
        raise ValueError("unexpected training directory")
    check_epochs(RUN)
    from regionssod.evaluation.runner import run_validation
    from regionssod.evaluation.size_recall import run_size_recall

    stage("validation960")
    if run_validation(RUN / "weights/best.pt", ROOT / "data/visdrone_area10/dataset_p1.yaml",
                      imgsz=960, batch=8, device="0", name=f"{NAME}_val960") != VAL:
        raise ValueError("unexpected validation directory")
    stage("size_recall")
    run_size_recall(RUN / "weights/best.pt", SOURCE, RECALL, imgsz=960,
                    batch=8, device="0", conf=.25, match_iou=.5)
    stage("comparing")
    collect()
    stage("completed", comparison=str(OUT / "comparison.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if STARTED:
            stage("failed", error=traceback.format_exc())
        raise
