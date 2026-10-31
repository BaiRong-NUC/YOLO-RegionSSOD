"""Conditional user-run 100-epoch pair; never starts from an unaccepted pilot."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime
import inspect
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import psutil
import yaml

from audit_p1_small_nwd import original_process_running
from regionssod.training.coverage_guard import run_coverage_guard_training
from regionssod.training.runner import load_experiment_config, run_training
from run_p1_scale_seed_pair import ROOT, digest, parameter_differences, read, replay, write

PILOT = ROOT / "outputs/p1_coverage_guard_pilot30_s0"
PROTOCOL = ROOT / "docs/paper/coverage_guard_full100_protocol.md"
SOURCE = Path("D:/Python/YOLO/data/visdrone/dataset.yaml")
ARMS = ("p1", "coverage_guard")


def configuration(arm):
    if arm not in ARMS:
        raise ValueError("unknown paired arm")
    config = deepcopy(load_experiment_config(ROOT / "configs/train_p1_10.yaml"))
    config["name"] = f"partial10_{arm}_full100_s0"
    config["train"]["epochs"] = 100
    if arm == "coverage_guard":
        config["assignment"] = "image_coverage_guard"
    return config


def paths(arm):
    name = configuration(arm)["name"]
    return {
        "out": ROOT / "outputs" / f"{arm}_full100_s0",
        "run": ROOT / "runs" / name,
        "val": ROOT / "runs/val" / f"{name}_val960",
        "recall": ROOT / "runs/size_recall" / name.removeprefix("partial10_"),
    }


def complete_epochs(path, epochs=100):
    with Path(path).open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if ([int(float(row["epoch"])) for row in rows] != list(range(1, epochs + 1))
            or not all(math.isfinite(float(value)) for row in rows for value in row.values())):
        raise ValueError(f"expected {epochs} continuous finite epochs: {path}")
    return rows


def pilot_eligible():
    required = ("launch_receipt.json", "status.json", "comparison.json", "independent_audit.json")
    if not all((PILOT / name).is_file() for name in required):
        raise ValueError("complete independently audited positive 30-epoch pilot is required")
    launch, status, audit, result = (
        read(PILOT / name) for name in
        ("launch_receipt.json", "status.json", "independent_audit.json", "comparison.json"))
    if (launch.get("exit_code") != 0 or status["stage"] != "completed"
            or audit.get("accepted") is not True or audit.get("epochs") != 30
            or audit.get("worth_larger_training") is not True
            or result.get("worth_larger_training") is not True):
        raise ValueError("pilot is incomplete or did not support larger training")
    if any(original_process_running(launch[key]) for key in
           ("wrapper_identity", "training_identity")):
        raise ValueError("pilot processes remain active")
    # Recompute the positive decision from frozen raw evidence, not a hand-edited flag.
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_p1_coverage_guard.py")],
                   check=True, capture_output=True)


def preflight(arm):
    import ultralytics

    if Path.cwd().resolve() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("use repository root and pinned yolo environment")
    pilot_eligible()
    config = configuration(arm)
    baseline, candidate = configuration("p1"), configuration("coverage_guard")
    if candidate != baseline | {"name": candidate["name"], "assignment": "image_coverage_guard"}:
        raise ValueError("formal arms differ beyond run name and the intended assignment rule")
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if (process.pid != os.getpid() and "--train" in command
                    and any(Path(item).name == Path(__file__).name for item in command)):
                raise ValueError("another formal arm is active; run paired arms sequentially")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    locations = paths(arm)
    if any(path.exists() for path in locations.values()):
        raise FileExistsError("existing formal outputs; inspect or audit, never restart or overwrite")
    frozen = read(PILOT / "preflight.json")["sha256"]
    for path, checksum in frozen.items():
        if digest(path) != checksum:
            raise ValueError(f"changed pilot source: {path}")
    own = [Path(__file__).resolve(), PROTOCOL,
           ROOT / "configs/train_p1_10.yaml",
           Path(inspect.getfile(run_training)),
           Path(inspect.getfile(run_coverage_guard_training))]
    relative = [path.relative_to(ROOT).as_posix() for path in own]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                   check=True, capture_output=True)
    subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *relative], check=True)
    hashes = dict(frozen)
    for path in own + [PILOT / "comparison.json", PILOT / "artifact_sha256.json"]:
        hashes[str(path)] = digest(path)
    return {"checked_at": now(), "arm": arm, "config": config, "sha256": hashes,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "epochs": 100, "train_tiles": 11310, "student_seed": 0, "split_seed": 0,
            "teacher": "same frozen B1 teacher and offline pseudo labels as the 30-epoch pair"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def stage(out, value, **extra):
    write(out / "status.json", {"stage": value, "updated_at": now(), **extra})
    print(value, extra, flush=True)


def measure(arm):
    locations = paths(arm)
    out, run, val, rec = (locations[key] for key in ("out", "run", "val", "recall"))
    receipt = read(out / "preflight.json")
    for path, checksum in receipt["sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"changed formal source: {path}")
    if load_experiment_config(out / "config.yaml") != receipt["config"]:
        raise ValueError("generated configuration differs from the frozen plan")
    exp = read(run / "experiment.json")
    if (exp["train_args"] != receipt["config"]["train"]
            or exp["p1"] != receipt["config"]["p1"]
            or exp["name"] != receipt["config"]["name"]
            or Path(exp["data"]) != ROOT / receipt["config"]["data"]
            or exp.get("assignment") != receipt["config"].get("assignment")):
        raise ValueError("actual experiment metadata differs from the planned arm")
    for path, checksum in exp["input_sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"changed run provenance: {path}")
    original = yaml.safe_load((ROOT / "runs/partial10_p1_pilot30_s0/args.yaml").read_text(encoding="utf-8"))
    actual = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    if actual["epochs"] != 100 or parameter_differences(
            original, actual, {"name", "save_dir", "epochs"}):
        raise ValueError("actual 100-epoch settings differ beyond planned duration and name")
    rows = complete_epochs(run / "results.csv")
    validation, recall = read(val / "metrics.json"), read(rec / "metrics.json")
    if (Path(validation["weights"]) != run / "weights/best.pt" or validation["imgsz"] != 960
            or Path(validation["data"]) != ROOT / receipt["config"]["data"]
            or recall["weights_sha256"] != digest(run / "weights/best.pt")
            or recall["images"] != 548 or recall["imgsz"] != 960
            or [recall["per_size"][size]["gt"] for size in ("small", "medium", "large")]
            != [26577, 11115, 1067]
            or (recall["conf"], recall["match_iou"], recall["nms_iou"], recall["max_det"])
            != (.25, .5, .7, 500)):
        raise ValueError("full-image validation or recall protocol differs")
    reference_rec = ROOT / "runs/size_recall/p1_pilot30_s0"
    reference_metrics = read(reference_rec / "metrics.json")
    if ((rec / "images.txt").read_bytes() != (reference_rec / "images.txt").read_bytes()
            or any(recall[key] != reference_metrics[key] for key in
                   ("data", "ultralytics_version", "size_pixel_space", "matching"))):
        raise ValueError("validation images or matching settings changed")
    if (validation["class_indices"] != list(range(10)) or len(validation["class_maps"]) != 10
            or not math.isclose(sum(validation["class_maps"]) / 10,
                validation["results"]["metrics/mAP50-95(B)"], abs_tol=1e-12)):
        raise ValueError("class AP values do not reconstruct overall AP")
    replay(rec)
    values = {"mAP50": validation["results"]["metrics/mAP50(B)"],
              "mAP50_95": validation["results"]["metrics/mAP50-95(B)"],
              **{f"{size}_recall": recall["per_size"][size]["recall"]
                 for size in ("small", "medium", "large")}}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite metrics")
    return {"arm": arm, "epochs": 100, "metrics": values,
            "student_time_seconds": float(rows[-1]["time"]),
            "scope": "fixed split, teacher and seed0; development val; not official AP_small or significance"}


def result_paths(arm):
    loc = paths(arm)
    return ([loc["run"] / name for name in
             ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "experiment.json")]
            + [loc["val"] / "metrics.json"]
            + [loc["recall"] / name for name in ("metrics.json", "images.txt", "predictions.jsonl")]
            + [loc["out"] / name for name in ("preflight.json", "config.yaml", "metrics.json")])


def audit(arm):
    out = paths(arm)["out"]
    if read(out / "status.json")["stage"] != "completed":
        raise ValueError("formal run has not completed all stages")
    if original_process_running(read(out / "process.json")):
        raise ValueError("training process remains active")
    measured = measure(arm)
    if measured != read(out / "metrics.json"):
        raise ValueError("independent measurement differs")
    for path, checksum in read(out / "artifact_sha256.json").items():
        if digest(path) != checksum:
            raise ValueError(f"changed formal artifact: {path}")
    write(out / "independent_audit.json", {"audited_at": now(), "accepted": True, **measured})
    print(measured, flush=True)
    if all((paths(item)["out"] / "independent_audit.json").exists() for item in ARMS):
        other = next(item for item in ARMS if item != arm)
        other_out = paths(other)["out"]
        if (read(other_out / "status.json")["stage"] != "completed"
                or original_process_running(read(other_out / "process.json"))):
            raise ValueError("other paired arm is not terminal")
        other_measured = measure(other)
        cached = read(other_out / "independent_audit.json")
        if cached.get("accepted") is not True or any(cached[key] != value
                                                   for key, value in other_measured.items()):
            raise ValueError("other paired audit is stale")
        for path, checksum in read(other_out / "artifact_sha256.json").items():
            if digest(path) != checksum:
                raise ValueError(f"other paired artifact changed: {path}")
        before, after = (read(paths(item)["out"] / "independent_audit.json") for item in ARMS)
        write(paths("coverage_guard")["out"] / "paired_comparison.json", {
            "reference": before, "candidate": after,
            "delta_coverage_guard_minus_p1_pp": {
                key: 100 * (after["metrics"][key] - before["metrics"][key])
                for key in before["metrics"]},
            "scope": "paired complete 100 epochs, fixed split/teacher/seed0; repeated seeds still required"})


def train(arm):
    receipt = preflight(arm)
    loc = paths(arm)
    out = loc["out"]
    out.mkdir(parents=True, exist_ok=False)
    config_path = out / "config.yaml"
    config_path.write_text(yaml.safe_dump(receipt["config"], sort_keys=False), encoding="utf-8")
    write(out / "preflight.json", receipt)
    process = psutil.Process()
    write(out / "process.json", {
        "pid": process.pid, "created_at": datetime.fromtimestamp(
            process.create_time(), ZoneInfo("Asia/Shanghai")).isoformat(),
        "command": process.cmdline()})
    try:
        stage(out, "training", epochs=100)
        runner = run_training if arm == "p1" else run_coverage_guard_training
        if runner(config_path) != loc["run"]:
            raise ValueError("unexpected formal training directory")
        complete_epochs(loc["run"] / "results.csv")
        from regionssod.evaluation.runner import run_validation
        from regionssod.evaluation.size_recall import run_size_recall
        stage(out, "validation960")
        if run_validation(loc["run"] / "weights/best.pt",
                          ROOT / receipt["config"]["data"], imgsz=960, batch=8,
                          device="0", name=f"{receipt['config']['name']}_val960") != loc["val"]:
            raise ValueError("unexpected full-image validation directory")
        stage(out, "size_recall")
        run_size_recall(loc["run"] / "weights/best.pt", SOURCE, loc["recall"],
                        imgsz=960, batch=8, device="0", conf=.25, match_iou=.5)
        write(out / "metrics.json", measure(arm))
        write(out / "artifact_sha256.json", {str(path): digest(path) for path in result_paths(arm)})
        stage(out, "completed", exit_code=0)
    except BaseException:
        stage(out, "interrupted" if sys.exc_info()[0] is KeyboardInterrupt else "failed",
              error=traceback.format_exc(), exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--audit", action="store_true")
    options = parser.parse_args()
    if options.plan_only:
        print(yaml.safe_dump({arm: configuration(arm) for arm in ARMS}, sort_keys=False))
        print("PLAN ONLY: no eligibility claim, output creation, GPU work or training.")
    elif options.arm is None:
        parser.error("--arm is required outside --plan-only")
    elif options.check_only:
        receipt = preflight(options.arm)
        print(f"Eligible, {len(receipt['sha256'])} frozen sources verified; no training started.")
    elif options.audit:
        audit(options.arm)
    else:
        train(options.arm)


if __name__ == "__main__":
    main()
