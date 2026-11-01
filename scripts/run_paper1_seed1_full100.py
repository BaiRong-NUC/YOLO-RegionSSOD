"""User-run seed1 confirmation of the completed 100-epoch B2/P1 pair."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import psutil
import yaml

from audit_p1_small_nwd import original_process_running
from regionssod.training.runner import load_experiment_config, run_training
import run_paper1_full100 as seed0
from run_p1_scale_seed_pair import ROOT, SOURCE, digest, parameter_differences, read, replay, write

ARMS = seed0.ARMS
PROTOCOL = ROOT / "docs/paper/paper1_seed1_full100_protocol.md"


def configuration(arm):
    config = deepcopy(seed0.configuration(arm))
    config["name"] = config["name"].removesuffix("_s0") + "_s1"
    config["train"]["seed"] = 1
    return config


def paths(arm):
    name = configuration(arm)["name"]
    return {
        "out": ROOT / "outputs" / f"paper1_{arm}_formal100_s1",
        "run": ROOT / "runs" / name,
        "val": ROOT / "runs/val" / f"{name}_val960",
        "recall": ROOT / "runs/size_recall" / name.removeprefix("partial10_"),
    }


def basis_hashes():
    seed0.audit("p1")
    hashes = dict(read(seed0.paths("p1")["out"] / "preflight.json")["sha256"])
    for filename, expected in hashes.items():
        if digest(filename) != expected:
            raise ValueError(f"changed seed0 source: {filename}")
    # Audit timestamps can change on read-only re-audit; freeze the underlying
    # evidence, not the timestamped acceptance receipts.
    for arm in ARMS:
        for path in seed0.artifacts(arm):
            hashes[str(path)] = digest(path)
    comparison = seed0.paths("p1")["out"] / "paired_comparison.json"
    hashes[str(comparison)] = digest(comparison)
    return hashes


def preflight(arm):
    import ultralytics

    if Path.cwd().resolve() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("use repository root and pinned yolo environment")
    if any(path.exists() for path in paths(arm).values()):
        raise FileExistsError("existing seed1 output; never overwrite or repeat")
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if (process.pid != os.getpid()
                    and any(Path(value).name in {Path(__file__).name, "run_paper1_full100.py"}
                            for value in command)
                    and any(flag in command for flag in ("--train", "--resume", "--worker"))):
                raise ValueError("another formal arm remains active; run sequentially")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    hashes = basis_hashes()
    for item in ARMS:
        before, after = seed0.configuration(item), configuration(item)
        expected = deepcopy(before)
        expected["name"] = after["name"]
        expected["train"]["seed"] = 1
        if after != expected:
            raise ValueError("seed1 may change only student seed and name")
    if configuration("b2")["train"] != configuration("p1")["train"]:
        raise ValueError("seed1 paired schedules differ")
    own = [Path(__file__).resolve(), PROTOCOL, Path(seed0.__file__).resolve()]
    relative = [path.relative_to(ROOT).as_posix() for path in own]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                   check=True, capture_output=True)
    subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *relative], check=True)
    hashes.update({str(path): digest(path) for path in own})
    return {
        "checked_at": seed0.now(), "arm": arm, "config": configuration(arm),
        "sha256": hashes, "epochs": 100, "student_seed": 1, "split_seed": 0,
        "train_tiles": 11095 if arm == "b2" else 11310,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": "repeat student randomness only; fixed split, teacher and pseudo labels",
    }


def measure(arm):
    loc = paths(arm)
    out, run, val, rec = (loc[key] for key in ("out", "run", "val", "recall"))
    receipt = read(out / "preflight.json")
    for filename, expected in receipt["sha256"].items():
        if digest(filename) != expected:
            raise ValueError(f"changed seed1 source/input: {filename}")
    config = configuration(arm)
    if receipt["config"] != config or load_experiment_config(out / "config.yaml") != config:
        raise ValueError("seed1 generated configuration differs")
    experiment = read(run / "experiment.json")
    if (experiment["train_args"] != config["train"] or experiment["name"] != config["name"]
            or experiment.get("method", "baseline") != config.get("method", "baseline")
            or experiment.get("p1") != config.get("p1")
            or Path(experiment["data"]) != ROOT / config["data"]
            or experiment.get("assignment") is not None):
        raise ValueError("actual seed1 method differs from planned configuration")
    for filename, expected in experiment["input_sha256"].items():
        if digest(filename) != expected:
            raise ValueError(f"changed experiment input: {filename}")
    pilot_name = "partial10_naive_t050_pilot30_s1" if arm == "b2" else "partial10_p1_pilot30_s1"
    pilot = yaml.safe_load((ROOT / "runs" / pilot_name / "args.yaml").read_text(encoding="utf-8"))
    actual = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    if (actual["epochs"] != 100 or actual["seed"] != 1 or actual["resume"]
            or parameter_differences(pilot, actual, {"epochs", "name", "save_dir"})):
        raise ValueError("actual seed1 arguments differ beyond intended duration/name")
    rows = seed0.complete_epochs(run / "results.csv")
    validation, recall = read(val / "metrics.json"), read(rec / "metrics.json")
    before = seed0.paths(arm)["recall"]
    protocol = read(before / "metrics.json")
    if (Path(validation["weights"]) != run / "weights/best.pt"
            or Path(validation["data"]) != ROOT / config["data"]
            or validation["imgsz"] != 960 or recall["weights_sha256"] != digest(run / "weights/best.pt")
            or recall["images"] != 548
            or (rec / "images.txt").read_bytes() != (before / "images.txt").read_bytes()
            or any(recall[key] != protocol[key] for key in
                   ("data", "imgsz", "conf", "match_iou", "nms_iou", "max_det",
                    "matching", "size_pixel_space", "ultralytics_version"))):
        raise ValueError("seed1 full-image evaluation protocol differs")
    counts = replay(rec)
    metrics = {
        "mAP50": validation["results"]["metrics/mAP50(B)"],
        "mAP50_95": validation["results"]["metrics/mAP50-95(B)"],
        **{f"{size}_recall": value["recall"] for size, value in counts.items()},
    }
    if (validation["class_indices"] != list(range(10)) or len(validation["class_maps"]) != 10
            or not all(math.isfinite(value) for value in [*metrics.values(), *validation["class_maps"]])
            or abs(sum(validation["class_maps"]) / 10 - metrics["mAP50_95"]) > 1e-12):
        raise ValueError("invalid overall or class metrics")
    return {
        "arm": arm, "epochs": 100, "student_seed": 1, "split_seed": 0,
        "train_tiles": receipt["train_tiles"], "student_seconds": float(rows[-1]["time"]),
        "metrics": metrics, "class_maps": validation["class_maps"], "names": validation["names"],
        "continuity": "uninterrupted",
    }


def artifacts(arm):
    loc = paths(arm)
    return ([loc["run"] / name for name in
             ("weights/best.pt", "weights/last.pt", "args.yaml", "experiment.json", "results.csv")]
            + [loc["val"] / "metrics.json"]
            + [loc["recall"] / name for name in ("metrics.json", "images.txt", "predictions.jsonl")]
            + [loc["out"] / name for name in ("preflight.json", "config.yaml", "metrics.json")])


def paired_summary(before, after):
    if any(record.get("epochs") != 100 for record in (before, after)):
        raise ValueError("only paired complete 100-epoch records can be compared")
    return {
        "reference": before, "candidate": after,
        "delta_p1_minus_b2_pp": {
            key: 100 * (value - before["metrics"][key]) for key, value in after["metrics"].items()},
        "scope": "fixed split/teacher, student seed1; equal epochs with unequal compute",
    }


def audit(arm):
    out = paths(arm)["out"]
    launch = read(out / "launch_receipt.json")
    if (launch.get("exit_code") != 0 or read(out / "status.json")["stage"] != "completed"
            or original_process_running(launch["training_identity"])):
        raise ValueError("real exit0 and terminal stopped worker are required")
    measured = measure(arm)
    if measured != read(out / "metrics.json"):
        raise ValueError("independent seed1 measurement differs")
    for filename, expected in read(out / "artifact_sha256.json").items():
        if digest(filename) != expected:
            raise ValueError(f"changed result artifact: {filename}")
    write(out / "independent_audit.json", {"accepted": True, "audited_at": seed0.now(), **measured})
    if arm == "p1":
        comparison = paired_summary(audit("b2"), measured)
        write(out / "paired_comparison.json", comparison)
        seed0.audit("p1")
        original = read(seed0.paths("p1")["out"] / "paired_comparison.json")
        write(out / "two_seed_comparison.json", {
            "epochs_per_student": 100, "student_seeds": [0, 1], "split_seed": 0,
            "per_seed": {"0": original, "1": comparison},
            "paired_delta_pp": {
                key: {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}
                for key in comparison["delta_p1_minus_b2_pp"]
                for values in [[original["delta_p1_minus_b2_pp"][key],
                                comparison["delta_p1_minus_b2_pp"][key]]]},
            "scope": "two student seeds, fixed split and teacher; seed0 B2 resumed; not significance",
        })
    return measured


def worker(arm):
    from regionssod.evaluation.runner import run_validation
    from regionssod.evaluation.size_recall import run_size_recall

    loc = paths(arm)
    out = loc["out"]
    try:
        seed0.stage(out, "training", epochs=100, student_seed=1)
        if run_training(out / "config.yaml") != loc["run"]:
            raise ValueError("unexpected seed1 training directory")
        seed0.complete_epochs(loc["run"] / "results.csv")
        seed0.stage(out, "validation960")
        config = read(out / "preflight.json")["config"]
        if run_validation(loc["run"] / "weights/best.pt", ROOT / config["data"],
                          name=f"{config['name']}_val960") != loc["val"]:
            raise ValueError("unexpected seed1 validation directory")
        seed0.stage(out, "size_recall")
        run_size_recall(loc["run"] / "weights/best.pt", SOURCE, loc["recall"])
        write(out / "metrics.json", measure(arm))
        write(out / "artifact_sha256.json", {str(path): digest(path) for path in artifacts(arm)})
        seed0.stage(out, "completed")
    except BaseException:
        seed0.stage(out, "interrupted" if sys.exc_info()[0] is KeyboardInterrupt else "failed",
                    error=traceback.format_exc())
        raise


def train(arm):
    if arm == "p1":
        audit("b2")
    receipt = preflight(arm)
    out = paths(arm)["out"]
    out.mkdir(exist_ok=False)
    write(out / "preflight.json", receipt)
    (out / "config.yaml").write_text(yaml.safe_dump(receipt["config"], sort_keys=False), encoding="utf-8")
    token = os.urandom(16).hex()
    write(out / "dispatch.json", {"started_at": seed0.now(), "token": token})
    with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (
            out / "stderr.log").open("w", encoding="utf-8") as stderr:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--arm", arm, "--worker", token]
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        identity = psutil.Process(process.pid)
        launch = {"started_at": seed0.now(), "training_identity": {
            "pid": process.pid, "created_at": datetime.fromtimestamp(
                identity.create_time(), ZoneInfo("Asia/Shanghai")).isoformat(),
            "command": identity.cmdline()}}
        write(out / "launch_receipt.json", launch)
        print(f"Training log: {out / 'stdout.log'}", flush=True)
        code = process.wait()
    write(out / "launch_receipt.json", launch | {"finished_at": seed0.now(), "exit_code": code})
    if code:
        raise SystemExit(code)
    print({"accepted": True, **audit(arm)}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--worker", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.check_only:
        receipt = preflight(options.arm)
        print(f"Seed1 preflight: {len(receipt['sha256'])} frozen files; no training or inference.")
    elif options.audit:
        print({"accepted": True, **audit(options.arm)})
    elif options.worker:
        out = paths(options.arm)["out"]
        if read(out / "dispatch.json")["token"] != options.worker:
            raise ValueError("worker token differs")
        identity = read(out / "launch_receipt.json")["training_identity"]
        if identity["pid"] != os.getpid() or not original_process_running(identity):
            raise ValueError("worker identity differs")
        worker(options.arm)
    else:
        train(options.arm)


if __name__ == "__main__":
    main()
