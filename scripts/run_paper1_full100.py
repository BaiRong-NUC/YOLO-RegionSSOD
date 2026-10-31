"""User-run B2/P1 formal pair following the dated research investment decision."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
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
from main_seed_tables import load_main_pair
from regionssod.training.runner import load_experiment_config, run_training
from run_coverage_guard_full100 import complete_epochs, now, stage
from run_p1_scale_seed_pair import ROOT, SOURCE, digest, parameter_differences, read, replay, write

ARMS = ("b2", "p1")
DECISION = ROOT / "docs/paper/training_investment_decision.json"
PROTOCOL = ROOT / "docs/paper/paper1_full100_protocol.md"
CONFIGS = {"b2": "configs/train_naive_10.yaml", "p1": "configs/train_p1_10.yaml"}
PILOTS = {"b2": "partial10_naive_t050_pilot30_s0", "p1": "partial10_p1_pilot30_s0"}


def configuration(arm):
    config = deepcopy(load_experiment_config(ROOT / CONFIGS[arm]))
    config["name"] = f"partial10_{arm}_formal100_s0"
    config["train"]["epochs"] = 100
    return config


def paths(arm):
    name = configuration(arm)["name"]
    return {
        "out": ROOT / "outputs" / f"paper1_{arm}_formal100_s0",
        "run": ROOT / "runs" / name,
        "val": ROOT / "runs/val" / f"{name}_val960",
        "recall": ROOT / "runs/size_recall" / name.removeprefix("partial10_"),
    }


def eligible_evidence():
    decision = read(DECISION)
    comparison = ROOT / "outputs/b2_pilot30_s1/comparison.json"
    if (decision["recommendation"] != "user_run_b2_p1_full100_seed0"
            or digest(comparison) != decision["paired_evidence_sha256"]):
        raise ValueError("dated research decision or paired evidence differs")
    if not (ROOT / "outputs/b2_pilot30_s1/independent_audit.json").is_file():
        raise ValueError("independent paired pilot audit is required")
    hashes = {}
    load_main_pair(hashes)
    for filename, expected in read(ROOT / "outputs/b2_pilot30_s1/preflight.json")["sha256"].items():
        if digest(filename) != expected:
            raise ValueError(f"changed paired pilot input: {filename}")
        hashes[filename] = expected
    return hashes


def preflight(arm):
    import ultralytics

    if Path.cwd().resolve() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("use repository root and pinned yolo environment")
    if any(path.exists() for path in paths(arm).values()):
        raise FileExistsError("existing formal outputs; inspect or audit instead of restarting")
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            entries = {Path(item).name for item in command}
            if (process.pid != os.getpid() and
                    entries.intersection({"run_paper1_full100.py", "run_coverage_guard_full100.py"})
                    and ("--train" in command or "--worker" in command)):
                raise ValueError("another formal arm is active; run arms sequentially")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    hashes = eligible_evidence()
    configs = {key: configuration(key) for key in ARMS}
    if (configs["b2"]["train"] != configs["p1"]["train"]
            or configs["b2"]["model"] != configs["p1"]["model"]):
        raise ValueError("paired optimization and initialization must agree")
    for key, config in configs.items():
        pilot = read(ROOT / "runs" / PILOTS[key] / "experiment.json")
        if config["train"] != pilot["train_args"] | {"epochs": 100}:
            raise ValueError("only duration may change from each pilot")
        if (ROOT / config["data"] != Path(pilot["data"])
                or config.get("method", "baseline") != pilot.get("method", "baseline")
                or config.get("p1") != pilot.get("p1")):
            raise ValueError("pilot data and method must remain frozen")
    own = [Path(__file__).resolve(), PROTOCOL, DECISION,
           ROOT / "scripts/run_coverage_guard_full100.py",
           ROOT / "scripts/run_p1_scale_seed_pair.py",
           ROOT / "scripts/main_seed_tables.py",
           ROOT / "scripts/audit_p1_small_nwd.py",
           *(ROOT / value for value in CONFIGS.values()),
           *(ROOT / "src/regionssod").rglob("*.py")]
    relative = [path.relative_to(ROOT).as_posix() for path in own]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                   check=True, capture_output=True)
    subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *relative], check=True)
    hashes.update({str(path): digest(path) for path in own})
    return {"checked_at": now(), "arm": arm, "config": configs[arm], "sha256": hashes,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "epochs": 100, "train_tiles": 11095 if arm == "b2" else 11310,
            "split_seed": 0, "student_seed": 0, "decision": read(DECISION)}


def measure(arm):
    loc = paths(arm)
    out, run, val, rec = (loc[key] for key in ("out", "run", "val", "recall"))
    receipt = read(out / "preflight.json")
    for filename, expected in receipt["sha256"].items():
        if digest(filename) != expected:
            raise ValueError(f"changed formal source/input: {filename}")
    if load_experiment_config(out / "config.yaml") != receipt["config"]:
        raise ValueError("generated config differs from frozen receipt")
    experiment = read(run / "experiment.json")
    config = receipt["config"]
    if (experiment["train_args"] != config["train"] or experiment["name"] != config["name"]
            or experiment.get("method", "baseline") != config.get("method", "baseline")
            or experiment["p1"] != config.get("p1")
            or Path(experiment["data"]) != ROOT / config["data"]
            or experiment.get("assignment") is not None):
        raise ValueError("actual experiment differs from planned main method")
    for filename, expected in experiment["input_sha256"].items():
        if digest(filename) != expected:
            raise ValueError(f"changed experiment source: {filename}")
    original = yaml.safe_load((ROOT / "runs" / PILOTS[arm] / "args.yaml").read_text(encoding="utf-8"))
    actual = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    if actual["epochs"] != 100 or parameter_differences(original, actual, {"name", "save_dir", "epochs"}):
        raise ValueError("actual settings differ beyond duration and run name")
    rows = complete_epochs(run / "results.csv")
    validation, recall = read(val / "metrics.json"), read(rec / "metrics.json")
    reference_rec = ROOT / f"runs/size_recall/{arm}_pilot30_s0"
    reference = read(reference_rec / "metrics.json")
    if (Path(validation["weights"]) != run / "weights/best.pt" or validation["imgsz"] != 960
            or Path(validation["data"]) != ROOT / config["data"]
            or recall["weights_sha256"] != digest(run / "weights/best.pt")
            or recall["images"] != 548
            or (rec / "images.txt").read_bytes() != (reference_rec / "images.txt").read_bytes()
            or any(recall[key] != reference[key] for key in
                   ("data", "imgsz", "conf", "match_iou", "nms_iou", "max_det",
                    "matching", "size_pixel_space", "ultralytics_version"))):
        raise ValueError("full-image evaluation protocol differs")
    if (validation["class_indices"] != list(range(10)) or len(validation["class_maps"]) != 10
            or abs(sum(validation["class_maps"]) / 10
                   - validation["results"]["metrics/mAP50-95(B)"]) > 1e-12):
        raise ValueError("class AP does not reconstruct overall AP")
    counts = replay(rec)
    metrics = {"mAP50": validation["results"]["metrics/mAP50(B)"],
               "mAP50_95": validation["results"]["metrics/mAP50-95(B)"],
               **{f"{size}_recall": values["recall"] for size, values in counts.items()}}
    if not all(math.isfinite(value) for value in [*metrics.values(), *validation["class_maps"]]):
        raise ValueError("non-finite evaluation metrics")
    return {"arm": arm, "epochs": 100, "train_tiles": receipt["train_tiles"],
            "student_seconds": float(rows[-1]["time"]),
            "metrics": metrics,
            "class_maps": validation["class_maps"], "names": validation["names"]}


def artifacts(arm):
    loc = paths(arm)
    return ([loc["run"] / name for name in
             ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "experiment.json")]
            + [loc["val"] / "metrics.json"]
            + [loc["recall"] / name for name in ("metrics.json", "images.txt", "predictions.jsonl")]
            + [loc["out"] / name for name in ("preflight.json", "config.yaml", "metrics.json")])


def audit(arm):
    out = paths(arm)["out"]
    launch = read(out / "launch_receipt.json")
    if (launch.get("exit_code") != 0 or read(out / "status.json")["stage"] != "completed"
            or original_process_running(launch["training_identity"])):
        raise ValueError("real exit0, completed stages and stopped worker are required")
    result = measure(arm)
    if read(out / "metrics.json") != result:
        raise ValueError("independent measurement differs")
    for filename, expected in read(out / "artifact_sha256.json").items():
        if digest(filename) != expected:
            raise ValueError(f"changed result artifact: {filename}")
    write(out / "independent_audit.json", {"accepted": True, "audited_at": now(), **result})
    if arm == "p1":
        before = audit("b2")
        write(out / "paired_comparison.json", {
            "reference": before, "candidate": result,
            "delta_p1_minus_b2_pp": {
                key: 100 * (value - before["metrics"][key]) for key, value in result["metrics"].items()},
            "scope": "one split/fixed teacher/student seed0; unequal tile counts and compute; not significance",
        })
    return result


def worker(arm):
    from regionssod.evaluation.runner import run_validation
    from regionssod.evaluation.size_recall import run_size_recall

    loc = paths(arm)
    out = loc["out"]
    stage(out, "training", epochs=100)
    try:
        if run_training(out / "config.yaml") != loc["run"]:
            raise ValueError("unexpected training directory")
        complete_epochs(loc["run"] / "results.csv")
        stage(out, "validation960")
        config = read(out / "preflight.json")["config"]
        if run_validation(loc["run"] / "weights/best.pt", ROOT / config["data"],
                          name=f"{config['name']}_val960") != loc["val"]:
            raise ValueError("unexpected validation directory")
        stage(out, "size_recall")
        run_size_recall(loc["run"] / "weights/best.pt", SOURCE, loc["recall"])
        write(out / "metrics.json", measure(arm))
        write(out / "artifact_sha256.json", {str(p): digest(p) for p in artifacts(arm)})
        stage(out, "completed")
    except BaseException:
        stage(out, "interrupted" if sys.exc_info()[0] is KeyboardInterrupt else "failed",
              error=traceback.format_exc())
        raise


def train(arm):
    if arm == "p1":
        audit("b2")
    receipt = preflight(arm)
    out = paths(arm)["out"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(yaml.safe_dump(receipt["config"], sort_keys=False), encoding="utf-8")
    write(out / "preflight.json", receipt)
    token = os.urandom(16).hex()
    write(out / "dispatch.json", {"started_at": now(), "token": token})
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--arm", arm, "--worker", token]
    with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (
            out / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=ROOT)
        identity = psutil.Process(process.pid)
        launch = {"started_at": now(), "training_identity": {
            "pid": process.pid, "created_at": datetime.fromtimestamp(
                identity.create_time(), ZoneInfo("Asia/Shanghai")).isoformat(), "command": identity.cmdline()}}
        write(out / "launch_receipt.json", launch)
        print(f"Training log: {out / 'stdout.log'}", flush=True)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            print("Controller interrupted; inspect the recorded worker before taking any action.", flush=True)
            raise
    write(out / "launch_receipt.json", launch | {"finished_at": now(), "exit_code": code})
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
        print(f"Preflight passed: {len(receipt['sha256'])} frozen files; no training or inference.")
    elif options.audit:
        print({"accepted": True, **audit(options.arm)})
    elif options.worker:
        if read(paths(options.arm)["out"] / "dispatch.json")["token"] != options.worker:
            raise ValueError("worker must be launched by the recorded controller")
        identity = read(paths(options.arm)["out"] / "launch_receipt.json")["training_identity"]
        if identity["pid"] != os.getpid() or not original_process_running(identity):
            raise ValueError("worker identity differs from the recorded launch")
        worker(options.arm)
    else:
        train(options.arm)


if __name__ == "__main__":
    main()
