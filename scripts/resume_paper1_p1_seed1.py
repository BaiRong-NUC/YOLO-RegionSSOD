"""Explicit continuation of the captured P1 seed1 interruption after epoch 55."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from functools import partial
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import psutil
import yaml

from audit_p1_small_nwd import original_process_running
from paper1_recovery import verify_resume_arguments
import run_paper1_seed1_full100 as repeat
from run_p1_scale_seed_pair import ROOT, SOURCE, digest, parameter_differences, read, replay, write

ENTRY = Path(__file__).resolve()
DOCUMENT = ROOT / "docs/paper/paper1_p1_seed1_recovery.md"
ARCHIVE = ROOT / "outputs/paper1_p1_formal100_s1_recovery_20261102"
LOC = repeat.paths("p1")
CONTINUITY = "checkpoint continuation after epoch55; not bitwise equivalent to uninterrupted training"


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return [{key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(handle)]


def validate_capture(*, originals=False):
    capture = read(ARCHIVE / "capture.json")
    if (capture["arm"] != "p1" or capture["student_seed"] != 1
            or capture["completed_epochs"] != 55):
        raise ValueError("capture is not the registered P1 seed1 interruption")
    for saved, item in capture["files"].items():
        if digest(saved) != item["sha256"]:
            raise ValueError(f"changed interruption archive: {saved}")
        if originals and digest(item["original"]) != item["sha256"]:
            raise ValueError(f"original evidence changed before continuation: {item['original']}")
    return capture


def verify_frozen():
    receipt = read(LOC["out"] / "preflight.json")
    for path, expected in receipt["sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"changed frozen seed1 source/input: {path}")
    config = repeat.configuration("p1")
    if (receipt["config"] != config
            or yaml.safe_load((LOC["out"] / "config.yaml").read_text(encoding="utf-8")) != config):
        raise ValueError("changed original P1 configuration")
    return receipt


def validate_checkpoint(checkpoint, original, rows):
    import torch

    if (checkpoint.get("epoch") != 54 or len(rows) != 55
            or [int(row["epoch"]) for row in rows] != list(range(1, 56))
            or not all(math.isfinite(float(v)) for row in rows for v in row.values())):
        raise ValueError("checkpoint and continuous finite epochs1-55 must agree")
    optimizer = checkpoint.get("optimizer")
    if (checkpoint.get("version") != "8.4.172" or not isinstance(optimizer, dict)
            or len(optimizer.get("state", {})) != 255 or checkpoint.get("ema") is None
            or checkpoint.get("updates") != 10496 or not isinstance(checkpoint.get("scaler"), dict)):
        raise ValueError("captured optimizer/EMA/scaler state differs or is missing")
    tensors = [*checkpoint["ema"].state_dict().values(),
               *(v for state in optimizer["state"].values()
                 for v in state.values() if torch.is_tensor(v))]
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("non-finite checkpoint or optimizer")
    if (checkpoint["train_args"] != original or original["epochs"] != 100
            or original["seed"] != 1 or original["resume"]):
        raise ValueError("checkpoint arguments differ from the original P1 seed1 plan")


def no_active_training():
    names = {ENTRY.name, "run_paper1_seed1_full100.py", "run_paper1_full100.py"}
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if (process.pid != os.getpid() and any(Path(v).name in names for v in command)
                    and any(flag in command for flag in ("--train", "--resume", "--worker"))):
                raise ValueError("another formal training/controller remains active")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def check():
    import torch
    import ultralytics

    if Path.cwd().resolve() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("use repository root and pinned yolo environment")
    no_active_training()
    out = LOC["out"]
    if any(path.exists() for path in (LOC["val"], LOC["recall"], out / "recovery.json",
                                     out / "resume_dispatch.json", out / "resume_launch_receipt.json")):
        raise FileExistsError("existing continuation/evaluation requires separate inspection")
    capture = validate_capture(originals=True)
    launch = read(out / "launch_receipt.json")
    if (read(out / "status.json")["stage"] != "interrupted"
            or original_process_running(launch["training_identity"])):
        raise ValueError("captured interrupted worker must have stopped")
    receipt = verify_frozen()
    original = yaml.safe_load((LOC["run"] / "args.yaml").read_text(encoding="utf-8"))
    checkpoint = LOC["run"] / "weights/last.pt"
    validate_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=False),
                        original, csv_rows(LOC["run"] / "results.csv"))
    repeat.audit("b2")
    own = [ENTRY, DOCUMENT]
    relative = [p.relative_to(ROOT).as_posix() for p in own]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                   check=True, capture_output=True)
    subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *relative], check=True)
    return {
        "archive": str(ARCHIVE), "capture_sha256": digest(ARCHIVE / "capture.json"),
        "completed_epochs": 55, "resume_epoch": 56, "total_epochs": 100,
        "student_seed": 1, "split_seed": 0, "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint), "frozen_source_count": len(receipt["sha256"]),
        "original_exit_code": capture["original_exit_code"],
        "recovery_sources": {str(p): digest(p) for p in own},
        "continuity": CONTINUITY,
    }


def recovery():
    record = read(LOC["out"] / "recovery.json")
    if (Path(record["archive"]) != ARCHIVE
            or record["capture_sha256"] != digest(ARCHIVE / "capture.json")
            or record["completed_epochs"] != 55 or record["total_epochs"] != 100
            or Path(record["checkpoint"]) != LOC["run"] / "weights/last.pt"):
        raise ValueError("continuation record differs from capture")
    validate_capture()
    verify_frozen()
    if set(record["recovery_sources"]) != {str(ENTRY), str(DOCUMENT)}:
        raise ValueError("continuation implementation manifest differs")
    for path, expected in record["recovery_sources"].items():
        if digest(path) != expected:
            raise ValueError(f"changed continuation implementation: {path}")
    return record


def student_time(run, archive=ARCHIVE):
    old = archive / "original_run/results.csv"
    current = csv_rows(run / "results.csv")
    previous = csv_rows(old)
    if not (run / "results.csv").read_bytes().startswith(old.read_bytes()):
        raise ValueError("continuation must preserve the exact original CSV prefix")
    segment = current[55:]
    if len(previous) != 55 or [int(row["epoch"]) for row in segment] != list(range(56, 101)):
        raise ValueError("continuation must complete exactly epochs56-100")
    times = [float(row["time"]) for row in segment]
    if not all(math.isfinite(v) and v > 0 for v in times) or any(
            b < a for a, b in zip(times, times[1:])):
        raise ValueError("invalid continuation segment times")
    return float(previous[-1]["time"]) + times[-1]


def verify_setup(trainer, checkpoint, run, *, engineering=False):
    import torch
    from regionssod.training.p1 import P1Trainer

    if (not isinstance(trainer, P1Trainer)
            or trainer.p1_options != {"soft_weights": True, "uncertain_ignore": True}
            or trainer.start_epoch != 55 or trainer.epochs != 100
            or trainer.scheduler.last_epoch != 54
            or Path(trainer.save_dir).resolve() != run.resolve()):
        raise ValueError("P1 trainer, options, continuation schedule or destination differs")
    original = yaml.safe_load((ARCHIVE / "original_run/args.yaml").read_text(encoding="utf-8"))
    actual = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    if engineering:
        original = original | {"save_dir": str(run), "name": run.name}
    verify_resume_arguments(original, actual, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = trainer.optimizer.state_dict()
    if restored.keys() != saved["optimizer"].keys():
        raise ValueError("optimizer structure differs")
    if restored["param_groups"] != saved["optimizer"]["param_groups"]:
        raise ValueError("optimizer parameter groups differ")
    if restored["state"].keys() != saved["optimizer"]["state"].keys():
        raise ValueError("optimizer state keys differ")
    for key, state in saved["optimizer"]["state"].items():
        if state.keys() != restored["state"][key].keys():
            raise ValueError("optimizer item keys differ")
        for name, value in state.items():
            target = restored["state"][key][name]
            equal = torch.equal(target.cpu(), value.cpu()) if torch.is_tensor(value) else target == value
            if not equal:
                raise ValueError("optimizer state did not restore exactly")
    if trainer.ema.updates != saved["updates"] or trainer.scaler.state_dict() != saved["scaler"]:
        raise ValueError("EMA updates or AMP scaler did not restore")
    ema = trainer.ema.ema.state_dict()
    expected_ema = saved["ema"].float().state_dict()
    if ema.keys() != expected_ema.keys() or any(
            not torch.equal(ema[key].cpu(), value.cpu()) for key, value in expected_ema.items()):
        raise ValueError("EMA weights did not restore exactly")
    return {
        "start_epoch_zero_based": 55, "next_epoch_one_based": 56, "total_epochs": 100,
        "optimizer_state_entries": len(restored["state"]), "optimizer_state_exactly_restored": True,
        "ema_updates": trainer.ema.updates, "ema_weights_exactly_restored": True,
        "scaler_restored": True,
        "scheduler_last_epoch": trainer.scheduler.last_epoch, "p1_trainer": True,
        "p1_options": trainer.p1_options, "actual_args_sha256": digest(run / "args.yaml"),
        "continuity": CONTINUITY,
    }


def train_from_checkpoint():
    from ultralytics import YOLO
    from regionssod.training.p1 import P1Trainer

    record = recovery()
    checkpoint = Path(record["checkpoint"])
    if digest(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("captured last.pt changed before continuation")
    model = YOLO(checkpoint)

    def setup(trainer):
        write(LOC["out"] / "resume_setup.json", verify_setup(trainer, checkpoint, LOC["run"]))

    model.add_callback("on_pretrain_routine_end", setup)
    result = model.train(resume=True, trainer=partial(
        P1Trainer, p1_options={"soft_weights": True, "uncertain_ignore": True}))
    if Path(result.save_dir).resolve() != LOC["run"]:
        raise ValueError("unexpected continued training destination")


def measure():
    record = recovery()
    out, run, val, rec = (LOC[key] for key in ("out", "run", "val", "recall"))
    receipt = read(out / "preflight.json")
    config = receipt["config"]
    experiment = read(run / "experiment.json")
    if (experiment["train_args"] != config["train"] or experiment["name"] != config["name"]
            or experiment.get("method") != "p1" or experiment.get("p1") != config.get("p1")
            or Path(experiment["data"]) != ROOT / config["data"]
            or experiment.get("assignment") is not None):
        raise ValueError("original P1 experiment differs")
    for path, expected in experiment["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"changed experiment input: {path}")
    original = yaml.safe_load((ARCHIVE / "original_run/args.yaml").read_text(encoding="utf-8"))
    actual = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    verify_resume_arguments(original, actual, record["checkpoint"])
    pilot = yaml.safe_load((ROOT / "runs/partial10_p1_pilot30_s1/args.yaml").read_text(encoding="utf-8"))
    if (original["epochs"] != 100 or original["seed"] != 1 or original["resume"]
            or parameter_differences(pilot, original, {"epochs", "name", "save_dir"})):
        raise ValueError("original seed1 arguments differ from intended duration/name")
    repeat.seed0.complete_epochs(run / "results.csv")
    seconds = student_time(run)
    setup = read(out / "resume_setup.json")
    if (setup["start_epoch_zero_based"] != 55 or setup["total_epochs"] != 100
            or setup["ema_updates"] != 10496 or setup["optimizer_state_entries"] != 255
            or not setup["p1_trainer"] or not setup["optimizer_state_exactly_restored"]
            or not setup["ema_weights_exactly_restored"] or not setup["scaler_restored"]
            or setup["scheduler_last_epoch"] != 54
            or setup["p1_options"] != {"soft_weights": True, "uncertain_ignore": True}
            or setup["actual_args_sha256"] != digest(run / "args.yaml")):
        raise ValueError("recorded continuation setup differs")
    validation, recall = read(val / "metrics.json"), read(rec / "metrics.json")
    before = repeat.seed0.paths("p1")["recall"]
    protocol = read(before / "metrics.json")
    if (Path(validation["weights"]) != run / "weights/best.pt"
            or Path(validation["data"]) != ROOT / config["data"] or validation["imgsz"] != 960
            or recall["weights_sha256"] != digest(run / "weights/best.pt") or recall["images"] != 548
            or (rec / "images.txt").read_bytes() != (before / "images.txt").read_bytes()
            or any(recall[key] != protocol[key] for key in
                   ("data", "imgsz", "conf", "match_iou", "nms_iou", "max_det",
                    "matching", "size_pixel_space", "ultralytics_version"))):
        raise ValueError("full-image evaluation protocol differs")
    counts = replay(rec)
    metrics = {
        "mAP50": validation["results"]["metrics/mAP50(B)"],
        "mAP50_95": validation["results"]["metrics/mAP50-95(B)"],
        **{f"{size}_recall": value["recall"] for size, value in counts.items()},
    }
    if (validation["class_indices"] != list(range(10)) or len(validation["class_maps"]) != 10
            or not all(math.isfinite(v) for v in [*metrics.values(), *validation["class_maps"]])
            or abs(sum(validation["class_maps"]) / 10 - metrics["mAP50_95"]) > 1e-12):
        raise ValueError("invalid overall or class metrics")
    return {"arm": "p1", "epochs": 100, "student_seed": 1, "split_seed": 0,
            "train_tiles": receipt["train_tiles"], "student_seconds": seconds, "metrics": metrics,
            "class_maps": validation["class_maps"], "names": validation["names"],
            "continuity": CONTINUITY}


def artifacts():
    return repeat.artifacts("p1") + [LOC["out"] / name for name in
                                   ("recovery.json", "resume_setup.json", "resume_dispatch.json")]


def audit():
    out = LOC["out"]
    launch = read(out / "resume_launch_receipt.json")
    if (launch.get("exit_code") != 0 or read(out / "resume_status.json")["stage"] != "completed"
            or original_process_running(launch["training_identity"])):
        raise ValueError("real continuation exit0 and stopped worker are required")
    measured = measure()
    if measured != read(out / "metrics.json"):
        raise ValueError("independent continued measurement differs")
    manifest = read(out / "artifact_sha256.json")
    if set(manifest) != {str(path) for path in artifacts()}:
        raise ValueError("continuation artifact manifest differs")
    for path, expected in manifest.items():
        if digest(path) != expected:
            raise ValueError(f"changed result artifact: {path}")
    comparison = repeat.paired_summary(repeat.audit("b2"), measured)
    repeat.seed0.audit("p1")
    original = read(repeat.seed0.paths("p1")["out"] / "paired_comparison.json")
    write(out / "independent_audit.json", {"accepted": True, "audited_at": repeat.seed0.now(), **measured})
    write(out / "paired_comparison.json", comparison)
    write(out / "two_seed_comparison.json", {
        "epochs_per_student": 100, "student_seeds": [0, 1], "split_seed": 0,
        "per_seed": {"0": original, "1": comparison},
        "paired_delta_pp": {
            key: {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}
            for key in comparison["delta_p1_minus_b2_pp"]
            for values in [[original["delta_p1_minus_b2_pp"][key],
                            comparison["delta_p1_minus_b2_pp"][key]]]},
        "scope": "fixed split/teacher; seed0 B2 and seed1 P1 resumed; not significance",
    })
    return measured


def stage(name, **extra):
    write(LOC["out"] / "resume_status.json", {"stage": name, "updated_at": repeat.seed0.now(), **extra})


def worker():
    from regionssod.evaluation.runner import run_validation
    from regionssod.evaluation.size_recall import run_size_recall

    try:
        stage("training", epochs=100, resume_epoch=56, student_seed=1)
        train_from_checkpoint()
        repeat.seed0.complete_epochs(LOC["run"] / "results.csv")
        stage("validation960")
        config = repeat.configuration("p1")
        if run_validation(LOC["run"] / "weights/best.pt", ROOT / config["data"],
                          name=f"{config['name']}_val960") != LOC["val"]:
            raise ValueError("unexpected validation destination")
        stage("size_recall")
        run_size_recall(LOC["run"] / "weights/best.pt", SOURCE, LOC["recall"])
        write(LOC["out"] / "metrics.json", measure())
        write(LOC["out"] / "artifact_sha256.json", {str(p): digest(p) for p in artifacts()})
        stage("completed")
    except BaseException:
        stage("interrupted" if sys.exc_info()[0] is KeyboardInterrupt else "failed",
              error=traceback.format_exc())
        raise


def resume():
    record = check()
    out = LOC["out"]
    write(out / "recovery.json", record)
    token = os.urandom(16).hex()
    write(out / "resume_dispatch.json", {"started_at": repeat.seed0.now(), "token": token})
    with (out / "resume_stdout.log").open("x", encoding="utf-8") as stdout, (
            out / "resume_stderr.log").open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen([sys.executable, "-u", str(ENTRY), "--worker", token],
                                   cwd=ROOT, stdout=stdout, stderr=stderr)
        identity = psutil.Process(process.pid)
        launch = {"started_at": repeat.seed0.now(), "training_identity": {
            "pid": process.pid, "created_at": datetime.fromtimestamp(
                identity.create_time(), ZoneInfo("Asia/Shanghai")).isoformat(),
            "command": identity.cmdline()}}
        wrapper = psutil.Process()
        launch["wrapper_identity"] = {
            "pid": wrapper.pid, "created_at": datetime.fromtimestamp(
                wrapper.create_time(), ZoneInfo("Asia/Shanghai")).isoformat(),
            "command": wrapper.cmdline()}
        pending = out / "resume_launch_receipt.pending.json"
        write(pending, launch)
        pending.replace(out / "resume_launch_receipt.json")
        print(f"Training log: {out / 'resume_stdout.log'}", flush=True)
        code = process.wait()
    write(out / "resume_launch_receipt.json",
          launch | {"finished_at": repeat.seed0.now(), "exit_code": code})
    if code:
        raise SystemExit(code)
    print({"accepted": True, **audit()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_only:
        print(check())
    elif args.audit:
        print({"accepted": True, **audit()})
    elif args.worker:
        out = LOC["out"]
        if read(out / "resume_dispatch.json")["token"] != args.worker:
            raise ValueError("continuation worker token differs")
        # The child may import faster than the parent can write its receipt.
        receipt = out / "resume_launch_receipt.json"
        deadline = time.monotonic() + 10
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        identity = read(receipt)["training_identity"]
        if identity["pid"] != os.getpid() or not original_process_running(identity):
            raise ValueError("continuation worker identity differs")
        worker()
    else:
        resume()


if __name__ == "__main__":
    main()
