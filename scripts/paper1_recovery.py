"""Explicit recovery of the captured B2 interruption; never retry automatically."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess

import yaml

from audit_p1_small_nwd import original_process_running
from run_p1_scale_seed_pair import ROOT, digest, parameter_differences, read, write

ARCHIVE = ROOT / "outputs/paper1_b2_formal100_recovery_20261031"
ENTRY = ROOT / "scripts/run_paper1_full100.py"
DOCUMENT = ROOT / "docs/paper/paper1_full100_recovery.md"


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_capture():
    capture = read(ARCHIVE / "capture.json")
    for path, item in capture["files"].items():
        if digest(path) != item["sha256"]:
            raise ValueError(f"changed interruption archive: {path}")
    return capture


def recovery(out):
    path = out / "recovery.json"
    if not path.exists():
        return None
    record = read(path)
    if Path(record["archive"]) != ARCHIVE or digest(ARCHIVE / "capture.json") != record["capture_sha256"]:
        raise ValueError("recovery capture differs")
    validate_capture()
    for path, expected in record["recovery_sources"].items():
        if digest(path) != expected:
            raise ValueError(f"changed recovery implementation: {path}")
    return record


def verify_sources(out, hashes):
    record = recovery(out)
    for path, expected in hashes.items():
        if digest(path) == expected:
            continue
        if (record is None or Path(path) != ENTRY
                or digest(ARCHIVE / "original_source/scripts/run_paper1_full100.py") != expected
                or digest(path) != record["recovery_sources"].get(str(ENTRY))):
            raise ValueError(f"changed formal source/input: {path}")
    return record


def verify_resume_arguments(original, actual, checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if (Path(actual["model"]).resolve() != checkpoint
            or not isinstance(actual.get("resume"), (str, Path))
            or Path(actual["resume"]).resolve() != checkpoint):
        raise ValueError("actual resume must identify the captured last.pt")
    if parameter_differences(original, actual, {"model", "resume"}):
        raise ValueError("resume changed settings beyond model/resume bookkeeping")


def student_time(run, record):
    current = rows(run / "results.csv")
    if record is None:
        return float(current[-1]["time"])
    old_csv = ARCHIVE / "original_run/results.csv"
    previous = rows(old_csv)
    if not (run / "results.csv").read_bytes().startswith(old_csv.read_bytes()):
        raise ValueError("resumed CSV must preserve the exact original prefix")
    count = record["completed_epochs"]
    if len(previous) != count or len(current) <= count:
        raise ValueError("resumed epochs are missing")
    segment = current[count:]
    if [int(float(row["epoch"])) for row in segment] != list(range(count + 1, 101)):
        raise ValueError("resumed CSV does not finish the original 100-epoch schedule")
    times = [float(row["time"]) for row in segment]
    if times[0] <= 0 or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("non-monotonic resumed segment time")
    return float(previous[-1]["time"]) + times[-1]


def check(arm, locations):
    import psutil
    import torch
    import ultralytics

    if arm != "b2":
        raise ValueError("this captured recovery applies only to B2")
    if Path.cwd().resolve() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("use repository root and pinned yolo environment")
    out, run = locations["out"], locations["run"]
    if any(path.exists() for path in (locations["val"], locations["recall"],
                                     out / "recovery.json", out / "resume_dispatch.json")):
        raise FileExistsError("existing recovery/evaluation requires inspection, not another retry")
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if (process.pid != os.getpid()
                    and any(Path(value).name == ENTRY.name for value in command)
                    and any(flag in command for flag in ("--train", "--resume", "--worker"))):
                raise ValueError("formal training or controller remains active")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    capture = validate_capture()
    if read(out / "status.json")["stage"] != "interrupted":
        raise ValueError("only the captured interrupted training may resume")
    launch = read(out / "launch_receipt.json")
    if original_process_running(launch["training_identity"]):
        raise ValueError("original training process remains active")
    # Every mutable input must still equal the capture before the first recovery.
    for path, item in capture["files"].items():
        if "original_source" not in Path(path).parts and digest(item["original"]) != item["sha256"]:
            raise ValueError(f"original evidence changed before recovery: {item['original']}")
    frozen = read(out / "preflight.json")["sha256"]
    source_snapshot = ARCHIVE / "original_source/scripts/run_paper1_full100.py"
    if frozen.get(str(ENTRY)) != digest(source_snapshot):
        raise ValueError("archived entry does not match the original frozen source")
    for path, expected in frozen.items():
        if Path(path) != ENTRY and digest(path) != expected:
            raise ValueError(f"changed original training input: {path}")
    checkpoint = run / "weights/last.pt"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    old = rows(run / "results.csv")
    count = ckpt.get("epoch", -1) + 1
    if (count != capture["completed_epochs"] or not 0 < count < 100
            or [int(float(row["epoch"])) for row in old] != list(range(1, count + 1))):
        raise ValueError("checkpoint epoch and continuous CSV disagree")
    if (ckpt.get("version") != "8.4.172"
            or not isinstance(ckpt.get("optimizer"), dict)
            or not ckpt["optimizer"].get("state") or ckpt.get("ema") is None
            or "scaler" not in ckpt):
        raise ValueError("checkpoint lacks required resumable training state")
    if not all(torch.isfinite(tensor).all() for tensor in ckpt["ema"].state_dict().values()):
        raise ValueError("non-finite checkpoint weights")
    optimizer_values = ckpt["optimizer"]["state"].values()
    if not all(torch.isfinite(value).all() for state in optimizer_values
               for value in state.values() if isinstance(value, torch.Tensor)):
        raise ValueError("non-finite optimizer state")
    actual = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    if ckpt["train_args"] != actual or actual["epochs"] != 100 or actual["resume"]:
        raise ValueError("checkpoint arguments differ from the interrupted run")
    implementation = [ENTRY, Path(__file__).resolve(), DOCUMENT]
    relative = [p.relative_to(ROOT).as_posix() for p in implementation]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                   check=True, capture_output=True)
    subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *relative], check=True)
    return {"archive": str(ARCHIVE), "capture_sha256": digest(ARCHIVE / "capture.json"),
            "completed_epochs": count, "resume_epoch": count + 1, "total_epochs": 100,
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
            "recovery_sources": {str(path): digest(path) for path in implementation},
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "original_exit_code": capture["original_exit_code"],
            "continuity": "checkpoint continuation; not bitwise equivalent to uninterrupted training"}


def train_from_checkpoint(out, run):
    from ultralytics import YOLO

    record = recovery(out)
    if record is None or digest(record["checkpoint"]) != record["checkpoint_sha256"]:
        raise ValueError("recorded recovery checkpoint changed")
    model = YOLO(record["checkpoint"])

    def verify_setup(trainer):
        actual = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        original = yaml.safe_load((ARCHIVE / "original_run/args.yaml").read_text(encoding="utf-8"))
        verify_resume_arguments(original, actual, record["checkpoint"])
        if (Path(trainer.save_dir).resolve() != run.resolve()
                or trainer.start_epoch != record["completed_epochs"] or trainer.epochs != 100):
            raise ValueError("resume start, target or save directory differs")
        if not trainer.optimizer.state:
            raise ValueError("optimizer state did not restore")
        write(out / "resume_setup.json", {
            "start_epoch_zero_based": trainer.start_epoch, "total_epochs": trainer.epochs,
            "optimizer_state_entries": len(trainer.optimizer.state),
            "ema_updates": trainer.ema.updates,
            "actual_args_sha256": digest(run / "args.yaml"),
            "continuity": record["continuity"],
        })

    model.add_callback("on_pretrain_routine_end", verify_setup)
    result = model.train(resume=True)
    if Path(result.save_dir).resolve() != run.resolve():
        raise ValueError("unexpected resumed training directory")
    return run
