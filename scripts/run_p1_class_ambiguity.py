"""Explicit user-run paired30: original P1 versus P1 with class-score allocation."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

import yaml

from diagnose_p1_class_ambiguity import ROOT, OUT as DIAGNOSTIC, sha, verify, read
from evaluate_paper1_small_pr import write
from export_p1_class_scores import OUT as SCORES
from regionssod.training.runner import load_experiment_config, run_training
from regionssod.data.annotations import replace_images_component
from run_p1_scale_seed_pair import check_epochs, parameter_differences

OUT = ROOT / "outputs/p1_class_ambiguity_pair30_s0"
CONFIG = ROOT / "configs/train_p1_class_ambiguity_10.yaml"
BASE = ROOT / "configs/train_p1_10.yaml"
NAMES = {"reference": "p1_class_ambiguity_reference30_s0", "candidate": "p1_class_ambiguity_candidate30_s0"}


def audit_diagnostic():
    """Retain exact populations/decisions; allow only float64 BLAS roundoff."""
    from diagnose_p1_class_ambiguity import analyze, lines
    actual, events = analyze()  # also verifies all frozen input/capture hashes
    differences = []

    def compare(a, b):
        if isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise ValueError("Diagnostic fields changed")
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                raise ValueError("Diagnostic population changed")
            for left, right in zip(a, b, strict=True):
                compare(left, right)
        elif isinstance(a, float) and isinstance(b, float):
            if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > 1e-12:
                raise ValueError("Diagnostic numeric replay differs beyond float64 tolerance")
            differences.append(abs(a-b))
        elif type(a) is not type(b) or a != b:
            raise ValueError("Diagnostic counts/decisions changed")

    compare(actual, read(DIAGNOSTIC / "analysis.json"))
    if events != lines(DIAGNOSTIC / "events.jsonl"):
        raise ValueError("Diagnostic events changed")
    return {"accepted": True, "float_absolute_tolerance": 1e-12,
            "max_float_difference": max(differences, default=0.), "events_counts_gates_exact": True}


def status(stage, **extra):
    write(OUT / "status.json", {"stage": stage, "at": datetime.now().astimezone().isoformat(), **extra})
    print(stage, extra, flush=True)


def preflight():
    import ultralytics
    import torch
    if Path.cwd().resolve() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("Run from repository root in the pinned yolo environment")
    if any((ROOT / "runs" / name).exists() for name in NAMES.values()) or (OUT / "dispatch.json").exists():
        raise FileExistsError("Existing paired run: inspect/audit; never overwrite or automatically restart")
    config, base = load_experiment_config(CONFIG), load_experiment_config(BASE)
    extension = {"enabled": True, "class_score_path": "outputs/p1_class_scores_20261102/class_scores.jsonl"}
    if config != base | {"name": NAMES["candidate"], "class_ambiguity": extension}:
        raise ValueError("Only name and the one classification extension may differ")
    diagnostic_audit = audit_diagnostic()
    subprocess.run([sys.executable, str(ROOT / "scripts/export_p1_class_scores.py"), "--audit"], check=True)
    if read(DIAGNOSTIC / "analysis.json")["worth_student_training"] is not True:
        raise ValueError("Prespecified diagnostic failed")
    data = yaml.safe_load((ROOT / config["data"]).read_text(encoding="utf-8"))
    listing = Path(data["train"])
    images = [Path(p) for p in listing.read_text(encoding="utf-8").splitlines()]
    if len(images) != 11310 or len(set(images)) != 11310:
        raise ValueError("Original training population changed")
    paths = [CONFIG, BASE, Path(config["model"]), ROOT / config["data"], listing,
             Path(data["p1_metadata"]), Path(data["p1_metadata"]).parent / "summary.json",
             SCORES / "class_scores.jsonl", SCORES / "receipt.json", SCORES / "plan.json",
             DIAGNOSTIC / "analysis.json",
             ROOT / "docs/paper/candidates/class_ambiguity/student_protocol.md"]
    paths += list((ROOT / "src/regionssod").rglob("*.py"))
    paths += [ROOT / "scripts" / name for name in (
        "run_p1_class_ambiguity.py", "evaluate_class_ambiguity_pair.py", "evaluate_paper1_small_pr.py",
        "run_p1_scale_seed_pair.py", "diagnose_p1_class_ambiguity.py", "export_p1_class_scores.py")]
    paths += list(Path(ultralytics.__file__).parent.rglob("*.py"))
    for image in images:
        paths += [image, replace_images_component(image).with_suffix(".txt")]
    hashes = {str(p.resolve()): sha(p) for p in paths}
    receipt = {"input_sha256": hashes, "training_images": len(images), "epochs_per_arm": 30,
        "arms": NAMES, "optimizer_steps_this_check": 0,
        "diagnostic_replay": diagnostic_audit,
        "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()}
    write(OUT / "preflight.json", receipt)
    print(f"Preflight passed: {len(hashes)} sources, two fresh arms x30 epochs. No training started.", flush=True)
    return receipt


def worker(arm):
    # Internal subprocess entry requires the parent dispatch and frozen inputs.
    if read(OUT / "dispatch.json")["preflight_sha256"] != sha(OUT / "preflight.json"):
        raise ValueError("Dispatch/preflight binding differs")
    verify(read(OUT / "preflight.json")["input_sha256"])
    if (ROOT / "runs" / NAMES[arm]).exists():
        raise FileExistsError("Arm already exists")
    if arm == "reference":
        run = run_training(BASE, name=NAMES[arm])
    else:
        from regionssod.training.class_ambiguity import run_class_ambiguity_training
        run = run_class_ambiguity_training(CONFIG)
    if run != ROOT / "runs" / NAMES[arm]:
        raise ValueError("Unexpected run location")
    check_epochs(run)


def paired_training_audit():
    verify(read(OUT / "preflight.json")["input_sha256"])
    runs = {a: ROOT / "runs" / name for a, name in NAMES.items()}
    args = {a: yaml.safe_load((p / "args.yaml").read_text(encoding="utf-8")) for a, p in runs.items()}
    differences = parameter_differences(args["reference"], args["candidate"], {"name", "save_dir"})
    if differences:
        raise ValueError(f"Actual paired training settings differ: {differences}")
    reference = read(runs["reference"] / "experiment.json")
    candidate = read(runs["candidate"] / "experiment.json")
    for field in ("model", "data", "train_args", "p1", "pseudo_summary", "ultralytics_version"):
        if reference[field] != candidate[field]:
            raise ValueError(f"Paired provenance differs: {field}")
    seconds = {}
    for arm, run in runs.items():
        seconds[arm] = float(check_epochs(run)[-1]["time"])
        verify(read(run / "experiment.json")["input_sha256"])
    snapshots = [json.loads(line) for line in
                 (runs["candidate"] / "classification_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    if ([r["epoch"] for r in snapshots] != list(range(1, 31))
            or any(r["status"] != "recorded" or r["epoch_gradient_estimate"] is not False for r in snapshots)):
        raise ValueError("Incomplete classification mechanism snapshots")
    for row in snapshots:
        for group, values in row["diagnostics"].items():
            if not isinstance(values, dict):
                if not math.isfinite(values):
                    raise ValueError("Nonfinite diagnostic")
                continue
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError("Nonfinite classification mechanism snapshot")
            for measure in ("mass", "weighted_mass"):
                if not math.isclose(values["old_"+measure], values["new_"+measure], rel_tol=1e-5, abs_tol=1e-5):
                    raise ValueError(f"Classification target mass drift: {group}")
    return {"accepted": True, "epochs_per_arm": 30, "student_time_seconds": seconds,
            "classification_snapshots": len(snapshots), "snapshot_scope": "first batch per epoch, not epoch totals"}


def train():
    receipt = preflight()
    # A verified zero-step engineering receipt is needed in addition to source tests.
    engineering = read(OUT / "engineering_check.json")
    verify(engineering["source_sha256"])
    if engineering.get("accepted") is not True or engineering.get("optimizer_steps") != 0:
        raise ValueError("Missing passing engineering check")
    with (OUT / "dispatch.json").open("x", encoding="utf-8") as stream:
        json.dump({"preflight_sha256": sha(OUT / "preflight.json"),
                   "engineering_sha256": sha(OUT / "engineering_check.json"),
                   "at": datetime.now().astimezone().isoformat()}, stream)
    try:
        for arm in NAMES:
            status("training", arm=arm, epochs=30, log=str(OUT / f"{arm}_stdout.log"))
            with (OUT / f"{arm}_stdout.log").open("x", encoding="utf-8") as log:
                process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                          "--worker", arm], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            write(OUT / f"{arm}_exit.json", {"exit_code": process.returncode})
            process.check_returncode()
        write(OUT / "training_audit.json", paired_training_audit())
        status("evaluating_548_full_images")
        from evaluate_class_ambiguity_pair import evaluate, audit
        result = evaluate({a: ROOT / "runs" / name / "weights/best.pt" for a, name in NAMES.items()},
                          OUT / "evaluation")
        write(OUT / "evaluation_audit.json", audit(OUT / "evaluation"))
        verify(receipt["input_sha256"])
        artifacts = [ROOT / "runs" / name / f for name in NAMES.values()
                     for f in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "experiment.json")]
        artifacts += [OUT / f for f in ("dispatch.json", "preflight.json", "engineering_check.json",
                      "training_audit.json", "evaluation_audit.json", "reference_exit.json", "candidate_exit.json",
                      "reference_stdout.log", "candidate_stdout.log")]
        artifacts.append(ROOT / "runs" / NAMES["candidate"] / "classification_diagnostics.jsonl")
        write(OUT / "artifact_sha256.json", {str(p): sha(p) for p in artifacts})
        status("completed", comparison=result, automatic_further_training=False)
    except BaseException:
        status("failed", error=traceback.format_exc(), automatic_retry=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--train", action="store_true")
    group.add_argument("--audit", action="store_true")
    group.add_argument("--worker", choices=NAMES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_only:
        preflight()
    elif args.worker:
        worker(args.worker)
    elif args.train:
        train()
    else:
        from evaluate_class_ambiguity_pair import audit
        verify(read(OUT / "artifact_sha256.json"))
        for arm in NAMES:
            if read(OUT / f"{arm}_exit.json")["exit_code"] != 0:
                raise ValueError("Training process did not exit normally")
        print(paired_training_audit())
        print(audit(OUT / "evaluation"))


if __name__ == "__main__":
    main()
