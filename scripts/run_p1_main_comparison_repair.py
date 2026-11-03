"""Train only AMP-safe candidate; reuse the completed paired B2/P1 read-only."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml

from diagnose_competitive_negative import ROOT, read, sha, verify
from evaluate_paper1_small_pr import write
from regionssod.training.runner import load_experiment_config
from regionssod.training.competitive_negative_v2 import IMPLEMENTATION
from run_p1_scale_seed_pair import check_epochs, parameter_differences
from run_p1_main_comparison import main_comparison, decision

OLD = ROOT / "outputs/p1_competitive_negative_main30_s0"
OLD_PREFLIGHT_SHA = "02737a5629435ed335f856b3a81576271cdd16c88fd72306f51b1feadb9b85d8"
FAILURE = OLD / "terminal_failure_20261103_140708_a233c4/failure_receipt.json"
OUT = ROOT / "outputs/p1_competitive_negative_main30_v2_s0"
CONFIG = ROOT / "configs/candidates/competitive_negative_v2/semantic30.yaml"
PROTOCOL = ROOT / "docs/paper/candidates/competitive_negative_v2/protocol.md"
VALIDATION = ROOT / "outputs/p1_competitive_negative_v2_validation_20261103/validation.json"
RUNS = {"baseline": ROOT / "runs/p1_maincmp_b2_30_s0",
        "reference": ROOT / "runs/p1_maincmp_p1_30_s0",
        "candidate": ROOT / "runs/p1_maincmp_improved_v2_30_s0"}
BEST = {"baseline": "76b678af53eb340860273d434a7404bd6edc4545d6d156adea92341c1ee30bcb",
        "reference": "6cc444decbcdcd67b4be7d9ed373e070d82aaefb0e522c6bd09debf1bb6e70a4"}


def reuse_bindings():
    """Refuse any drift in original inputs, completed arms or failed evidence."""
    if sha(OLD / "preflight.json") != OLD_PREFLIGHT_SHA:
        raise ValueError("Original preflight identity changed")
    if read(OLD / "dispatch.json")["preflight_sha256"] != OLD_PREFLIGHT_SHA:
        raise ValueError("Original dispatch binding changed")
    original = read(OLD / "preflight.json")
    verify(original["input_sha256"])
    failure = read(FAILURE)
    hashes = {str(OLD / n): sha(OLD / n) for n in ("preflight.json", "dispatch.json", "status.json")}
    hashes[str(FAILURE)] = sha(FAILURE)
    for path, item in failure["snapshots"].items():
        hashes[path] = item["sha256"]
    for arm in ("baseline", "reference"):
        run = RUNS[arm]
        if read(OLD / f"{arm}_exit.json")["exit_code"] != 0:
            raise ValueError("Reused arm did not exit zero")
        check_epochs(run)
        verify(read(run / "experiment.json")["input_sha256"])
        if sha(run / "weights/best.pt") != BEST[arm]:
            raise ValueError("Reused best weight differs from terminal evidence")
        for path, item in failure["arms"][arm]["weights"].items():
            hashes[path] = item["sha256"]
    verify(hashes)
    actual = {a: yaml.safe_load((RUNS[a] / "args.yaml").read_text(encoding="utf-8"))
              for a in ("baseline", "reference")}
    if parameter_differences(actual["baseline"], actual["reference"], {"name", "save_dir", "data"}):
        raise ValueError("Reused B2/P1 training settings differ")
    for arm, config in (("baseline", ROOT / "configs/candidates/competitive_negative/baseline30.yaml"),
                        ("reference", ROOT / "configs/train_p1_10.yaml")):
        expected = load_experiment_config(config)
        if Path(actual[arm]["data"]).resolve() != (ROOT / expected["data"]).resolve():
            raise ValueError("Reused dataset differs")
        if any(actual[arm].get(k) != v for k, v in expected["train"].items()) or actual[arm]["resume"]:
            raise ValueError("Reused arm violates frozen train settings")
    return original, hashes


def preflight():
    import torch
    import ultralytics
    if Path.cwd().resolve() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("Repository root and pinned yolo environment required")
    if OUT.exists() or RUNS["candidate"].exists():
        raise FileExistsError("New output/run already exists; no overwrite or retry")
    original, reused = reuse_bindings()
    config = load_experiment_config(CONFIG)
    expected = load_experiment_config(ROOT / "configs/candidates/competitive_negative/semantic30.yaml")
    expected["name"] = RUNS["candidate"].name
    if config != expected:
        raise ValueError("Candidate config may change only run name")
    validation = read(VALIDATION)
    if (validation["accepted"] is not True or validation["optimizer_steps"] != 0
            or validation["backward_calls"] != 0 or validation["torch_version"] != torch.__version__):
        raise ValueError("Missing compatible zero-step engineering validation")
    verify(validation["source_sha256"])
    verify(validation["artifact_sha256"])
    mapping = original["input_sha256"] | reused | validation["source_sha256"]
    sources = [CONFIG, PROTOCOL, Path(__file__).resolve(), VALIDATION,
               ROOT / "scripts/validate_competitive_negative_v2.py"]
    sources += list((ROOT / "src/regionssod").rglob("*.py"))
    sources += list((ROOT / "tests").glob("test_competitive_negative_v2*.py"))
    mapping.update({str(p): sha(p) for p in sources})
    mapping.update(validation["artifact_sha256"])
    verify(mapping)
    return {"input_sha256": mapping, "epochs_per_arm": 30, "train_arms": ["candidate"],
        "arms": {k: str(v) for k, v in RUNS.items()}, "configuration": config,
        "reuse_original_preflight_sha256": OLD_PREFLIGHT_SHA,
        "reused_best_sha256": BEST, "implementation": IMPLEMENTATION,
        "optimizer_steps_this_check": 0, "inference_this_check": False,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "at": datetime.now().astimezone().isoformat()}


def verify_dispatch():
    receipt = read(OUT / "preflight.json")
    if (read(OUT / "dispatch.json") != {"preflight_sha256": sha(OUT / "preflight.json"), "train_arms": ["candidate"]}
            or receipt["train_arms"] != ["candidate"]
            or receipt["arms"] != {k: str(v) for k, v in RUNS.items()}
            or receipt["implementation"] != IMPLEMENTATION):
        raise ValueError("Candidate-only dispatch identity changed")
    verify(receipt["input_sha256"])
    return receipt


def audit_training():
    receipt = verify_dispatch()
    seconds = {}
    actual = {a: yaml.safe_load((p / "args.yaml").read_text(encoding="utf-8")) for a, p in RUNS.items()}
    for arm, run in RUNS.items():
        exit_path = (OUT if arm == "candidate" else OLD) / f"{arm}_exit.json"
        if read(exit_path)["exit_code"] != 0:
            raise ValueError("Nonzero worker exit")
        seconds[arm] = float(check_epochs(run)[-1]["time"])
        verify(read(run / "experiment.json")["input_sha256"])
    if parameter_differences(actual["reference"], actual["candidate"], {"name", "save_dir"}):
        raise ValueError("Candidate actual training settings differ from reused P1")
    exp = read(RUNS["candidate"] / "experiment.json")
    if exp["implementation"] != IMPLEMENTATION or exp["config"] != receipt["configuration"]:
        raise ValueError("Candidate implementation/config binding differs")
    rows = [json.loads(s) for s in (RUNS["candidate"] / "classification_diagnostics.jsonl").read_text().splitlines()]
    if [r["epoch"] for r in rows] != list(range(1, 31)):
        raise ValueError("Incomplete candidate mechanism snapshots")
    for row in rows:
        d = row["diagnostics"]
        if (row["status"] != "recorded" or d["implementation"] != IMPLEMENTATION
                or d["changed_positive_target_weights"] != 0 or d["changed_assigned_class_weights"] != 0
                or d["target_tensor_unchanged"] is not True
                or not math.isfinite(d["removed_negative_logit_gradient_l1_before_gain"])
                or d["removed_negative_logit_gradient_l1_before_gain"] < 0):
            raise ValueError("Negative-only engineering invariant failed")
    return {"accepted": True, "epochs_per_arm": 30, "student_seconds": seconds,
            "reused_arms": ["baseline", "reference"], "newly_trained_arms": ["candidate"]}


def comparison_and_decision(models):
    comparison = main_comparison(models)
    comparison["scope"] = ("repaired candidate30 + reused B2/P1 from main30_s0; same split0/teacher/"
        "COCO init/student seed0; different source commits disclosed, not three newly trained arms")
    comparison["implementation"] = IMPLEMENTATION
    comparison["reused_best_sha256"] = BEST
    classes = {a: models[a]["coco"].get("per_class_AP_small", {}) for a in RUNS}
    if any(set(row) != set(classes["reference"]) for row in classes.values()):
        raise ValueError("Per-class AP_small category sets differ")
    comparison["per_class_AP_small_fraction"] = classes
    comparison["per_class_AP_small_delta_pp"] = {
        name: {c: None if classes[left][c] is None or classes[right][c] is None
               else 100 * (classes[right][c] - classes[left][c]) for c in classes["reference"]}
        for name, left, right in (("P1_minus_B2", "baseline", "reference"),
            ("improved_minus_P1", "reference", "candidate"), ("improved_minus_B2", "baseline", "candidate"))}
    result = decision(models)
    result["main_comparison"] = comparison
    return comparison, result


def table_text(comparison, seconds):
    metrics = list(comparison["absolute_fraction"]["reference"])
    lines = ["# B2／P1复用与AMP修订候选：30轮主对照", "", comparison["scope"], "",
        "百分数／百分点pp；COCO-style非VisDrone官方AP；FP为全尺寸计数。", "",
        "| 方法 | " + " | ".join(metrics) + " | FP@.25 | 原学生累计秒 |",
        "|---|" + "---:|" * (len(metrics) + 2)]
    for arm in RUNS:
        values = comparison["absolute_fraction"][arm]
        title = {"baseline": "B2（复用）", "reference": "P1（复用）", "candidate": "P1＋竞争负项V2"}[arm]
        numbers = ["不可达" if values[m] is None else f"{100*values[m]:.4f}" for m in metrics]
        lines.append(f"| {title} | " + " | ".join(numbers) + f" | {comparison['conf025_FP_counts'][arm]} | {seconds[arm]:.2f} |")
    lines += ["", "| 差值 | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
    for label, row in comparison["delta_pp"].items():
        lines.append(f"| {label} | " + " | ".join("不可达" if row[m] is None else f"{row[m]:+.4f}" for m in metrics) + " |")
    lines += ["", "全部类别AP_small（绝对百分数／差值pp，保留负向结果）：", "",
        "| 类别 | B2 | P1 | 改进V2 | P1−B2 | 改进−P1 | 改进−B2 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for category in comparison["per_class_AP_small_fraction"]["reference"]:
        absolute = [comparison["per_class_AP_small_fraction"][a][category] for a in RUNS]
        delta = [row[category] for row in comparison["per_class_AP_small_delta_pp"].values()]
        numbers = ["无有效GT" if v is None else f"{100*v:.4f}" for v in absolute]
        numbers += ["不可计算" if v is None else f"{v:+.4f}" for v in delta]
        lines.append(f"| {category} | " + " | ".join(numbers) + " |")
    lines += ["", "仅新训练candidate；无strength，机制NOT_TESTED；七项护栏见decision.json。",
        "两个复用run的原训练成本单列，新调度墙钟成本另见candidate_exit.json；评估成本见各analysis.json。", ""]
    return "\n".join(lines)


def models_and_binding():
    manifest = read(OUT / "evaluation/manifest.json")
    expected = {a: str(p / "weights/best.pt") for a, p in RUNS.items()}
    if manifest["weights"] != expected:
        raise ValueError("Evaluation weights are not the three intended frozen best checkpoints")
    models = {a: read(OUT / "evaluation" / a / "analysis.json") for a in RUNS}
    if any(models[a]["weights_sha256"] != sha(expected[a]) for a in RUNS):
        raise ValueError("Evaluation weight binding differs")
    if any(len(models[a]["coco"]["per_class_AP_small"]) != 10 for a in RUNS):
        raise ValueError("Incomplete ten-class AP_small reporting")
    return models


def finish_comparison():
    from evaluate_class_ambiguity_pair import evaluate, audit
    training = audit_training()
    write(OUT / "training_audit.json", training)
    write(OUT / "status.json", {"stage": "evaluating", "automatic_retry": False})
    evaluate({a: p / "weights/best.pt" for a, p in RUNS.items()}, OUT / "evaluation")
    # Replace only the newly generated evaluator title; never edit original artifacts.
    import matplotlib.pyplot as plt
    import numpy as np
    from regionssod.evaluation.precision_recall import precision_constrained_point
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for arm in RUNS:
        with np.load(OUT / "evaluation" / arm / "micro_iou0.5.npz") as curve:
            axes[0].plot(curve["recall"], curve["precision"], label=arm)
            grid = np.linspace(.5, .95, 91)
            points = [precision_constrained_point(curve, p) for p in grid]
            axes[1].plot(grid, [p["small_recall"] if p else np.nan for p in points], label=arm)
    axes[0].set(xlabel="All-size recall", ylabel="All-size precision")
    axes[1].set(xlabel="Required all-size precision", ylabel="Small recall")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.25)
    fig.suptitle("Reused B2/P1 + AMP-safe candidate | 30 epochs | IoU0.50")
    bindings = read(OUT / "evaluation/artifact_sha256.json")
    for ext in ("png", "pdf"):
        path = OUT / f"evaluation/paired_pr.{ext}"
        fig.savefig(path, dpi=180)
        bindings[str(path)] = sha(path)
    plt.close(fig)
    write(OUT / "evaluation/artifact_sha256.json", bindings)
    write(OUT / "evaluation_audit.json", audit(OUT / "evaluation"))
    models = models_and_binding()
    comparison, result = comparison_and_decision(models)
    write(OUT / "main_comparison.json", comparison)
    write(OUT / "decision.json", result)
    (OUT / "main_comparison.md").write_text(table_text(comparison, training["student_seconds"]), encoding="utf-8")
    verify_dispatch()
    paths = [p for p in OUT.rglob("*") if p.is_file() and p.name not in {"status.json", "artifact_sha256.json"}]
    # Include nested evaluator artifact manifests too.
    paths.append(OUT / "evaluation/artifact_sha256.json")
    paths += [p / f for p in RUNS.values() for f in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "experiment.json")]
    paths.append(RUNS["candidate"] / "classification_diagnostics.jsonl")
    write(OUT / "artifact_sha256.json", {str(p): sha(p) for p in paths})
    write(OUT / "status.json", {"stage": "completed", "automatic_further_training": False})


def worker():
    receipt = verify_dispatch()
    if RUNS["candidate"].exists():
        raise FileExistsError("Candidate run already exists")
    snapshot = OUT / "candidate_config.yaml"
    if load_experiment_config(snapshot) != receipt["configuration"]:
        raise ValueError("Candidate snapshot differs")
    from regionssod.training.competitive_negative_v2 import run_competitive_negative_training_v2
    run_competitive_negative_training_v2(snapshot)
    check_epochs(RUNS["candidate"])


def train():
    receipt = preflight()
    OUT.mkdir()
    snapshot = OUT / "candidate_config.yaml"
    snapshot.write_bytes(CONFIG.read_bytes())
    receipt["input_sha256"][str(snapshot)] = sha(snapshot)
    write(OUT / "preflight.json", receipt)
    write(OUT / "dispatch.json", {"preflight_sha256": sha(OUT / "preflight.json"), "train_arms": ["candidate"]})
    try:
        write(OUT / "status.json", {"stage": "training", "arm": "candidate", "epochs": 30, "automatic_retry": False})
        start = time.monotonic()
        with (OUT / "candidate_stdout.log").open("x", encoding="utf-8") as stream:
            process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        write(OUT / "candidate_exit.json", {"exit_code": process.returncode, "worker_wall_seconds": time.monotonic()-start})
        process.check_returncode()
        finish_comparison()
    except BaseException:
        write(OUT / "status.json", {"stage": "failed", "error": traceback.format_exc(), "automatic_retry": False})
        raise


def audit():
    from evaluate_class_ambiguity_pair import audit as evaluation_audit
    verify(read(OUT / "artifact_sha256.json"))
    training = audit_training()
    if training != read(OUT / "training_audit.json"):
        raise ValueError("Training audit replay differs")
    evaluated = evaluation_audit(OUT / "evaluation")
    if evaluated != read(OUT / "evaluation_audit.json"):
        raise ValueError("Evaluation audit replay differs")
    comparison, result = comparison_and_decision(models_and_binding())
    if comparison != read(OUT / "main_comparison.json") or result != read(OUT / "decision.json"):
        raise ValueError("Comparison/decision replay differs")
    if (OUT / "main_comparison.md").read_text(encoding="utf-8") != table_text(comparison, training["student_seconds"]):
        raise ValueError("Markdown table replay differs")
    return {"accepted": True, "training": training, "evaluation": evaluated,
            "mechanism": result["mechanism_control"]["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check-only", action="store_true")
    choice.add_argument("--train", action="store_true")
    choice.add_argument("--audit", action="store_true")
    choice.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_only:
        receipt = preflight()
        path = ROOT / f"outputs/p1_main_comparison_v2_preflights/{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        write(path, receipt)
        print(f"Accepted: train candidate only, reuse B2/P1; zero training/inference. {path}")
    elif args.train:
        train()
    elif args.worker:
        worker()
    else:
        print(json.dumps(audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
