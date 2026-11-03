"""User-run three-arm30 experiment. Never resumes, overwrites or expands training."""
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

from diagnose_competitive_negative import ROOT, OUT as DIAG, read, sha, verify
from evaluate_paper1_small_pr import write
from regionssod.training.runner import load_experiment_config, run_training
from run_p1_scale_seed_pair import check_epochs, parameter_differences

OUT = ROOT / "outputs/p1_competitive_negative_three30_s0"
BASE = ROOT / "configs/train_p1_10.yaml"
CONFIGS = {m: ROOT / f"configs/candidates/competitive_negative/{m}30.yaml" for m in ("semantic", "strength")}
NAMES = {"reference": "p1_compneg_reference30_s0", "candidate": "p1_compneg_semantic30_s0", "strength": "p1_compneg_strength30_s0"}
PROTOCOL = ROOT / "docs/paper/candidates/competitive_negative/student_protocol.md"


def preflight():
    import ultralytics
    if Path.cwd().resolve() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("Use repository root and pinned yolo environment")
    if OUT.exists() or any((ROOT / "runs" / n).exists() for n in NAMES.values()):
        raise FileExistsError("Existing experiment; audit, do not restart")
    verify(read(DIAG / "input_sha256.json"))
    verify(read(DIAG / "artifact_sha256.json"))
    if not all(read(DIAG / "analysis.json")["teacher"][g]["information_gate"] for g in ("design", "confirmation")):
        raise ValueError("Exploratory information gate failed")
    base = load_experiment_config(BASE)
    for mode, p in CONFIGS.items():
        c = load_experiment_config(p)
        expected = base | {"name": NAMES["candidate" if mode == "semantic" else mode],
            "competitive_negative": {"mode": mode, "class_score_path": "outputs/p1_class_scores_20261102/class_scores.jsonl"}}
        if c != expected:
            raise ValueError("Only negative classification extension and run name may differ")
    sources = [BASE, PROTOCOL, Path(__file__), *CONFIGS.values(), DIAG / "analysis.json", DIAG / "artifact_sha256.json"]
    sources += list((ROOT / "src/regionssod").rglob("*.py"))
    sources += [ROOT / "scripts" / n for n in ("evaluate_class_ambiguity_pair.py", "evaluate_paper1_small_pr.py", "run_p1_scale_seed_pair.py", "diagnose_competitive_negative.py")]
    sources += list((ROOT / "tests").glob("test_competitive_negative*.py"))
    # Include all previously bound images/labels/teacher/init/implementation files.
    mapping = dict(read(DIAG / "input_sha256.json"))
    for p in sources:
        mapping[str(p)] = sha(p)
    verify(mapping)
    return {"input_sha256": mapping, "epochs_per_arm": 30, "arms": NAMES,
        "optimizer_steps_this_check": 0, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "at": datetime.now().astimezone().isoformat(),
        "engineering_scope": "CPU fixture autograd; actual full-data CUDA integration not yet exercised"}


def decision(models):
    from evaluate_class_ambiguity_pair import continuation
    reference, semantic, strength = [models[k] for k in ("reference", "candidate", "strength")]
    old = continuation(reference, semantic)
    def delta(a, b, iou, p):
        x, y = [r["precision_constraints"][iou][p] for r in (a, b)]
        return None if x is None or y is None else 100 * (y["small_recall"] - x["small_recall"])
    strict = delta(reference, semantic, "0.75", "0.9")
    specificity = delta(strength, semantic, "0.5", "0.9")
    ref_fp, sem_fp = [m["conf025"]["0.5"]["fp"] for m in (reference, semantic)]
    checks = old["gates"] | {
        "strict_iou_P90_no_drop": strict is not None and strict >= 0,
        "conf025_FP_increase_at_most_2percent": sem_fp <= 1.02 * ref_fp,
        "semantic_P90_vs_strength_at_least_015pp": specificity is not None and specificity >= .15,
        "semantic_APsmall_vs_strength_no_drop": semantic["coco"]["metrics"]["AP_small"] >= strength["coco"]["metrics"]["AP_small"],
    }
    return {"delta_vs_reference": old["delta_pp"], "strict_P90_delta_pp": strict,
        "semantic_vs_strength_P90_delta_pp": specificity, "gates": checks,
        "recommend_second_seed30": all(checks.values()), "automatic_further_training": False,
        "interpretation": "development investment criteria, not significance, deployment guarantees or unique causality"}


def audit_training():
    verify(read(OUT / "preflight.json")["input_sha256"])
    runs = {a: ROOT / "runs" / n for a, n in NAMES.items()}
    actual = {a: yaml.safe_load((p / "args.yaml").read_text(encoding="utf-8")) for a, p in runs.items()}
    for a in ("candidate", "strength"):
        if parameter_differences(actual["reference"], actual[a], {"name", "save_dir"}):
            raise ValueError("Actual training arguments differ")
        snapshots = [json.loads(s) for s in (runs[a] / "classification_diagnostics.jsonl").read_text().splitlines()]
        if [r["epoch"] for r in snapshots] != list(range(1, 31)):
            raise ValueError("Incomplete mechanism snapshots")
        if any(r["status"] != "recorded" or r["diagnostics"]["changed_positive_target_weights"] != 0 for r in snapshots):
            raise ValueError("Negative-only mechanism failed")
        if any(not math.isfinite(r["diagnostics"]["removed_negative_logit_gradient_l1_before_gain"]) for r in snapshots):
            raise ValueError("Nonfinite mechanism snapshot")
    seconds = {}
    for a, p in runs.items():
        if read(OUT / f"{a}_exit.json")["exit_code"] != 0:
            raise ValueError("Nonzero worker exit")
        seconds[a] = float(check_epochs(p)[-1]["time"])
        verify(read(p / "experiment.json")["input_sha256"])
    return {"accepted": True, "epochs_per_arm": 30, "student_seconds": seconds}


def worker(arm):
    if read(OUT / "dispatch.json")["preflight_sha256"] != sha(OUT / "preflight.json"):
        raise ValueError("Dispatch/preflight binding mismatch")
    verify(read(OUT / "preflight.json")["input_sha256"])
    if (ROOT / "runs" / NAMES[arm]).exists():
        raise FileExistsError("Arm already exists")
    if arm == "reference":
        run_training(BASE, name=NAMES[arm])
    else:
        from regionssod.training.competitive_negative import run_competitive_negative_training
        run_competitive_negative_training(CONFIGS["semantic" if arm == "candidate" else arm])
    check_epochs(ROOT / "runs" / NAMES[arm])


def train():
    receipt = preflight()
    # Three-arm control is indispensable: no two-arm shortcut or historical B2 substitution.
    OUT.mkdir(parents=True)
    write(OUT / "preflight.json", receipt)
    write(OUT / "dispatch.json", {"preflight_sha256": sha(OUT / "preflight.json")})
    try:
        for arm in NAMES:
            write(OUT / "status.json", {"stage": "training", "arm": arm, "epochs": 30, "automatic_retry": False})
            with (OUT / f"{arm}_stdout.log").open("x", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", arm],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            write(OUT / f"{arm}_exit.json", {"exit_code": result.returncode})
            result.check_returncode()
        write(OUT / "training_audit.json", audit_training())
        from evaluate_class_ambiguity_pair import evaluate, audit
        write(OUT / "status.json", {"stage": "evaluating"})
        evaluate({a: ROOT / "runs" / n / "weights/best.pt" for a, n in NAMES.items()}, OUT / "evaluation")
        # Reuse the frozen numerical evaluator; give NEW plots their correct identity
        # before accepting/hashing this experiment. No historical plot is touched.
        import matplotlib.pyplot as plt
        import numpy as np
        from regionssod.evaluation.precision_recall import precision_constrained_point
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for arm in NAMES:
            with np.load(OUT / "evaluation" / arm / "micro_iou0.5.npz") as curve:
                axes[0].plot(curve["recall"], curve["precision"], label=arm)
                levels = np.linspace(.5, .95, 91)
                points = [precision_constrained_point(curve, p) for p in levels]
                axes[1].plot(levels, [p["small_recall"] if p else np.nan for p in points], label=arm)
        axes[0].set(xlabel="All-size recall", ylabel="All-size precision")
        axes[1].set(xlabel="Required all-size precision", ylabel="Small recall")
        for ax in axes:
            ax.legend()
            ax.grid(alpha=.25)
        fig.suptitle("P1 competitive negative | three arms x 30 epochs | IoU0.50")
        bindings = read(OUT / "evaluation/artifact_sha256.json")
        for ext in ("png", "pdf"):
            path = OUT / f"evaluation/paired_pr.{ext}"
            fig.savefig(path, dpi=180)
            bindings[str(path)] = sha(path)
        plt.close(fig)
        write(OUT / "evaluation/artifact_sha256.json", bindings)
        write(OUT / "evaluation_audit.json", audit(OUT / "evaluation"))
        models = {a: read(OUT / "evaluation" / a / "analysis.json") for a in NAMES}
        write(OUT / "decision.json", decision(models))
        verify(receipt["input_sha256"])
        paths = [p for p in OUT.rglob("*") if p.is_file() and p.name != "status.json"]
        paths += [ROOT / "runs" / n / f for n in NAMES.values() for f in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "experiment.json")]
        paths += [ROOT / "runs" / NAMES[a] / "classification_diagnostics.jsonl" for a in ("candidate", "strength")]
        write(OUT / "artifact_sha256.json", {str(p): sha(p) for p in paths})
        write(OUT / "status.json", {"stage": "completed", "automatic_further_training": False})
    except BaseException:
        write(OUT / "status.json", {"stage": "failed", "error": traceback.format_exc(), "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check-only", action="store_true")
    choice.add_argument("--train", action="store_true")
    choice.add_argument("--audit", action="store_true")
    choice.add_argument("--worker", choices=NAMES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_only:
        receipt = preflight()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = ROOT / f"outputs/p1_competitive_negative_preflights/{stamp}.json"
        write(path, receipt)
        print(f"Preflight accepted; zero training/inference. Receipt: {path}")
    elif args.worker:
        worker(args.worker)
    elif args.train:
        train()
    else:
        verify(read(OUT / "artifact_sha256.json"))
        print(audit_training())
        from evaluate_class_ambiguity_pair import audit
        print(audit(OUT / "evaluation"))
        models = {a: read(OUT / "evaluation" / a / "analysis.json") for a in NAMES}
        if decision(models) != read(OUT / "decision.json"):
            raise ValueError("Decision replay differs")
        print(read(OUT / "decision.json"))


if __name__ == "__main__":
    main()
