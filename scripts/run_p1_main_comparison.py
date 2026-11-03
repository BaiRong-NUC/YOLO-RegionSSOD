"""User-run B2/P1/improved main comparison; optional separate strength control."""
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

from run_p1_competitive_negative import decision as mechanism_decision

from diagnose_competitive_negative import ROOT, OUT as DIAG, read, sha, verify
from evaluate_paper1_small_pr import write
from regionssod.training.runner import load_experiment_config, run_training
from run_p1_scale_seed_pair import check_epochs, parameter_differences

OUT = ROOT / "outputs/p1_competitive_negative_main30_s0"
B2_CONFIG = ROOT / "configs/candidates/competitive_negative/baseline30.yaml"
BASE = ROOT / "configs/train_p1_10.yaml"
CONFIGS = {m: ROOT / f"configs/candidates/competitive_negative/{m}30.yaml" for m in ("semantic", "strength")}
ALL_NAMES = {"baseline": "p1_maincmp_b2_30_s0", "reference": "p1_maincmp_p1_30_s0", "candidate": "p1_maincmp_improved30_s0", "strength": "p1_maincmp_strength30_s0"}
NAMES = {k: v for k, v in ALL_NAMES.items() if k != "strength"}
PROTOCOL = ROOT / "docs/paper/candidates/competitive_negative/main_comparison_protocol.md"


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
        expected = base | {"name": "p1_compneg_semantic30_s0" if mode == "semantic" else "p1_compneg_strength30_s0",
            "competitive_negative": {"mode": mode, "class_score_path": "outputs/p1_class_scores_20261102/class_scores.jsonl"}}
        if c != expected:
            raise ValueError("Only negative classification extension and run name may differ")
    b2 = load_experiment_config(B2_CONFIG)
    expected_b2 = load_experiment_config(ROOT / "configs/train_naive_10.yaml")
    expected_b2["name"] = ALL_NAMES["baseline"]
    expected_b2["train"]["epochs"] = 30
    if b2 != expected_b2 or b2["train"] != base["train"]:
        raise ValueError("B2 must match student settings at 30 epochs")
    dataset = yaml.safe_load((ROOT / b2["data"]).read_text(encoding="utf-8"))
    listing = Path(dataset["train"])
    images = [Path(p) for p in listing.read_text(encoding="utf-8").splitlines()]
    if len(images) != 11095 or len(set(images)) != 11095:
        raise ValueError("Frozen B2 training population differs")
    pseudo_summary = ROOT / "data/visdrone_area10/pseudo/naive_t050/summary.json"
    summary = read(pseudo_summary)
    teacher = read(ROOT / "outputs/p1_class_scores_20261102/plan.json")["settings"]
    if (Path(summary["weights"]).resolve() != Path(teacher["weights"]).resolve()
            or sha(summary["weights"]) != teacher["teacher_sha256"]
            or summary["threshold"] != .5 or summary["conf_floor"] != .05
            or summary["accepted_boxes"] != 27667 or summary["accepted_images"] != 4624):
        raise ValueError("B2/P1 must share the frozen teacher and original pseudo labels")
    from regionssod.data.annotations import replace_images_component
    sources = [BASE, B2_CONFIG, ROOT / "configs/train_naive_10.yaml", ROOT / b2["data"],
               listing, pseudo_summary, PROTOCOL, Path(__file__),
               ROOT / "scripts/run_p1_competitive_negative.py", *CONFIGS.values(),
               DIAG / "analysis.json", DIAG / "artifact_sha256.json"]
    for image in images:
        sources += [image, replace_images_component(image).with_suffix(".txt")]
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


def main_comparison(models):
    """Same-run absolute results and three direct within-30epoch differences."""
    def row(m):
        def recall(iou, precision):
            p = m["precision_constraints"][iou][precision]
            return None if p is None else p["small_recall"]
        return {"AP50": m["ultralytics"]["metrics/mAP50(B)"],
            "AP50_95": m["ultralytics"]["metrics/mAP50-95(B)"],
            "AP_small": m["coco"]["metrics"]["AP_small"],
            "small_conf025": None if m["conf025"]["0.5"] is None else m["conf025"]["0.5"]["small_recall"],
            "small_P70": recall("0.5", "0.7"), "small_P90": recall("0.5", "0.9"),
            "small_strict_P90": recall("0.75", "0.9")}
    rows = {k: row(models[k]) for k in ("baseline", "reference", "candidate")}
    comparisons = {name: {metric: None if rows[left][metric] is None or rows[right][metric] is None
        else 100 * (rows[right][metric] - rows[left][metric]) for metric in rows[left]}
        for name, left, right in (("P1_minus_B2", "baseline", "reference"),
            ("improved_minus_P1", "reference", "candidate"), ("improved_minus_B2", "baseline", "candidate"))}
    return {"epochs": 30, "absolute_fraction": rows, "delta_pp": comparisons,
        "conf025_FP_counts": {k: None if models[k]["conf025"]["0.5"] is None else models[k]["conf025"]["0.5"]["fp"] for k in rows},
        "scope": "fresh B2/P1/improved, split0, one teacher, student seed0; 30epochs only"}


def decision(models):
    from evaluate_class_ambiguity_pair import continuation
    reference, candidate = models["reference"], models["candidate"]
    checks = continuation(reference, candidate)["gates"]
    a, b = [m["precision_constraints"]["0.75"]["0.9"] for m in (reference, candidate)]
    checks["strict_iou_P90_no_drop"] = a is not None and b is not None and b["small_recall"] >= a["small_recall"]
    fixed_ref, fixed_candidate = [m["conf025"]["0.5"] for m in (reference, candidate)]
    checks["conf025_FP_increase_at_most_2percent"] = (fixed_ref is not None and fixed_candidate is not None
        and fixed_candidate["fp"] <= 1.02 * fixed_ref["fp"])
    mechanism = {"status": "NOT_TESTED", "recommend_second_seed30": False}
    if "strength" in models:
        mechanism = (mechanism_decision(models) if all(m["conf025"]["0.5"] is not None for m in models.values())
                     else {"status": "UNREACHABLE_POINT", "recommend_second_seed30": False})
    return {"main_comparison": main_comparison(models), "performance_gates_vs_P1": checks,
        "performance_conditions_passed": all(checks.values()), "mechanism_control": mechanism,
        "recommend_separate_control30": all(checks.values()) and "strength" not in models,
        "recommend_second_seed30": all(checks.values()) and mechanism["recommend_second_seed30"],
        "automatic_further_training": False}


def write_main_table(comparison, seconds):
    metrics = {"AP50": "AP50", "AP50_95": "AP50–95", "AP_small": "AP_small", "small_conf025": "small R@conf.25",
               "small_P70": "small R@P70", "small_P90": "small R@P90", "small_strict_P90": "small R@IoU.75/P90"}
    def fmt(value):
        return "不可达" if value is None else f"{100*value:.4f}"
    text = ["# B2 / P1 / P1＋竞争负项：新配对30轮", "", comparison["scope"], "",
        "数值为百分数；FP为全尺寸计数；时长为学生累计秒。不是VisDrone官方AP。", "",
        "| 方法 | " + " | ".join(metrics.values()) + " | FP@conf.25 | 学生秒 |",
        "|---|" + "---:|" * (len(metrics) + 2)]
    for key, title in (("baseline", "B2基线"), ("reference", "原P1"), ("candidate", "P1＋竞争负项")):
        values = comparison["absolute_fraction"][key]
        text.append("| " + title + " | " + " | ".join(fmt(values[m]) for m in metrics) +
                    f" | {comparison['conf025_FP_counts'][key]} | {seconds[key]:.2f} |")
    text += ["", "以下为百分点pp，直接使用同一新配对的未舍入数计算。", "",
             "| 差值 | " + " | ".join(metrics.values()) + " |", "|---|" + "---:|" * len(metrics)]
    for name, values in comparison["delta_pp"].items():
        text.append("| " + name + " | " + " | ".join("不可达" if values[m] is None else f"{values[m]:+.4f}" for m in metrics) + " |")
    text += ["", "强度控制单独记录于decision.json；没有控制时不宣称类别语义的独立因果。", ""]
    (OUT / "main_comparison.md").write_text("\n".join(text), encoding="utf-8")


def audit_training():
    verify(read(OUT / "preflight.json")["input_sha256"])
    runs = {a: ROOT / "runs" / n for a, n in NAMES.items()}
    actual = {a: yaml.safe_load((p / "args.yaml").read_text(encoding="utf-8")) for a, p in runs.items()}
    for arm in NAMES:
        expected_data = load_experiment_config(B2_CONFIG if arm == "baseline" else BASE)["data"]
        if Path(actual[arm]["data"]).resolve() != (ROOT / expected_data).resolve():
            raise ValueError("Actual arm dataset binding differs")
    for a in NAMES:
        excluded = {"name", "save_dir", "data"} if a == "baseline" else {"name", "save_dir"}
        if parameter_differences(actual["reference"], actual[a], excluded):
            raise ValueError("Actual training arguments differ")
        if a not in {"candidate", "strength"}:
            continue
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
    if arm == "baseline":
        run_training(B2_CONFIG)
    elif arm == "reference":
        run_training(BASE, name=NAMES[arm])
    else:
        from regionssod.training.competitive_negative import run_competitive_negative_training
        mode = "semantic" if arm == "candidate" else arm
        config = load_experiment_config(CONFIGS[mode])
        config["name"] = NAMES[arm]
        snapshot = OUT / f"{arm}_config.yaml"
        if snapshot.exists():
            raise FileExistsError(snapshot)
        snapshot.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_competitive_negative_training(snapshot)
    check_epochs(ROOT / "runs" / NAMES[arm])


def train():
    receipt = preflight()
    # Main comparison has three fresh arms. Strength control is separate and optional.
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
        for arm in ("baseline", "reference", "candidate"):
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
        fig.suptitle("B2 / P1 / improved | fresh 30-epoch comparison | IoU0.50")
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
        comparison = main_comparison(models)
        write(OUT / "main_comparison.json", comparison)
        write_main_table(comparison, read(OUT / "training_audit.json")["student_seconds"])
        verify(receipt["input_sha256"])
        paths = [p for p in OUT.rglob("*") if p.is_file() and p.name != "status.json"]
        paths += [ROOT / "runs" / n / f for n in NAMES.values() for f in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "experiment.json")]
        paths += [ROOT / "runs" / NAMES[a] / "classification_diagnostics.jsonl" for a in ("candidate", "strength") if a in NAMES]
        write(OUT / "artifact_sha256.json", {str(p): sha(p) for p in paths})
        write(OUT / "status.json", {"stage": "completed", "automatic_further_training": False})
    except BaseException:
        write(OUT / "status.json", {"stage": "failed", "error": traceback.format_exc(), "automatic_retry": False})
        raise


def main():
    global NAMES
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check-only", action="store_true")
    choice.add_argument("--train", action="store_true")
    choice.add_argument("--audit", action="store_true")
    choice.add_argument("--worker", choices=ALL_NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--with-control", action="store_true", help="also run strength control as a fourth arm")
    args = parser.parse_args()
    if args.worker or args.audit:
        NAMES = read(OUT / "preflight.json")["arms"]
    elif args.with_control:
        NAMES = dict(ALL_NAMES)
    else:
        NAMES = {k: v for k, v in ALL_NAMES.items() if k != "strength"}
    if args.check_only:
        receipt = preflight()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = ROOT / f"outputs/p1_main_comparison_preflights/{stamp}.json"
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
        if main_comparison(models) != read(OUT / "main_comparison.json"):
            raise ValueError("Main table replay differs")
        if decision(models) != read(OUT / "decision.json"):
            raise ValueError("Decision replay differs")
        print(read(OUT / "decision.json"))


if __name__ == "__main__":
    main()
