"""Cross-fit scale-conditioned inference thresholds on accepted cached predictions."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np

import evaluate_paper1_small_pr as base
from regionssod.evaluation.error_diagnosis import json_ready
from regionssod.evaluation.operating_point import (
    confidence_grid, evaluate_policy, select_global, select_scale,
)

ROOT = base.ROOT
OUT = ROOT / "outputs/paper1_scale_operating_point_20261102"
DIAGNOSTIC = ROOT / "outputs/paper1_high_precision_diagnostic_20261102"
PROTOCOL = ROOT / "docs/paper/scale_operating_point_protocol.md"
DOCUMENT = ROOT / "docs/paper/scale_operating_point_results.md"
SPLITS = (20261102, 20261103, 20261104, 20261105, 20261106)
TARGETS = (.9, .95)
GRID = confidence_grid()


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def verify_inputs():
    diag = base.read(DIAGNOSTIC / "independent_audit.json")
    if diag.get("accepted") is not True or diag.get("exact_cached_replay") is not True:
        raise ValueError("accepted diagnostic replay is required")
    base.verify_hashes(base.read(DIAGNOSTIC / "manifest.json")["input_sha256"])
    base.verify_hashes(base.read(DIAGNOSTIC / "artifact_sha256.json"))
    base.verify_hashes(base.read(base.OUT / "manifest.json")["input_sha256"])
    base.verify_hashes(base.read(base.OUT / "artifact_sha256.json"))
    base.verify_captures()


def verify_result_manifest():
    """Use the archived compute source when presentation-only code changed."""
    expected = base.read(OUT / "manifest.json")["input_sha256"]
    snapshot = OUT / "compute_source_snapshot.py"
    adjusted = {}
    for filename, value in expected.items():
        if Path(filename) == Path(__file__).resolve() and snapshot.is_file():
            adjusted[str(snapshot)] = value
        else:
            adjusted[filename] = value
    base.verify_hashes(adjusted)


def manifest():
    verify_inputs()
    paths = [PROTOCOL, Path(__file__).resolve(), ROOT / "src/regionssod/evaluation/operating_point.py"]
    paths += [DIAGNOSTIC / name for name in ("manifest.json", "diagnosis.json", "artifact_sha256.json",
               "independent_audit.json")]
    for key in base.KEYS:
        paths += [DIAGNOSTIC / key / "diagnostic_arrays.npz", base.OUT / key / "predictions.jsonl"]
    return {"created_at": now(), "split_seeds": SPLITS, "precision_targets": TARGETS,
            "grid_size": len(GRID), "grid_first_last": [GRID[0].item(), GRID[-1].item()],
            "numpy_random_generator": "numpy.random.default_rng(PCG64) with version recorded by runtime",
            "versions": {"numpy": np.__version__, "python": __import__("sys").version},
            "input_sha256": {str(path): base.digest(path) for path in paths}}


def model_data(key, gt):
    arrays = np.load(DIAGNOSTIC / key / "diagnostic_arrays.npz")
    records, _ = base.load_predictions(key, gt)
    scores = np.concatenate([np.asarray(row["scores"], dtype=float) for row in records]) if records else np.empty(0)
    if not np.array_equal(scores, arrays["scores"]):
        raise ValueError("raw prediction scores differ from accepted diagnostic arrays")
    predicted_area = np.concatenate([
        np.maximum(0, np.asarray(row["xyxy"], dtype=float).reshape(-1, 4)[:, 2:] -
                   np.asarray(row["xyxy"], dtype=float).reshape(-1, 4)[:, :2]).prod(1)
        for row in records])
    image_ids = np.concatenate([np.full(len(row["scores"]), image["id"], dtype=int)
                                for image, row in zip(gt["images"], records)])
    if len(image_ids) != len(scores):
        raise ValueError("prediction lengths differ")
    return {"scores": scores, "matches": arrays["iou0.5_match"].astype(int),
            "gt_areas": arrays["iou0.5_areas"], "predicted_small": predicted_area < 1024,
            "image_ids": image_ids}


def fit_policy(x, indices, policy, target):
    scores, matches, areas = x["scores"][indices], x["matches"][indices], x["gt_areas"][indices]
    if policy == "global":
        return select_global(scores, matches, areas, GRID, target)
    return select_scale(scores, matches, areas, x["predicted_small"][indices], GRID, target)


def eval_policy(x, indices, fitted):
    if fitted is None:
        return None
    return evaluate_policy(x["scores"][indices], x["matches"][indices], x["gt_areas"][indices],
                           x["predicted_small"][indices], fitted)


def merged_eval(parts):
    if any(part is None for part in parts):
        return None
    predictions = sum(part["predictions"] for part in parts)
    tp = sum(part["tp"] for part in parts)
    small_tp = sum(part["small_tp"] for part in parts)
    return {"images": 548, "predictions": predictions, "tp": tp,
            "precision": tp / predictions if predictions else None,
            "small_tp": small_tp, "small_recall": small_tp / 26577}


def summarize(records):
    keys = ("precision", "small_recall", "predictions", "small_tp")
    result = {}
    for arm in ("b2", "p1"):
        result[arm] = {}
        for policy in ("global", "scale"):
            result[arm][policy] = {}
            for metric in keys:
                values = [record[arm][policy][metric] for record in records]
                result[arm][policy][metric] = {
                    "mean": float(np.mean(values)) if all(value is not None for value in values) else None,
                    "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 and all(value is not None for value in values) else None,
                    "values": values}
    result["comparisons"] = {"p1_minus_b2": {}, "scale_minus_global": {}}
    for policy in ("global", "scale"):
        values = [record["small_recall_gap_pp"][policy] for record in records]
        result["comparisons"]["p1_minus_b2"][policy] = {
            "mean_pp": float(np.mean(values)), "sample_sd_pp": float(np.std(values, ddof=1)), "values": values}
    for arm in ("b2", "p1"):
        values = [record["scale_minus_global_pp"][arm] for record in records]
        result["comparisons"]["scale_minus_global"][arm] = {
            "mean_pp": float(np.mean(values)), "sample_sd_pp": float(np.std(values, ddof=1)), "values": values}
    deltas = [record["delta_gap_pp"] for record in records]
    result["comparisons"]["delta_gap_pp"] = {"mean_pp": float(np.mean(deltas)),
        "sample_sd_pp": float(np.std(deltas, ddof=1)), "values": deltas,
        "positive_splits": int(sum(value > 0 for value in deltas)), "n_splits": len(deltas)}
    return result


def compute(save):
    gt = base.read(base.OUT / "ground_truth.json")
    image_ids = np.arange(1, 549)
    data = {key: model_data(key, gt) for key in base.KEYS}
    all_records, summaries = [], {}
    for target in TARGETS:
        target_tag = f"{target:.2f}"
        summaries[target_tag] = {}
        for student_seed in (0, 1):
            records = []
            for split_seed in SPLITS:
                permutation = np.random.default_rng(split_seed).permutation(image_ids)
                folds = ((permutation[:274], permutation[274:]), (permutation[274:], permutation[:274]))
                arm_folds = {arm: {"global": [], "scale": []} for arm in ("b2", "p1")}
                fold_details = []
                for calibration_ids, evaluation_ids in folds:
                    fold = {"split_seed": split_seed, "target": target, "student_seed": student_seed,
                            "calibration_images": int(len(calibration_ids)), "evaluation_images": int(len(evaluation_ids))}
                    for arm in ("b2", "p1"):
                        key = f"{arm}_s{student_seed}"
                        x = data[key]
                        cal = np.isin(x["image_ids"], calibration_ids)
                        ev = np.isin(x["image_ids"], evaluation_ids)
                        fold[arm] = {}
                        for policy in ("global", "scale"):
                            fitted = fit_policy(x, cal, policy, target)
                            evaluated = eval_policy(x, ev, fitted)
                            arm_folds[arm][policy].append(evaluated)
                            fold[arm][policy] = {"fitted": fitted, "evaluation": None if evaluated is None else
                                                 {k: v for k, v in evaluated.items() if k != "selected"}}
                    fold_details.append(fold)
                record = {"split_seed": split_seed, "semantic_target": target, "arms": {}}
                for arm in ("b2", "p1"):
                    record[arm] = {policy: merged_eval(arm_folds[arm][policy]) for policy in ("global", "scale")}
                record["small_recall_gap_pp"] = {
                    policy: 100 * (record["p1"][policy]["small_recall"] - record["b2"][policy]["small_recall"])
                    for policy in ("global", "scale")}
                record["scale_minus_global_pp"] = {
                    arm: 100 * (record[arm]["scale"]["small_recall"] - record[arm]["global"]["small_recall"])
                    for arm in ("b2", "p1")}
                record["delta_gap_pp"] = record["small_recall_gap_pp"]["scale"] - record["small_recall_gap_pp"]["global"]
                record["fold_details"] = fold_details
                records.append(record)
                all_records.append(record)
            summaries[target_tag][str(student_seed)] = summarize(records)
    continuation = {}
    for student_seed in (0, 1):
        summary = summaries["0.90"][str(student_seed)]
        precisions = [summary[arm][policy]["precision"]["mean"] for arm in ("b2", "p1") for policy in ("global", "scale")]
        p1_improvement = summary["comparisons"]["scale_minus_global"]["p1"]["mean_pp"]
        b2_improvement = summary["comparisons"]["scale_minus_global"]["b2"]["mean_pp"]
        delta = summary["comparisons"]["delta_gap_pp"]
        continuation[str(student_seed)] = {
            "delta_gap_mean_ge_0_20": delta["mean_pp"] >= .2,
            "delta_gap_positive_at_least_4_of_5": delta["positive_splits"] >= 4,
            "minimum_arm_policy_mean_eval_precision_ge_0_88": min(precisions) >= .88,
            "minimum_arm_policy_mean_eval_precision": min(precisions),
            "p1_scale_gain_gt_b2": p1_improvement > b2_improvement}
    return {"scope": "five fixed image-split seeds, two-fold disjoint calibration/evaluation, zero inference/training",
            "models": list(base.KEYS), "summaries": summaries, "continue_rule_items": continuation,
            "continue_rule_pass": all(all(values[key] for key in values if key != "minimum_arm_policy_mean_eval_precision")
                                      for values in continuation.values()),
            "records": all_records if save else None,
            "limits": ["same repeatedly used VisDrone val; internal image cross-fitting is not external confirmation",
                "calibration requires labels; hidden GT cannot be used for pseudo-label generation",
                "scale-conditioned thresholds are an inference policy, not a trained mechanism or student improvement",
                "evaluation precision can differ from calibration target and is reported for every split/fold",
                "two studentseeds and five image splits do not prove significance or cross-dataset generalization",
                "saved candidate floor/NMS/maxDet500 and training recovery constraints remain"]}


def flatten_fold_records(result):
    fields = ["semantic_target", "split_seed", "student_seed", "arm", "policy", "calibration_images",
              "evaluation_images", "calibration_threshold", "calibration_other_threshold",
              "calibration_precision", "evaluation_precision", "evaluation_small_recall"]
    rows = []
    for record in result["records"]:
        for fold in record["fold_details"]:
            for arm in ("b2", "p1"):
                for policy in ("global", "scale"):
                    fitted, evaluated = fold[arm][policy]["fitted"], fold[arm][policy]["evaluation"]
                    rows.append({"semantic_target": record["semantic_target"], "split_seed": record["split_seed"],
                        "student_seed": fold["student_seed"], "arm": arm, "policy": policy,
                        "calibration_images": fold["calibration_images"], "evaluation_images": fold["evaluation_images"],
                        "calibration_threshold": None if not fitted else fitted.get("threshold"),
                        "calibration_other_threshold": None if not fitted else fitted.get("other_threshold"),
                        "calibration_precision": None if not fitted else fitted.get("calibration_precision", fitted.get("precision")),
                        "evaluation_precision": None if not evaluated else evaluated["precision"],
                        "evaluation_small_recall": None if not evaluated else evaluated["small_recall"]})
    return fields, rows


def render_tables(result):
    lines = ["## 跨拟合数字表", "",
             "所有评估均来自不相交的另一半原图；各split seed的两折合计覆盖548图。", ""]
    for target in ("0.90", "0.95"):
        lines += [f"### 校准precision目标 {target}", "",
                  "| seed | split | Global P1−B2 / pp | Scale P1−B2 / pp | Δgap / pp | B2 Scale−Global / pp | P1 Scale−Global / pp | 最低评估precision |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for seed in ("0", "1"):
            summary = result["summaries"][target][seed]
            for index, split_seed in enumerate(SPLITS):
                global_gap = summary["comparisons"]["p1_minus_b2"]["global"]["values"][index]
                scale_gap = summary["comparisons"]["p1_minus_b2"]["scale"]["values"][index]
                delta = summary["comparisons"]["delta_gap_pp"]["values"][index]
                b2_improved = summary["comparisons"]["scale_minus_global"]["b2"]["values"][index]
                p1_improved = summary["comparisons"]["scale_minus_global"]["p1"]["values"][index]
                precisions = [summary[arm][policy]["precision"]["values"][index]
                              for arm in ("b2", "p1") for policy in ("global", "scale")]
                lines.append(f"| {seed} | {split_seed} | {global_gap:+.4f} | {scale_gap:+.4f} | {delta:+.4f} | "
                             f"{b2_improved:+.4f} | {p1_improved:+.4f} | {min(precisions):.6f} |")
        lines += ["", "| seed | Global gap均值±SD | Scale gap均值±SD | Δgap均值±SD | B2 improvement均值±SD | P1 improvement均值±SD |",
                  "|---|---:|---:|---:|---:|---:|"]
        for seed in ("0", "1"):
            summary = result["summaries"][target][seed]
            def stat(values):
                return f"{values['mean_pp']:+.4f} ± {values['sample_sd_pp']:.4f}"
            lines.append(f"| {seed} | "
                f"{stat(summary['comparisons']['p1_minus_b2']['global'])} | "
                f"{stat(summary['comparisons']['p1_minus_b2']['scale'])} | "
                f"{stat(summary['comparisons']['delta_gap_pp'])} | "
                f"{stat(summary['comparisons']['scale_minus_global']['b2'])} | "
                f"{stat(summary['comparisons']['scale_minus_global']['p1'])} |")
        lines.append("")
    lines += ["### 预先固定的0.90继续线", "",
              "| seed | Δgap均值≥+0.20 | 至少4/5 split为正 | 最低平均评估precision≥0.88 | P1增益大于B2 |",
              "|---|---:|---:|---:|---:|"]
    for seed in ("0", "1"):
        item = result["continue_rule_items"][seed]
        lines.append(f"| {seed} | {item['delta_gap_mean_ge_0_20']} | {item['delta_gap_positive_at_least_4_of_5']} | "
                     f"{item['minimum_arm_policy_mean_eval_precision_ge_0_88']} | {item['p1_scale_gain_gt_b2']} |")
    lines += ["", f"总体继续线：**{result['continue_rule_pass']}**。", ""]
    return "\n".join(lines)


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed, color in ((0, "#0072B2"), (1, "#D55E00")):
        values = result["summaries"]["0.90"][str(seed)]["comparisons"]["delta_gap_pp"]["values"]
        positions = np.arange(len(SPLITS))
        axes[0].plot(positions, values, marker="o", color=color, label=f"Student seed{seed}")
    axes[0].axhline(.2, linestyle="--", color="gray", label="Continue threshold +0.20pp")
    axes[0].axhline(0, color="black", lw=.7)
    axes[0].set_xlabel("Image split seed")
    axes[0].set_xticks(np.arange(len(SPLITS)), [str(seed) for seed in SPLITS], rotation=20)
    axes[0].set_ylabel("Change in P1−B2 small recall / pp")
    axes[0].set_title("Scale policy improvement over global policy")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.25)
    labels = [f"seed{seed} {arm.upper()} {policy}" for seed in (0, 1)
              for arm in ("b2", "p1") for policy in ("global", "scale")]
    means, errors = [], []
    for seed in (0, 1):
        for arm in ("b2", "p1"):
            for policy in ("global", "scale"):
                summary = result["summaries"]["0.90"][str(seed)][arm][policy]["small_recall"]
                means.append(100 * summary["mean"])
                errors.append(100 * summary["sample_sd"])
    x = np.arange(len(labels))
    colors = ["#56B4E9", "#0072B2", "#E69F00", "#D55E00"] * 2
    axes[1].bar(x, means, yerr=errors, capsize=3, color=colors)
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel("Small recall / %")
    axes[1].set_title("Evaluation-half means ± sample SD across 5 splits")
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Scale-conditioned high-precision operating point | Target calibration precision90%", fontsize=11)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"scale_operating_point.{ext}", dpi=180)
    plt.close(fig)


def run():
    if (OUT / "crossfit.json").exists():
        raise FileExistsError("existing result; use --audit instead of overwriting")
    info = manifest()
    if (OUT / "manifest.json").exists():
        old = base.read(OUT / "manifest.json")
        if old["input_sha256"] != info["input_sha256"]:
            raise ValueError("failed-attempt manifest differs from frozen input")
    else:
        base.write(OUT / "manifest.json", info)
    result = json_ready(compute(True))
    base.write(OUT / "crossfit.json", result)
    fields, rows = flatten_fold_records(result)
    with (OUT / "fold_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    plot(result)
    base.verify_hashes(info["input_sha256"])
    paths = [OUT / name for name in ("crossfit.json", "fold_results.csv")]
    paths += [OUT / f"scale_operating_point.{ext}" for ext in ("png", "pdf", "svg")]
    base.write(OUT / "artifact_sha256.json", {str(path): base.digest(path) for path in paths})
    base.write(OUT / "status.json", {"stage": "completed", "at": now(), "inference": 0, "training_steps": 0})


def audit():
    verify_inputs()
    verify_result_manifest()
    base.verify_hashes(base.read(OUT / "artifact_sha256.json"))
    result = json_ready(compute(False))
    stored = base.read(OUT / "crossfit.json")
    stored_without_records = {key: value for key, value in stored.items() if key != "records"}
    result_without_records = {key: value for key, value in result.items() if key != "records"}
    if stored_without_records != result_without_records:
        raise ValueError("cross-fitted numeric summary did not replay exactly")
    base.write(OUT / "independent_audit.json", {"accepted": True, "at": now(),
        "input_hashes": len(base.read(OUT / "manifest.json")["input_sha256"]),
        "image_splits": len(SPLITS), "two_fold_crossfit": True, "exact_cached_replay": True,
        "inference": 0, "training_steps": 0})
    print("Accepted exact cross-fit replay;0 inference,0 training.", flush=True)


def check_doc():
    expected = render_tables(base.read(OUT / "crossfit.json"))
    text = DOCUMENT.read_text(encoding="utf-8")
    actual = text.split("<!-- scale-table:begin -->\n", 1)[1].split("<!-- scale-table:end -->", 1)[0]
    if actual != expected:
        raise ValueError("scale-operating-point document tables differ from frozen JSON")
    print("Scale operating-point document tables match accepted cross-fit JSON.", flush=True)


def render():
    result = base.read(OUT / "crossfit.json")
    plot(result)
    paths = [OUT / name for name in ("crossfit.json", "fold_results.csv")]
    paths += [OUT / f"scale_operating_point.{ext}" for ext in ("png", "pdf", "svg")]
    base.write(OUT / "artifact_sha256.json", {str(path): base.digest(path) for path in paths})
    base.write(OUT / "status.json", {"stage": "rendered", "at": now(), "inference": 0, "training_steps": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--audit", action="store_true")
    group.add_argument("--render", action="store_true")
    group.add_argument("--check-doc", action="store_true")
    args = parser.parse_args()
    try:
        run() if args.run else render() if args.render else check_doc() if args.check_doc else audit()
    except Exception:
        base.write(OUT / "failure.json", {"at": now(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
