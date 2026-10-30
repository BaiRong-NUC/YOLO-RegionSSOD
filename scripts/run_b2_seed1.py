"""Repeat frozen B2 with student seed1 and audit two-seed P1-versus-B2 pairs."""
from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import statistics
import traceback
from zoneinfo import ZoneInfo

import yaml

from run_p1_scale_seed_pair import ROOT, SOURCE, check_epochs, digest, parameter_differences, read, replay, write
from regionssod.training.runner import load_experiment_config

NAME = "partial10_naive_t050_pilot30_s1"
CONFIG = ROOT / "configs/train_naive_10_seed1.yaml"
OUT = ROOT / "outputs/b2_pilot30_s1"
RUN = ROOT / "runs" / NAME
VAL = ROOT / "runs/val" / f"{NAME}_val960"
RECALL = ROOT / "runs/size_recall/b2_pilot30_s1"
BASE = ROOT / "runs/partial10_naive_t050_pilot30_s0"
METRICS = ("mAP50", "mAP50_95", "small_recall", "medium_recall", "large_recall")


def stage(name, **extra):
    write(OUT / "status.json", {"stage": name, "updated_at": datetime.now(
        ZoneInfo("Asia/Shanghai")).isoformat(), **extra})
    print(name, extra, flush=True)


def assert_config():
    expected = load_experiment_config(ROOT / "configs/train_naive_10.yaml")
    expected["name"] = NAME
    expected["train"].update(epochs=30, seed=1)
    if load_experiment_config(CONFIG) != expected:
        raise RuntimeError("only student seed/name may change from the 30-epoch B2 pilot")


def preflight():
    import ultralytics
    if Path.cwd().resolve() != ROOT or ultralytics.__version__ != "8.4.172":
        raise RuntimeError("run from repository root in frozen yolo environment")
    if any(p.exists() for p in (RUN, VAL, RECALL, OUT / "dispatch.json")):
        raise FileExistsError("refusing duplicate dispatch or overwrite")
    assert_config()
    hashes = dict(read(ROOT / "outputs/p1_scale_threshold_pair_s1/preflight.json")["sha256"])
    historical_sources = {}
    for manifest in (ROOT / "data/visdrone_area10/pseudo/naive_t050/SHA256SUMS.txt", BASE / "SHA256SUMS.txt"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, filename = line.split("  ", 1)
            # Pseudo-label generation predates P1 source changes; preserve its
            # provenance separately. Freeze current training sources below.
            if Path(filename).is_relative_to(ROOT / "src"):
                historical_sources[filename] = expected
                continue
            if filename in hashes and hashes[filename] != expected:
                raise RuntimeError(f"conflicting frozen hash: {filename}")
            hashes[filename] = expected
    config = load_experiment_config(CONFIG)
    spec = yaml.safe_load((ROOT / config["data"]).read_text(encoding="utf-8"))
    if len(Path(spec["train"]).read_text().splitlines()) != 11095 or spec.get("p1_metadata"):
        raise RuntimeError("B2 must use original 11095-tile dataset and baseline loss")
    for p in [CONFIG, Path(__file__).resolve()] + list((ROOT / "src/regionssod").rglob("*.py")):
        hashes.setdefault(str(p), digest(p))
    for seed in (0, 1):
        p1 = ROOT / f"runs/partial10_p1_pilot30_s{seed}"
        for filename in ("weights/best.pt", "weights/last.pt", "args.yaml", "experiment.json", "results.csv"):
            p = p1 / filename
            hashes[str(p)] = digest(p)
    for i, (filename, expected) in enumerate(hashes.items(), 1):
        if digest(filename) != expected:
            raise RuntimeError(f"frozen input changed: {filename}")
        if i % 10000 == 0:
            print(f"Verified {i}/{len(hashes)} frozen files", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "split_seed": 0, "student_seed": 1, "epochs": 30, "train_tiles": 11095,
               "sha256": hashes, "teacher_sha256": read(ROOT / "outputs/gate2_20261027/comparison.json")["protocol"]["teacher_sha256"]}
    receipt["historical_pseudo_generation_source_sha256"] = historical_sources
    write(OUT / "preflight.json", receipt)
    print(f"Preflight passed: {len(hashes)} files", flush=True)


def collect():
    frozen = read(OUT / "preflight.json")
    if any(digest(p) != h for p, h in frozen["sha256"].items()):
        raise RuntimeError("frozen inputs changed during training")
    assert_config()
    all_pairs, artifacts, records, actuals = {}, [], {}, {}
    for seed in (0, 1):
        pair = {}
        for method in ("B2", "P1"):
            run = ROOT / (f"runs/partial10_naive_t050_pilot30_s{seed}" if method == "B2"
                          else f"runs/partial10_p1_pilot30_s{seed}")
            val = ROOT / "runs/val" / f"{run.name}_val960"
            recall_dir = ROOT / f"runs/size_recall/{method.lower()}_pilot30_s{seed}"
            rows = check_epochs(run)
            args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
            exp = read(run / "experiment.json")
            actuals[method, seed] = args
            if args["seed"] != seed or args["epochs"] != 30 or args["batch"] != 8 or args["imgsz"] != 640:
                raise RuntimeError("unexpected seed or schedule")
            if method == "P1" and exp["p1"] != {"soft_weights": True, "uncertain_ignore": True}:
                raise RuntimeError("P1 loss options changed")
            if method == "B2" and (exp.get("method", "baseline") != "baseline" or exp.get("p1") is not None):
                raise RuntimeError("B2 must use the baseline loss")
            if any(digest(p) != h for p, h in exp.get("input_sha256", {}).items()):
                raise RuntimeError("run inputs changed")
            if method == "P1" and exp["pseudo_summary"]["teacher_sha256"] != frozen["teacher_sha256"]:
                raise RuntimeError("teacher mismatch")
            metrics, recall = read(val / "metrics.json"), read(recall_dir / "metrics.json")
            best = digest(run / "weights/best.pt")
            if (Path(metrics["weights"]) != run / "weights/best.pt" or metrics["imgsz"] != 960
                    or recall["weights_sha256"] != best or recall["images"] != 548):
                raise RuntimeError("evaluation weights or full val scope mismatch")
            reference = read(ROOT / "runs/size_recall/b2_pilot30_s0/metrics.json")
            keys = ("data", "imgsz", "conf", "match_iou", "nms_iou", "max_det", "ultralytics_version", "matching", "size_pixel_space")
            if any(recall[k] != reference[k] for k in keys):
                raise RuntimeError("recall protocol mismatch")
            if (recall_dir / "images.txt").read_bytes() != (ROOT / "data/visdrone_area10/splits/val.txt").read_bytes():
                raise RuntimeError("validation order differs")
            counts = replay(recall_dir)
            pair[method] = {"mAP50": metrics["results"]["metrics/mAP50(B)"],
                            "mAP50_95": metrics["results"]["metrics/mAP50-95(B)"],
                            **{f"{s}_recall": counts[s]["recall"] for s in counts}}
            if not all(math.isfinite(v) for v in pair[method].values()):
                raise RuntimeError("non-finite metrics")
            records[f"{method}_s{seed}"] = {"run": str(run), "metrics": pair[method],
                "best_sha256": best, "last_sha256": digest(run / "weights/last.pt"),
                "student_seconds": float(rows[-1]["time"]), "replayed_recall": counts,
                "class_maps": metrics["class_maps"], "names": metrics["names"]}
            artifacts += [run / n for n in ("weights/best.pt", "weights/last.pt", "args.yaml", "experiment.json", "results.csv")]
            artifacts += [val / "metrics.json", recall_dir / "metrics.json", recall_dir / "images.txt", recall_dir / "predictions.jsonl"]
        if parameter_differences(actuals["B2", seed], actuals["P1", seed], {"data", "name", "save_dir"}):
            raise RuntimeError("actual same-seed B2/P1 hyperparameters differ")
        pair["delta_p1_minus_b2_pp"] = {k: 100 * (pair["P1"][k] - pair["B2"][k]) for k in METRICS}
        all_pairs[str(seed)] = pair
    for method in ("B2", "P1"):
        if parameter_differences(actuals[method, 0], actuals[method, 1], {"seed", "name", "save_dir"}):
            raise RuntimeError("actual seed0/seed1 hyperparameters differ")
    aggregate = {m: {k: {"mean_percent": statistics.mean(p[m][k] * 100 for p in all_pairs.values()),
                          "sample_std_pp": statistics.stdev(p[m][k] * 100 for p in all_pairs.values())}
                      for k in METRICS} for m in ("B2", "P1")}
    delta = {k: {"mean_pp": statistics.mean(p["delta_p1_minus_b2_pp"][k] for p in all_pairs.values()),
                 "sample_std_pp": statistics.stdev(p["delta_p1_minus_b2_pp"][k] for p in all_pairs.values())} for k in METRICS}
    comparison = {"split_seed": 0, "student_seeds": [0, 1], "epochs": 30, "per_seed": all_pairs,
                  "runs": records, "aggregate": aggregate, "paired_delta_p1_minus_b2_pp": delta,
                  "limitations": ["one split and fixed teacher, two student seeds; no significance claim",
                                  "Ultralytics development mAP and fixed-point recall, not AP_small/AR500",
                                  "equal student epochs, unequal data and compute; original Gate3 unchanged"]}
    write(OUT / "comparison.json", comparison)
    artifacts += [OUT / "comparison.json", OUT / "preflight.json", CONFIG, Path(__file__).resolve()]
    audit = {"audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
             "acceptance_checks_passed": True, "frozen_files_checked": len(frozen["sha256"]),
             "parameter_differences": {}, "replayed_images_per_run": 548, "runs_replayed": 4,
             "sha256": {str(p): digest(p) for p in artifacts}}
    write(OUT / "acceptance_audit.json", audit)
    write(RUN / "audit.json", audit)
    (RUN / "SHA256SUMS.txt").write_text("".join(f"{h}  {p}\n" for p, h in audit["sha256"].items()), encoding="utf-8")
    lines = ["# P1 与 B2 的配对学生 seed 复验", "", "固定 split seed0 与教师；差值为 P1 − B2。", "",
             "| 学生 seed | 指标 | B2 / % | P1 / % | 差值 / pp |", "|---|---|---:|---:|---:|"]
    for seed, pair in all_pairs.items():
        lines += [f"| {seed} | {k} | {100*pair['B2'][k]:.4f} | {100*pair['P1'][k]:.4f} | {pair['delta_p1_minus_b2_pp'][k]:+.4f} |" for k in METRICS]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stage("completed", comparison=str(OUT / "comparison.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if args.collect_only:
        collect()
        return
    preflight()
    if args.check_only:
        return
    with (OUT / "dispatch.json").open("x", encoding="utf-8") as handle:
        import json
        json.dump({"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, handle)
    from regionssod.training.runner import run_training
    from regionssod.evaluation.runner import run_validation
    from regionssod.evaluation.size_recall import run_size_recall
    stage("training", run=str(RUN), epochs=30)
    if run_training(CONFIG) != RUN:
        raise RuntimeError("unexpected training directory")
    check_epochs(RUN)
    stage("validation960")
    if run_validation(RUN / "weights/best.pt", ROOT / load_experiment_config(CONFIG)["data"], name=f"{NAME}_val960") != VAL:
        raise RuntimeError("unexpected validation directory")
    stage("size_recall")
    run_size_recall(RUN / "weights/best.pt", SOURCE, RECALL, imgsz=960, batch=8, device="0", conf=.25, match_iou=.5)
    stage("acceptance")
    collect()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if (OUT / "dispatch.json").exists():
            stage("failed", error=traceback.format_exc())
        raise
