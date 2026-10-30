"""Independently verify saved B2/P1 paired-seed outputs without inference."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from main_seed_tables import load_main_pair
from run_b2_seed1 import OUT, METRICS
from run_p1_scale_seed_pair import ROOT, check_epochs, digest, parameter_differences, read, replay, write


def main():
    checks, hashes = {}, {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError(name)

    check("pipeline_completed", read(OUT / "status.json")["stage"] == "completed")
    pair = load_main_pair(hashes, include_independent=False)
    frozen = read(OUT / "preflight.json")
    check("all_frozen_inputs_match", all(digest(p) == h for p, h in frozen["sha256"].items()))
    teacher = ROOT / "runs/partial10_sup_s0/weights/best.pt"
    check("same_teacher", digest(teacher) == frozen["teacher_sha256"])
    summary = read(ROOT / "data/visdrone_area10/pseudo/naive_t050/summary.json")
    check("b2_original_pseudo_scope", summary["threshold"] == .5 and summary["accepted_images"] == 4624 and summary["accepted_boxes"] == 27667)
    protocol = read(ROOT / "runs/size_recall/b2_pilot30_s0/metrics.json")
    protocol_keys = ("data", "imgsz", "conf", "match_iou", "nms_iou", "max_det",
                     "ultralytics_version", "matching", "size_pixel_space")
    val_list = ROOT / "data/visdrone_area10/splits/val.txt"
    recovery_path = OUT / "recovery.json"
    seen_archives = set()
    while recovery_path.exists():
        recovery = read(recovery_path)
        archive = Path(recovery["archive"]).resolve()
        check(f"{archive.name}_unique_workspace_archive", archive.is_relative_to(ROOT / "outputs")
              and archive not in seen_archives)
        seen_archives.add(archive)
        receipt_path = archive / "recovery.json"
        check(f"{archive.name}_receipt", digest(receipt_path) == recovery["receipt_sha256"])
        receipt = read(receipt_path)
        original_out = ROOT / "outputs/b2_pilot30_s1"
        original_run = ROOT / "runs/partial10_naive_t050_pilot30_s1"
        for filename, expected in receipt["original_paths_sha256"].items():
            original = Path(filename)
            base, label = (original_out, "pipeline") if original.is_relative_to(original_out) else (original_run, "run")
            archived = archive / label / original.relative_to(base)
            check(f"{archive.name}_{label}_{original.relative_to(base).as_posix()}", digest(archived) == expected)
            hashes[str(archived)] = expected
        check(f"{archive.name}_same_execution_script", digest(archive / "run_b2_seed1.py") == digest(ROOT / "scripts/run_b2_seed1.py"))
        check(f"{archive.name}_same_config", digest(archive / "train_naive_10_seed1.yaml") == digest(ROOT / "configs/train_naive_10_seed1.yaml"))
        hashes[str(receipt_path)] = recovery["receipt_sha256"]
        hashes[str(recovery_path)] = digest(recovery_path)
        recovery_path = archive / "pipeline/recovery.json"
    launch_path = OUT / "launch_receipt.json"
    if launch_path.exists():
        launch = read(launch_path)
        check("independent_launch_wrapper", digest(launch["wrapper"]) == launch["wrapper_sha256"]
              and launch["entry"] == "scripts/run_b2_seed1.py")
        hashes[str(launch_path)] = digest(launch_path)
        hashes[launch["wrapper"]] = launch["wrapper_sha256"]
    all_args = {}
    for seed in (0, 1):
        for method in ("B2", "P1"):
            key = f"{method}_s{seed}"
            record = pair["runs"][key]
            run = ROOT / record["run"]
            args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
            all_args[method,seed] = args
            check(f"{key}_30_finite_epochs", len(check_epochs(run)) == 30)
            check(f"{key}_correct_seed", args["seed"] == seed)
            check(f"{key}_full_schedule", args["epochs"] == 30 and args["imgsz"] == 640
                  and args["batch"] == 8 and args["optimizer"] == "SGD"
                  and args["close_mosaic"] == 10 and not args["resume"])
            experiment = read(run / "experiment.json")
            check(f"{key}_baseline_or_p1_loss", (experiment.get("method", "baseline") == "baseline" and experiment.get("p1") is None)
                  if method == "B2" else experiment["p1"] == {"soft_weights": True, "uncertain_ignore": True})
            check(f"{key}_best_last_hashes", digest(run / "weights/best.pt") == record["best_sha256"]
                  and digest(run / "weights/last.pt") == record["last_sha256"])
            metrics = read(ROOT / "runs/val" / f"{run.name}_val960/metrics.json")
            recall_dir = ROOT / f"runs/size_recall/{method.lower()}_pilot30_s{seed}"
            recall = read(recall_dir / "metrics.json")
            check(f"{key}_full_val960", metrics["imgsz"] == recall["imgsz"] == 960 and recall["images"] == 548)
            check(f"{key}_evaluation_weights", Path(metrics["weights"]) == run / "weights/best.pt"
                  and recall["weights_sha256"] == record["best_sha256"])
            check(f"{key}_recall_operation_point", all(recall[k] == protocol[k] for k in protocol_keys))
            check(f"{key}_same_val_order", (recall_dir / "images.txt").read_bytes() == val_list.read_bytes())
            check(f"{key}_complete_gt_counts", {s: recall["per_size"][s]["gt"] for s in ("small", "medium", "large")}
                  == {"small": 26577, "medium": 11115, "large": 1067})
            check(f"{key}_saved_predictions_replay", replay(recall_dir) == record["replayed_recall"])
            check(f"{key}_frozen_training_inputs", all(digest(p) == h for p, h in experiment.get("input_sha256", {}).items()))
            raw = {"mAP50": metrics["results"]["metrics/mAP50(B)"],
                   "mAP50_95": metrics["results"]["metrics/mAP50-95(B)"],
                   **{f"{s}_recall": recall["per_size"][s]["recall"] for s in ("small", "medium", "large")}}
            check(f"{key}_comparison_equals_raw", raw == record["metrics"] == pair["per_seed"][str(seed)][method])
            check(f"{key}_class_mean", metrics["class_indices"] == list(range(10)) and bool(np.isclose(
                np.mean(metrics["class_maps"]), raw["mAP50_95"], atol=1e-12, rtol=0)))
        check(f"seed{seed}_actual_parameters_paired", not parameter_differences(
            all_args["B2",seed], all_args["P1",seed], {"data", "name", "save_dir"}))
    for method in ("B2", "P1"):
        check(f"{method}_only_seed_name_changed", not parameter_differences(
            all_args[method,0], all_args[method,1], {"seed", "name", "save_dir"}))
    for key in METRICS:
        delta = np.array([100*(p["P1"][key]-p["B2"][key]) for p in pair["per_seed"].values()])
        stats = pair["paired_delta_p1_minus_b2_pp"][key]
        check(f"{key}_independent_statistics", bool(np.isclose(delta.mean(), stats["mean_pp"], atol=1e-12, rtol=0))
              and bool(np.isclose(delta.std(ddof=1), stats["sample_std_pp"], atol=1e-12, rtol=0)))
    hashes[__file__] = digest(__file__)
    write(OUT / "independent_audit.json", {"audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "acceptance_checks_passed": all(checks.values()), "checks": checks,
        "frozen_files_checked": len(frozen["sha256"]), "sha256": hashes,
        "scope": "four complete saved evaluation replays; fixed split and teacher, two student seeds"})
    print(f"Independent acceptance passed: {len(checks)} checks; {len(frozen['sha256'])} frozen files; 4x548 predictions replayed")


if __name__ == "__main__":
    main()
