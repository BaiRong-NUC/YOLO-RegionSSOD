"""Independently recheck the completed paired repeat without GPU inference."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from regionssod.data.annotations import iter_images, label_path_for_image, read_label_file
from regionssod.data.dataset import load_source_dataset
from run_p1_scale_seed_pair import (
    ARMS, CONFIGS, METRICS, OUT, ROOT, SOURCE, assert_config, check_epochs,
    digest, parameter_differences, paths, read, replay, write,
)
from scale_seed_tables import load_pair


def main():
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError(name)

    check("pipeline_completed", read(OUT / "status.json")["stage"] == "completed")
    hashes = {}
    # The new audit must not include the previous version of its own output.
    pair = load_pair(hashes, include_independent=False)
    preflight = read(OUT / "preflight.json")
    check("runner_unchanged", digest(ROOT / "scripts/run_p1_scale_seed_pair.py") == preflight["script_sha256"])
    check("all_frozen_hashes_match", all(digest(p) == h for p, h in preflight["sha256"].items()))
    teacher = ROOT / "runs/partial10_sup_s0/weights/best.pt"
    teacher_hash = digest(teacher)
    source = load_source_dataset(SOURCE)
    images = iter_images(source.images_val)
    gt_count = sum(len(read_label_file(label_path_for_image(p, source.images_val, source.labels_val))) for p in images)
    check("full_val_548_images_38759_gt", len(images) == 548 and gt_count == 38759)
    expected_images = [str(p.resolve()) for p in images]
    args_by_arm, per_class = {}, []
    val_by_arm = {}
    for arm in ARMS:
        assert_config(arm)
        run, val_dir, recall_dir = paths(arm)
        check(f"{arm}_thirty_finite_epochs", len(check_epochs(run)) == 30)
        args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        args_by_arm[arm] = args
        old_run = ROOT / "runs" / ARMS[arm].replace("_s1", "_s0")
        old_args = yaml.safe_load((old_run / "args.yaml").read_text(encoding="utf-8"))
        check(f"{arm}_only_student_seed_changed", args["seed"] == 1 and not parameter_differences(
            old_args, args, {"name", "save_dir", "seed"}))
        experiment, old_experiment = read(run / "experiment.json"), read(old_run / "experiment.json")
        check(f"{arm}_run_inputs_match", all(digest(p) == h for p, h in experiment["input_sha256"].items()))
        check(f"{arm}_same_pseudo_data", experiment["pseudo_summary"] == old_experiment["pseudo_summary"])
        check(f"{arm}_same_teacher", experiment["pseudo_summary"]["teacher_sha256"] == teacher_hash)
        check(f"{arm}_same_loss", experiment["p1"] == {"soft_weights": True, "uncertain_ignore": True})
        check(f"{arm}_seed1_config_matches", digest(CONFIGS[arm]) == experiment["input_sha256"][str(CONFIGS[arm])])
        check(f"{arm}_best_last_hashes", digest(run / "weights/best.pt") == pair["seed1"][arm]["weights_sha256"]
              and digest(run / "weights/last.pt") == pair["seed1"][arm]["last_sha256"])
        val, recall = read(val_dir / "metrics.json"), read(recall_dir / "metrics.json")
        val_by_arm[arm] = val
        check(f"{arm}_full_images_order", (recall_dir / "images.txt").read_text(encoding="utf-8").splitlines() == expected_images)
        check(f"{arm}_recall_protocol", recall["images"] == 548 and recall["imgsz"] == val["imgsz"] == 960
              and recall["conf"] == .25 and recall["match_iou"] == .5 and recall["nms_iou"] == .7
              and recall["max_det"] == 500 and recall["ultralytics_version"] == "8.4.172")
        replayed = replay(recall_dir)
        check(f"{arm}_saved_predictions_replay", replayed == pair["seed1"][arm]["replayed_recall"])
        check(f"{arm}_gt_size_counts", [replayed[s]["gt"] for s in ("small", "medium", "large")] == [26577, 11115, 1067])
        check(f"{arm}_class_mean_matches_map", val["class_indices"] == list(range(10))
              and bool(np.isfinite(val["class_maps"]).all())
              and bool(np.isclose(np.mean(val["class_maps"]), val["results"]["metrics/mAP50-95(B)"], atol=1e-12, rtol=0)))
    check("actual_seed1_parameters_paired", not parameter_differences(
        args_by_arm["scale_aware"], args_by_arm["global_t050"], {"data", "name", "save_dir"}))
    for key in METRICS:
        differences = np.array([100 * (p["global_t050"][key] - p["scale_aware"][key])
                                for p in pair["per_seed"].values()])
        stats = pair["paired_delta_global_minus_scale_aware_pp"][key]
        check(f"{key}_independent_paired_statistics", bool(np.isclose(differences.mean(), stats["mean_pp"], atol=1e-12, rtol=0))
              and bool(np.isclose(differences.std(ddof=1), stats["sample_std_pp"], atol=1e-12, rtol=0)))
    for i in range(10):
        a, b = (val_by_arm[arm]["class_maps"][i] for arm in ARMS)
        per_class.append({"class_id": i, "name": val_by_arm["scale_aware"]["names"][str(i)],
                          "scale_aware": a, "global_t050": b, "delta_global_minus_scale_pp": 100 * (b - a)})
    hashes[str(Path(__file__).resolve())] = digest(__file__)
    record = {"audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "acceptance_checks_passed": all(checks.values()), "checks": checks,
              "frozen_files_checked": len(preflight["sha256"]), "per_class_seed1": per_class,
              "class_improved_global": sum(p["delta_global_minus_scale_pp"] > 0 for p in per_class),
              "class_declined_global": sum(p["delta_global_minus_scale_pp"] < 0 for p in per_class),
              "sha256": hashes, "scope": "read-only acceptance of completed runs; no training/inference"}
    write(OUT / "independent_audit.json", record)
    print(f"Independent acceptance passed: {len(checks)} checks; {len(preflight['sha256'])} frozen files; "
          f"two 548-image prediction replays; global class changes {record['class_improved_global']} up/"
          f"{record['class_declined_global']} down.")


if __name__ == "__main__":
    main()
