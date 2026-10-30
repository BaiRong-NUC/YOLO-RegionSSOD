"""Audit the completed A-paper soft-weight ablation without training/inference."""
from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from run_p1_no_soft_weights import EVIDENCE, FULL, RECALL, ROOT, RUN, VAL
from run_p1_scale_seed_pair import (
    check_epochs, digest, parameter_differences, read, replay, write,
)


def continuation(delta: dict) -> bool:
    """Exploratory follow-up line recorded before this full-data ablation."""
    return ((delta["mAP50_95"] >= .3 and delta["small_recall"] >= 0)
            or (delta["small_recall"] >= 1.0 and delta["mAP50_95"] >= -.1))


def main() -> None:
    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)
        if not passed:
            raise ValueError(name)

    check("pipeline_completed", read(EVIDENCE / "status.json")["stage"] == "completed")
    launch = read(EVIDENCE / "launch_receipt.json")
    check("background_process_exited_successfully", launch.get("exit_code") == 0)
    receipt = read(EVIDENCE / "preflight.json")
    check("frozen_inputs_unchanged", all(digest(p) == h for p, h in receipt["sha256"].items()))
    hashes = {}
    for line in (RUN / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        checksum, filename = line.split("  ", 1)
        hashes[filename] = checksum
    check("result_artifact_hashes", all(digest(p) == h for p, h in hashes.items()))
    check("complete_30_epochs", len(check_epochs(RUN)) == 30 and len(check_epochs(FULL)) == 30)
    actual = yaml.safe_load((RUN / "args.yaml").read_text(encoding="utf-8"))
    reference = yaml.safe_load((FULL / "args.yaml").read_text(encoding="utf-8"))
    check("paired_actual_hyperparameters",
          not parameter_differences(reference, actual, {"name", "save_dir"}))
    experiment, old = read(RUN / "experiment.json"), read(FULL / "experiment.json")
    check("only_soft_weight_method_change",
          experiment["p1"] == {"soft_weights": False, "uncertain_ignore": True}
          and old["p1"] == {"soft_weights": True, "uncertain_ignore": True}
          and experiment["pseudo_summary"] == old["pseudo_summary"]
          and experiment["train_args"] == old["train_args"]
          and experiment["model"] == old["model"] and experiment["data"] == old["data"])
    check("run_input_hashes_match",
          all(digest(p) == h for p, h in experiment["input_sha256"].items()))
    weight = RUN / "weights/best.pt"
    weight_hash = digest(weight)
    comparison = read(EVIDENCE / "comparison.json")
    val, recall = read(VAL / "metrics.json"), read(RECALL / "metrics.json")
    old_val = read(ROOT / "runs/val/partial10_p1_pilot30_s0_val960/metrics.json")
    old_recall_dir = ROOT / "runs/size_recall/p1_pilot30_s0"
    old_recall = read(old_recall_dir / "metrics.json")
    check("val960_same_data_weights", val["imgsz"] == 960
          and Path(val["weights"]) == weight and val["data"] == old_val["data"]
          and recall["weights_sha256"] == weight_hash
          and comparison["ablation"]["weights_sha256"] == weight_hash)
    check("same_full_val_list", (RECALL / "images.txt").read_bytes()
          == (old_recall_dir / "images.txt").read_bytes())
    check("recall_protocol_paired", all(recall[k] == old_recall[k] for k in (
        "images", "imgsz", "conf", "match_iou", "nms_iou", "max_det",
        "ultralytics_version", "size_pixel_space", "matching")))
    check("full_val548_operating_point", recall["images"] == 548
          and recall["conf"] == .25 and recall["match_iou"] == .5
          and recall["nms_iou"] == .7 and recall["max_det"] == 500)
    replayed, old_replayed = replay(RECALL), replay(old_recall_dir)
    check("all_gt_sizes_preserved",
          [replayed[s]["gt"] for s in ("small", "medium", "large")] == [26577, 11115, 1067])
    check("both_predictions_replayed",
          replayed == recall["per_size"] and old_replayed == old_recall["per_size"])
    delta = {}
    for label, key in (("mAP50", "metrics/mAP50(B)"),
                       ("mAP50_95", "metrics/mAP50-95(B)"),
                       ("precision", "metrics/precision(B)"), ("recall", "metrics/recall(B)")):
        delta[label] = 100 * (val["results"][key] - old_val["results"][key])
    for scale in ("small", "medium", "large"):
        delta[f"{scale}_recall"] = 100 * (replayed[scale]["recall"] - old_replayed[scale]["recall"])
    reported = comparison["delta_no_soft_weights_minus_full_p1_percentage_points"]
    check("deltas_recomputed", reported.keys() == delta.keys()
          and all(math.isclose(delta[k], reported[k], abs_tol=1e-12) for k in delta))
    check("class_maps_consistent", val["class_indices"] == list(range(10))
          and bool(np.isfinite(val["class_maps"]).all())
          and math.isclose(float(np.mean(val["class_maps"])),
                           val["results"]["metrics/mAP50-95(B)"], abs_tol=1e-12))
    per_class = [{"class_id": i, "name": val["names"][str(i)],
                  "delta_no_soft_minus_full_pp": 100 * (val["class_maps"][i] - old_val["class_maps"][i])}
                 for i in range(10)]
    hashes[str(Path(__file__).resolve())] = digest(__file__)
    write(EVIDENCE / "independent_audit.json", {
        "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "acceptance_checks_passed": all(checks.values()), "checks": checks,
        "frozen_files_checked": len(receipt["sha256"]), "sha256": hashes,
        "delta_no_soft_minus_full_pp": delta, "per_class": per_class,
        "no_soft_candidate_for_followup": continuation(delta),
        "scope": "single split/teacher/student seed; no official AP_small or significance claim",
    })
    print(f"Passed {len(checks)} checks, {len(receipt['sha256'])} frozen files and two full-val replays")
    print(delta)
    print("No-soft follow-up candidate:", continuation(delta))


if __name__ == "__main__":
    main()
