"""Diagnose high-precision loss using accepted caches only: no model/inference/training."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import time
import traceback

import numpy as np

import evaluate_paper1_small_pr as base
from regionssod.evaluation.error_diagnosis import (
    ERROR_NAMES, LOST_NAMES, box_iou_matrix, classify_false_predictions,
    gt_candidate_summary, json_ready, lost_gt_reason, paired_geometry, transition_counts,
)
from regionssod.evaluation.precision_recall import (
    assign_predictions, precision_constrained_point, threshold_curve,
)

ROOT = base.ROOT
OUT = ROOT / "outputs/paper1_high_precision_diagnostic_20261102"
PROTOCOL = ROOT / "docs/paper/high_precision_diagnostic_protocol.md"
DOCUMENT = ROOT / "docs/paper/high_precision_diagnostic_results.md"
LEVELS = (.7, .9)
IOUS = (.5, .75)


def verify_inputs():
    audit = base.read(base.OUT / "independent_audit.json")
    if audit.get("accepted") is not True or audit.get("cached_replay_exact") is not True:
        raise ValueError("accepted four-arm cache is required")
    base.verify_hashes(base.read(base.OUT / "manifest.json")["input_sha256"])
    base.verify_hashes(base.read(base.OUT / "artifact_sha256.json"))
    base.verify_captures()


def input_manifest():
    verify_inputs()
    paths = [PROTOCOL, Path(__file__).resolve(), ROOT / "src/regionssod/evaluation/error_diagnosis.py",
             ROOT / "src/regionssod/evaluation/precision_recall.py",
             ROOT / "scripts/evaluate_paper1_small_pr.py"]
    paths += [base.OUT / name for name in ("manifest.json", "ground_truth.json", "comparison.json",
              "capture_complete.json", "independent_audit.json", "artifact_sha256.json")]
    for key in base.KEYS:
        paths += [base.OUT / key / name for name in
                  ("predictions.jsonl", "inference.json", "analysis.json", "micro_iou0.5.npz", "micro_iou0.75.npz")]
    return {"created_at": base.now(), "selection": "H2 existence/geometric evidence separation, unconfirmed",
            "precision_levels": LEVELS, "matching_ious": IOUS,
            "input_sha256": {str(path): base.digest(path) for path in paths}}


def describe(values):
    array = np.asarray(values)
    return {"n": len(array), "mean": float(array.mean()) if len(array) else None,
            "q10_q50_q90": np.quantile(array, [.1, .5, .9]).tolist() if len(array) else None}


def truth_arrays(gt):
    annotations = gt["annotations"]
    if [item["id"] for item in annotations] != list(range(1, len(annotations) + 1)):
        raise ValueError("fixed GT identifiers are not consecutive")
    boxes = np.asarray([item["bbox"] for item in annotations], float)
    boxes[:, 2:] += boxes[:, :2]
    return {"boxes": boxes, "areas": np.asarray([item["area"] for item in annotations]),
            "classes": np.asarray([item["category_id"] - 1 for item in annotations]),
            "image_ids": np.asarray([item["image_id"] for item in annotations])}


def model_diagnostics(key, gt, truth, original, save):
    records, _ = base.load_predictions(key, gt)
    n_gt = len(gt["annotations"])
    candidates = {name: np.zeros(n_gt) for name in
                  ("best_same_iou", "best_other_iou", "score_iou0.5", "score_iou0.75",
                   "index_iou0.5", "index_iou0.75")}
    sequences = {str(iou): {"match": [], "errors": [], "areas": []} for iou in IOUS}
    scores, predicted_areas, predicted_classes = [], [], []
    full_hit_scores = {str(iou): np.full(n_gt, -1.) for iou in IOUS}
    geometry = {name: np.full(n_gt, np.nan) for name in
                ("iou", "normalized_center_error", "normalized_edge_error", "score")}
    by_image = {image["id"]: [] for image in gt["images"]}
    for annotation in gt["annotations"]:
        by_image[annotation["image_id"]].append(annotation["id"] - 1)
    for image_index, (image, row) in enumerate(zip(gt["images"], records)):
        ids = np.asarray(by_image[image["id"]], dtype=int)
        classes = np.asarray(row["classes"], dtype=int)
        confidence = np.asarray(row["scores"])
        boxes = np.asarray(row["xyxy"]).reshape(-1, 4)
        overlaps = box_iou_matrix(boxes, truth["boxes"][ids])
        summary = gt_candidate_summary(overlaps, classes, confidence, truth["classes"][ids])
        for name, values in summary.items():
            if name.startswith("index_"):
                values = np.where(values >= 0, values + len(scores), -1)
            candidates[name][ids] = values
        scores.extend(confidence)
        predicted_areas.extend(np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1))
        predicted_classes.extend(classes)
        for iou in IOUS:
            local = assign_predictions(classes, confidence, boxes, truth["classes"][ids], truth["boxes"][ids], iou)
            errors = classify_false_predictions(local, overlaps, classes, truth["classes"][ids], iou)
            valid = local >= 0
            global_ids = np.full(len(local), -1, dtype=int)
            global_ids[valid] = ids[local[valid]]
            areas = np.full(len(local), -1.)
            areas[valid] = truth["areas"][global_ids[valid]]
            full_hit_scores[str(iou)][global_ids[valid]] = confidence[valid]
            seq = sequences[str(iou)]
            seq["match"].extend(global_ids)
            seq["errors"].extend(errors)
            seq["areas"].extend(areas)
            # Check actual prefix matching rather than assuming a threshold
            # can be applied after an incompatible assignment procedure.
            for level in LEVELS:
                point = original["models"][key]["precision_constraints"][str(iou)][str(level)]
                threshold = point["threshold"] if point else 2.
                selected = confidence >= threshold
                direct = assign_predictions(classes[selected], confidence[selected], boxes[selected],
                                            truth["classes"][ids], truth["boxes"][ids], iou)
                if not np.array_equal(direct, local[selected]):
                    raise ValueError("prefix matching differs from direct subset matching")
            if iou == .5:
                paired = paired_geometry(boxes[valid], truth["boxes"][global_ids[valid]])
                for name, values in paired.items():
                    geometry[name][global_ids[valid]] = values
                geometry["score"][global_ids[valid]] = confidence[valid]
        if image_index % 150 == 149:
            print(f"{key}: {image_index + 1}/548 images", flush=True)
    scores = np.asarray(scores)
    predicted_areas, predicted_classes = np.asarray(predicted_areas), np.asarray(predicted_classes)
    points, hits, errors_at_points, coarse_at_points, oracle = {}, {}, {}, {}, {}
    counts = {"small": 26577, "medium": 11115, "large": 1067}
    for iou, seq in sequences.items():
        seq = {name: np.asarray(values) for name, values in seq.items()}
        sequences[iou] = seq
        curve = threshold_curve(scores, seq["areas"], counts)
        stored = dict(np.load(base.OUT / key / f"micro_iou{iou}.npz"))
        for name in curve:
            np.testing.assert_array_equal(curve[name], stored[name])
        points[iou], hits[iou], errors_at_points[iou] = {}, {}, {}
        for level in LEVELS:
            point = precision_constrained_point(curve, level)
            if point != original["models"][key]["precision_constraints"][iou][str(level)]:
                raise ValueError("operating point differs from accepted evaluation")
            points[iou][str(level)] = point
            selected = scores >= point["threshold"] if point else np.zeros(len(scores), bool)
            hit = np.zeros(n_gt, bool)
            valid = selected & (seq["match"] >= 0)
            hit[seq["match"][valid].astype(int)] = True
            hits[iou][str(level)] = hit
            counts_fp = {name: int((selected & (seq["errors"] == index)).sum())
                         for index, name in enumerate(ERROR_NAMES) if index}
            if point and sum(counts_fp.values()) != point["fp"]:
                raise ValueError("FP partition is not exhaustive")
            errors_at_points[iou][str(level)] = {
                "counts": counts_fp, "per1000_predictions": {
                    name: 1000 * count / int(selected.sum()) if selected.any() else None
                    for name, count in counts_fp.items()},
                "share_of_fp": {name: count / sum(counts_fp.values()) if sum(counts_fp.values()) else None
                                for name, count in counts_fp.items()},
                "threshold_tie_group": int((scores == point["threshold"]).sum()) if point else 0}
            groups = {"predicted_small": predicted_areas < 1024,
                      "predicted_medium": (predicted_areas >= 1024) & (predicted_areas < 9216),
                      "predicted_large": predicted_areas >= 9216}
            groups.update({f"predicted_class:{c}": predicted_classes == c for c in range(10)})
            errors_at_points[iou][str(level)]["prediction_groups"] = {
                name: {"predictions": int((selected & mask).sum()),
                       "tp": int((selected & mask & (seq["match"] >= 0)).sum()),
                       "fp": int((selected & mask & (seq["match"] < 0)).sum()),
                       "matched_gt_small_tp": int((selected & mask & (seq["areas"] >= 0) & (seq["areas"] < 1024)).sum())}
                for name, mask in groups.items()}
        # The counterfactual keeps GT-defined TP and removes one FP category.
        # This is an oracle diagnostic, never a candidate deployment policy.
        oracle[iou] = {}
        for index, name in enumerate(ERROR_NAMES):
            if index == 0:
                continue
            keep = seq["errors"] != index
            point = precision_constrained_point(threshold_curve(scores[keep], seq["areas"][keep], counts), .9)
            original_point = points[iou]["0.9"]
            at_original = scores >= original_point["threshold"] if original_point else np.zeros(len(scores), bool)
            removed_at_original = int((at_original & ~keep).sum())
            oracle[iou][name] = {"point": point, "delta_small_recall_pp": None if point is None or original_point is None
                                else 100 * (point["small_recall"] - original_point["small_recall"]),
                "removed_fp_at_original_threshold": removed_at_original,
                "precision_at_original_threshold_after_removal": original_point["tp"] / (original_point["predictions"] - removed_at_original) if original_point else None,
                "small_recall_at_original_threshold_unchanged": original_point["small_recall"] if original_point else None,
                "new_threshold_minus_original": point["threshold"] - original_point["threshold"] if point and original_point else None,
                "new_small_tp": point["small_tp"] - original_point["small_tp"] if point and original_point else None}
    for level in LEVELS:
        hit = hits["0.5"][str(level)] & (truth["areas"] < 1024)
        own_geometry_low = hit & (geometry["iou"] < .75)
        selected = scores >= points["0.5"][str(level)]["threshold"]
        m50, m75 = sequences["0.5"]["match"].astype(int), sequences["0.75"]["match"].astype(int)
        same_pred_different_gt = selected & (m50 >= 0) & (m75 >= 0) & (m50 != m75)
        inverse50, inverse75 = np.full(n_gt, -1), np.full(n_gt, -1)
        for assignment, inverse in ((m50, inverse50), (m75, inverse75)):
            valid = selected & (assignment >= 0)
            inverse[assignment[valid]] = np.flatnonzero(valid)
        common_strict = (truth["areas"] < 1024) & (inverse50 >= 0) & (inverse75 >= 0)
        coarse_at_points[str(level)] = {"small_tp50": int(hit.sum()),
            "paired_gt_iou_below075": int(own_geometry_low.sum()),
            "fraction_below075": float(own_geometry_low.sum() / hit.sum()) if hit.any() else None,
            "small_gt_covered075_rematching": int(((truth["areas"] < 1024) & (inverse75 >= 0)).sum()),
            "tp50_predictions_unmatched075": int((selected & (m50 >= 0) & (m75 < 0)).sum()),
            "predictions_switching_gt_identity": int(same_pred_different_gt.sum()),
            "common_small_gt_between_iou_thresholds": int(common_strict.sum()),
            "common_small_gt_switching_prediction": int((common_strict & (inverse50 != inverse75)).sum())}
    result = {"points": points, "fp_partition": errors_at_points,
              "coarse_tp50_geometry": coarse_at_points, "oracle_remove_fp": oracle}
    if save:
        arrays = {"scores": scores, **{f"candidate_{k}": v for k, v in candidates.items()},
                  **{f"geometry_{k}": v for k, v in geometry.items()}}
        for iou, seq in sequences.items():
            arrays.update({f"iou{iou}_{name}": values for name, values in seq.items()})
        path = OUT / key
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "diagnostic_arrays.npz", **arrays)
    return result, {"candidates": candidates, "hits": hits, "geometry": geometry,
                    "full_hit_scores": full_hit_scores, "match_ids": {iou: seq["match"] for iou, seq in sequences.items()}}


def paired_diagnostics(seed, models, state, gt, truth):
    b2key, p1key = f"b2_s{seed}", f"p1_s{seed}"
    a, b = state[b2key], state[p1key]
    small = truth["areas"] < 1024
    events, turnovers, transitions, geometry = [], {}, {}, {}
    names = {item["id"] - 1: item["name"] for item in gt["categories"]}
    for iou in map(str, IOUS):
        turnovers[iou] = {}
        for level in map(str, LEVELS):
            before, after = a["hits"][iou][level], b["hits"][iou][level]
            gained, lost = small & ~before & after, small & before & ~after
            turnovers[iou][level] = {"gained": int(gained.sum()), "lost": int(lost.sum()),
                                    "net": int(gained.sum() - lost.sum())}
        transitions[iou] = transition_counts(a["hits"][iou]["0.7"], b["hits"][iou]["0.7"],
                                             a["hits"][iou]["0.9"], b["hits"][iou]["0.9"], small)
        for direction, current_key, missing_state, reference, hit in (
            ("b2_only", p1key, b, a["hits"][iou]["0.9"], b["hits"][iou]["0.9"]),
            ("p1_only", b2key, a, b["hits"][iou]["0.9"], a["hits"][iou]["0.9"])):
            target = small & reference & ~hit
            point = models[current_key]["points"][iou]["0.9"]
            threshold = point["threshold"] if point else 2.
            for index in np.flatnonzero(target):
                annotation = gt["annotations"][index]
                event = {"seed": seed, "iou": float(iou), "direction": direction,
                    "gt_id": index + 1, "image_id": annotation["image_id"],
                    "image": gt["images"][annotation["image_id"] - 1]["file_name"],
                    "class": names[int(truth["classes"][index])], "area": truth["areas"][index].item(),
                    "reason": lost_gt_reason(missing_state["candidates"], index, float(iou), threshold),
                    "best_same_iou": missing_state["candidates"]["best_same_iou"][index].item(),
                    "best_other_iou": missing_state["candidates"]["best_other_iou"][index].item(),
                    "best_eligible_score": missing_state["candidates"][f"score_iou{iou}"][index].item(),
                    "threshold": threshold,
                    "full_cache_gt_matched": bool(missing_state["full_hit_scores"][iou][index] >= 0),
                    "full_cache_actual_match_score": missing_state["full_hit_scores"][iou][index].item(),
                    "has_same_class_qualified_candidate": bool(missing_state["candidates"][f"score_iou{iou}"][index] >= 0),
                    "has_coarse_same_class_candidate": bool(missing_state["candidates"]["score_iou0.5"][index] >= 0),
                    "has_other_class_qualified_candidate": bool(missing_state["candidates"]["best_other_iou"][index] >= float(iou))}
                pred_index = int(missing_state["candidates"][f"index_iou{iou}"][index])
                event["best_eligible_prediction_index"] = pred_index
                event["that_prediction_full_assigned_gt_id"] = int(missing_state["match_ids"][iou][pred_index]) + 1 if pred_index >= 0 else 0
                events.append(event)
    for level in map(str, LEVELS):
        common = small & a["hits"]["0.5"][level] & b["hits"]["0.5"][level]
        groups = {"all_common_small": small,
            "area_lt16sq": small & (truth["areas"] < 256),
            "area_16sq_to32sq": small & (truth["areas"] >= 256)}
        groups.update({f"class:{name}": small & (truth["classes"] == cls) for cls, name in names.items()})
        geometry[level] = {}
        for name, population in groups.items():
            mask = population & common
            before, after = a["hits"]["0.5"][level], b["hits"]["0.5"][level]
            group = {"n": int(mask.sum()), "sparse_lt5": int(mask.sum()) < 5,
                "support": {"both": int((population & before & after).sum()),
                    "b2_only": int((population & before & ~after).sum()),
                    "p1_only": int((population & ~before & after).sum()),
                    "neither": int((population & ~before & ~after).sum())}}
            for metric in ("iou", "normalized_center_error", "normalized_edge_error"):
                av, bv = a["geometry"][metric][mask], b["geometry"][metric][mask]
                group[metric] = {"b2": describe(av), "p1": describe(bv), "paired_p1_minus_b2": describe(bv - av)}
            group["fraction_iou_below075"] = {
                "b2": float((a["geometry"]["iou"][mask] < .75).mean()) if mask.any() else None,
                "p1": float((b["geometry"]["iou"][mask] < .75).mean()) if mask.any() else None}
            geometry[level][name] = group
    reasons = {}
    for iou in map(str, IOUS):
        reasons[iou] = {}
        for direction in ("b2_only", "p1_only"):
            count = Counter(event["reason"] for event in events if str(event["iou"]) == iou and event["direction"] == direction)
            reasons[iou][direction] = {name: count[name] for name in LOST_NAMES}
    oracle_changes = {}
    for iou in map(str, IOUS):
        oracle_changes[iou] = {}
        for name in ERROR_NAMES[1:]:
            av = models[b2key]["oracle_remove_fp"][iou][name]
            bv = models[p1key]["oracle_remove_fp"][iou][name]
            original_gap = 100 * (models[p1key]["points"][iou]["0.9"]["small_recall"] - models[b2key]["points"][iou]["0.9"]["small_recall"])
            after_gap = 100 * (bv["point"]["small_recall"] - av["point"]["small_recall"]) if av["point"] and bv["point"] else None
            oracle_changes[iou][name] = {"original_pair_gap_pp": original_gap,
                "oracle_pair_gap_pp": after_gap, "change_in_pair_gap_pp": after_gap - original_gap if after_gap is not None else None}
    return {"small_gt_turnover": turnovers, "transition_cells_70_to90": transitions,
            "missing_candidate_reasons90": reasons, "common_gt_geometry": geometry,
            "oracle_pair_gap_changes": oracle_changes}, events


def compute(save):
    gt = base.read(base.OUT / "ground_truth.json")
    truth, original = truth_arrays(gt), base.read(base.OUT / "comparison.json")
    models, states = {}, {}
    for key in base.KEYS:
        print(f"DIAGNOSE {key} {base.now()}", flush=True)
        models[key], states[key] = model_diagnostics(key, gt, truth, original, save)
    pairs, events = {}, []
    for seed in (0, 1):
        pairs[str(seed)], current = paired_diagnostics(seed, models, states, gt, truth)
        events.extend(current)
    return {"scope": "GT-assisted descriptive diagnostics on frozen saved validation candidates; no inference/training",
            "selected_direction": "H2 existence/geometric evidence separation; unconfirmed",
            "models": models, "pairs": pairs, "events90": events,
            "limits": ["GT reason priority is a descriptive partition, not an identified training cause",
                "all-size precision constraints are not small-only precision or equal absolute false-positive counts",
                "cache score floor/NMS/500 cap prevent conclusions about raw uncensored model candidates",
                "common-GT pairing controls object identity but is selected by both methods' successes",
                "oracle FP removal uses validation GT; no deployable method or new student result",
                "same split/teacher and only2 studentseeds, training recoveries retained"]}


def tables(result):
    lines = ["## 零训练诊断数字表", "", "### 实际操作点与GT尺寸TP", "",
             "| 模型 | IoU | P下限 | 实际P | 阈值 | 预测数 | small/medium/large TP | 阈值同分数 |",
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    for key, model in result["models"].items():
        for iou, levels in model["points"].items():
            for level, point in levels.items():
                if point:
                    ties = model["fp_partition"][iou][level]["threshold_tie_group"]
                    lines.append(f"| {key} | {iou} | {level} | {point['precision']:.6f} | {point['threshold']:.6f} | "
                        f"{point['predictions']} | {point['small_tp']}/{point['medium_tp']}/{point['large_tp']} | {ties} |")
    lines += ["", "### 70%与90%precision约束的small逐目标增失", "",
             "| seed | IoU | precision下限 | P1新增 | P1丢失 | 净增 |", "|---|---:|---:|---:|---:|---:|"]
    for seed, pair in result["pairs"].items():
        for iou, levels in pair["small_gt_turnover"].items():
            for level, values in levels.items():
                lines.append(f"| {seed} | {iou} | {level} | {values['gained']} | {values['lost']} | {values['net']:+d} |")
    for iou in map(str, IOUS):
        lines += ["", f"### IoU{iou}：B2在90%命中、P1未命中的small GT", "",
                  "| P1缓存中的描述性原因 | seed0 | seed1 |", "|---|---:|---:|"]
        for name in LOST_NAMES:
            values = [result["pairs"][str(seed)]["missing_candidate_reasons90"][iou]["b2_only"][name] for seed in (0, 1)]
            lines.append(f"| {name} | {values[0]} | {values[1]} |")
    lines += ["", "### IoU0.50、90%precision操作点的FP组成（数量 / 每1000预测）", "",
              "| 类型 | B2 seed0 | P1 seed0 | B2 seed1 | P1 seed1 |", "|---|---:|---:|---:|---:|"]
    for name in ERROR_NAMES[1:]:
        values = []
        for key in base.KEYS:
            record = result["models"][key]["fp_partition"]["0.5"]["0.9"]
            values.append(f"{record['counts'][name]} / {record['per1000_predictions'][name]:.4f}")
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += ["", "### 共同命中small GT的配对几何（IoU0.50选定操作点）", "",
              "| seed | precision | 共同GT数 | 平均IoU P1−B2 | 归一化中心误差差 | 归一化边误差差 |", "|---|---:|---:|---:|---:|---:|"]
    for seed, pair in result["pairs"].items():
        for level, groups in pair["common_gt_geometry"].items():
            group = groups["all_common_small"]
            values = [group[metric]["paired_p1_minus_b2"]["mean"] for metric in
                      ("iou", "normalized_center_error", "normalized_edge_error")]
            lines.append(f"| {seed} | {level} | {group['n']} | " + " | ".join(f"{v:+.6f}" for v in values) + " |")
    lines += ["", "### Oracle剔除一类GT定义FP后，90%precision small recall变化 / pp", "",
              "此表只表示GT oracle条件容量，不能作为后处理或学生提升。", "",
              "| 删除类别 | B2 seed0 | P1 seed0 | B2 seed1 | P1 seed1 |", "|---|---:|---:|---:|---:|"]
    for name in ERROR_NAMES[1:]:
        values = [result["models"][key]["oracle_remove_fp"]["0.5"][name]["delta_small_recall_pp"] for key in base.KEYS]
        lines.append(f"| {name} | " + " | ".join("NA" if v is None else f"{v:+.4f}" for v in values) + " |")
    lines += ["", "### Oracle后两臂small recall差值变化 / pp（后差值减原差值）", "",
              "| 删除类别 | seed0 | seed1 |", "|---|---:|---:|"]
    for name in ERROR_NAMES[1:]:
        values = [result["pairs"][str(seed)]["oracle_pair_gap_changes"]["0.5"][name]["change_in_pair_gap_pp"] for seed in (0, 1)]
        lines.append(f"| {name} | " + " | ".join("NA" if v is None else f"{v:+.4f}" for v in values) + " |")
    return "\n".join(lines) + "\n"


def plot_result(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    labels = ["Duplicate/competition", "Wrong class", "Localization near", "Near other class", "Unexplained by GT"]
    for index, key in enumerate(base.KEYS):
        values = result["models"][key]["fp_partition"]["0.5"]["0.9"]["per1000_predictions"]
        axes[0].bar(np.arange(5) + (index - 1.5) * .18, [values[n] for n in ERROR_NAMES[1:]],
                    width=.18, label=key)
    axes[0].set_xticks(np.arange(5), labels, rotation=30, ha="right")
    axes[0].set_ylabel("False predictions per1000 predictions")
    axes[0].set_title("Actual operating point: precision >=90%, IoU0.50")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=.25)
    x = np.arange(7)
    for seed, shift in ((0, -.18), (1, .18)):
        values = result["pairs"][str(seed)]["missing_candidate_reasons90"]["0.5"]["b2_only"]
        axes[1].barh(x + shift, [values[name] for name in LOST_NAMES], height=.35, label=f"Student seed{seed}")
    axes[1].set_yticks(x, ["Active competition", "Compatible box below threshold", "Coarse only", "Wrong-class candidate",
                          "Localization near", "Near other class", "Saved-candidate gap"], fontsize=8)
    axes[1].set_xlabel("Small GT hit by B2 but missed by P1")
    axes[1].set_title("Candidate evidence for lost GT (not training causes)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="x", alpha=.25)
    fig.suptitle("Cache-only high-precision diagnostics | Full-size precision, GT-small recall", fontsize=11)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"high_precision_diagnostic.{ext}", dpi=180)
    plt.close(fig)


def run():
    if (OUT / "diagnosis.json").exists() or any((OUT / f"arrays_{key}.npz").exists() for key in base.KEYS):
        raise FileExistsError("existing diagnostic; use --audit instead of overwriting")
    if (OUT / "manifest.json").exists():
        manifest = base.read(OUT / "manifest.json")
        base.verify_hashes(manifest["input_sha256"])
        expected = input_manifest()["input_sha256"]
        if manifest["input_sha256"] != expected:
            raise ValueError("failed-attempt manifest does not match current frozen inputs")
    else:
        manifest = input_manifest()
        base.write(OUT / "manifest.json", manifest)
    result = compute(True)
    result = json_ready(result)
    base.write(OUT / "diagnosis.json", result)
    (OUT / "results_tables.md").write_text(tables(result), encoding="utf-8")
    plot_result(result)
    with (OUT / "events90.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(result["events90"][0]) if result["events90"] else ["seed", "gt_id"]
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(result["events90"])
    base.verify_hashes(manifest["input_sha256"])
    paths = [OUT / name for name in ("diagnosis.json", "results_tables.md", "events90.csv")]
    paths += [OUT / key / "diagnostic_arrays.npz" for key in base.KEYS]
    paths += [OUT / f"high_precision_diagnostic.{ext}" for ext in ("png", "pdf", "svg")]
    base.write(OUT / "artifact_sha256.json", {str(path): base.digest(path) for path in paths})
    base.write(OUT / "status.json", {"stage": "completed", "at": base.now(), "inference": 0, "training_steps": 0})


def check_doc():
    expected = tables(base.read(OUT / "diagnosis.json"))
    text = DOCUMENT.read_text(encoding="utf-8")
    actual = text.split("<!-- diagnostic-tables:begin -->\n", 1)[1].split("<!-- diagnostic-tables:end -->", 1)[0]
    if actual != expected:
        raise ValueError("diagnostic document tables differ from frozen JSON")
    print("All diagnostic document tables match frozen JSON.", flush=True)


def audit():
    verify_inputs()
    base.verify_hashes(base.read(OUT / "manifest.json")["input_sha256"])
    base.verify_hashes(base.read(OUT / "artifact_sha256.json"))
    result = compute(False)
    if result != base.read(OUT / "diagnosis.json"):
        raise ValueError("diagnostic numeric replay differs")
    if (OUT / "results_tables.md").read_text(encoding="utf-8") != tables(result):
        raise ValueError("rendered table replay differs")
    base.write(OUT / "independent_audit.json", {"accepted": True, "at": base.now(),
        "input_hashes": len(base.read(OUT / "manifest.json")["input_sha256"]),
        "images_per_model": 548, "models": base.KEYS, "exact_cached_replay": True,
        "inference": 0, "training_steps": 0})
    print("Accepted: exact cached diagnostic replay,0 new inference,0 training.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--check-doc", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    try:
        if args.run:
            run()
        elif args.audit:
            audit()
        else:
            check_doc()
    except Exception:
        base.write(OUT / "failure.json", {"at": base.now(), "traceback": traceback.format_exc(),
                                         "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
