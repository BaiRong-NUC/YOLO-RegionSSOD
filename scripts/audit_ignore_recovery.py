"""Read-only capacity and noise analysis of frozen P1 ignore candidates.

Locks existing positive matches, uses all GT sizes jointly, exports aggregates
only. Hidden GT is an oracle diagnostic, never a training-label selector.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from audit_paper1_gaps import DATA, PSEUDO, SIZES, match_iou, overlaps, size
from regionssod.data.annotations import (
    Tile, YoloBox, clip_box_to_tile, label_path_for_image, read_label_file,
    replace_images_component,
)
from run_p1_scale_seed_pair import ROOT, digest, read

OUT = ROOT / "outputs/paper1_ignore_recovery_20261030"
PRIOR = ROOT / "outputs/paper1_gap_audit_20261030"
FLOORS = (.25, .20, .15, .10, .05)


def maximum_pairs(adjacency: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-cardinality matching via augmenting paths; no confidence weights."""
    if adjacency.ndim != 2:
        raise ValueError("expected a prediction by GT adjacency matrix")
    neighbors = [[int(g) for g in np.flatnonzero(row)] for row in adjacency]
    assigned = [-1] * adjacency.shape[1]

    def augment(pred: int, seen: set[int]) -> bool:
        for target in neighbors[pred]:
            if target in seen:
                continue
            seen.add(target)
            if assigned[target] < 0 or augment(assigned[target], seen):
                assigned[target] = pred
                return True
        return False

    for pred in range(adjacency.shape[0]):
        augment(pred, set())
    return [(pred, target) for target, pred in enumerate(assigned) if pred >= 0]


def residual_analysis(predictions: list[YoloBox], gt: list[YoloBox],
                      locked_gt: set[int]) -> dict:
    """Characterize extra candidates without stealing baseline true positives."""
    if not locked_gt.issubset(set(range(len(gt)))):
        raise ValueError("invalid locked GT indices")
    residual_indices = [g for g in range(len(gt)) if g not in locked_gt]
    residual = [gt[g] for g in residual_indices]
    iou = overlaps(predictions, gt)
    same = np.asarray([p.class_id for p in predictions])[:, None] == np.asarray([g.class_id for g in gt])[None, :]
    edges = same & (iou >= .5)
    residual_edges = edges[:, residual_indices]
    locked_indices = sorted(locked_gt)
    supports_new = residual_edges.any(axis=1)
    supports_old = edges[:, locked_indices].any(axis=1)
    other_class = (~same & (iou >= .5)).any(axis=1)
    greedy = match_iou(predictions, residual)
    maximum = maximum_pairs(residual_edges)
    if not len(greedy) <= len(maximum) <= int(residual_edges.any(axis=0).sum()):
        raise ValueError("invalid matching capacity")
    return {
        "predictions": len(predictions),
        "supports_unmatched_gt": int(supports_new.sum()),
        "supports_both_locked_and_unmatched_gt": int((supports_new & supports_old).sum()),
        "only_overlaps_locked_gt": int((~supports_new & supports_old).sum()),
        "wrong_class_only": int((~supports_new & ~supports_old & other_class).sum()),
        "no_class_aware_iou_ge_050": int((~supports_new & ~supports_old & ~other_class).sum()),
        "raw_unmatched_gt_coverage": int(residual_edges.any(axis=0).sum()),
        "greedy_extra_matches": len(greedy),
        "oracle_max_extra_matches": len(maximum),
        "greedy_extra_gt_indices": [residual_indices[g] for _, g in greedy],
        "oracle_extra_gt_indices": [residual_indices[g] for _, g in maximum],
    }


def analyse() -> tuple[dict, dict[str, str]]:
    receipt = read(PRIOR / "manifest.json")
    hashes = dict(receipt["input_sha256"])
    for filename, expected in hashes.items():
        if digest(filename) != expected:
            raise ValueError(f"frozen diagnostic source changed: {filename}")
    if digest(PRIOR / "analysis.json") != receipt["analysis_sha256"]:
        raise ValueError("previous diagnostic output changed")
    for path in (PRIOR / "manifest.json", PRIOR / "analysis.json", Path(__file__).resolve()):
        hashes[str(path)] = digest(path)
    source, split, summary = (read(DATA / name) for name in ("source.json", "summary.json", "pseudo/p1_s030_m050_l050/summary.json"))
    records = [json.loads(line) for line in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = {r["source_image"]: r for r in map(json.loads, (PSEUDO / "metadata.jsonl").read_text(encoding="utf-8").splitlines())}
    candidates = {r["source_image"]: r for r in map(json.loads, (PSEUDO / "candidates.jsonl").read_text(encoding="utf-8").splitlines())}
    totals: dict[str, Counter] = defaultdict(Counter)
    baseline = Counter()
    seen: set[str] = set()
    for record in records:
        label = label_path_for_image(Path(record["source_image"]), Path(source["images_train"]), Path(source["labels_train"]))
        boxes = read_label_file(label)
        width, height = record["source_size"]
        for entry in record["unlabeled_tiles"]:
            image = entry["image"]
            if image in seen:
                raise ValueError("duplicate tile")
            seen.add(image)
            tile = Tile(*entry["tile"])
            gt = [g for b in boxes if (g := clip_box_to_tile(b, tile, width, height, split["min_box_visibility"])) is not None]
            if len(gt) != entry["visible_gt_boxes_hidden_from_training"]:
                raise ValueError("GT scope differs")
            row = metadata.get(image)
            positives = [] if row is None else read_label_file(replace_images_component(Path(row["image"])).with_suffix(".txt"))
            locked_pairs = match_iou(positives, gt)
            locked = {g for _, g in locked_pairs}
            baseline["gt"] += len(gt)
            baseline["positive_predictions"] += len(positives)
            baseline["matches"] += len(locked)
            for g, target in enumerate(gt):
                baseline[f"{size(target,tile)}_gt"] += 1
                baseline[f"{size(target,tile)}_matched"] += int(g in locked)
            ignores = [item for item in candidates[image]["candidates"] if item["state"] == "ignore"]
            scenarios = {"kept_all_ignore" if row else "excluded_all_ignore": ignores}
            if row:
                scenarios.update({f"kept_small_conf_ge_{floor:.2f}": [item for item in ignores if item["scale"] == "small" and item["confidence"] >= floor]
                                  for floor in FLOORS})
            for name, selected in scenarios.items():
                pred = [YoloBox(item["class_id"], *item["xywhn"]) for item in selected]
                result = residual_analysis(pred, gt, locked)
                target = totals[name]
                target["tiles"] += 1
                for key, value in result.items():
                    if isinstance(value, int):
                        target[key] += value
                for key, prefix in (("greedy_extra_gt_indices", "greedy"), ("oracle_extra_gt_indices", "oracle")):
                    for g in result[key]:
                        target[f"{prefix}_{size(gt[g],tile)}_extra"] += 1
                # Ordinary combined matching can reassign baseline pairs; record this separately.
                union = match_iou(positives + pred, gt)
                target["union_iou_greedy_match_delta"] += len(union) - len(locked_pairs)
    if seen != set(candidates) or len(seen) != 6471 or len(metadata) != 4839:
        raise ValueError("incomplete candidate scope")
    if (baseline["gt"], baseline["positive_predictions"], baseline["matches"],
        baseline["small_gt"], baseline["small_matched"]) != (38245, 35067, 25669, 22665, 13681):
        raise ValueError("baseline replay differs")
    if totals["kept_all_ignore"]["predictions"] != summary["ignore_boxes"]:
        raise ValueError("in-training ignore count differs")
    scenarios_out = {}
    for name, counter in sorted(totals.items()):
        if sum(counter[k] for k in ("supports_unmatched_gt", "only_overlaps_locked_gt", "wrong_class_only", "no_class_aware_iou_ge_050")) != counter["predictions"]:
            raise ValueError("candidate categories do not partition predictions")
        scenarios_out[name] = {**dict(counter),
            "greedy_novel_match_fraction": counter["greedy_extra_matches"] / counter["predictions"] if counter["predictions"] else None,
            "oracle_novel_match_fraction": counter["oracle_max_extra_matches"] / counter["predictions"] if counter["predictions"] else None}
    result = {"baseline": dict(baseline), "scenarios": scenarios_out,
        "protocol": "lock P1 positive matches; same-class IoU>=0.50; ignore-to-residual-GT matching across all sizes jointly",
        "maximum_matching": "deterministic maximum-cardinality bipartite matching on residual GT; total capacity only",
        "selection": "fixed confidence floors chosen before this audit; predicted-small ignore candidates in kept training tiles",
        "limitations": ["hidden GT is diagnostic oracle only, no inference/training or label export",
            "upper bound conditional on locked baseline matches, not an implementable recovery algorithm",
            "per-size counts describe one deterministic total-cardinality solution, not separate per-size maxima",
            "novel-match fraction penalizes duplicates of locked GT; not standard overall pseudo-label precision",
            "excluded tiles are outside current P1 training and must not be silently counted as current gains",
            "ordinary union IoU-greedy matching can reassign positives; reported separately from locked matching",
            "saved candidates are post-NMS at conf>=0.05; no claim about pre-NMS or below-floor predictions",
            "no confidence-floor row proves improved student mAP or permits choosing training labels using GT"]}
    return result, hashes


def markdown(result: dict) -> str:
    rows = ["# Ignore 候选可恢复增量：离线诊断", "",
        "固定原 P1 正框匹配，ignore 只与剩余 GT 类别一致、IoU≥0.50 一对一匹配；所有尺度联合。",
        "最大匹配是依赖隐藏 GT 的容量上界，不是新算法或学生精度。", "",
        "| 候选范围 | 新增框 | 覆盖剩余GT | 贪心新增匹配 | 最大新增匹配上界 | 贪心新增 small | 仅覆盖已有GT框 | 贪心新增匹配比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["scenarios"].items():
        fraction = row["greedy_novel_match_fraction"]
        rows.append(f"| {name} | {row['predictions']} | {row['raw_unmatched_gt_coverage']} | {row['greedy_extra_matches']} | {row['oracle_max_extra_matches']} | {row.get('greedy_small_extra',0)} | {row['only_overlaps_locked_gt']} | {100*fraction:.2f}% |" if fraction is not None else f"| {name} | 0 | 0 | 0 | 0 | 0 | 0 | — |")
    rows += ["", "excluded 行来自没有 positive 的排除 tile，单列且不属于当前入训数据。",
             "新增独立匹配比例为固定旧匹配后新增匹配/新增框，重复覆盖旧GT也不记为新检出；不等同于总体 precision。",
             "所有候选规则只按固定置信度和预测尺寸取框；隐藏GT不作为训练标签或新筛选器。", ""]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    receipt: dict = {}
    if not args.check and OUT.exists():
        raise FileExistsError("refusing overwrite of frozen recovery audit")
    if args.check:
        receipt = read(OUT / "manifest.json")
        for filename, expected in receipt["input_sha256"].items():
            if digest(filename) != expected:
                raise ValueError(f"input changed: {filename}")
        for filename, expected in receipt["output_sha256"].items():
            if digest(OUT / filename) != expected:
                raise ValueError(f"output changed: {filename}")
    result, hashes = analyse()
    report = markdown(result)
    if args.check:
        if result != read(OUT / "analysis.json") or report != (OUT / "report.md").read_text(encoding="utf-8") or hashes != receipt["input_sha256"]:
            raise ValueError("recovery audit is not reproducible")
        print(f"Verified locked matching, oracle capacity and {len(hashes)} input hashes over 6471 tiles")
        return
    OUT.mkdir(parents=True)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    (OUT / "report.md").write_text(report, encoding="utf-8")
    receipt = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "input_sha256": hashes,
               "output_sha256": {name: digest(OUT / name) for name in ("analysis.json", "report.md")}}
    (OUT / "manifest.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
