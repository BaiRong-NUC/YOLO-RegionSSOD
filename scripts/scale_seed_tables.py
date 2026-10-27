"""Validate and format the two-seed threshold repeat for the paper generator."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from run_p1_scale_seed_pair import ARMS, METRICS, OUT, check_epochs, digest, paths


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired_statistics(per_seed):
    aggregate = {arm: {key: {
        "mean_percent": statistics.mean(pair[arm][key] * 100 for pair in per_seed.values()),
        "sample_std_pp": statistics.stdev(pair[arm][key] * 100 for pair in per_seed.values())}
        for key in METRICS} for arm in ARMS}
    delta = {key: {
        "mean_pp": statistics.mean(pair["delta_global_minus_scale_aware_pp"][key] for pair in per_seed.values()),
        "sample_std_pp": statistics.stdev(pair["delta_global_minus_scale_aware_pp"][key] for pair in per_seed.values())}
        for key in METRICS}
    return aggregate, delta


def load_pair(hashes, *, include_independent=True):
    comparison = read(OUT / "comparison.json")
    audit = read(OUT / "acceptance_audit.json")
    if not audit["acceptance_checks_passed"] or audit["parameter_differences"]:
        raise ValueError("paired seed acceptance did not pass")
    if comparison["split_seed"] != 0 or comparison["student_seeds"] != [0, 1] or comparison["epochs"] != 30:
        raise ValueError("unexpected split, seeds, or schedule")
    for filename, expected in audit["sha256"].items():
        if digest(filename) != expected:
            raise ValueError(f"paired result artifact changed: {filename}")
        hashes[filename] = expected
    root = OUT.parents[1]
    old = read(root / "outputs/p1_global_t050_pilot30_s0/comparison.json")
    if comparison["per_seed"]["0"] != {
            "scale_aware": old["reference"]["metrics"], "global_t050": old["global_t050"]["metrics"],
            "delta_global_minus_scale_aware_pp": old["delta_global_minus_scale_aware_pp"]}:
        raise ValueError("seed0 values differ from frozen source")
    for arm in ARMS:
        run, val_dir, recall_dir = paths(arm)
        check_epochs(run)
        val, recall = read(val_dir / "metrics.json"), read(recall_dir / "metrics.json")
        raw = {"mAP50": val["results"]["metrics/mAP50(B)"],
               "mAP50_95": val["results"]["metrics/mAP50-95(B)"],
               **{f"{s}_recall": recall["per_size"][s]["recall"] for s in ("small", "medium", "large")}}
        if raw != comparison["per_seed"]["1"][arm] or raw != comparison["seed1"][arm]["metrics"]:
            raise ValueError("seed1 comparison differs from raw evaluations")
    for pair in comparison["per_seed"].values():
        expected = {k: 100 * (pair["global_t050"][k] - pair["scale_aware"][k]) for k in METRICS}
        if expected != pair["delta_global_minus_scale_aware_pp"]:
            raise ValueError("paired differences are incorrect")
        if not all(math.isfinite(v) for arm in ARMS for v in pair[arm].values()):
            raise ValueError("non-finite repeat result")
    aggregate, delta = paired_statistics(comparison["per_seed"])
    if aggregate != comparison["aggregate"] or delta != comparison["paired_delta_global_minus_scale_aware_pp"]:
        raise ValueError("two-seed statistics differ from raw pairs")
    for name in ("comparison.json", "acceptance_audit.json", "preflight.json"):
        path = OUT / name
        hashes[str(path)] = digest(path)
    independent_path = OUT / "independent_audit.json"
    if include_independent and independent_path.exists():
        independent = read(independent_path)
        if not independent["acceptance_checks_passed"] or not all(independent["checks"].values()):
            raise ValueError("independent paired acceptance did not pass")
        for filename, expected in independent["sha256"].items():
            if digest(filename) != expected:
                raise ValueError(f"independent audit input changed: {filename}")
            hashes[filename] = expected
        hashes[str(independent_path)] = digest(independent_path)
    return comparison


def table_sections(pair, table):
    rows = []
    for seed, values in pair["per_seed"].items():
        for key in METRICS:
            rows.append([seed, key, f"{100*values['scale_aware'][key]:.4f}",
                         f"{100*values['global_t050'][key]:.4f}",
                         f"{values['delta_global_minus_scale_aware_pp'][key]:+.4f}"])
    summaries = []
    for key in METRICS:
        cells = []
        for arm in ARMS:
            stats = pair["aggregate"][arm][key]
            cells.append(f"{stats['mean_percent']:.4f} ± {stats['sample_std_pp']:.4f}")
        delta = pair["paired_delta_global_minus_scale_aware_pp"][key]
        summaries.append([key, *cells, f"{delta['mean_pp']:+.4f} ± {delta['sample_std_pp']:.4f}"])
    return ["## 表 4-S3 固定 split 的尺度阈值配对 seed 复验",
            table(["学生 seed", "指标", "尺度分层 / %", "统一 0.50 / %", "统一−尺度 / pp"], rows),
            table(["指标", "尺度分层均值 ± SD / %", "统一阈值均值 ± SD / %", "配对差均值 ± SD / pp"], summaries),
            "注：固定 split seed0、同一冻结教师和两套伪标签，仅学生训练种子为 0、1。"
            "每组独立训练 30 epoch，完整 548 图 val960；固定操作点 recall 口径同表 4-2。"
            "SD 为两学生种子的样本标准差（ddof=1）；配对差的 SD 由两组种子内差值直接计算。"
            "这是同一 split/教师下的重复，不是多 split、独立教师或统计显著性证据，亦不改变原 Gate 3。"]
