"""Validate and format two-seed P1-versus-B2 results for the manuscript."""
from __future__ import annotations

import math
import statistics

from run_b2_seed1 import OUT, METRICS
from run_p1_scale_seed_pair import digest, read


def load_main_pair(hashes, *, include_independent=True):
    audit = read(OUT / "acceptance_audit.json")
    pair = read(OUT / "comparison.json")
    if not audit["acceptance_checks_passed"] or audit["parameter_differences"]:
        raise ValueError("B2 seed1 acceptance did not pass")
    if pair["split_seed"] != 0 or pair["student_seeds"] != [0, 1] or pair["epochs"] != 30:
        raise ValueError("unexpected main pair scope")
    for filename, expected in audit["sha256"].items():
        if digest(filename) != expected:
            raise ValueError(f"main pair artifact changed: {filename}")
        hashes[filename] = expected
    for seed, values in pair["per_seed"].items():
        for method in ("B2", "P1"):
            if values[method] != pair["runs"][f"{method}_s{seed}"]["metrics"]:
                raise ValueError("main pair summary differs from per-run values")
        for key in METRICS:
            expected = 100 * (values["P1"][key] - values["B2"][key])
            if not math.isfinite(expected) or expected != values["delta_p1_minus_b2_pp"][key]:
                raise ValueError("main paired delta differs")
    for key in METRICS:
        data = [p["delta_p1_minus_b2_pp"][key] for p in pair["per_seed"].values()]
        if pair["paired_delta_p1_minus_b2_pp"][key] != {"mean_pp": statistics.mean(data), "sample_std_pp": statistics.stdev(data)}:
            raise ValueError("main pair mean or SD differs")
    for name in ("comparison.json", "acceptance_audit.json", "preflight.json"):
        hashes[str(OUT / name)] = digest(OUT / name)
    independent_path = OUT / "independent_audit.json"
    if include_independent and independent_path.exists():
        independent = read(independent_path)
        if not independent["acceptance_checks_passed"] or not all(independent["checks"].values()):
            raise ValueError("independent main-pair acceptance failed")
        for filename, expected in independent["sha256"].items():
            if digest(filename) != expected:
                raise ValueError(f"independent main-pair input changed: {filename}")
            hashes[filename] = expected
        hashes[str(independent_path)] = digest(independent_path)
    return pair


def main_pair_sections(pair, table):
    rows, summaries = [], []
    for seed, values in pair["per_seed"].items():
        for key in METRICS:
            rows.append([seed, key, f"{100*values['B2'][key]:.4f}", f"{100*values['P1'][key]:.4f}",
                         f"{values['delta_p1_minus_b2_pp'][key]:+.4f}"])
    for key in METRICS:
        cells = []
        for method in ("B2", "P1"):
            data = [100*p[method][key] for p in pair["per_seed"].values()]
            expected = {"mean_percent": statistics.mean(data), "sample_std_pp": statistics.stdev(data)}
            if pair["aggregate"][method][key] != expected:
                raise ValueError("per-method main statistics differ")
            cells.append(f"{expected['mean_percent']:.4f} ± {expected['sample_std_pp']:.4f}")
        delta = pair["paired_delta_p1_minus_b2_pp"][key]
        summaries.append([key, *cells, f"{delta['mean_pp']:+.4f} ± {delta['sample_std_pp']:.4f}"])
    return ["## 表 4-S4 固定 split 下 B2/P1 的配对学生 seed 复验",
            table(["学生 seed", "指标", "B2 / %", "P1 / %", "P1−B2 / pp"], rows),
            table(["指标", "B2 均值 ± SD / %", "P1 均值 ± SD / %", "配对差均值 ± SD / pp"], summaries),
            "注：固定 split seed0 和同一 B1 教师，学生 seed0/1，各自独立 30 epoch；完整 548 图 val960。"
            "SD 为两学生 seed 的样本标准差（ddof=1），配对差先在各 seed 内计算。"
            "B2 使用原全局 0.50 伪标签和标准损失，P1 使用尺度阈值、软权重与 ignore；相同 epoch 不代表相同数据或计算量。"
            "两个学生 seed 不是多 split、独立教师或显著性检验，固定操作点 recall 不是 AP_small，原 Gate 3 判定不变。"]
