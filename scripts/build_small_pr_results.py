"""Render the small-object evaluation tables from its accepted cached audit."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import numpy as np

from evaluate_paper1_small_pr import OUT, ROOT, KEYS, LEVELS, digest, read, write, load_predictions, verify_hashes
from regionssod.evaluation.precision_recall import precision_constrained_point

TARGET = ROOT / "docs/paper/small_pr_results.md"


def pct(value):
    return "NA" if value is None else f"{value * 100:.4f}"


def signed(value):
    return "NA" if value is None else f"{value:+.4f}"


def mean_sd(values):
    if any(value is None for value in values):
        return "NA"
    return f"{np.mean(values):+.4f} ± {np.std(values, ddof=1):.4f}"


def render():
    audit = read(OUT / "independent_audit.json")
    if audit.get("accepted") is not True or audit.get("cached_replay_exact") is not True:
        raise ValueError("accepted exact cached replay is required before writing results")
    result = read(OUT / "comparison.json")
    models, pairs = result["models"], result["pairs"]
    lines = ["# 正式100轮小目标 AP、完整PR与等精度召回补评估", "",
        "日期：2026-11-02。四个冻结权重的548原图低置信度推理、缓存分析及来源绑定审计已完成；本轮零训练。", "",
        "## 1. 本轮发现", "",
        "本轮能加强的是**适用精度区间内的小目标覆盖证据**。COCO-style AP_small的两个学生seed方向不一致；",
        "不能把固定阈值召回增益扩写为小目标AP稳定提高。全尺寸precision约束下，IoU0.50时的",
        "收益主要位于中等精度区间，高精度端存在退化；这些负向结果与正向结果一起保留。", "",
        "这是相同split0、固定教师、两个学生seed上的回顾性验证分析，未产生新训练方法或外部泛化证明。",
        "旧30轮结果、原Gate3/4及中断恢复记录保持原判定；本次COCO-style AP不冒充预登记官方指标。", "",
        "## 2. 评估范围和两套口径", "",
        "协议见[固定补评估协议](small_pr_protocol.md)。Ultralytics 8.4.172、FP32、960、batch8、",
        "conf≥0.001、NMS IoU0.70、max_det500、无TTA；四臂重新计算的Ultralytics mAP50和mAP50–95",
        "均与原正式记录精确一致。所有权重在推理前后保持原哈希。", "",
        "- AP采用pycocotools 2.0.11，IoU0.50:0.05:0.95、101点插值、10类宏平均，取maxDets500。",
        "  GT由现有YOLO标签转换为原图像素框；没有恢复VisDrone原始忽略区域，因此**不是VisDrone官方AP**。",
        "  COCO面积忽略和旧诊断全尺寸联合匹配分别实现；small面积边界按各自工具保留。",
        "- 等精度召回采用全尺寸、类别一致、置信度降序的一对一匹配，按被匹配GT面积统计small<32²。",
        "  precision=全部TP/全部预测，是micro全尺寸precision，不能解释为small专用precision或COCO类别宏平均。",
        "- R_small@P≥p取所有实际分数阈值中满足precision约束的最大small recall；同分框整组纳入。",
        "  实际precision可略高于约束，阈值在同一验证集估计，不是独立部署政策或风险保证。", "",
        f"原图GT共38759个；诊断small/medium/large为{result['gt_counts_diagnostic_strict']}；",
        f"COCO默认包含边界的计数为{result['gt_counts_coco_inclusive']}，本批GT两者计数恰好一致。", "",
        "## 3. COCO-style面积分组结果", "",
        "以下均为百分数；每个模型仍只对应同一548图，重复seed不增加独立验证样本数。", "",
        "| 学生 | AP_all | AP_small | AP50_small | AP75_small |",
        "|---|---:|---:|---:|---:|"]
    for key in KEYS:
        metrics = models[key]["coco"]["metrics"]
        lines.append(f"| {key} | " + " | ".join(pct(metrics[name]) for name in
                     ("AP_all", "AP_small", "AP50_small", "AP75_small")) + " |")
    lines += ["", "| P1−B2 / pp | seed0 | seed1 | 配对均值 ± 样本SD |", "|---|---:|---:|---:|"]
    for name in ("AP_all", "AP_small", "AP50_small", "AP75_small", "AP_medium", "AP_large"):
        values = [pairs[str(seed)]["delta_coco_pp"][name] for seed in (0, 1)]
        lines.append(f"| {name} | {signed(values[0])} | {signed(values[1])} | {mean_sd(values)} |")
    lines += ["", "AP_small的seed方向不一致，平均正增幅不能证明稳定改善。AP_large是探索性次要结果，",
        "其GT仅1067个，类别支持也不同；不因它更大而事后替换原定小目标主问题。", "",
        "## 4. 全尺寸precision约束下的小目标recall", "",
        "表中每个seed单元格为B2 / P1 / 差值，前两项为%，差值和均值为pp。",
        "两臂各用满足同一precision下限的实际阈值，不对PR曲线做虚构的精确插值。", ""]
    for iou in ("0.5", "0.75"):
        lines += [f"### 匹配IoU={iou}", "", "| 全尺寸precision下限 | seed0：B2 / P1 / Δ | seed1：B2 / P1 / Δ | Δ均值 ± 样本SD / pp |",
                  "|---|---:|---:|---:|"]
        for level in LEVELS:
            values, cells = [], []
            for seed in (0, 1):
                b2 = models[f"b2_s{seed}"]["precision_constraints"][iou][str(level)]
                p1 = models[f"p1_s{seed}"]["precision_constraints"][iou][str(level)]
                delta = pairs[str(seed)]["delta_small_recall_at_precision_pp"][iou][str(level)]
                values.append(delta)
                cells.append("NA" if b2 is None or p1 is None else
                             f"{pct(b2['small_recall'])} / {pct(p1['small_recall'])} / {signed(delta)}")
            lines.append(f"| {level:.0%} | {cells[0]} | {cells[1]} | {mean_sd(values)} |")
        lines.append("")
    lines += ["IoU0.50下50%、60%、70%精度约束的small recall在两个seed均增加，80%方向不一致，",
        "90%均下降；IoU0.75下50%、60%两seed同向增加，其余档位并不稳定。这是条件性前沿，",
        "不能声称全精度区间领先、误检减少、定位机制已改善或校准机制已有效。", "",
        "## 5. 与B2原conf0.25精度对齐的补充锚点", "",
        "此处B2的conf0.25也来自**本轮validator缓存**；与旧predict路径的计数差异见第7节。", "",
        "| seed | B2/P1实际precision / % | B2/P1 small recall / % | Δrecall / pp | small净增 | P1阈值 |",
        "|---|---:|---:|---:|---:|---:|"]
    for seed in (0, 1):
        anchor = pairs[str(seed)]["b2_conf025_precision_anchor"]
        a, b = anchor["b2"], anchor["p1"]
        lines.append(f"| {seed} | {pct(a['precision'])} / {pct(b['precision'])} | "
                     f"{pct(a['small_recall'])} / {pct(b['small_recall'])} | {signed(anchor['delta_small_recall_pp'])} | "
                     f"{b['small_tp'] - a['small_tp']:+d} | {b['threshold']:.6f} |")
    lines += ["", "这表明在B2约72%的全尺寸precision附近，P1仍保留部分small覆盖收益。",
        "但此表不把两条推理路径的差值相减来估计“多少收益来自分数平移”，也不把调出的阈值用于部署。", "",
        "## 6. 逐类别AP_small差值", "", "| 类别 | seed0 P1−B2 / pp | seed1 P1−B2 / pp |", "|---|---:|---:|"]
    for name in pairs["0"]["delta_per_class_AP_small_pp"]:
        lines.append(f"| {name} | {signed(pairs['0']['delta_per_class_AP_small_pp'][name])} | "
                     f"{signed(pairs['1']['delta_per_class_AP_small_pp'][name])} |")
    lines += ["", "bicycle和truck的AP_small在两seed均下降；AP_small与旧逐类全尺寸AP是不同指标，",
        "不能互相替代，也不把正向类别挑出后称为全部类别改善。", "",
        "## 7. 候选截断与旧固定点兼容性", "",
        "| 模型 | 保存预测数 | 达到每图500框的图数/548 | 新validator−旧predict small匹配 | medium | large |",
        "|---|---:|---:|---:|---:|---:|"]
    for key in KEYS:
        row = models[key]
        delta = row["old_predict_recall_count_deltas"]
        lines.append(f"| {key} | {row['prediction_count']} | {row['images_capped_at_500']} | "
                     f"{delta['small']:+d} | {delta['medium']:+d} | {delta['large']:+d} |")
    lines += ["", "相同imgsz与batch不代表两条路径有相同像素输入。已核对安装版本源码：validator的数据集",
        "按宽高比分批并使用val pad=0.5；predict的LetterBox取决于每批原图是否同形。二者的分组/填充",
        "机制不同，与本次计数差异相容，但本轮未逐因素隔离其贡献。旧结果保持原值，",
        "新等精度比较只在统一validator路径内配对；原Ultralytics mAP精确重放另行验证。", "",
        "大量图像达到500框，完整曲线仅指所保存候选范围内的阈值扫描；不覆盖低于0.001、",
        "被NMS移除或超过500框的预测。本轮保持原cap做公平比较，不因看到结果而提高cap。", "",
        "## 8. 文献与可证伪的新方向", "",
        "[文献侧报告与Zotero清单](small_pr_literature.md)记录5篇实际采用论文、原始链接和确切阅读范围。",
        "本轮新增STOD，其余4篇查重复用；目标Research / YOLO-RegionSSOD，collection BMQTT9Y7。",
        "4篇有PDF，STOD为摘要/元数据，不能把它当作已全文核验的创新边界。", "",
        "以下为AI提出的研究假设，不是已实现/验证的新方法：", "",
        "1. **按精度区间约束监督覆盖。** 现有收益集中在中等precision区间；未来可研究有限标注区域",
        "   提供的可靠性如何迁移到隐藏区域，同时限制高精度端的错误接受。单调全局分数重标定",
        "   不改善固定候选的类别内排序/AP；按尺度校准本身亦已有先例。新机制必须改变可归因的决策，",
        "   并在独立校准/评测划分上检验，不能在本val拟合后再称外部验证。",
        "2. **区分存在证据与几何证据。** AP_small/严格IoU未稳定改善，提示进一步分层检查弱分框",
        "   的类别、重复与边界误差；不能据此诊断某个回归模块故障。STOD、Soft Teacher与UTv2",
        "   已有质量加权或解耦先例，单纯增加IoU/不确定性头不是已确立的新颖性。",
        "3. **给ignore分配空间预算。** 可用既有教师候选审计新增独立隐藏目标保护与屏蔽面积，",
        "   避免重复候选消耗保护预算；隐藏GT仅作诊断。最终学生val预测不能重建训练时负梯度，",
        "   该设想需要另行证据及配对训练，本轮未实现。", "",
        "前两项的部分必要条件由本轮PR回答；更具体的分层定位、风险迁移及空间预算机制均未验证。",
        "既有置信度校准、回归加权等探索及负向结果继续保留，不以本轮假设追认其成功。", "",
        "## 9. 来源、验收和复现", "",
        "正式输出：`outputs/paper1_small_pr_20261102/bound_v2/`。四臂低分预测、inference回执、",
        "manifest、ground_truth、comparison、CSV、NPZ、PNG/PDF/SVG、退出码与独立缓存审计分别保留。",
        f"缓存审计核对{audit['input_hashes']}项输入哈希、548图×4臂，COCO/PR/等精度数字完整重放精确一致。",
        "这验证缓存评估可重放，不是重新训练的逐位可复现性。首次工程预检失败和第一版未绑定产物",
        "保留在父目录；bound_v2重建绑定时没有改变任何推理/指标条件，未覆盖旧训练或评估。", "",
        "主要图表：同输出目录的`small_pr_comparison.{png,pdf,svg}`与`small_recall_precision_delta.{png,pdf,svg}`。",
        "后者突出两seed在各precision区间的差值，零线和负向区间均保留；生成图不进入Git。", "",
        "各模型`coco_class_pr.npz`保存全部类别×面积×IoU的101点precision/score及max recall；",
        "`small_class_pr_iou50.{png,pdf,svg}`给出10类小目标PR。类别轴按实际COCO category_id保存，",
        "并重算核验其宏平均与已验收曲线一致。NPZ可用于IoU0.75及其他面积范围的逐类重绘。", "",
        "在仓库根目录、yolo环境，只读复核与文档数字核对：", "", "```bash",
        "python scripts/evaluate_paper1_small_pr.py --audit",
        "python scripts/build_small_pr_results.py --check", "```", "",
        "若需要重建分析/图表，`--analyze`仅重放已有缓存；文档和差值图由",
        "`python scripts/build_small_pr_results.py`生成。已有预测时不要再次执行`--predict`。",
        "本页由脚本从冻结JSON生成；改变结果或解释需同步生成器，`--check`会检测不一致。", ""]
    return "\n".join(lines)


def delta_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, iou in zip(axes, ("0.5", "0.75")):
        for seed, color in ((0, "#0072B2"), (1, "#D55E00")):
            a = dict(np.load(OUT / f"b2_s{seed}/micro_iou{iou}.npz"))
            b = dict(np.load(OUT / f"p1_s{seed}/micro_iou{iou}.npz"))
            grid = np.linspace(.5, .95, 91)
            differences = []
            for level in grid:
                before, after = precision_constrained_point(a, level), precision_constrained_point(b, level)
                differences.append(np.nan if before is None or after is None else
                                   100 * (after["small_recall"] - before["small_recall"]))
            ax.plot(grid * 100, differences, label=f"Student seed{seed}", color=color)
        ax.axhline(0, color="black", lw=.8)
        ax.set_title(f"Matching IoU {iou}")
        ax.set_xlabel("Required all-size precision (%)")
        ax.set_ylabel("P1 - B2 small recall (percentage points)")
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Recall benefit depends on the precision constraint\nFrozen students; empirical validation frontiers, not deployment guarantees", fontsize=11)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"small_recall_precision_delta.{ext}", dpi=180)
    plt.close(fig)


def export_class_pr():
    """Export the entire COCO tensor from the same accepted native predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dataset = read(OUT / "ground_truth.json")
    for key in KEYS:
        path = OUT / key / "coco_class_pr.npz"
        if path.exists():
            continue
        print(f"Exporting complete class PR tensor: {key}", flush=True)
        _, detections = load_predictions(key, dataset)
        with redirect_stdout(io.StringIO()):
            truth = COCO()
            truth.dataset = dataset
            truth.createIndex()
            evaluator = COCOeval(truth, truth.loadRes(detections), "bbox")
            evaluator.params.maxDets = [1, 100, 500]
            evaluator.evaluate()
            evaluator.accumulate()
        values = evaluator.eval["precision"][:, :, :, :, -1]
        expected = dict(np.load(OUT / key / "coco_pr.npz"))
        for area_index, area in enumerate(evaluator.params.areaRngLbl):
            for t, tag in ((0, "50"), (5, "75")):
                current = values[t, :, :, area_index]
                support = (current >= 0).sum(1)
                mean = np.divide(np.maximum(current, 0).sum(1), support,
                                 out=np.zeros(len(support)), where=support > 0)
                np.testing.assert_array_equal(mean, expected[f"{area}_iou{tag}"])
        np.savez_compressed(path, precision=values, scores=evaluator.eval["scores"][:, :, :, :, -1],
            max_recall=evaluator.eval["recall"][:, :, :, -1], category_ids=evaluator.params.catIds,
            area_labels=evaluator.params.areaRngLbl, iou_grid=evaluator.params.iouThrs,
            recall_grid=evaluator.params.recThrs, max_det=500)
    categories = sorted(dataset["categories"], key=lambda category: category["id"])
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for index, category in enumerate(categories):
        ax = axes.flat[index]
        for seed, style in ((0, "-"), (1, "--")):
            for arm, color in (("b2", "#0072B2"), ("p1", "#D55E00")):
                values = np.load(OUT / f"{arm}_s{seed}/coco_class_pr.npz")
                ids = values["category_ids"].tolist()
                precision = values["precision"][0, :, ids.index(category["id"]), 1]
                precision = np.where(precision < 0, np.nan, precision)
                ax.plot(values["recall_grid"], precision, style, color=color, lw=1,
                        label=f"{arm.upper()} seed{seed}")
        ax.set_title(category["name"], fontsize=10)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
        ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Small-object class PR at IoU0.50 | COCO-style on YOLO labels, maxDet500", fontsize=12)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"small_class_pr_iou50.{ext}", dpi=180)
    plt.close(fig)


def workflow_reports():
    for name in ("predict", "analyze", "audit"):
        if read(OUT / f"{name}_exit.json")["exit_code"] != 0:
            raise ValueError(f"{name} did not finish with real exit0")
    text = """## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-02
- Verification Status: ANALYZED
- Version Label: small_pr_validation_v1
- Verification scope: descriptive interpretation; accepted exact cached evaluation replay is recorded separately; training reproducibility not tested

## Validation Report

Source: bound_v2/comparison.json, manifest.json, capture_complete.json and independent_audit.json.
四个冻结模型的推理/分析/审计包装均真实exit0。缓存重算精确一致，四臂原Ultralytics
mAP精确重放；修正绑定后再次推理的原始预测与第一版字节一致。无优化器步骤。

### Statistical findings

AP_small两seed方向不同，不能称为稳定提高。IoU0.50下50/60/70%全尺寸precision约束
small召回两seed同向增加，90%两seed下降；所有档位、类别与严格IoU结果均保留。
配对差均值±样本SD只描述两个学生seed，不是CI/SE或显著性。没有进行假设检验。
经验阈值在同一val上估计，不具备留出部署性能或风险覆盖保证。

### Fallacy scan — 11/11 checked

| Pattern | Scope-specific finding |
|---|---|
| Simpson's paradox | Checked: class/seed and IoU/precision regimes differ; aggregate mean never substitutes for their directions. |
| Ecological fallacy | Checked: image/target/seed units separated; no claim every image or target benefits. |
| Berkson's paradox | Checked: all548 images and all4 frozen models retained; existing YOLO label exclusions and candidate filtering are explicit selection boundaries. |
| Collider bias | Checked: conditioning on empirical precision is an operating-frontier diagnostic, not an adjusted causal estimate. |
| Base-rate neglect | Checked: fullGT/size denominators stated; all-size microprecision is not small-only precision; category-macro AP reported separately. |
| Regression to mean | Checked: no new checkpoint selection; pre-existing best checkpoints and repeatedly used val remain a selection/generalization limitation. |
| Survivorship bias | Checked: expected4 arms complete; original training resumes, engineering preflight failure and first unbound evaluation preserved, none silently omitted. |
| Look-elsewhere effect | Checked: all5 fixedprecision levels, bothIoUs and negative classes shown; exploratory largeAP cannot replace small-object primary question. |
| Garden of forking paths | Checked: inference/metrics/grid fixed before results; code fixes changed provenance/category naming, not scientific settings; sameval thresholds remain retrospective. |
| Correlation vs causation | Checked: AP/PR patterns do not identify calibration, ignore or regression as the cause; literature hypotheses remain untested. |
| Reverse causality | Checked: no causal direction claimed from final confidence/localization associations; final student boxes do not recover training-time teacher/gradient states. |

### Reproducibility and remaining limits

Cached evaluation replay: accepted, exact, input/output hashes checked. Training rerun: NOT TESTED.
COCO-style on currentYOLO labels is not officialVisDrone AP. Candidate confidence floor/NMS/500cap,
validator-vs-predict preprocessing, fixedsplit/teacher, two seeds and recovery history remain disclosed.
Human-subject tests are not applicable. No new training, calibration fit, deployment policy or independent test was run.
"""
    (OUT / "validation_report.md").write_text(text, encoding="utf-8")
    seconds = sum(read(OUT / key / "inference.json")["elapsed_seconds"] for key in KEYS)
    text = f"""## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-11-02
- Verification Status: UNVERIFIED
- Version Label: small_pr_execution_v1

## Experiment Result

- ID: paper1_small_pr_20261102_bound_v2
- Type: analysis (frozen-model inference and cached evaluation)
- Status: completed
- Working Directory: D:/Python/YOLO-RegionSSOD
- Commands: python scripts/evaluate_paper1_small_pr.py --predict / --analyze / --audit
- Exit Codes: 0 / 0 / 0
- Bound-v2 inference elapsed total: {seconds:.3f} seconds; not a training or deployment speed benchmark
- Population: 548 full images per frozen arm, B2/P1 studentseed0/1; one validation population
- Training steps: 0
- Outputs: manifest/GT/4 raw prediction caches/COCO+micro curves/fixedprecision CSV/comparison and audit
- Anomalies: initial directory-loader preflight failure beforeGPU; initial unbound results retained; provenance and category-axis fixes tested before bound_v2 capture
- Validation handoff: validation_report.md and independent_audit.json; this execution carrier alone is not scientific verification
"""
    (OUT / "experiment_result.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    text = render()
    if options.check:
        if TARGET.read_text(encoding="utf-8") != text:
            raise ValueError("small-object result documentation differs from accepted JSON")
        verify_hashes(read(OUT / "documentation_sha256.json"))
        print("Small-object AP, all precision constraints, anchor, per-class and compatibility tables match accepted JSON.")
    else:
        TARGET.write_text(text, encoding="utf-8")
        delta_plot()
        export_class_pr()
        workflow_reports()
        # Audit timestamps may refresh without changing measurements. Freeze
        # the comparison/manifest and reports rather than that mutable nonce.
        paths = [TARGET, Path(__file__).resolve(), OUT / "comparison.json", OUT / "manifest.json",
                 OUT / "validation_report.md", OUT / "experiment_result.md"]
        paths += [OUT / f"small_recall_precision_delta.{ext}" for ext in ("png", "pdf", "svg")]
        paths += [OUT / key / "coco_class_pr.npz" for key in KEYS]
        paths += [OUT / f"small_class_pr_iou50.{ext}" for ext in ("png", "pdf", "svg")]
        write(OUT / "documentation_sha256.json", {str(path): digest(path) for path in paths})
        print(TARGET)


if __name__ == "__main__":
    main()
