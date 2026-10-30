"""Build manuscript tables from frozen evaluations; --check verifies saved tables."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from scale_seed_tables import OUT as SEED_PAIR_OUT, load_pair, table_sections
from main_seed_tables import OUT as MAIN_PAIR_OUT, load_main_pair, main_pair_sections

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def pct(value):
    return f"{value * 100:.3f}"


def no_soft_sections(hashes):
    evidence = ROOT / "outputs/p1_no_soft_weights_pilot30_s0"
    audit = load(evidence / "independent_audit.json")
    assert audit["acceptance_checks_passed"] and all(audit["checks"].values())
    for filename, expected in audit["sha256"].items():
        assert digest(filename) == expected, filename
        hashes[filename] = expected
    hashes[str(evidence / "independent_audit.json")] = digest(evidence / "independent_audit.json")
    comparison = load(evidence / "comparison.json")
    assert comparison["epochs"] == 30 and comparison["train_tiles"] == 11310
    reference = comparison["reference"]["metrics"]
    ablation = comparison["ablation"]["metrics"]
    delta = comparison["delta_no_soft_weights_minus_full_p1_percentage_points"]
    matches = {}
    for method, folder in (("reference", "p1_pilot30_s0"),
                           ("ablation", "p1_no_soft_weights_pilot30_s0")):
        recall_path = ROOT / f"runs/size_recall/{folder}/metrics.json"
        recall = load(recall_path)
        assert recall["weights_sha256"] == comparison[method]["weights_sha256"]
        matches[method] = recall["per_size"]
        hashes[str(recall_path)] = digest(recall_path)
    count_note = "，".join(
        f"{size}匹配数{matches['reference'][size]['matched']}→"
        f"{matches['ablation'][size]['matched']}（"
        f"{matches['ablation'][size]['matched']-matches['reference'][size]['matched']:+d}）"
        for size in ("small", "medium", "large"))
    rows = []
    for key in reference:
        expected = 100 * (ablation[key] - reference[key])
        assert math.isclose(expected, delta[key], abs_tol=1e-12)
        assert math.isclose(expected, audit["delta_no_soft_minus_full_pp"][key], abs_tol=1e-12)
        rows.append(["全体" if not key.endswith("_recall") else "尺寸", key,
                     pct(reference[key]), pct(ablation[key]), f"{expected:+.3f}"])
    reference_val = load(ROOT / "runs/val/partial10_p1_pilot30_s0_val960/metrics.json")
    ablation_val = load(ROOT / "runs/val/partial10_p1_no_soft_weights_pilot30_s0_val960/metrics.json")
    for row in audit["per_class"]:
        cls = row["class_id"]
        before, after = reference_val["class_maps"][cls], ablation_val["class_maps"][cls]
        expected = 100 * (after - before)
        assert math.isclose(expected, row["delta_no_soft_minus_full_pp"], abs_tol=1e-12)
        rows.append(["类别", row["name"], pct(before), pct(after), f"{expected:+.3f}"])
    sections = ["## 表 4-S5 分类软权重的全量单因素消融",
                table(["范围", "指标", "完整 P1 / %", "去软权重 / %", "去软−P1 / pp"], rows),
                "注：固定10%面积split seed0、同一教师和11310个训练tile，学生seed0、30epoch；"
                "仅关闭教师置信度分类软权重，保留尺度阈值和ignore，完整548图val960。"
                f"{count_note}。"
                "分尺寸recall操作点同表4-2，不是AP_small。"
                "去软权重降低mAP50-95并提高small recall，体现取舍；单seed不证明稳定效应，原Gate3不变。"]
    return comparison, sections


def main(check=False):
    gate2 = load(ROOT / "outputs/gate2_20261027/comparison.json")
    gate3 = load(ROOT / "outputs/gate3_20261027/comparison.json")
    ablation = load(ROOT / "outputs/p1_global_t050_pilot30_s0/comparison.json")
    ablation_acceptance = load(ROOT / "outputs/p1_global_t050_pilot30_s0/acceptance_audit.json")
    assert ablation_acceptance["acceptance_checks_passed"] is True
    assert ablation_acceptance["parameter_differences"] == {}
    assert ablation["reference"]["weights_sha256"] == gate3["best_sha256"]
    models = [(r["method"], ROOT / "runs" / r["run"], Path(r["validation"]), r["best_sha256"])
              for r in gate2["runs"]]
    models.append(("P1", Path(gate3["run"]), Path(gate3["p1_validation"]), gate3["best_sha256"]))
    metrics, recalls, times, hashes, actual_args = {}, {}, {}, {}, {}
    for method, run, evaluation, expected in models:
        weights = run / "weights/best.pt"
        assert digest(weights) == expected, weights
        metrics[method] = load(evaluation)
        assert Path(metrics[method]["weights"]).resolve() == weights.resolve()
        assert metrics[method]["imgsz"] == 960
        recalls[method] = load(ROOT / f"runs/size_recall/{method.lower()}_pilot30_s0/metrics.json")
        assert recalls[method]["weights_sha256"] == expected
        assert recalls[method]["images"] == 548 and recalls[method]["imgsz"] == 960
        assert recalls[method]["conf"] == .25 and recalls[method]["match_iou"] == .5
        assert recalls[method]["nms_iou"] == .7 and recalls[method]["max_det"] == 500
        assert [recalls[method]["per_size"][size]["gt"] for size in ("small", "medium", "large")] == [26577, 11115, 1067]
        csv_path = run / "results.csv"
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
        assert [int(row["epoch"]) for row in rows] == list(range(1, 31))
        assert all(math.isfinite(float(value)) for row in rows for value in row.values())
        times[method] = float(rows[-1]["time"])
        actual_args[method] = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        assert actual_args[method]["seed"] == 0 and actual_args[method]["epochs"] == 30
        assert actual_args[method]["imgsz"] == 640 and actual_args[method]["batch"] == 8
        for path in [weights, evaluation, csv_path, run / "args.yaml",
                     ROOT / f"runs/size_recall/{method.lower()}_pilot30_s0/metrics.json"]:
            hashes[str(path)] = digest(path)
    for method in ("B2", "P1"):
        assert all(actual_args[method].get(key) == value for key,value in actual_args["B1"].items()
                   if key not in {"data", "name", "save_dir"}), "student hyperparameters differ"
    ablation_run = Path(ablation["global_t050"]["run"])
    ablation_weights = ablation_run / "weights/best.pt"
    ablation_eval_path = ROOT / "runs/val/partial10_p1_global_t050_pilot30_s0_val960/metrics.json"
    ablation_recall_path = ROOT / "runs/size_recall/p1_global_t050_pilot30_s0/metrics.json"
    ablation_results_path = ablation_run / "results.csv"
    ablation_args_path = ablation_run / "args.yaml"
    ablation_experiment_path = ablation_run / "experiment.json"
    ablation_eval = load(ablation_eval_path)
    ablation_recall = load(ablation_recall_path)
    assert digest(ablation_weights) == ablation["global_t050"]["weights_sha256"]
    assert Path(ablation_eval["weights"]).resolve() == ablation_weights.resolve()
    assert ablation_eval["imgsz"] == 960
    assert ablation_recall["weights_sha256"] == digest(ablation_weights)
    assert ablation_recall["images"] == 548 and ablation_recall["imgsz"] == 960
    assert ablation_recall["conf"] == .25 and ablation_recall["match_iou"] == .5
    assert ablation_recall["nms_iou"] == .7 and ablation_recall["max_det"] == 500
    assert [ablation_recall["per_size"][size]["gt"] for size in ("small", "medium", "large")] == [26577, 11115, 1067]
    ablation_rows_log = list(csv.DictReader(ablation_results_path.open(encoding="utf-8-sig")))
    assert [int(row["epoch"]) for row in ablation_rows_log] == list(range(1, 31))
    assert all(math.isfinite(float(value)) for row in ablation_rows_log for value in row.values())
    ablation_args = yaml.safe_load(ablation_args_path.read_text(encoding="utf-8"))
    assert ablation_args["seed"] == 0 and ablation_args["epochs"] == 30
    assert ablation_args["imgsz"] == 640 and ablation_args["batch"] == 8
    assert all(ablation_args.get(key) == value for key,value in actual_args["P1"].items()
               if key not in {"data", "name", "save_dir"}), "ablation hyperparameters differ"
    p1_experiment = load(ROOT / "runs/partial10_p1_pilot30_s0/experiment.json")
    ablation_experiment = load(ablation_experiment_path)
    assert p1_experiment["train_args"] == ablation_experiment["train_args"]
    assert p1_experiment["p1"] == ablation_experiment["p1"] == {"soft_weights": True, "uncertain_ignore": True}
    assert p1_experiment["pseudo_summary"]["teacher_sha256"] == ablation_experiment["pseudo_summary"]["teacher_sha256"]
    assert ablation_experiment["pseudo_summary"]["policy"]["small_threshold"] == .5
    assert ablation_experiment["pseudo_summary"]["policy"]["medium_threshold"] == .5
    assert ablation_experiment["pseudo_summary"]["policy"]["large_threshold"] == .5
    for metric_key in ("mAP50", "mAP50_95"):
        result_key = "metrics/mAP50(B)" if metric_key == "mAP50" else "metrics/mAP50-95(B)"
        assert math.isclose(ablation["global_t050"]["metrics"][metric_key], ablation_eval["results"][result_key], abs_tol=1e-12)
        assert math.isclose(ablation["reference"]["metrics"][metric_key], metrics["P1"]["results"][result_key], abs_tol=1e-12)
    for size in ("small", "medium", "large"):
        assert math.isclose(ablation["global_t050"]["metrics"][f"{size}_recall"], ablation_recall["per_size"][size]["recall"], abs_tol=1e-12)
        assert math.isclose(ablation["reference"]["metrics"][f"{size}_recall"], recalls["P1"]["per_size"][size]["recall"], abs_tol=1e-12)
    for path in [ablation_weights, ablation_run / "weights/last.pt", ablation_eval_path,
                 ablation_recall_path, ablation_results_path, ablation_args_path,
                 ablation_experiment_path, ROOT / "outputs/p1_global_t050_pilot30_s0/comparison.json",
                 ROOT / "outputs/p1_global_t050_pilot30_s0/acceptance_audit.json"]:
        hashes[str(path)] = digest(path)
    assert digest(ablation_run / "weights/last.pt") == ablation_acceptance["weights"]["last_sha256"]
    q = {method: load(ROOT / "data/visdrone_area10/pseudo" / folder / "quality_audit.json")
         for method, folder in [("B2", "naive_t050"), ("P1", "p1_s030_m050_l050")]}
    summaries = {method: load(ROOT / "data/visdrone_area10/pseudo" / folder / "summary.json")
                 for method, folder in [("B2", "naive_t050"), ("P1", "p1_s030_m050_l050")]}
    teacher = ROOT / "runs/partial10_sup_s0/weights/best.pt"
    assert digest(teacher) == gate2["protocol"]["teacher_sha256"] == summaries["P1"]["teacher_sha256"]
    assert all(q[m]["teacher_sha256"] == digest(teacher) for m in q)
    long = load(ROOT / "runs/partial10_sup_s0/validation/metrics.json")
    assert Path(long["weights"]).resolve() == teacher.resolve()
    for method, folder in [("B2", "naive_t050"), ("P1", "p1_s030_m050_l050")]:
        for filename in ["quality_audit.json", "summary.json"]:
            path = ROOT / "data/visdrone_area10/pseudo" / folder / filename
            hashes[str(path)] = digest(path)
    for filename in ["data/visdrone_area10/audit.json", "data/visdrone_area10/summary.json",
                     "runs/partial10_sup_s0/validation/metrics.json", "runs/partial10_sup_s0/results.csv",
                     "outputs/gate2_20261027/comparison.json", "outputs/gate3_20261027/comparison.json",
                     "outputs/gate3_20261027/acceptance_audit.json"]:
        hashes[str(ROOT / filename)] = digest(ROOT / filename)
    hashes[str(teacher)] = digest(teacher)
    sections = ["# B1 B2 P1 论文结果表格", "所有检测结果来自 10% 标注面积、split seed0、训练 seed0，"
                "在 548 张完整验证原图上以 960 输入评估。表中百分数保留三位小数，差值由未舍入数据计算。"
                "本轮为单 seed 开发对照；指标采用 Ultralytics 8.4.172。来源与权重见 [证据索引](evidence.md)。"]
    keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    sections += ["## 表 4-1 相同学生训练轮数下的检测结果", table(
        ["方法", "学生 epoch", "Precision / %", "Recall / %", "mAP50 / %", "mAP50-95 / %"],
        [[m, 30] + [pct(metrics[m]["results"][k]) for k in keys] for m in metrics]),
        "注：三个学生均从同一 COCO 预训练权重独立初始化，640 训练、batch8、SGD。B2/P1 共用额外训练 100 epoch 的 B1 教师。"
        "Precision/Recall 为 Ultralytics 返回值，表 4-2 为另行固定置信度的召回，二者操作点不同。"]
    delta_rows = [[f"{a}−{b}"] + [f"{100*(metrics[a]['results'][k]-metrics[b]['results'][k]):+.3f}" for k in keys]
                  for a,b in [("B2", "B1"), ("P1", "B2"), ("P1", "B1")]]
    sections += [table(["配对差值", "ΔPrecision / pp", "ΔRecall / pp", "ΔmAP50 / pp", "ΔmAP50-95 / pp"], delta_rows),
                 "注：pp 表示百分点。以下表注中的差值也使用百分点，数量和耗时另列单位。"]
    sections += ["## 表 4-2 完整原图上固定操作点的分尺寸召回", table(
        ["方法", "Small recall / %", "Medium recall / %", "Large recall / %", "Small 匹配数 / GT数"],
        [[m] + [pct(recalls[m]["per_size"][s]["recall"]) for s in ("small", "medium", "large")]
         + [f"{recalls[m]['per_size']['small']['matched']} / 26577"] for m in metrics]),
        "注：conf=0.25，匹配 IoU≥0.50，NMS IoU=0.70，max_det=500。按原图 GT 面积分为 small<32²、"
        "32²≤medium<96²、large≥96²；各组 GT 数为 26577、11115、1067。所有尺寸联合进行类别一致、"
        "预测置信度降序的一对一匹配，再按 GT 尺寸统计。该诊断指标不是 AP_small 或官方 AR500。"]
    sections += ["## 表 4-3 未标注训练 tile 的伪标签覆盖", table(
        ["方法", "候选 tile", "入训伪标签 tile", "tile 覆盖率 / %", "正伪标签框", "Small 正框", "入训 ignore 框"],
        [[m, summaries[m]["processed_images"], summaries[m]["accepted_images"], pct(q[m]["accepted_tile_fraction"]),
          summaries[m]["accepted_boxes"], q[m]["groups"]["small"]["predictions"], summaries[m].get("ignore_boxes", "不使用")]
         for m in q]),
        "注：覆盖率=有正伪标签的 tile 数/6471。B2 阈值为 0.50；P1 的 small/medium/large 阈值为 0.30/0.50/0.50。"
        "两者均排除无正伪标签的 tile。ignore 数量只统计入训 tile，且不计入正框。"]
    quality_rows = []
    for m in q:
        for s in ("overall", "small", "medium", "large"):
            values = q[m]["overall"] if s == "overall" else q[m]["groups"][s]
            quality_rows.append([m, "全部" if s == "overall" else s,
                                 values["hidden_gt"] if s == "overall" else values["gt"],
                                 values["predictions"], pct(values["precision"]), pct(values["recall"])])
    sections += ["## 表 4-4 可见隐藏 GT 上的正伪标签质量", table(
        ["方法", "范围", "隐藏 GT 数", "正伪标签框", "伪标签 Precision / %", "伪标签 Recall / %"], quality_rows),
        "注：该表只用于离线诊断。GT 由原训练标注裁剪到未标注 tile，并沿用可见面积比例≥0.35 的规则；"
        "隐藏 GT 不进入伪标签生成或学生训练。类别一致、IoU≥0.50、IoU 降序贪心一对一匹配，"
        "匹配后分别按预测框尺寸统计 precision、按 GT 尺寸统计 recall，二者分组中的匹配数可不同。"
        "尺寸以原始 tile 的框面积计算，与表 4-2 的完整原图 GT 尺寸分别定义。"]
    class_rows=[]
    for cls in metrics["B1"]["class_indices"]:
        values=[metrics[m]["class_maps"][int(cls)] for m in metrics]
        class_rows.append([metrics["P1"]["names"][str(cls)]] + [pct(v) for v in values] + [f"{100*(values[2]-values[1]):+.3f}"])
    sections += ["## 表 4-5 完整原图验证集的逐类别 AP50-95", table(
        ["类别", "B1 / %", "B2 / %", "P1 / %", "P1−B2 / pp"], class_rows),
        "注：三者均为 30 epoch 学生。该表为 Ultralytics 的逐类别 AP50-95，不是 AP50。"]
    tiles={"B1":6471,"B2":11095,"P1":11310}
    sections += ["## 表 4-6 学生训练规模与日志累计耗时", table(
        ["方法", "训练 tile", "正标签行数", "每轮 batch 数", "学生 epoch", "日志累计耗时 / s"],
        [[m,tiles[m],105058+(summaries[m]["accepted_boxes"] if m in summaries else 0),math.ceil(tiles[m]/8),30,f"{times[m]:.2f}"] for m in metrics]),
        "注：正标签行数为落盘原始行数；标准加载器各删除一个重复人工框。B2/P1 共享教师的 100 epoch 日志累计耗时为 11460.10 s，"
        "另有伪标签推理成本，均未并入学生耗时。累计耗时取各 run 的 results.csv 最后一行 time，"
        "用于描述本轮实际运行开销，不能解释为同等计算预算下的比较或独立模块耗时基准。"]
    sections += ["## 表 4-S1 B1 的训练轮数参考", table(
        ["部分监督模型", "epoch", "mAP50 / %", "mAP50-95 / %", "用途"],
        [["B1",30,pct(metrics["B1"]["results"][keys[2]]),pct(metrics["B1"]["results"][keys[3]]),"配对学生对照"],
         ["B1",100,pct(long["results"][keys[2]]),pct(long["results"][keys[3]]),"教师与较长训练参考"]]),
        "注：B1 100 epoch 是部分监督模型，不能称为全监督上界 B0，也不能与 30 epoch 的 B2/P1 直接比较方法增益。"
        "两次 B1 均独立使用完整训练轮数的调度；100 epoch run 的第 30 行不替代独立 30 epoch run。"]
    ablation_table_rows = [
        ["全体", "mAP50", pct(metrics["P1"]["results"][keys[2]]), pct(ablation_eval["results"][keys[2]]),
         f"{100*(ablation_eval['results'][keys[2]]-metrics['P1']['results'][keys[2]]):+.3f}"],
        ["全体", "mAP50-95", pct(metrics["P1"]["results"][keys[3]]), pct(ablation_eval["results"][keys[3]]),
         f"{100*(ablation_eval['results'][keys[3]]-metrics['P1']['results'][keys[3]]):+.3f}"],
    ]
    for size, label in (("small", "Small"), ("medium", "Medium"), ("large", "Large")):
        ablation_table_rows.append([label, "recall", pct(recalls["P1"]["per_size"][size]["recall"]),
                                    pct(ablation_recall["per_size"][size]["recall"]),
                                    f"{100*(ablation_recall['per_size'][size]['recall']-recalls['P1']['per_size'][size]['recall']):+.3f}"])
    for cls in ablation_eval["class_indices"]:
        index = ablation_eval["class_indices"].index(cls)
        p1_index = metrics["P1"]["class_indices"].index(cls)
        p1_value = metrics["P1"]["class_maps"][p1_index]
        global_value = ablation_eval["class_maps"][index]
        ablation_table_rows.append(["类别", ablation_eval["names"][str(cls)], pct(p1_value),
                                    pct(global_value), f"{100*(global_value-p1_value):+.3f}"])
    sections += ["## 表 4-S2 尺度阈值全量确认", table(
        ["范围", "指标", "完整 P1 / %", "统一 0.50 / %", "统一−P1 / pp"], ablation_table_rows),
        "注：两组均为同一训练种子、数据源和教师下的独立 30 epoch 学生，并在 548 张完整原图上以 960 输入评估。"
        "唯一方法变量是伪标签正阈值为 0.30/0.50/0.50（完整 P1）或 0.50/0.50/0.50（统一阈值），两组均保留软权重与 ignore。"
        f"固定操作点 small 匹配数为 {recalls['P1']['per_size']['small']['matched']}→{ablation_recall['per_size']['small']['matched']}；"
        "recall 操作点及口径同表 4-2。类别行使用 Ultralytics 逐类别 AP50-95，差值为统一阈值减完整 P1。"]
    seed_pair = None
    if (SEED_PAIR_OUT / "acceptance_audit.json").exists():
        seed_pair = load_pair(hashes)
        sections += table_sections(seed_pair, table)
    main_pair = None
    if (MAIN_PAIR_OUT / "acceptance_audit.json").exists():
        main_pair = load_main_pair(hashes)
        sections += main_pair_sections(main_pair, table)
    no_soft = None
    if (ROOT / "outputs/p1_no_soft_weights_pilot30_s0/independent_audit.json").exists():
        no_soft, extra_sections = no_soft_sections(hashes)
        sections += extra_sections
    text="\n\n".join(sections)+"\n"
    dest=ROOT/"docs/paper/results_tables.md"
    if check:
        assert dest.read_text(encoding="utf-8")==text,"paper tables differ from frozen source values"
    else:
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(text,encoding="utf-8")
    output=ROOT/"outputs/paper_tables_20261027"
    output.mkdir(parents=True,exist_ok=True)
    receipt={"checked_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),"table_file":str(dest),
             "table_sha256":digest(dest),"tables_match_frozen_evaluations":True,"input_sha256":hashes,
             "metrics":metrics,"recalls":recalls,"times":times,"pseudo_quality":q,
             "global_threshold_ablation":{"comparison":ablation,"acceptance":ablation_acceptance},
             "scale_threshold_seed_pair":seed_pair,"main_method_seed_pair":main_pair,
             "no_soft_weights_ablation":no_soft}
    (output/"audit.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    recall_count = (6 if seed_pair else 4) + (1 if main_pair else 0) + (1 if no_soft else 0)
    print(f"{'Verified' if check else 'Wrote'} {dest}; {len(hashes)} sources hashed; {recall_count} recall evaluations share GT counts.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    main(parser.parse_args().check)
