# 代码、改进点与证据分轨索引

日期：2026-11-02。本索引通过链接归类，保留实际文件路径；
历史审计绑定了源码/配置路径及哈希，直接迁移目录可能破坏复核。

## A. 已有 P1：保留的论文主线

| 环节 | 文件 | 当前身份 |
|---|---|---|
| 区域标注协议 | `src/regionssod/data/partial.py` | 10%面积split0；隐藏GT仅审计 |
| 尺度正阈值、候选/ignore导出 | `src/regionssod/pseudo/scale.py`、`configs/pseudo_p1_10.yaml` | 原P1生成策略 |
| 分类置信度软权重、uncertain-ignore | `src/regionssod/training/p1.py`、`configs/train_p1_10.yaml` | 原P1默认训练语义 |
| B2/P1正式100轮 | `scripts/run_paper1_full100.py`、`scripts/run_paper1_seed1_full100.py` | 两seed已完成；不是待跑队列 |
| 固定点召回 | `src/regionssod/evaluation/size_recall.py` | 旧predict操作点，与AP_small分开 |
| 原论文表 | `scripts/build_paper_tables.py`、`docs/paper/results_tables.md` | 30轮主对照/消融保留为历史独立表 |
| 正式主结果 | [seed0](p1_b2_full100_results.md)、[seed1/两seed](p1_b2_seed1_full100_results.md) | 写作主结果 |

不通过改原默认配置把新候选偷偷变成P1。`p1.py`中的原教师最高分类分数c缩放匹配
伪正anchor的整个类别BCE向量，分母仍是未加权TAL target-score之和；
这不等于完整类别分布的软目标蒸馏。

## B. 已完成单因素消融与扩展：各自保留

| 方向 | 源码/入口 | 已有结果与处置 |
|---|---|---|
| no-ignore | `scripts/run_p1_no_ignore.py` | 30轮，去掉后AP+0.0217pp、small−0.1392pp；见原表4-S1 |
| no-soft-weights | `scripts/run_p1_no_soft_weights.py` | 30轮，去掉后AP−0.1623pp、small+0.9143pp；原表4-S5 |
| 全局阈值 vs 尺度阈值 | `scripts/run_p1_global_t050.py`、`scripts/run_p1_scale_seed_pair.py` | 30轮两学生seed，尺度−全局平均AP+0.4299pp、small+0.4929pp |
| 小框NWD | `src/regionssod/training/nwd.py`、`scripts/run_p1_small_nwd.py` | [完整30轮](small_nwd30_results.md)：AP−0.2123pp、small−0.6698pp |
| 回归置信度 | `src/regionssod/training/regression_confidence.py`、`scripts/run_p1_regression_confidence.py` | [修正版30轮](regression_confidence30_results.md)：AP−0.1755pp、small+0.9444pp |
| 覆盖保护分配 | `src/regionssod/training/coverage_guard.py`、`scripts/run_p1_coverage_guard.py` | [完整30轮](coverage_guard30_results.md)：AP−0.0652pp、small+0.1016pp |

各行比较对象和长度以原报告为准，不能与100轮P1拼成同一排名表。
NWD、回归置信度和覆盖保护均未达原继续条件，不默认叠加到新候选。
早期5轮筛选仅作工程/探索记录，不与30轮确认互换。

## C. 新增评估代码：研究分析工具，不是训练改进

| 代码 | 命令入口 | 作用 |
|---|---|---|
| `src/regionssod/evaluation/precision_recall.py` | `scripts/evaluate_paper1_small_pr.py`、`scripts/build_small_pr_results.py` | COCO-style AP_small、PR、等精度召回与来源绑定 |
| `src/regionssod/evaluation/error_diagnosis.py` | `scripts/diagnose_paper1_high_precision.py` | 高精度逐GT增失、FP分区、共同目标几何、oracle容量 |
| `src/regionssod/evaluation/operating_point.py` | `scripts/evaluate_scaled_operating_point.py` | 尺度阈值跨拟合代理诊断；不是新训练模型 |
| `scripts/preserve_paper1.py` | `--create` / `--audit` | 本次新增的独立成果快照与逐文件哈希复核 |
| `scripts/audit_paper1_frozen.py` | `--stage all` | 复用既有缓存审计，把新回执写到独立目录；不改写历史验收时间戳 |

上述工具不会替换原P1损失。尺度操作点代码继承完整缓存的匹配关系，
没有对独立尺度筛选后的预测集重新一对一匹配；数值应称代理诊断。

## D. 已完成30轮探索：P1 + 类别歧义感知分类监督

单独入口：[候选目录](candidates/class_ambiguity/README.md)。
已完成[教师分数诊断](candidates/class_ambiguity/diagnostic_results.md)，通过预定门槛；
全量35067原正框类别分数及独立实现后，已于2026-11-03完成两臂各30轮和独立验收。
[结果](candidates/class_ambiguity/results30.md)：IoU.50/P90 small +0.8203pp，
但AP50–95 −0.2631pp、AP_small −0.4162pp、P70 small −0.5418pp，未通过固定继续线。
训练产物用 `p1_class_ambiguity_*` 独立名称，不复用 `partial10_p1_*`。
当前仅运行`python scripts/run_p1_class_ambiguity.py --audit`只读复核；不重跑或直接扩到100轮。
[固定条件](candidates/class_ambiguity/student_protocol.md)保持原判定，正负结果均保留。
新候选只扩展分类监督；原尺度阈值、正伪框、原c软权重、ignore、回归和网络保持固定。
分类监督影响共享特征和后续训练分配，因此不能声称训练动态完全不受影响。

## E. 复现与保存

2026-11-03用户澄清成果对照后，当前入口更新为`run_p1_main_comparison.py`：
新B2/原P1/P1＋竞争负项主三臂各30轮，报告P1−B2、改进−P1、改进−B2。
[新合同](candidates/competitive_negative/main_comparison_protocol.md)独立保存，
强度控制可选第四组，主结果与机制结果分列；旧合同/实现保留，未启动训练。

2026-11-03新增独立候选：[竞争类别负项衰减](candidates/competitive_negative/README.md)。
`training/competitive_negative.py`与`configs/candidates/competitive_negative/`单独保存，
原P1及类别分数实现/合同未改。`diagnose_competitive_negative.py`完成六缓存重放；
`run_p1_competitive_negative.py`提供新原P1/语义动作/强度控制三臂30轮的预检与手动入口。
当前仅工程准备，未训练/未取得新学生效果；不替换上文任何已完成结果。

[成果保存说明](preservation.md)记录Git保存点和本地大文件副本；
[论文结构与证据](paper1_writeup_plan.md)负责写作导航；
[文献地图](literature_map.md)把参考论文分别连接到已完成主线和未验证候选。
README/baseline顶部提供只读复核命令。历史训练命令保留作协议，当前不重新执行。
