# 论文结果与实现证据索引

## 结果文件与权重

以下路径相对于仓库根目录，数据和 run 文件位于本机忽略目录。正文与表格中的百分数由 JSON 的 0–1 原始值乘 100 后保留三位小数，差值在舍入前计算。

| 模型 | 完整原图 val960 结果 | 分尺寸 recall 结果 |
|---|---|---|
| B1 30 epoch | `runs/partial10_sup_pilot30_s0/validation/metrics.json` | `runs/size_recall/b1_pilot30_s0/metrics.json` |
| B2 30 epoch | `runs/partial10_naive_t050_pilot30_s0/validation/metrics.json` | `runs/size_recall/b2_pilot30_s0/metrics.json` |
| P1 30 epoch | `runs/val/partial10_p1_pilot30_s0_val960/metrics.json` | `runs/size_recall/p1_pilot30_s0/metrics.json` |
| B1 100 epoch | `runs/partial10_sup_s0/validation/metrics.json` | 本轮表格不使用其分尺寸 recall |

| 模型 | best.pt SHA256 |
|---|---|
| B1 30 epoch | `5c9e531a5e04929261fbeb57854f78703c011db00b226a8250683cbe232f86f4` |
| B2 30 epoch | `f22629ae5aae1fd61ac83dc3a2bcd165ac3a542847198c25780c84744688d45e` |
| P1 30 epoch | `8d20c9645328d9f1a18429c7223d40e3a123f211ec04e86b2f697a844a51e115` |
| B1 100 epoch，共享教师 | `58f40e1e7b397c3f0b873bbc9cbd9e50660b41c48c9cadb1e7ef3d0f0b0ebf54` |

B1 分尺寸 recall 在本次写作整理时使用已冻结的 30 epoch best.pt 补评估；没有重新训练。B1/B2/P1 三次评估共享 548 张完整原图，GT 分组计数均为 small 26577、medium 11115、large 1067，操作点与匹配规则一致。补评估日志为 `outputs/b1_size_recall_paper_20261027.log`。

## 表格来源

| 表格 | 主要来源 |
|---|---|
| 表 4-1 主结果与差值 | 上述三份 val960 `metrics.json`，B1/B2 冻结验收见 `outputs/gate2_20261027/comparison.json` |
| 表 4-2 分尺寸 recall | 三份 `runs/size_recall/*/metrics.json`，权重 SHA256 与冻结 best.pt 对应 |
| 表 4-3 伪标签覆盖 | `data/visdrone_area10/pseudo/{naive_t050,p1_s030_m050_l050}/summary.json`、`quality_audit.json` |
| 表 4-4 伪标签质量 | 上述两个 `quality_audit.json`；隐藏 GT 总数由 `data/visdrone_area10/summary.json` 核对 |
| 表 4-5 逐类别 AP50-95 | 三份 val960 `metrics.json` 的 `class_indices`、`class_maps` 和 `names` |
| 表 4-6 规模与耗时 | 学生样本清单、`summary.json`、各 run 的 `results.csv` 最后一行 `time` |
| 表 4-S1 较长训练参考 | B1 30/100 epoch 的独立 val960 结果；对应各自独立的 `results.csv` |

P1 的完整性验收为 `outputs/gate3_20261027/acceptance_audit.json`，19 项检查通过；原自动比较为同目录 `comparison.json`。原 Gate 3 的 mAP 条件未达到，小目标条件达到；用户结果后的工作决策另记在 `decision_note.json`，不作为实验数值来源。

## 第①组 no-ignore 消融证据

第①组 no-ignore 消融已完成：训练为 `runs/partial10_p1_no_ignore_pilot30_s0`，
完整原图评估为 `runs/val/partial10_p1_no_ignore_pilot30_s0_val960/metrics.json`，
分尺寸评估为 `runs/size_recall/p1_no_ignore_pilot30_s0/metrics.json`。其 best.pt
SHA256 为 `9c98ea42a1d656135ca562e84c86adf8ab53894fbaed4f0e6487643611f8aef0`。
独立验收见 `outputs/p1_no_ignore_pilot30_s0/acceptance_audit.json`，含冻结文件
哈希、实际参数、两组保存预测的 GT 匹配重算和逐类别差值；自动对比表为
同目录 `comparison.md`，差值为“去掉 ignore − 完整 P1”。本组尚未并入原表
4-1 至 4-6 的生成器，单列于实验章节第 4.9 节。

## 方法描述与代码对应

| 方法内容 | 实现或配置 |
|---|---|
| 面积预算、目标中心采样、抖动、未标注低重叠采样 | `src/regionssod/data/partial.py` |
| 框裁剪和可见性过滤 | `src/regionssod/data/annotations.py` |
| B2 全局阈值、仅纳入有正框的 tile | `src/regionssod/pseudo/generator.py` |
| 原始 tile 尺度分层与候选保存 | `src/regionssod/pseudo/scale.py`、`configs/pseudo_p1_10.yaml` |
| 增强后的置信度/ignore 对齐、正框优先掩码、原分母分类软权重 | `src/regionssod/training/p1.py` |
| P1 学生超参数 | `configs/train_p1_10.yaml`，实际值以 run 的 `args.yaml` 为准 |
| 完整验证与分尺寸匹配 | `src/regionssod/evaluation/{runner,size_recall}.py` |

分类损失的分母是原始 TAL target-score 总和，正 anchor 的整个 BCE 类别向量乘教师置信度；正样本分配、回归项和 DFL 的计算规则不增加该乘数。不确定框只屏蔽中心落入其中的背景 anchor 分类项，已匹配正 anchor 优先。方法初稿按这些已实现行为写作。

## 数字核对与完整精度记录

运行 `python scripts/build_paper_tables.py --check` 会核对三组 best.pt 哈希、各 run 的完整 30 行日志、验证分辨率、相同 GT 分组计数和操作点，并逐字比较表格与重新计算的文本。表格使用的 27 个源文件哈希及完整精度指标保存在 `outputs/paper_tables_20261027/audit.json`。

## 引言与相关工作核对

[引言](introduction.md)中的主结果和固定操作点召回来自表 4-1、4-2；面积预算与裁片构造来自上述数据实现。引言不将 10% 面积等同于目标数量或实测人工工时，也不将单 seed 开发结果解释为跨 seed 稳定收益。

[S³OD 差异对照](s3od_comparison.md)分别记录了 2025 年正式发表信息、2023 年可读预印本的章节与公式、以及作者公开代码提交 `d7806b78947cdc403d525e67a28193a88a4f8598` 的静态核对结果。正式发表 DOI 为 [10.1016/j.isprsjprs.2025.01.037](https://doi.org/10.1016/j.isprsjprs.2025.01.037)。预印本 PDF 本机缓存位于忽略目录 `outputs/literature/s3od/2310.14718.pdf`；期刊全文未取得，不能把预印本参数直接归于期刊版。公开代码的 GMM 阈值配置已与预印本百分位方案区分。

Soft Teacher 的权重对象依据其[第 3.2 节式(4)–(5)](https://arxiv.org/html/2106.09018v3#S3.SS2)核对，方法和相关工作已明确区分其背景可靠性权重与 P1 正伪框前景置信度权重。详细对照没有产生新的训练或同协议相关方法结果。

论文表格之外的复现命令见 [baseline_protocol.md](../baseline_protocol.md)，研究进度见 [paper_completion_plan.md](../paper_completion_plan.md)。引用文献与最终章节编号可在整合论文时统一处理，当前正文不使用虚构的文献编号或缺失结果。
