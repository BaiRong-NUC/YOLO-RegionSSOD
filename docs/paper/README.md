# RegionSSOD 论文初稿

本目录保存基于已验收 B1、B2、P1 结果整理的中文论文初稿。原 seed0 主对照中 P1 相对 B2 的 mAP50-95 提高 0.335 pp、小目标 recall 提高 1.878 pp；2026-10-30 补齐固定 split/教师、学生 seed0/1 的 B2/P1 复验，两项平均配对增益为 +0.2893±0.0644 pp、+1.7854±0.1304 pp（样本 SD）。

| 文件 | 内容 |
|---|---|
| [introduction.md](introduction.md) | 第 1 章引言初稿：区域标注问题、相关工作衔接、方法概述、三项工作与初步结果 |
| [related_work.md](related_work.md) | 第 2 章相关工作初稿：SSOD、缺失标注、航拍小目标与最接近工作的关系 |
| [methods.md](methods.md) | 第 3 章方法初稿：区域协议、固定教师、尺度筛选、分类软权重、ignore 掩码与损失公式 |
| [experiments.md](experiments.md) | 第4章实验分析初稿：实验设置、整体/分尺寸/逐类别结果、伪标签质量和开销，含4.14分类软权重消融 |
| [results_tables.md](results_tables.md) | 表4-1至4-6和补充表4-S1至4-S5，含尺度阈值、B2/P1配对seed及分类软权重全量消融 |
| [evidence.md](evidence.md) | 表格来源、权重 SHA256、补评估记录和数字核对方式 |
| [ignore_recovery_analysis.md](ignore_recovery_analysis.md) | 第一篇追加的零训练诊断：锁定已有匹配，核对 ignore 条件容量上界、重复与低阈值候选代价，不作为学生增益 |
| [cross_view_flip_analysis.md](cross_view_flip_analysis.md) | 固定随机小样本的水平翻转候选筛选，记录精度与覆盖权衡，未进入学生训练 |
| [candidate_retention_analysis.md](candidate_retention_analysis.md) | 128 个独立抽样 tile 的 NMS/max-det 候选覆盖检查及停止结论，非学生精度 |
| [scale_support_analysis.md](scale_support_analysis.md) | 跨输入尺度支持的探索与不重叠样本确认，局部信号未通过确认，不进入学生训练 |
| [a_keyword_research.md](a_keyword_research.md) | 六类关键词的原论文调研、阅读范围及A的单机制优先级；新假设尚未实施，不是实验结果 |
| [case_studies.md](case_studies.md) | 已冻结 seed0 的成功/退化/共同漏检/未匹配预测四类案例，含可复现选图规则、裁剪坐标和全验证集事件计数 |
| [s3od_comparison.md](s3od_comparison.md) | S³OD 差异对照：正式发表版本、预印本公式、公开代码与 P1 实现、贡献范围及待验证问题 |
| [references.md](references.md) | 16 篇核心文献、阅读优先级、引用边界；[references.bib](references.bib) 供文献管理工具导入 |

表格采用三个独立 30 epoch 学生进行主对照。B1 100 epoch 单列为教师和较长训练参考；20 图工程烟测不进入这些结果表。表 4-1 至 4-6 的 B1/B2/P1 主对照仍为 seed0；表 4-S4 和第 4.12 节单列 B2/P1 的两学生 seed 复验，B1 未重复，全部为 Ultralytics 开发结果；尺度阈值另有固定 split/教师、学生 seed0/1 的配对复验（表 4-S3、第 4.11 节）。尺度分层在两学生 seed 中的 mAP50-95 和 small recall 均高于统一 0.50，平均配对增益为 +0.4299±0.2657 pp 和 +0.4929±0.2128 pp（样本 SD）。小目标诊断 recall 与官方 AP_small/AR500 分别表述，不将两学生 seed 解释为统计显著性或多 split 验证。

论文正文围绕方法和实验发现组织，项目 Gate 的验收记录与结果后的工作决策保留在 [paper_completion_plan.md 第 3.8 节](../paper_completion_plan.md)。当前原 Gate 3 数值门槛未达到，用户选择保留正向初步成果并优先写作；正文没有据此改写实验数字或声称通过原门槛。

核对或重新生成表格，在仓库根目录的 `yolo` 环境运行：

```powershell
python scripts/build_paper_tables.py --check
python scripts/build_paper_tables.py
```

`--check`只核对表格与冻结结果是否一致；第二条命令重新生成表格文本。完整精度数值、
输入哈希和生成核对记录位于忽略目录`outputs/paper_tables_20261027/audit.json`。
目前已有第1–4章初稿和S³OD方法差异对照；S³OD引用已更新为2025年期刊版，
公式明确标注预印本来源，公开代码按固定提交核对。尺度阈值两学生seed复验、
四类seed0案例、B2学生seed1及116项独立验收已完成。分类软权重全量消融
现已通过16项独立验收，表4-S5和实验4.14保留mAP与small recall取舍；
当前表格核对137个来源和8组recall。large recall的不一致方向及所有负面结果保留。
正式矩阵和最终排版仍待补充；引用暂沿用现有编号，最终按首次出现顺序重排。
