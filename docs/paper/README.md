# RegionSSOD 论文初稿

## 当前阅读入口（2026-11-02整理）

本次将正式100轮结果放到正文主线，并保留旧实验文件与所有正负结果。
按顺序阅读：[1 引言](introduction.md) → [2 相关工作](related_work.md) →
[3 方法](methods.md) → [4 正式结果](main_results.md) →
[5 讨论](discussion.md) → [6 结论](conclusion.md) → [参考文献](references.md)。
当前是有证据支撑的中文正文初稿，摘要、最终格式、引用统一和投稿缺口仍待处理。

| 管理入口 | 用途 |
|---|---|
| [结构与证据地图](paper1_writeup_plan.md) | 当前可写的主张、全部关键正负指标、图表安排与证据缺口 |
| [成果保存点](preservation.md) | 原P1 Git基点、本地4.20 GiB产物副本、哈希及只读复核 |
| [代码与改进点分轨](research_tracks.md) | 原P1、旧消融、评估工具、独立新候选分别定位 |
| [P1＋类别歧义候选](candidates/class_ambiguity/README.md) | 诊断及两臂30轮已完成；P90收益伴随AP/AP_small/P70退化，未通过扩大条件 |
| [类别分数完整30轮结果](candidates/class_ambiguity/results30.md) | 两臂审计通过、全部PR/逐类/成本与固定继续线；当前不扩训 |
| [类别歧义诊断结果](candidates/class_ambiguity/diagnostic_results.md) | 控制top1/面积/类别后的信息增益、原图bootstrap与全量旧框身份核对 |
| [文献与Zotero地图](literature_map.md) | 主线/候选的文献职责、MCP检索与集合核对、阅读范围 |
| [本轮验收与待办](organization_validation.md) | 208项测试、数值复核、回执恢复、缩写检查及文稿待完成项 |

以下为按时间保留的实验说明；旧`experiments.md`和`results_tables.md`包含原30轮矩阵，
不把它们当作100轮主结果。新候选无效也不影响原P1主线继续写作。

本目录保存基于已验收 B1、B2、P1 结果整理的中文论文初稿。
2026-11-01完成B2/P1正式100轮seed0配对：P1的mAP50-95提高0.0528pp，
小目标recall提高2.1259pp（净多565个），伴随未匹配预测增加1221个。
正式配对结果与原30轮证据分开报告。原30轮seed0主对照中P1相对B2的
mAP50-95提高0.335pp、小目标recall提高1.878pp；2026-10-30补齐同split/教师、
学生seed0/1复验，两项平均配对增益为+0.2893±0.0644pp、+1.7854±0.1304pp
（样本SD）。不将30轮与100轮混合统计。

2026-11-02完成同seed1 P1的55轮后恢复及独立验收：相对B2的AP50-95
提高0.3351pp、small recall提高2.4194pp（净+643），未匹配预测增加1435个。
正式100轮两学生seed平均配对增益为+0.1939±0.1996pp、
+2.2726±0.2075pp（样本SD）。建议本轮主线训练收束，优先写作与已有机制分析；
整体AP增幅随seed变化，原Gate3/4不追认为通过，论文证据缺口仍须按主张盘点。

| 文件 | 内容 |
|---|---|
| [small_pr_results.md](small_pr_results.md) | 四个正式100轮权重的COCO-style AP_small、完整PR、五档等精度召回、类别退化/候选截断与来源审计；零训练 |
| [small_pr_protocol.md](small_pr_protocol.md) | 新增低置信度评估的固定协议、两套匹配口径、来源绑定和复现命令 |
| [small_pr_literature.md](small_pr_literature.md) | 5篇论文的机制边界、确切阅读范围、可证伪假设与Zotero实际归档清单 |
| [high_precision_diagnostic_results.md](high_precision_diagnostic_results.md) | 90%precision小目标增失、候选去向、FP组成、共同GT几何与GT-oracle容量；H2未被支持为主因 |
| [scale_operating_point_results.md](scale_operating_point_results.md) | 5个图像split seed、两折校准/评估的尺度条件阈值；同时提高B2/P1但不形成P1独有优势 |
| [high_precision_diagnostic_protocol.md](high_precision_diagnostic_protocol.md) | 高精度诊断的固定70%/90%档位、逐GT互斥原因、双向增失、配对几何与oracle边界 |
| [scale_operating_point_protocol.md](scale_operating_point_protocol.md) | 尺度条件阈值的201点网格、5×2图像交拟、精度目标与固定继续线 |
| [introduction.md](introduction.md) | 第 1 章引言初稿：区域标注问题、相关工作衔接、方法概述、三项工作与初步结果 |
| [related_work.md](related_work.md) | 第 2 章相关工作初稿：SSOD、缺失标注、航拍小目标与最接近工作的关系 |
| [methods.md](methods.md) | 第 3 章方法初稿：区域协议、固定教师、尺度筛选、分类软权重、ignore 掩码与损失公式 |
| [experiments.md](experiments.md) | 第4章实验分析初稿：实验设置、整体/分尺寸/逐类别结果、伪标签质量和开销，含4.14分类软权重、4.15小框NWD及4.16回归置信度全量确认 |
| [results_tables.md](results_tables.md) | 表4-1至4-6和补充表4-S1至4-S5，含尺度阈值、B2/P1配对seed及分类软权重全量消融 |
| [evidence.md](evidence.md) | 表格来源、权重 SHA256、补评估记录和数字核对方式 |
| [p1_b2_full100_results.md](p1_b2_full100_results.md) | 正式100轮seed0配对的AP/尺寸、逐目标增失、误检/类别/成本及研究决定 |
| [b2_seed1_full100_results.md](b2_seed1_full100_results.md) | 正式100轮B2学生seed1的独立验收、原图指标、匹配数和P1手动交接 |
| [p1_b2_seed1_full100_results.md](p1_b2_seed1_full100_results.md) | 已完成seed1配对及两seed正式100轮汇总；AP/覆盖/类别/误检/成本、恢复限制和收束建议 |
| [paper1_p1_seed1_recovery.md](paper1_p1_seed1_recovery.md) | seed1 P1原55轮中断的冻结恢复说明；已执行完毕，当前只读验收 |
| [paper1_seed1_full100_protocol.md](paper1_seed1_full100_protocol.md) | 用户顺序运行学生seed1 B2/P1各100轮的固定协议、命令、预算与解释边界 |
| [ignore_recovery_analysis.md](ignore_recovery_analysis.md) | 第一篇追加的零训练诊断：锁定已有匹配，核对 ignore 条件容量上界、重复与低阈值候选代价，不作为学生增益 |
| [cross_view_flip_analysis.md](cross_view_flip_analysis.md) | 固定随机小样本的水平翻转候选筛选，记录精度与覆盖权衡，未进入学生训练 |
| [candidate_retention_analysis.md](candidate_retention_analysis.md) | 128 个独立抽样 tile 的 NMS/max-det 候选覆盖检查及停止结论，非学生精度 |
| [scale_support_analysis.md](scale_support_analysis.md) | 跨输入尺度支持的探索与不重叠样本确认，局部信号未通过确认，不进入学生训练 |
| [a_keyword_research.md](a_keyword_research.md) | 六类关键词的原论文调研、阅读范围及A的单机制优先级；新假设尚未实施，不是实验结果 |
| [positive_localization_analysis.md](positive_localization_analysis.md) | 已有正伪框跨尺度定位稳定性诊断；预先协议与分层样本不足的停止结论，不进入训练 |
| [case_studies.md](case_studies.md) | 已冻结 seed0 的成功/退化/共同漏检/未匹配预测四类案例，含可复现选图规则、裁剪坐标和全验证集事件计数 |
| [s3od_comparison.md](s3od_comparison.md) | S³OD 差异对照：正式发表版本、预印本公式、公开代码与 P1 实现、贡献范围及待验证问题 |
| [references.md](references.md) | 17篇核心/候选依据文献、阅读优先级、引用边界；[references.bib](references.bib)供文献管理工具导入 |
| [nwd_mechanism_positioning.md](nwd_mechanism_positioning.md) | NWD原文、S³OD与当前候选的作用位置，12/24epoch原文观察、单因素归因与后续对照边界 |
| [small_nwd30_results.md](small_nwd30_results.md) | 全量30轮NWD独立验收、完整原图960结果/类别/尺寸取舍、耗时和未达到继续线的记录；与五轮探索分开 |
| [assignment_geometry_results.md](assignment_geometry_results.md) | 固定P1和256训练增强输入的零训练分配诊断；框外监督减少但目标覆盖保护失败，初版不训练 |
| [assignment_coverage_results.md](assignment_coverage_results.md) | 新128基础tile确认整图覆盖保护规则；目标保留100%但框外质量仅降低48.10%，未达线，不训练 |
| [regression_confidence_diagnostic.md](regression_confidence_diagnostic.md) | 原教师c用于回归监督的零推理质量诊断，通过但不是学生收益；唯一30轮确认协议另列 |
| [regression_confidence30_failure.md](regression_confidence30_failure.md) | 首轮验证工程失败、真实exit1和原始证据；不能判断候选性能 |
| [regression_confidence30_valfix_protocol.md](regression_confidence30_valfix_protocol.md) | 验证兼容修正版、实际验证工程检查及独立30轮用户入口；沿用原继续线 |
| [regression_confidence30_results.md](regression_confidence30_results.md) | 修正版完整30轮独立验收；small recall+0.9444pp、mAP50-95−0.1755pp，未达扩大训练条件 |
| [regression_confidence_mass_results.md](regression_confidence_mass_results.md) | 全量35067正伪框的原c尺度/类别份额、实际损失归因限制及RUPL/UTv2机制边界；零推理零训练 |

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

2026-10-31补齐小框混合NWD全量30轮确认及独立验收，第4.15节和独立结果表
保留mAP50-95−0.2123pp、small recall−0.6698pp的负向结果。
该新增表的来源由`python scripts/audit_p1_small_nwd.py`核对，
不计入上述既有表格137来源/8组recall；不据此扩大训练或改写原Gate3。

同日回归置信度修正版完整30轮和独立验收完成，第4.16节及独立结果表
保留small recall+0.9444pp（+251匹配）与mAP50-95−0.1755pp的取舍。
两条预登记继续线均未达到；来源由
`python scripts/audit_p1_regression_confidence.py --validation-fix`核对，
同样不计入上述137来源/8组recall。当前持续目标保持用户设置的paused。
