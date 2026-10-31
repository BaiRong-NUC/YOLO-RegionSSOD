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
| 表 4-S2 尺度阈值全量确认 | `outputs/p1_global_t050_pilot30_s0/{comparison,acceptance_audit}.json`；两组 val960、分尺寸 recall 和逐类别结果 |
| 表 4-S3 固定 split 配对学生 seed | `outputs/p1_scale_threshold_pair_s1/{comparison,acceptance_audit,independent_audit}.json`；seed0 来源同表 4-S2，seed1 来源及哈希见下文 |
| 表 4-S4 B2/P1 主对照配对学生 seed | `outputs/b2_pilot30_s1/{comparison,acceptance_audit,independent_audit}.json`；两 seed 四组原始评估及样本 SD，来源/权重见下文 |
| 表 4-S5 分类软权重全量消融 | `outputs/p1_no_soft_weights_pilot30_s0/{comparison,independent_audit}.json`；同数据、教师、学生seed0、30epoch，来源/权重见下文 |

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

## 尺度阈值全量确认证据

统一 0.50 阈值确认组已完成 30 epoch 训练。训练为
`runs/partial10_p1_global_t050_pilot30_s0`，完整原图评估为
`runs/val/partial10_p1_global_t050_pilot30_s0_val960/metrics.json`，分尺寸评估为
`runs/size_recall/p1_global_t050_pilot30_s0/metrics.json`。其 best.pt SHA256 为
`94571aae799119becc033fa6a7296eb0326dbf7fb83e44f650ef7ab488728a4a`，
last.pt SHA256 为 `cd96237d62d121ec8132cdfd30dbce94aefe39afd887bfde6a39759ebae3b5ff`。

独立验收见 `outputs/p1_global_t050_pilot30_s0/acceptance_audit.json`。验收确认
连续 30 epoch、训练值有限、实际超参数配对、同一教师、相同软权重和 ignore 开关、
完整 val960 与固定操作点 recall 协议一致，并重放两组保存预测进行 GT 匹配。
自动对比为同目录 `comparison.json` / `comparison.md`，差值为“统一 0.50 − 完整
P1”。结果已由 `scripts/build_paper_tables.py` 纳入表 4-S2，包含整体指标、
分尺寸 recall 与逐类别 AP50-95。

## 尺度阈值配对学生 seed 复验证据

2026-10-27 23:47 已完成两组 seed1 的训练、完整评估和自动验收，随后独立复核通过。
split seed 仍为 0，教师和两套伪标签均复用；只将学生训练 seed 改为 1。

| seed1 方案 | 训练目录（`runs/` 下） | best.pt SHA256 | last.pt SHA256 |
|---|---|---|---|
| 尺度分层 | `partial10_p1_pilot30_s1` | `18b241f1dea83c9fba6dbf23bc10c6d57c3ee7fb7bd13c3ed17db38d8b6667b4` | `07efa451d614ece5b10ed6236f5e567a814312b781e9ac2f849af1db8f53f048` |
| 统一 0.50 | `partial10_p1_global_t050_pilot30_s1` | `5a2f82de198d1d7326d179b56c899b9ea70f54fe9bc5eeb1ac3d1f9d8a2d761c` | `1e4e12d86388d578c8b18aec70099534c755e5bf44398970349cadb7f5943e4a` |

两组 val960 来源分别为
`runs/val/partial10_p1_pilot30_s1_val960/metrics.json` 和
`runs/val/partial10_p1_global_t050_pilot30_s1_val960/metrics.json`；固定操作点 recall
分别为 `runs/size_recall/p1_pilot30_s1/` 和
`runs/size_recall/p1_global_t050_pilot30_s1/` 下的 `metrics.json`、`images.txt`、
`predictions.jsonl`。两 run 均保存 `experiment.json`、`args.yaml`、连续 30 行
`results.csv`、`audit.json` 和 `SHA256SUMS.txt`。

可复现入口为 `scripts/run_p1_scale_seed_pair.py`；既有 run 防重复覆盖，
`--collect-only` 仅重做已有评估的验收和汇总。执行脚本的 SHA256 保存于
`outputs/p1_scale_threshold_pair_s1/preflight.json`，本次训练期间未修改该脚本或方法源码。
另以 `python scripts/audit_p1_scale_seed_pair.py` 独立复核 36 项检查，记录为
同目录 `independent_audit.json`：33,047 个冻结文件、实际参数仅改变学生 seed、
同一教师/伪标签/损失、best/last 与评估对应、548 图/38759 GT，以及两组保存预测
重放均通过。配对统计还使用 NumPy 独立核算均值和 ddof=1 的标准差。

表 4-S3 来自 `comparison.json` 中的 seed0/1 单次值和配对差；论文生成器复查
自动验收产物哈希、原始评估与两种统计口径，独立验收的哈希另纳入论文表证据。
这是一个 split/固定教师下两个学生 seed 的尺度阈值复验，不补齐 B1/B2/P1 的
多 seed 主表，不报告统计显著性，不改变原 Gate 3。各项数值和不一致方向见实验第 4.11 节。

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

运行 `python scripts/build_paper_tables.py --check` 会核对主对照及尺度阈值 seed0/1 结果的 best.pt 哈希、各 run 的完整 30 行日志、验证分辨率、相同 GT 分组计数和操作点，并逐字比较表格与重新计算的文本。表 4-S3、4-S4 另核对种子内配对差、各方案均值及样本标准差；表4-S5要求分类软权重独立验收通过、产物哈希一致并重算差值。当前核对137个来源及8组不同目录的recall（GT分组计数一致）。表格使用的全部源文件哈希及完整精度指标保存在 `outputs/paper_tables_20261027/audit.json`。

## 引言与相关工作核对

[引言](introduction.md)中的主结果和固定操作点召回来自表 4-1、4-2；面积预算与裁片构造来自上述数据实现。引言不将 10% 面积等同于目标数量或实测人工工时，也不将单 seed 开发结果解释为跨 seed 稳定收益。

[S³OD 差异对照](s3od_comparison.md)分别记录了 2025 年正式发表信息、2023 年可读预印本的章节与公式、以及作者公开代码提交 `d7806b78947cdc403d525e67a28193a88a4f8598` 的静态核对结果。正式发表 DOI 为 [10.1016/j.isprsjprs.2025.01.037](https://doi.org/10.1016/j.isprsjprs.2025.01.037)。预印本 PDF 本机缓存位于忽略目录 `outputs/literature/s3od/2310.14718.pdf`；期刊全文未取得，不能把预印本参数直接归于期刊版。公开代码的 GMM 阈值配置已与预印本百分位方案区分。

Soft Teacher 的权重对象依据其[第 3.2 节式(4)–(5)](https://arxiv.org/html/2106.09018v3#S3.SS2)核对，方法和相关工作已明确区分其背景可靠性权重与 P1 正伪框前景置信度权重。详细对照没有产生新的训练或同协议相关方法结果。

论文表格之外的复现命令见 [baseline_protocol.md](../baseline_protocol.md)，研究进度见 [paper_completion_plan.md](../paper_completion_plan.md)。引用文献与最终章节编号可在整合论文时统一处理，当前正文不使用虚构的文献编号或缺失结果。

## B2 学生 seed1 与案例图来源（2026-10-30，验收完成）

用户在终端执行 `scripts/run_b2_seed1.py`，2026-10-30 **10:33:25** 派发，
**12:11:53** 完成训练、完整评估和自动验收。训练日志连续 1–30 epoch、数值有限，
B2 seed1 累计训练日志耗时 5845.51 秒；P1 seed1 复用既有完整运行，没有重训。

新增配置为 `configs/train_naive_10_seed1.yaml`。预检冻结 37,695 个文件，含原训练/
验证数据、B1 教师、B2 伪标签、COCO 初始化、P1 两 seed 产物、方法源码及执行配置。
B2 原伪标签生成时的历史源码哈希单列于 `preflight.json` 的
`historical_pseudo_generation_source_sha256`，不将历史生成脚本当成本次训练源码。
执行入口 `scripts/run_b2_seed1.py` 的 SHA256 为
`7671ebdbb131e211e8245f74d7eeef876480c0da78c38f4e5a1a985a6f5bfb95`，与各次
归档快照一致，训练期间配置和方法源码均未修改；教师与伪标签未重生成。

| 方法/学生 seed | 训练目录（`runs/` 下） | best.pt SHA256 | last.pt SHA256 |
|---|---|---|---|
| B2_s0 | `partial10_naive_t050_pilot30_s0` | `f22629ae5aae1fd61ac83dc3a2bcd165ac3a542847198c25780c84744688d45e` | `b5c5ed3af29392726871a7cceda9c689380adce61ee3ade7fc4771e85b0a04b1` |
| P1_s0 | `partial10_p1_pilot30_s0` | `8d20c9645328d9f1a18429c7223d40e3a123f211ec04e86b2f697a844a51e115` | `03cf3b6c4f9bb238001179a44667610c5a9344935fd3ed9cc820add79c7253ca` |
| B2_s1 | `partial10_naive_t050_pilot30_s1` | `558c086f1f8d11de9c6cb2062c5d2b5d08ff749f221ecd88e873797112172a10` | `2ae4bea8ca7d89e0e98632f619ccf2ec77d57dc20194367d272ce656c3d0317e` |
| P1_s1 | `partial10_p1_pilot30_s1` | `18b241f1dea83c9fba6dbf23bc10c6d57c3ee7fb7bd13c3ed17db38d8b6667b4` | `07efa451d614ece5b10ed6236f5e567a814312b781e9ac2f849af1db8f53f048` |

四组完整评估为 `runs/val/<上述训练目录名>_val960/metrics.json`，固定操作点
来源为 `runs/size_recall/{b2,p1}_pilot30_s{0,1}/` 下的 `metrics.json`、`images.txt`
和 `predictions.jsonl`。B2 seed1 的 run 另保存 `audit.json`、`SHA256SUMS.txt`、
`args.yaml`、`experiment.json` 和 `results.csv`。对照主表的旧 B2 seed0
`validation/metrics.json` 与本次 `runs/val/.../metrics.json` 数值一致，独立来源均保留。

`scripts/audit_b2_seed1.py` 独立复核 **116 项检查**通过：全部冻结输入、四组连续
训练调度/有限值、同 seed 的实际超参数配对、同方法 seed0 到 seed1 仅 seed/name
变化、标准损失与 P1 损失开关、best/last 和评估权重哈希、548 图顺序、原图尺寸
GT 计数 26,577/11,115/1,067、recall 操作点与四组保存预测 GT 匹配重放，以及
NumPy 独立计算的配对均值和 ddof=1 样本 SD。固定 conf=0.25、匹配 IoU=0.50、
NMS IoU=0.70、max_det=500。自动与独立记录为
`outputs/b2_pilot30_s1/{comparison,acceptance_audit,independent_audit}.json`；表 4-S4
与完整精度/来源哈希由论文表生成器核对，不手填或筛选不利结果。

### 中断与手动交接来源

此前三次未完成运行完整保留，均不拼接到最终日志，不进入论文统计：

| 归档目录（`outputs/` 下） | 情况 | 完整保留的文件数 |
|---|---|---:|
| `b2_seed1_interrupted_20261030_091251/` | 10 月 28 日运行在第 3 epoch 中断，CSV 完成 1–2，无能确认原因的异常栈 | 10 |
| `b2_seed1_interrupted_20261030_101330/` | 10 月 30 日 09:13 重跑在第 16 epoch 中断，CSV 完成 1–15；系统未重启，退出原因未确认 | 12 |
| `b2_seed1_interrupted_20261030_102720/` | 10:15 独立进程启动，10:27 按用户手动训练要求主动停止，CSV 完成 1–3 | 13 |

当前 `outputs/b2_pilot30_s1/recovery.json` 链接第三次归档，其内部递归链接前两次；
独立验收逐一核对三份 receipt、35 个归档文件的原路径哈希、脚本/配置快照。
第三次归档中保存 WMI 启动的 `launch_receipt.json` 及包装源码哈希；最终完整运行
由用户终端直接执行原入口，其日志为当前 `stdout.log`/`stderr.log`。未把外层启动
包装调整解释为训练方法变动。进度日志保留原中断、交接及完成时间。

### 案例图来源

案例来自事先冻结的 seed0 B2/P1 `runs/size_recall/{b2,p1}_pilot30_s0/` 保存预测，
不按新增 seed1 结果重新选图。manifest 保存 1,106 个输入 SHA256、四类关注位置/
裁剪坐标、全验证事件计数与 5 张 PNG 的 SHA256。总览 `cases_overview.png` 哈希为
`36116e8ab3a59b58fcd316a313102e8d4911697fdda0f0bba94f6e722cd84a9d`。
`python scripts/build_detection_cases.py --check` 再次通过选图、全验证计数和输出图
核验；人工查看总览确认 GT/B2/P1 三列及关注位置可读。图像位于忽略目录
`outputs/paper_cases_b2_p1_s0/`，图注及边界见 `case_studies.md`。

## 分类软权重全量消融来源（2026-10-30）

- 训练：`runs/partial10_p1_no_soft_weights_pilot30_s0/`，连续30epoch、11310tile，
  CSV累计7016.44秒；仅关闭分类软权重，保留原教师、正标签、ignore与实际训练参数。
- best.pt SHA256：`94cbc8f289c4d446ab821e9d2a4fb80d943bd7f1ee117c7d216b2b2bfe5ea883`。
- last.pt SHA256：`6113b6d8da80e17933e2886aa36dc55cefaa30125c26c6c76970ffbebf44c510`。
- 全图评估：`runs/val/partial10_p1_no_soft_weights_pilot30_s0_val960/metrics.json`；
  recall与保存预测：`runs/size_recall/p1_no_soft_weights_pilot30_s0/`。
- 比较、预检、进程退出和独立验收：
  `outputs/p1_no_soft_weights_pilot30_s0/{comparison,preflight,launch_receipt,independent_audit}.json`。
  包装进程于北京时间16:28:18退出，exit_code=0；`python scripts/audit_p1_no_soft_weights.py`
  通过16项检查、23783冻结输入和两组各548图预测重放。
- 结果方向为去软权重mAP50-95−0.1623pp、small recall+0.9143pp（+243），逐类3升7降。
  只有单split/教师/学生seed；表4-S5及实验4.14保留取舍，不宣称所有模块均提高全部指标。

## 原图上下文类别纠正探索（2026-10-30，不入学生结果表）

- 执行前协议`context_rectification_analysis.md`及配置先由`ef8f565`提交；
  实际128图推理源码`f1c997e`。原教师、既有正框/权重/ignore均未改。
- `outputs/context_rectification_20261030/discovery/{protocol,analysis,costs,manifest}.json`
  保存固定样本、全部统计、4.3874秒推理成本和372来源SHA256；原始预测
  `context.jsonl`和逐纠正`diagnostics.jsonl`可完整重放。
- 98接受tile/799正框，仅1次类别纠正且改错；总体匹配562→561、
  small302→301。固定继续线失败，确认入口重放后拒绝；确认/学生训练均未执行。
- 只读命令`python scripts/audit_context_rectification.py --check`通过。
  详细范围与小样本限制见`context_rectification_results.md`；
  这是伪标签诊断，不是学生检测结果、独立盲测或官方AP_small。

## 开放区域置信度校准探索（2026-10-30，不入学生结果表）

- `f3d77cd`先提交`confidence_calibration_protocol.md`及固定配置/代码。
  新推理仅512开放标注tile，使用原B1教师，耗时含warmup8.1703秒；
  校准目标来自`labeled/labels`，7548正框，未使用源图隐藏GT拟合。
- `outputs/confidence_calibration_20261030/{fit,discovery}/`保存协议、预测、映射、
  逐正框统计及SHA256；拟合1045来源/探索474来源只读重放一致。
- 来源不重叠256tile探索有1421正框、分组覆盖96.4110%；整体Brier原分数
  0.158864、分组0.160310，固定整体条件失败；small虽改善，不重定义通过。
  确认入口拒绝已实测，无确认/训练。详细表及分布迁移限制见
  `confidence_calibration_results.md`，只读命令为
  `python scripts/audit_confidence_calibration.py --check`。

## 小框混合NWD五轮筛查（2026-10-30，不入主表）

- 执行前协议`small_nwd_protocol.md`由`626fa9e`先提交，使用同一1024源图/
  1765tile、原COCO初始化、5epoch/640/batch8/SGD/seed0/close_mosaic0，
  只新增一组NWD学生；参考为已验收同预算完整P1。
- `outputs/screens/p1_small_nwd_1024_e5/`保存23794输入预检、GPU检查、
  启动/退出及独立验收。训练18:16启动、18:22:23包装exit0，CSV累计299.989秒。
- 两组548原图val960和保存预测匹配重放一致；NWD−P1的mAP50-95−0.1746pp、
  small recall+1.5464pp，继续线失败。保留取舍、不调参数追试、不扩训。
  数据、代码/参数、权重/产物SHA256通过独立入口
  `python scripts/audit_small_nwd_screen.py`核验。
- 详细表及范围见`small_nwd_results.md`，本组是子集探索，不是全量方法结论或官方AP。

## 小框混合NWD全量30轮确认（2026-10-31）

- 用户终端从初始化独立训练30轮，未拼接旧13轮中断；09:02:42启动、
  11:18:53包装exit0，status=completed，CSV累计8092.09秒，11310tile。
- 训练：`runs/partial10_p1_small_nwd_pilot30_s0/`；完整val：
  `runs/val/partial10_p1_small_nwd_pilot30_s0_val960/metrics.json`；
  recall及保存预测：`runs/size_recall/p1_small_nwd_pilot30_s0/`。
- 冻结输入/来源、回执、比较及独立验收：
  `outputs/p1_small_nwd_pilot30_s0/{preflight,launch_receipt,status,comparison,independent_audit}.json`。
  `python scripts/audit_p1_small_nwd.py`核对23793项来源、实际参数与产物哈希，
  并重放两组各548图预测，accepted=true；worth_larger_training=false。
- best SHA256：`3a0357ae000a922441f2c472b8a07c03f18552dad77134a6809d24923dfc81e6`；
  last SHA256：`e50dcbd46c0aa7ece4345bfb75721433aba2df9300f2496da6a994e8cedf58c5`。
- mAP50-95−0.2123pp、small recall−0.6698pp（−178匹配），medium/large亦下降；
  类别4升6降。完整结果与限制见`small_nwd30_results.md`及实验4.15。
  新表单独由NWD验收入口核对，不纳入既有表格生成器的137来源/8组recall统计。

## 原c回归加权全量30轮确认（2026-10-31）

- 科学变量和继续线在两个回归置信度30轮协议中预登记；修正版源码提交
  `12b68866bf849586189e6c9deb8b76ee6099e17b`，与旧工程失败/用户中断分开归档。
  13:36:03独立启动，15:41:20包装exit0，status=completed，连续1–30轮、
  11310tile；验收时原包装22020和训练22512均已停止。
- 输出：`outputs/p1_regression_confidence_valfix_pilot30_s0/`的preflight、
  launch_receipt、status、comparison、artifact_sha256及independent_audit。
  `python scripts/audit_p1_regression_confidence.py --validation-fix`核对
  23809冻结来源、实际args、权重/产物，并重放两组各548图固定操作点预测；
  accepted=true、worth_larger_training=false。
- 训练：`runs/partial10_p1_regression_confidence_valfix_pilot30_s0/`；
  完整val：`runs/val/partial10_p1_regression_confidence_valfix_pilot30_s0_val960/metrics.json`；
  recall和保存预测：`runs/size_recall/p1_regression_confidence_valfix_pilot30_s0/`。
- best SHA256：`3899b907206b00440ec5a2aa51227afe0f0b4d1454d47484cf362b1c03c9a563`；
  last SHA256：`35386d2c882964efc4643ad1791d6b7f3571af7fb38b8f7b6808ea0bf929e18b`。
- small recall+0.9444pp（+251匹配）、mAP50-95−0.1755pp，类别1升9降，
  未达两条继续线。完整精度/尺寸/类别/耗时及限制见
  `regression_confidence30_results.md`和实验4.16。
  新结果表由本候选独立验收核对，不计入既有137来源/8组recall统计。
