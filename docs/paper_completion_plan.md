# 论文完成计划与项目交接

> 最后更新：2026-10-30<br>
> 当前执行范围：只推进A第一篇方法稿；B/C、蒸馏剪枝和树莓派部署留在远期规划，不并行启动。<br>
> 最新补证据：分类软权重全量30epoch消融已完成独立验收；去软权重mAP50-95−0.1623pp、small recall+0.9143pp，未达到扩大训练条件；表4-S5/实验4.14已同步，当前不扩训。<br>
> 最新安排：用户要求至少30epoch才判断性能，中文提交后push；NWD五轮取舍保留为历史探索，按独立协议准备唯一全量30epoch确认。<br>
> 当前阶段：B2/P1 学生 seed0/1 主对照复验及四类案例已完成验收；mAP50-95 与 small recall 两 seed 同向，平均配对增益 +0.2893/+1.7854 pp；large recall 方向不同。论文表 4-S4、第 4.12–4.13 节及证据更新，原 Gate 3 不变<br>
> 论文暂定题目：**面向高分辨率航拍图像的区域局部标注小目标半监督检测方法**  
> 仓库：`D:\Python\YOLO-RegionSSOD`  
> 计划起点：2026-10-26，可按实际启动日期整体平移

## 0. 一句话交接

本论文研究“高分辨率训练图只有局部区域有标签”时的航拍小目标半监督检测：
未标注区域中的目标仍然存在，不能当作纯背景。主实验固定 VisDrone、YOLO11s、
640 训练、完整原图 960 验证，先建立 B1 部分监督和 B2 朴素自训练基线，再验证
P1 尺度感知伪标签与 uncertain-ignore 是否带来稳定收益。

## 1. 论文定位

### 1.1 研究问题

每张高分辨率训练图只标注一个目标中心 tile，tile 外仍有大量未标注目标。
训练时如果把所有未标注像素当作背景，会产生错误负样本；如果直接丢弃 tile 外
区域，又会损失上下文和潜在目标信息。

### 1.2 核心假设

- H1：在不使用未标注区域时，B1 会明显低于完整监督参考 B0。
- H2：只用全局阈值伪标签的 B2 能获得一定收益，但对小目标存在系统性漏检。
- H3：按目标尺度设置伪标签阈值、按教师置信度加权，并把低置信度候选视为
  unknown 而不是背景，能在相同训练预算下优于 B2。

### 1.3 论文贡献

1. 建立区域局部标注下的高分辨率航拍小目标检测协议，记录固定 split 和未标注
   区域中的隐藏 GT，用于分析伪标签漏检。
2. 提出 tile-aware 自训练流程，将尺度感知阈值和教师置信度权重用于伪标签筛选。
3. 将低置信度教师预测作为 unknown/ignore 区域，减少小目标漏检造成的背景负样本
   污染，并按尺寸、密度和类别报告机制证据。

不要在论文中使用“首次”或“全面解决”表述。贡献必须由 B0/B1/B2/P1 对照、
消融和机制分析支撑。

## 2. 当前范围与边界

- 当前论文只使用图像检测，不切换到视频。
- 当前主数据集为 VisDrone train/val；外部确认集优先 UAVDT，但必须完成类别映射
  和版本审计后才纳入。
- 当前方法实现基于 YOLO11s，不把 backbone、neck、head、P2 或注意力改造作为
  第一轮变量。
- `docs/RGB_IR_Reliability_Experiment_Design.md` 是另一项 RGB-IR 可靠性融合
  设计，不属于本论文。不得混用数据、模型、指标或结论。
- 现有 VisDrone test-dev 已被旧项目多次查看，不再作为新论文的盲测集。

## 3. 当前状态

以下状态以 2026-10-30 为快照。后续会话必须先运行 `git status` 并核对实际产物，
不能只依赖本文档的旧描述。

| 项目 | 状态 | 证据或位置 | 下一步 |
|---|---|---|---|
| Git 仓库 | B2 seed1、案例与论文更新已提交 `80e9f2b`；新增三篇文章/学位论文/树莓派路线为规划文档修改 | 第 3.20–3.21 节；`git status --short --branch` | 规划不等于训练/投稿或专利申请已启动 |
| 数据构造 | 已实现；10% seed0 已生成 | `src/regionssod/data/partial.py` | 生成正式 5%/20% split |
| B0 完整监督 | 配置已加入，未正式训练 | `configs/train_full.yaml` | 复用旧结果或重训 seed 0 |
| B1 部分监督 | 正式 100 epoch 和独立 30 epoch pilot 均完成；val960 mAP50-95 分别为 0.248835、0.229114 | `runs/partial10_sup_s0/`、`runs/partial10_sup_pilot30_s0/` | 冻结为正式参考与 pilot 对照 |
| B2 朴素自训练 | seed0/1 独立 30 epoch 均已验收；seed1 val960 mAP50-95=23.3688%、small recall=34.4471%；100 epoch 未运行 | `outputs/b2_pilot30_s1/{comparison,acceptance_audit,independent_audit}.json`；第 3.20 节 | 冻结两 seed 对照，保留单 split/固定教师边界 |
| P1 尺度感知方法 | seed0/1 相对 B2 的 mAP50-95 与 small recall 均提高；平均配对增益 +0.2893±0.0644/+1.7854±0.1304 pp；large recall 方向不同 | 表 4-S4；第 3.8、3.20 节；实验第 4.12 节 | 保留两个学生 seed 的正向总体/小目标结果，不作显著性或跨 split 主张；原 Gate 3 不变 |
| 第①组 no-ignore 消融 | 30 epoch、val960、固定操作点 recall 及独立验收完成；相对完整 P1 的 mAP50-95 +0.0217 pp、small recall −0.1392 pp | `outputs/p1_no_ignore_pilot30_s0/{comparison,acceptance_audit}.json`；第 3.13 节 | 不能声称 ignore 独立提高整体 mAP；继续拆分软权重与尺度阈值，补配对 seed |
| 软权重低成本筛选 | 1024 源图、两组各 5 epoch 和 val960 完成；mAP50-95 +0.0884 pp、small recall +0.0790 pp，低于 0.3 pp 候选线 | `outputs/screens/p1_soft_weights_1024_e5/{comparison,acceptance_audit}.json`；第 3.15 节 | 保留当时中性停止结论；最新A补证据授权下单独启动正式30epoch消融，见第3.26节，不追认短筛过线 |
| 软权重全量单因素消融 | 30epoch/val960/recall/独立验收完成；去软权重mAP50-95−0.1623pp、small recall+0.9143pp | `outputs/p1_no_soft_weights_pilot30_s0/independent_audit.json`；第3.26节；表4-S5 | 未达扩大训练条件；保留完整P1主配置及取舍结果，不自动扩训 |
| 尺度阈值低成本筛选 | 统一 0.50 对比尺度分层：mAP50-95 +0.2237 pp、small recall −3.1230 pp | `outputs/screens/p1_global_t050_1024_v2_e5/{comparison,acceptance_audit}.json`；第 3.16 节 | 自动候选线未触发，但小目标信号值得全量 30 epoch 确认；数据已准备，用户手动启动 |
| 尺度阈值全量确认 | 30 epoch、val960、固定操作点 recall 及独立验收完成；统一 0.50 相对完整 P1 的 mAP50-95 −0.6177 pp、small recall −0.3424 pp，逐类 1 升/9 降 | `outputs/p1_global_t050_pilot30_s0/{comparison,acceptance_audit}.json`；表 4-S2；第 3.18 节 | 全量结果支持尺度分层阈值更有利于总体检测和小目标覆盖；仍为单 split/seed，不扩写为跨 seed 稳定性 |
| 尺度阈值配对学生 seed | seed1 两组各 30 epoch、完整 val960 和预测重放完成；尺度分层相对统一 0.50 的 mAP50-95 +0.2420 pp、small recall +0.6434 pp，与 seed0 两项同向 | `outputs/p1_scale_threshold_pair_s1/{comparison,acceptance_audit,independent_audit}.json`；表 4-S3；第 3.19 节 | 保留尺度分层；单 split/固定教师下两学生 seed 的平均配对增益为 +0.4299/+0.4929 pp，不作显著性或跨 split 主张 |
| 正式 5/10/20% 数据 | 10% seed0 已生成；20 图同 seed 检查通过；5%/20% 未生成 | `data/visdrone_area10/summary.json`、`audit.json` | 生成 5%/20% split |
| 正式 100 epoch 主矩阵 | 已完成 B1 10% seed0；B2/P1 未完成 | `runs/partial10_sup_s0`；已有 P1 30 epoch 正向初步结果 | 先整理已有成果，后续按论文需要决定补充验证规模 |
| 论文正文 | 第1–4章、两seed统计/案例，以及表4-S5/实验4.14软权重全量消融已同步；原seed0主表保留 | `docs/paper/{experiments,results_tables,evidence,case_studies}.md`；第3.20、3.26节 | 正式矩阵与最终排版仍待补，后续先选一个低成本问题，不自动扩训 |
| 文章与树莓派部署规划 | 用户新增三篇文章目标及最终树莓派部署要求；论文按录用、专利按“申请通过”，硬件与学校节点待确认 | `docs/publication_and_edge_deployment_plan.md`；第 3.21 节 | 先 A 方法稿、C 轻量化/部署稿，再扩展 B 可靠补全稿；树莓派先做导出与板端基线，均未启动 |
| 第一篇实验缺口 | 已完成零训练漏检归因；6,471 tile/38,245 hidden GT 精确重放，按低成本列出 L0→L1→L2 | `docs/paper1_experiment_gaps.md`；`scripts/audit_paper1_gaps.py`；第 3.22 节 | 先做 ignore 可恢复增量与 NMS/max-det 候选覆盖小试；未过线不启动训练 |

### 3.1 已完成的烟测

- 20 张 VisDrone 图像，10% 标注面积：20 个标注 tile、321 个监督框。
- 20 个未标注 tile 中有 141 个隐藏真实框，说明“未标注不等于纯背景”。
- B2 使用 1 epoch、320 分辨率教师：20 个未标注 tile 中接受 13 个，生成 38 个框；
  合并后训练集为 33 张 tile。
- B1/B2 的伪标签生成、列表合并、训练和完整 val 验证链路均成功。
- 最后一次记录：`pytest` 9/9 通过。

**证据边界：**烟测只有 20 张图、1 epoch，B2 烟测 `mAP50-95 ≈ 0.0252`。
这只证明工程链路，不是方法效果，不得写入论文结果表，也不得据此判断研究方向。

### 3.2 正式 10% seed0 数据

- 2026-10-26 在 Git Bash 中加载 `D:/MiniConda/etc/profile.d/conda.sh`，激活已有
  `yolo` 环境后运行 `make-partial`；命令见 README 和 baseline 协议。
- 6471 张训练图全部使用，无跳过；6471 个标注 tile、6471 个未标注 tile。
- 标注 tile 共保留 105058 个监督框，未标注 tile 中共有 38245 个可见隐藏 GT 框。
- 实际平均标注面积比例为 9.993%；完整原图验证集仍为 548 张。
- `data/visdrone_area10/audit.json` 保存了 12 项基础检查、当前 commit 和关键文件
  SHA256；检查包括 split/manifest 一致、标签合法、隐藏 GT 未导出为训练标签、
  验证路径指向完整原图。

**证据边界：**该次数据构造只完成基础审计。后续的 20 图工程验收和正式 B1
训练分别见第 3.3、3.4 节。

### 3.3 Gate 1 工程验收通过（2026-10-26）

- 用户已完成 `partial10_sup_smoke5`：20 张标注 tile、5 epoch、640、batch 8、
  SGD、seed 0，`results.csv` 包含完整 5 行；所有数值有限，best/last 权重齐全。
- box loss 从 2.04970 降至 1.77198；该趋势仅用于排除明显工程异常。
- 本次补跑 `best.pt` 的完整原图 960 验证：548 张图、38759 个实例；Ultralytics
  开发指标 mAP50=0.117764、mAP50-95=0.080882。
- 同 seed 重生成 20 图数据，manifest 在归一化输出路径后完全一致；40 张裁剪
  图像与 20 个标签文件的 SHA256 均一致，且与正式数据的前 20 图对应。
- 正式数据基础审计所记录的 8 个文件哈希均未变化；`python -m pytest` 9/9 通过。
- 证据：`runs/smoke/partial10_sup_smoke5/gate1_audit.json`、
  `runs/val/partial10_sup_smoke5_val960/metrics.json`、
  `data/gate1_checks/area10_seed0_20261026`。

**结论：**第 9 节定义的 Gate 1 已通过。上述数字仍是 20 图工程烟测，不能进入
论文结果表，也不能证明 B1/B2/P1 的正式效果。当时建议的 B1 seed0、100 epoch
现已完成，见第 3.4 节；仍需完成 B1/B2 的 30 epoch pilot，才能决定是否实现 P1。

### 3.4 正式 B1 完成，B2 数据准备完成（2026-10-26）

- 用户已完成 `runs/partial10_sup_s0`：6471 张标注 tile、100 epoch、YOLO11s、
  640、batch 8、SGD、seed 0；从 COCO 预训练初始化。日志共 100 行、全部数值
  有限，CSV 累计训练耗时 11460.1 秒（约 3.18 小时）。训练 commit 为 `14eaf2c`。
- 本次补跑 `best.pt` 的完整原图 960 验证：548 张图、38759 个实例；Ultralytics
  mAP50=0.417503、mAP50-95=0.248835、precision=0.559604、recall=0.408240。
  训练日志的 640 验证最佳 mAP50-95 为 0.16704（epoch 90），须与 val960 区分。
- 8 项正式 run 审计通过；原 split/manifest 等关键哈希未变。证据和 SHA256 保存在
  `runs/partial10_sup_s0/audit.json`、`SHA256SUMS.txt`、`validation/metrics.json`。
- 固定教师 `runs/partial10_sup_s0/weights/best.pt`，SHA256 为
  `58f40e1e7b397c3f0b873bbc9cbd9e50660b41c48c9cadb1e7ef3d0f0b0ebf54`。
- 正式伪标签首轮暴露了批处理缺陷：Python 路径列表使 Ultralytics 将全部 6471
  图当作一个 batch，触发 CUDA OOM。已改为持久化文件清单输入，按 batch 8
  流式读取；回归测试在旧实现上复现问题，修复后全部 11 项测试通过。
  失败目录归档为 `data/visdrone_area10/pseudo/naive_t050_failed_all_images_batch_20261026`。
- 修复后以 threshold=0.50、conf-floor=0.05、imgsz=960、batch=8 完成 6471 图
  推理；接受 4624 个 tile（71.46%）、27667 个框，10 类均有伪标签。合并后
  11095 张训练 tile、132725 个框，完整验证集保持不变，未将空预测 tile 当背景。
- 离线隐藏 GT 诊断：按类别、IoU>=0.50、降序 IoU 一对一匹配，伪标签 precision
  为 81.47%、recall 为 58.94%，漏检隐藏 GT 15704 个；小目标 recall 为 46.99%。
  尺寸按原始 tile 中裁剪后框面积划分（small<32²、medium<96²），这不是 AP_small
  或官方评测，不能证明 B2 学生已有收益。隐藏 GT 只用于离线分析。
- B2 数据通过 10 项审计；其标签、清单、配置、源代码及教师共 11106 个文件的
  SHA256 已保存。证据位于 `data/visdrone_area10/pseudo/naive_t050/` 下的
  `summary.json`、`audit.json`、`quality_audit.json`、`SHA256SUMS.txt`；合并配置为
  `data/visdrone_area10/dataset_naive_t050.yaml`。诊断脚本在忽略目录
  `outputs/audit_naive_t050_20261026.py`。

**当时的下一步：**按 baseline 协议第 4.1 节运行 B1/B2 各 30 epoch 的独立学生并
完成完整 val960，现已完成，见第 3.5 节。B2 和后续 P1 pilot 固定使用上述 B1
100 epoch 教师；学生训练轮数配对，实际训练数据量、批次数和耗时分别记录，
教师成本另记。不得将 B2 30 epoch 与 B1 100 epoch 或其第 30 行 CSV 作 gate
比较。官方 AP、AP_small 和多 seed 仍待补。

### 3.5 Gate 2 配对 pilot 通过（2026-10-27）

- 用户已完成 `partial10_sup_pilot30_s0` 和 `partial10_naive_t050_pilot30_s0`，
  训练 commit 均为 `f7b711c`。两者均从 COCO 预训练独立开始，完整训练 30 epoch；
  640、batch 8、SGD、seed 0、增强及其他实际训练参数一致，仅训练数据和 run
  名称不同。训练 CSV 数值全部有限，best/last 权重齐全。
- 本次分别补跑两者的 `best.pt` 完整原图 960 验证，均为 548 张图、38759 个实例。

| 方法 | 训练 tile | 学生训练耗时 | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| B1 30 epoch | 6471 | 3523.76 秒 | 0.382766 | 0.229114 | 0.512319 | 0.385004 |
| B2 30 epoch | 11095 | 5073.12 秒 | 0.389202 | 0.234534 | 0.520438 | 0.386509 |

- B2−B1 的 mAP50-95 为 +0.005421，即 **+0.5421 个百分点**；mAP50 为
  +0.6436 个百分点。Gate 2 要求差值不低于 -0.002，因此 **Gate 2 通过**。
- 每类 AP 的方向为 9 类上升、1 类下降；awning-tricycle 为 -0.1817 个百分点。
  这是单 split/seed 的开发 pilot，不做显著性或跨 seed 稳定性结论。
- 核对了原数据 8 个关键文件和伪标签包 11106 个文件的哈希，全部与冻结记录一致；
  固定教师未变。两个 run 下均保存了 `audit.json`、`SHA256SUMS.txt` 和
  `validation/metrics.json`。汇总证据为 `outputs/gate2_20261027/comparison.json`，
  输入审计为同目录 `input_integrity.json`。
- 本对照固定的是学生 epoch；B2 的数据量、每轮批次数和训练耗时更多，并额外使用
  B1 100 epoch 教师。不能据此宣称同等计算量下的收益。指标仍为 Ultralytics
  开发指标，未补官方 AP、AP_small、AR500 或多 seed。

**当时的下一步：**开始实现 P1 的尺度分层阈值、教师置信度软权重和 uncertain-ignore。
先完成候选/置信度保留和损失侧的权重、ignore 路径及测试，再使用同一固定教师、
同一 split、COCO 初始化和 30 epoch 学生协议对照 B2。Gate 3 要求 P1 相对 B2
至少 +0.5 个百分点 mAP50-95（即当前基线下至少 0.239534），并且 AP_small 或
小目标 recall 同向改善。Gate 3 通过前不添加网络模块或扩展 P1 正式主矩阵。
本次仅完成训练验收、验证与记录；P1 代码和训练均尚未开始。

### 3.6 P1 已实现，全量数据就绪，30 epoch pilot 已启动（2026-10-27）

- P1 第一轮只加入尺度阈值、教师置信度分类软权重、uncertain-ignore，没有修改
  YOLO11s 结构或 box/DFL gains。绑定 Ultralytics 8.4.172；方法、阈值边界、
  增强传递方式和损失公式语义见 `docs/baseline_protocol.md` 第 4.3 节。
- 固定原 tile 面积 small<32²、medium<96²、large>=96²；正阈值分别为
  0.30/0.50/0.50，下限 0.05。仅含正框的 tile 入训，其余候选仍保存在离线记录。
  教师、split、960 推理、batch8、NMS 和 max_det 均沿用 B2。
- `python -m pytest --basetemp outputs/pytest_p1_20261027` **28/28 通过**。覆盖
  尺度边界、阈值合法性、单位权重时与原 loss 完全一致、空正框、分类梯度、
  ignore/正样本优先、回归不变、跨图隔离、mosaic/翻转/close_mosaic 对齐、
  元信息与标签错配拒绝、实际流式推理 batch，以及固定 recall 匹配。
- 真实 GPU 工程烟测：同一教师对前 20 个未标注 tile 生成 17 个伪标签 tile、
  139 个正框和 325 个 ignore 框；与 20 个标注 tile 合并为 37 张，640/batch8
  完成 1 epoch 和 548 图训练内完整验证。best/last 齐全，CSV 全部有限，独立
  进程重新加载 `P1DetectionModel` 推理正常。证据为
  `runs/smoke/p1_integration_20261027`、`outputs/p1_smoke_*_20261027.log`。
  **这些烟测结果不进入论文结果表，也不用于判断 Gate 3。**
- 全量 P1 生成完成：6471 个未标注 tile 中接受 4839 个，35067 个正框、52117
  个 ignore 框；正框按尺度为 small 20666、medium 12637、large 1764。合并学生
  训练集 11310 张，原始正标签行共 140125。标准 Ultralytics loader 会删除人工
  标签中的 1 个重复框，实际加载 140124 个正框；没有修改原标签文件，去重逻辑
  与 B1/B2 相同。数据在 `data/visdrone_area10/pseudo/p1_s030_m050_l050/`。
- 隐藏 GT 仅用于**离线**诊断：P1 precision=73.20%、recall=67.12%、小目标
  recall=60.36%；B2 分别为 81.47%、58.94%、46.99%。增加覆盖的同时精度下降，
  不能据此声称学生性能改善。相同 IoU/匹配规则和边界可见性规则的证据为 P1
  目录下 `quality_audit.json`，脚本为 `outputs/audit_p1_20261027.py`。
- B2 学生已补固定操作点完整原图分尺寸诊断：548 张、38759 GT，conf=0.25、
  IoU>=0.50、max_det=500；小目标 recall=8987/26577=0.33814952778718443。
  证据为 `runs/size_recall/b2_pilot30_s0/metrics.json`，其权重 SHA256 与 Gate 2
  一致。此指标不是官方 AP_small/AR500，也不是上述伪标签 recall。
- 全量预检通过：8 个原 split/manifest 关键文件和 B2 的 11103 个数据/权重文件
  哈希未变；全部 P1 正框与置信度元信息一致，完整 val548 路径保持原图。配置、
  源码/测试和全部训练/验证输入已冻结 SHA256，脏源码快照存于
  `outputs/gate3_20261027/source_snapshot/`；预检为同目录 `preflight.json`，
  生成审计脚本为 `outputs/prepare_p1_20261027.py`。没有把新源码哈希与 B2 的
  历史源码哈希混作“未变化”证据。
- 北京时间 **2026-10-27 10:33** 启动后台 `scripts/run_p1_pilot.py`，run 为
  `runs/partial10_p1_pilot30_s0`：同 COCO 初始化、640、30 epoch、SGD、batch8、
  seed0。P1/B2 每 epoch 批次数分别为 1414/1387；学生耗时与教师成本另记。
  脚本会依次完成训练参数/CSV 审核、完整 val960、同操作点小目标 recall、
  权重哈希和 `comparison.json`；失败写入 `status.json` 并停止，不会扩训。
  日志为 `outputs/p1_pilot30_20261027.{stdout,stderr}.log`，当前状态必须读取
  `outputs/gate3_20261027/status.json`，不得仅凭本启动记录认定训练已完成。

**当时尚未完成：**30 epoch pilot 和 Gate 3 的最终验收，现已完成，见第 3.8 节。
原通过条件为 P1 mAP50-95
至少 0.239534294688062，且小目标 recall 严格高于 0.33814952778718443。
未测得两个条件前不扩展正式 P1 主矩阵，不宣称 P1 有效。

### 3.7 P1 训练进度核查：尚未完成（2026-10-27 11:04，北京时间）

- 后台 Python 进程 PID 9500 仍在运行，命令为 `scripts/run_p1_pilot.py`；
  `status.json` 仍处于 `training`。该文件仅记录阶段切换，实时 epoch 进度应以
  stdout 日志和 `results.csv` 为准，不把阶段时间戳较旧误判为训练停滞。
- CSV 已记录完整、连续的 epoch 1–6，全部数值有限；最近一行累计训练时间
  1618.76 秒。11:04:53 日志已到第 7/30 epoch 的 1344/1414 batch（约 95%）。
  best.pt、last.pt 均存在；stderr 为 0 字节，核查时未见训练错误。
- 实际 `args.yaml` 与 B2 对比，除 data/name/save_dir 外无参数差异。训练内的
  640 验证值属于中间曲线，不能当作最终完整原图 val960 结果或提前判定 Gate 3。
- P1 的完整 val960 `metrics.json`、分尺寸 recall 和 Gate 3 `comparison.json`
  均尚未生成。因此本次仅完成**训练进度核查**，不能验收最终效果；Gate 3 是
  **尚未测量**，不是通过或失败。原后台训练继续运行，完成后由已启动脚本自动
  执行完整验证、固定操作点小目标 recall 和比较，不另行启动重复训练。
- 核查证据：`outputs/gate3_20261027/progress_check_20261027_110453.json`。

### 3.8 P1 结果验收：取得初步收益，保留阶段成果（2026-10-27）

P1 的 30 epoch 训练、完整原图 val960 和固定操作点小目标 recall 已于北京时间
**12:36:10** 全部完成，12:44 完成独立验收。**结果显示正向收益，可作为方法的
阶段成果整理。**原 Gate 3 的 +0.5 个百分点 mAP50-95 门槛未达到，小目标 recall
同向改善条件已达到，故原 Gate 3 判定为未通过。

| 方法（独立 30 epoch，seed0） | 训练 tile | 学生耗时（秒） | mAP50 | mAP50-95 | 小目标 recall |
|---|---:|---:|---:|---:|---:|
| B2 全局阈值自训练 | 11095 | 5073.12 | 38.9202% | 23.4534% | 33.8150% |
| P1 尺度阈值 + 软权重 + ignore | 11310 | 7286.61 | 39.4990% | 23.7883% | 35.6925% |
| P1−B2（百分点） | — | — | **+0.5788** | **+0.3349** | **+1.8776** |

- 两者使用同一完整原图 val：548 张、38759 个 GT。小目标 recall 固定
  conf=0.25、匹配 IoU>=0.50、max_det=500、原图 GT 框面积<32²，匹配规则一致。
  P1 匹配小目标 9486/26577，B2 为 8987/26577，即多匹配 499 个小目标。
  这是学生完整验证集的诊断 recall，与隐藏训练 GT 的伪标签 recall 分开报告。
- 原门槛要求 P1 mAP50-95>=0.239534294688062，实际为
  0.23788331849556607，差 0.1651 个百分点；因此自动判定与人工复核均为
  `gate3_passed=false`。该检查项未通过不等于没有成果：mAP50、mAP50-95
  和小目标 recall 均为正向变化。
- **验收检查 19/19 通过**：30 行 epoch 连续、CSV 数值全部有限、best/last 齐全、
  训练参数与 B2 除 data/name/save_dir 外一致；共 23773 个冻结输入、19 个 run
  记录的源码/输入和 8 个输出文件哈希均匹配。教师与 B2 所用教师一致，P1
  评估权重对应本轮 best.pt，均可追溯；
  两次分尺寸评估的 548 图列表/顺序和 GT 数一致，stderr 为空。
- best.pt SHA256：
  `8d20c9645328d9f1a18429c7223d40e3a123f211ec04e86b2f697a844a51e115`。
  last.pt SHA256：
  `03cf3b6c4f9bb238001179a44667610c5a9344935fd3ed9cc820add79c7253ca`。
- 证据：`runs/partial10_p1_pilot30_s0/{audit.json,SHA256SUMS.txt,results.csv}`、
  `runs/val/partial10_p1_pilot30_s0_val960/metrics.json`、
  `runs/size_recall/{b2_pilot30_s0,p1_pilot30_s0}/metrics.json`、
  `outputs/gate3_20261027/{comparison,acceptance_audit}.json`。
  独立验收脚本为 `outputs/audit_gate3_acceptance_20261027.py`。
- 当前证据是单 split/seed 的 Ultralytics 开发结果。可以表述为“在本次配对
  pilot 中，P1 提高整体检测指标和小目标召回”；跨 seed 稳定性和统计显著性
  尚待补充，官方 AP/AP_small/AR500 也需与这些开发指标分别报告。
- **结果后的工作决策：**用户明确要求“有提升，有成果就好”。因此将本轮作为
  积极的初步成果保留，优先整理方法、配对结果表、伪标签覆盖/精度取舍和案例。
  原 Gate 3 数值和判定保留用于复盘，后续工作不以本次未达 0.5 个百分点门槛
  为由否定已有成果。若需要增强论文证据，可按具体需要选择单因素消融或重复
  seed 验证；不把扩大完整矩阵作为整理现有成果的前提。

### 3.9 论文结果表格与方法和实验初稿（2026-10-27）

- 写作入口为 `docs/paper/README.md`。`methods.md` 已写出区域协议、固定离线
  教师、原始 tile 尺度阈值、置信度分类软权重、ignore 掩码及总体损失公式；
  描述与现有 P1 实现对应。`experiments.md` 已写出设置、整体结果、分尺寸
  recall、伪标签精度与覆盖取舍、逐类别表现和运行开销分析。
- `results_tables.md` 包含表 4-1 至 4-6、表 4-S1 和 4-S2，共八组：主结果/差值、
  分尺寸 recall、伪标签覆盖、伪标签质量、逐类别 AP50-95、学生规模/耗时，
  以及单列的 B1 30/100 epoch 参考。主对照只使用三个独立 30 epoch 学生，
  未混入工程烟测；百分数保留三位小数，差值用未舍入值计算。
- 为补齐三组的相同操作点评估，使用冻结 B1 30 epoch best.pt 补跑完整 val960
  `size-recall`，没有重新训练。结果 small/medium/large recall 分别为
  34.379%/76.302%/89.316%，small 匹配 9137/26577；权重哈希仍为
  `5c9e531a5e04929261fbeb57854f78703c011db00b226a8250683cbe232f86f4`。
  证据为 `runs/size_recall/b1_pilot30_s0/metrics.json`，命令见 baseline 协议。
- 分尺寸分析明确记录：B2 整体 mAP 提高，但固定 conf=0.25 下 small recall
  低于 B1；P1 相对 B1/B2 均提高该 recall。逐类别 AP 中 P1 相对 B2 为
  **7 类升、3 类降**；论文草稿据此描述收益分布，没有表述为全部类别改善。
- 新增 `scripts/build_paper_tables.py`，从冻结 JSON/CSV 生成表格并用
  `--check` 逐字核对；同时校验 best.pt 哈希、30 行日志、配对参数、548 图
  和 GT 分组/操作点。27 个来源的哈希、完整精度数值和核对记录保存为
  `outputs/paper_tables_20261027/audit.json`；写作来源与权重索引见
  `docs/paper/evidence.md`。
- 当前交付为**方法与实验章节初稿**。引言、相关工作、案例图和最终排版可在
  此基础上继续整合。原 Gate 3 与用户保留初步成果的决定按第 3.8 节记录。
- `python scripts/build_paper_tables.py --check` 核对通过；
  `python -m pytest --basetemp outputs/pytest_paper_writing_20261027` 28/28 通过。

### 3.10 相关工作与参考论文整理（2026-10-27）

- 新增 `docs/paper/related_work.md`，按半监督目标检测、漏标/稀疏标注、航拍
  小目标检测及与本文方法的关系组织，并直接服务于后续第 2 章写作。
- 新增 `docs/paper/references.md` 和 `references.bib`，共 16 项文献。核心必读
  为 S³OD、Soft Teacher、Consistent-Teacher、SparseDet 和 Density Crop-guided
  SSOD；另整理阅读重点及引用限制，后续不应只按题名堆积参考文献。
- **最近的参考工作是 S³OD**（arXiv:2310.14718，当前按预印本引用）：其尺寸感知
  自适应阈值、尺度重平衡标签分配和教师引导负学习与本文尺度阈值及不确定区域
  处理直接相关。相关工作已明确，本文不能声称首次结合尺度阈值和教师负学习；
  当前可支持的差异主要是区域局部标注协议、不修改 YOLO11s 网络的实现，以及
  完整原图上的伪标签覆盖/精度和大小目标收益分析。正式投稿前应核对其预印本
  后续版本，并尽可能做同协议对照或单因素消融。
- `methods.md` 和 `experiments.md` 已加入关键引用。文献编号当前按基础检测器/
  数据、SSOD、缺失标注、航拍尺度工作的顺序编排；完成引言后需按全文首次引用
  顺序统一重排。
- 文献线索经论文搜索和官方出版页/论文页核对；S³OD 与 Density Crop-guided SSOD
  均标注为预印本。未把不同数据集上的论文数值直接与本文 VisDrone 开发结果比较。

### 3.11 S³OD 方法差异对照与引言初稿（2026-10-27）

- 新增 `docs/paper/s3od_comparison.md`：按标注预算、区域构造、师生更新、尺度
  阈值、正样本分配、分类权重、负学习、框类型和评测口径逐项对照，并连接
  S³OD 原文段落、公式与 P1 实现；不从不同协议的数值推断方法优劣。
- 更新第 3.10 节的历史文献信息：S³OD 已以 *S³OD: Size-unbiased semi-supervised
  object detection in aerial images* 正式发表于 ISPRS Journal of Photogrammetry
  and Remote Sensing，2025，221：179–192，DOI `10.1016/j.isprsjprs.2025.01.037`。
  `references.md` 与 BibTeX 已更新；详细公式仍明确归于已读的 2023 预印本。
- 期刊全文本次未取得。核对作者公开 master 提交
  `d7806b78947cdc403d525e67a28193a88a4f8598`：其 DOTA 配置含 GMM 自适应阈值，
  与预印本百分位方案不同；另核对 GWD top-K 分配和教师 RoI 负候选处理，
  没有声称复现或运行 S³OD。期刊摘要的实验集为 DOTA-v1.5 与 SODA-A。
- 新增 `docs/paper/introduction.md`，包含研究背景、区域面积预算、错误负监督
  问题、已有工作关系、方法概述、三项工作和单 seed 初步结果。明确目标中心
  采样为合成协议，面积预算不能等同于实测工时节省，不声称首次或领先 S³OD。
- 按 Soft Teacher 第 3.2 节修正方法/相关工作/文献阅读说明：其具体公式对负候选
  背景分类项按教师背景分数加权，与 P1 的正 anchor 前景置信度权重不同。
- README 与论文入口、证据索引已同步；现有引用编号暂保持一致，最终排版时
  再按全文首次引用顺序重排。本次为文稿与文献核对，不新增实验或改变 Gate 3。
- `python scripts/build_paper_tables.py --check` 通过，27 个结果来源核对一致。
  忽略目录中的 `outputs/verify_writing_20261027.py` 核对 9 份论文 Markdown、
  36 个本地链接、16 项文献/BibTeX 及引言 6 个结果数值，全部通过；审计记录为
  `outputs/literature/s3od/writing_audit_20261027.json`。`git diff --check` 通过。
  未改训练源码，未重跑训练测试或启动训练。

### 3.12 第①组去掉 uncertain-ignore 的训练入口就绪（2026-10-27）

- 按用户要求新增 `configs/train_p1_no_ignore_10.yaml`，run 为
  `partial10_p1_no_ignore_pilot30_s0`。与完整 P1 的冻结配置相比仅更改名称和
  `p1.uncertain_ignore=false`；保留软权重、尺度阈值、11310 个训练 tile、元信息、
  增强链路、同一 COCO 初始化、640/30 epoch/batch8/SGD/seed0。
- 新增 `scripts/run_p1_no_ignore.py`，执行后依次训练、验证连续 30 epoch 和实际
  参数、完整原图 val960、相同操作点分尺寸 recall，再与完整 P1 自动生成 JSON
  及 Markdown 对照。差值为“去掉 ignore − 完整 P1”，不预设结果方向，不改变
  原 Gate 3；失败写入阶段记录并停止，已有 run/派发记录时拒绝重复启动。
- 已在原 `yolo` 环境执行 `python scripts/run_p1_no_ignore.py --check-only`：
  **23781 个冻结输入与完整 P1 参考产物哈希核对通过**，确认仅变更目标开关。
  预检在 `outputs/p1_no_ignore_pilot30_s0/preflight.json`，不表示训练已完成。
- `python -m pytest --basetemp outputs/pytest_p1_no_ignore_20261027` 28/28 通过；
  README 和 baseline 协议第 4.4 节已加入自动入口与独立训练/评估命令。
- **当时状态仅为配置、脚本与预检完成，尚未启动训练**，按用户“给我训练指令”
  提供可直接运行的 `python -u scripts/run_p1_no_ignore.py`。启动后的阶段文件为
  `outputs/p1_no_ignore_pilot30_s0/status.json`；全部完成后查看同目录
  `comparison.md`、`comparison.json`；本组现已完成并验收，见第 3.13 节。

### 3.13 第①组 no-ignore 消融完成验收（2026-10-27）

- 用户已完成 `runs/partial10_p1_no_ignore_pilot30_s0`，自动流程于北京时间
  **17:12:53** 完成。CSV 为连续 1–30 epoch，所有数值有限；best/last 权重齐全，
  实际训练参数与完整 P1 除 name/save_dir 外一致。唯一方法变量为
  `p1.uncertain_ignore=true→false`，软权重、数据、教师和训练源码保持一致。
- 完整原图 val960 与固定操作点分尺寸 recall 均已完成。差值定义为“去掉
  ignore − 完整 P1”，单位为百分点：

| 指标 | 完整 P1 / % | 去掉 ignore / % | 差值 / pp |
|---|---:|---:|---:|
| mAP50 | 39.4990 | 39.5405 | +0.0415 |
| mAP50-95 | 23.7883 | 23.8100 | +0.0217 |
| Small recall | 35.6925 | 35.5533 | −0.1392 |
| Medium recall | 77.5709 | 77.1390 | −0.4318 |
| Large recall | 89.4096 | 88.0037 | −1.4058 |

- recall 操作点保持 conf=0.25、匹配 IoU=0.50、NMS IoU=0.70、max_det=500，
  两组为相同 548 张完整原图和 38759 个 GT。small 匹配数 9486→9449（−37），
  medium 为 8622→8574（−48），large 为 954→939（−15）；此指标不是 AP_small。
- 逐类别 AP50-95 为 4 类升、6 类降。去掉 ignore 后，bus +1.2027 pp、
  tricycle +0.8362 pp；truck −0.7854 pp、van −0.7845 pp。总体近似持平包含
  不同类别的增减抵消，不能描述为所有类别一致改善。
- 日志累计学生耗时为 7169.38 秒，完整 P1 为 7286.61 秒；单次运行的差值不能
  解释为 ignore 模块的精确耗时。no-ignore best.pt SHA256 为
  `9c98ea42a1d656135ca562e84c86adf8ab53894fbaed4f0e6487643611f8aef0`。
- 独立审计 `outputs/audit_p1_no_ignore_acceptance_20261027.py` 已通过：23795 个
  冻结输入/源码/参考及输出文件哈希匹配，run 记录的 19 个输入哈希匹配；
  用两组保存预测分别重算全部 548 图的 GT 匹配，分尺寸计数与报告一致。
  审计为 `outputs/p1_no_ignore_pilot30_s0/acceptance_audit.json`。
- **结论：**本轮去掉 ignore 后整体 mAP 几乎不变，固定操作点召回下降。
  不支持将整体 mAP 收益主要归因于 ignore，也不能从单 seed 断言该机制稳定
  有效或无效。保留两组结果，下一步优先做去掉软权重的同条件对照，随后补
  尺度阈值消融及重复 seed；本次没有启动新训练或 GPU 评估，原 Gate 3 不变。

### 3.14 用户约束更新与软权重低成本筛选启动（2026-10-27）

- 用户要求按步骤继续实验，每个完成实验提交并推送 Gitee，提交信息用中文；
  随后补充“每次验证实验成本不要太大，当找到了值得后续训练的东西后给出
  训练指令停止”。后续以这项低成本及停止条件为当前执行范围，不自动扩训
  30/100 epoch 或完整主矩阵。
- 第①组验收内容已在 `4fe664d`，本次用 `git ls-remote origin refs/heads/main`
  核实 Gitee main 为同一提交，初始工作区干净，不重复提交已有结果。
- 准备了全量去软权重配置/入口 `configs/train_p1_no_soft_weights_10.yaml`、
  `scripts/run_p1_no_soft_weights.py`，仅关闭软权重、保留 ignore。23781 个冻结
  文件预检通过，但按用户更新要求**没有启动这组全量 30 epoch 训练**。
- 已于北京时间 **17:44:44** 启动 `scripts/screen_p1_soft_weights.py`，PID **5784**。
  固定抽取 1024 张源训练图，样本选择 seed=20261027，不读取隐藏 GT 作筛选。
  对应 1024 个标注 tile、741 个已接受伪标签 tile，共 1765 个训练 tile，
  两组共用同一列表与相同元信息；分别保留/去掉软权重，均保留 ignore。
- 两组从同一 COCO 权重独立训练 5 epoch，640/batch8/SGD/seed0；短程对照
  统一 `close_mosaic=0`，默认 warmup=3 epoch。每组完成后各做完整原图 val960
  和固定操作点 recall。提前固定的候选信号为：mAP50-95 差绝对值至少 0.3 pp，
  small recall 同向。它是探索性筛选标准，不是论文效应门槛或显著性检验。
- 31/31 tests 通过，其中新增独立梯度检查，验证仅关闭某一开关时另一个机制
  仍起作用。原论文表格的 27 个来源核对通过，原 Gate 3 判定不变。
- 当时仅已启动筛选，尚无结果；阶段和日志在
  `outputs/screens/p1_soft_weights_1024_e5/status.json`、同层 stdout/stderr 日志。
  此轮结果不得进入正式论文表或与既有 30 epoch 结果直接作方法增益比较。

### 3.15 软权重低成本配对筛选完成（2026-10-27）

- 配对筛选于北京时间 **17:53:05** 完成，PID 5784 已结束。两组各完成连续
  5 epoch，CSV 数值有限；共享固定 1024 张源图、1765 个训练 tile 和相同
  val960/recall 操作点，唯一方法变量为 `soft_weights`。
- 保留软权重与去掉软权重结果如下；15% 左右的 mAP 和 28% 左右的 recall 是
  1024 图子集、5 epoch 模型的探索值，不能与已有全量 30 epoch 数字并列：

| 指标 | 保留软权重 / % | 去掉软权重 / % | 差值 / pp |
|---|---:|---:|---:|
| mAP50 | 26.8181 | 26.5932 | −0.2249 |
| mAP50-95 | 15.3751 | 15.4635 | +0.0884 |
| Small recall | 28.0129 | 28.0920 | +0.0790 |

- 去掉软权重后 mAP50-95 与 small recall 同向为正，但绝对值均小于预先固定的
  0.3 个百分点筛选线，mAP50 下降，因此 `promising_direction=null`。**结论为
  中性，不启动全量去软权重 30 epoch 训练**；不能据此说软权重无效。
- 独立审计 27 项检查全部通过：1024 图/1765 tile、两开关、5 epoch、权重哈希、
  完整 val960、548 图 recall 操作点和差值复算均一致；本轮总日志学生耗时
  417.319 秒。记录在 `outputs/screens/p1_soft_weights_1024_e5/`。
- 全量确认配置和入口保留，但只作为之后需要时的人工入口，不自动派发。下一步
  继续下一项低成本候选筛选，仍须先固定筛选信号，完成一个实验后提交并推送。

### 3.16 尺度阈值低成本筛选完成（2026-10-27）

- 沿用相同的 1024 张训练原图。尺度分层组复用软权重筛选中的完整 P1 5 epoch
  臂；新组仅把 small/medium/large 阈值统一为 0.50，软权重和 ignore 均保留。
  `scripts/screen_p1_scale_threshold.py` 从冻结教师候选重划正/ignore，并新写
  对应标签文件，避免复用尺度分层标签；新组为 713 个伪标签 tile、1737 个训练 tile。
- 新组于北京时间 **18:00:37** 完成，学生训练 238.414 秒。独立审计验证 713 条
  候选的状态和标签逐项可从冻结候选复现，5 epoch、权重哈希、完整 val960、
  548 图 recall 与基线均一致，全部检查通过。

| 指标 | 尺度分层 / % | 统一 0.50 / % | 统一−分层 / pp |
|---|---:|---:|---:|
| mAP50 | 26.8181 | 26.7279 | −0.0903 |
| mAP50-95 | 15.3751 | 15.5988 | +0.2237 |
| Small recall | 28.0129 | 24.8899 | −3.1230 |
| Medium recall | 67.1615 | 68.2681 | +1.1066 |
| Large recall | 83.9738 | 85.7545 | +1.7807 |

- 自动候选规则要求 mAP 变化至少 0.3 个百分点且 small recall 同向，本轮因总体
  增量 0.2237 且方向相反，结果为 `promising_direction=null`。不过本研究的主要
  目标是小目标覆盖，统一阈值导致 small recall 下降 3.1230 个百分点、约少匹配
  830 个目标，而中、大目标更好。该现象支持尺度分层对小目标覆盖的预期方向，
  值得进行一次全量 30 epoch 确认，而非直接宣称机制成立。
- 证据：`outputs/screens/p1_global_t050_1024_v2_e5/{comparison,acceptance_audit}.json`。
  结果仍属于 1024 图、5 epoch 探索筛选，不进入论文正式表。

### 3.17 全量尺度阈值确认数据已准备（2026-10-27）

- 新增 `configs/{pseudo_p1_global_t050,train_p1_global_t050_10}.yaml` 和
  `scripts/run_p1_global_t050.py`。全量数据从冻结教师候选重划，不重跑教师：
  4624 个伪标签 tile、27667 个正框、57219 个 ignore 框，合并 11095 个训练 tile。
- `python scripts/run_p1_global_t050.py --prepare-only` 已通过：23781 个冻结输入、
  4624 条 metadata/标签对应关系、训练列表 11095、数据集 YAML 指向和教师摘要均
  通过。生成数据与审计在 `data/visdrone_area10/pseudo/`、
  `outputs/p1_global_t050_pilot30_s0/dataset_audit.json`（忽略目录）。
- 确认配置保留 `soft_weights=true`、`uncertain_ignore=true`，只替换数据且将三个
  尺度阈值统一为 0.50；学生仍为 COCO 初始化、640、30 epoch、batch8、SGD、
  seed0、`close_mosaic=10`。脚本将自动完成训练、val960、分尺寸 recall 和与完整
  P1 对比。
- **当时仅完成数据准备与预检，全量训练尚未启动。**按用户最新约束，给出
  `python -u scripts/run_p1_global_t050.py` 后停止，不自动占用 GPU 跑 30 epoch。

### 3.18 全量尺度阈值确认完成验收（2026-10-27）

- 用户执行全量确认，自动流程于北京时间 **20:00:53** 完成。新 run 为
  `runs/partial10_p1_global_t050_pilot30_s0`；CSV 为连续 1–30 epoch、全部数值有限，
  累计训练日志耗时 6915.15 秒。实际学生超参数与完整 P1 相同，配置差异仅指向
  阈值策略生成的另一数据集；两组共用冻结教师、软权重和 ignore 处理。
- 阈值策略只把 small/medium/large 正阈值从 0.30/0.50/0.50 改为
  0.50/0.50/0.50，从同一教师候选重新生成数据：新组为 4624 个伪标签 tile、
  27667 个正框、57219 个 ignore 框，合并 11095 个训练 tile。完整 P1 为 4839 个
  伪标签 tile、35067 个正框、52117 个 ignore 框、11310 个训练 tile。阈值造成的
  正/ignore 划分差异会改变入训 tile 数和每轮 batch 数，因此不能写成等计算量对照。
- 差值为“统一 0.50 − 完整 P1”，单位为百分点：

| 指标 | 完整 P1 / % | 统一 0.50 / % | 差值 / pp |
|---|---:|---:|---:|
| mAP50 | 39.4990 | 38.6992 | −0.7998 |
| mAP50-95 | 23.7883 | 23.1706 | −0.6177 |
| Small recall | 35.6925 | 35.3501 | −0.3424 |
| Medium recall | 77.5709 | 77.9847 | +0.4139 |
| Large recall | 89.4096 | 89.5033 | +0.0937 |

- recall 操作点仍为 conf=0.25、匹配 IoU=0.50、NMS IoU=0.70、max_det=500，
  两组使用相同 548 张完整原图。small 匹配数由 9486 降至 9395，少 91 个；medium
  和 large 分别由 8622→8668、954→955。该指标不是 AP_small 或官方 AR500。
- 逐类别 AP50-95 为 **1 类升、9 类降**。统一 0.50 仅使 tricycle 提高 0.6425 pp；
  people、motor、bus、car、van、pedestrian 分别下降 1.2516、1.1630、1.0498、
  0.8112、0.8046、0.8024 pp。总体和小目标差异不是由单一类别或中等/大目标支配。
- 独立验收入口 `outputs/audit_p1_global_t050_acceptance_20261027.py` 已通过：24 个
  dataset/run 输入及输出哈希匹配，30 epoch、实际参数、教师和损失开关配对，完整
  val960 与 recall 协议一致；对两组各自 548 图保存预测重新进行 GT 匹配，分尺寸
  计数与报告一致。验收记录为 `outputs/p1_global_t050_pilot30_s0/acceptance_audit.json`，
  run 审计为 `runs/partial10_p1_global_t050_pilot30_s0/audit.json`。
- best.pt / last.pt SHA256 分别为
  `94571aae799119becc033fa6a7296eb0326dbf7fb83e44f650ef7ab488728a4a` /
  `cd96237d62d121ec8132cdfd30dbce94aefe39afd887bfde6a39759ebae3b5ff`。
  结果已接入 `scripts/build_paper_tables.py` 并生成表 4-S2。
- **结论：**全量 30 epoch 与此前 5 epoch 筛选的 small recall 方向一致，但总体
  mAP50-95 方向相反（筛选中统一阈值 +0.2237 pp，全量中 −0.6177 pp）。全量
  结果支持尺度分层相对统一 0.50 同时改善总体检测和小目标覆盖。此节证据为单 split/seed 的
  Ultralytics 开发对照，不改变原 Gate 3，不据此声称尺度阈值首次提出或跨 seed 稳定。

### 3.19 尺度阈值配对学生 seed 复验完成（2026-10-27）

- 用户授权优先补尺度分层阈值的配对 seed，并在完成后更新论文、记录和提交 Git。
  本轮固定 split seed0、同一 B1 教师、两套既有全量伪标签与损失，仅学生 seed
  改为 1；不生成新 split，不扩展正式 100 epoch 主矩阵或其他消融。
- `scripts/run_p1_scale_seed_pair.py` 与两份 seed1 配置已准备。33,047 个冻结文件
  预检通过，统一阈值 metadata/标签对应关系核对通过；测试 36/36。
- 北京时间 20:31 启动顺序流水线，主进程 PID 29248；先运行
  `partial10_p1_pilot30_s1`，随后 `partial10_p1_global_t050_pilot30_s1`，各 30 epoch。
  日志、状态及后续验收位于 `outputs/p1_scale_threshold_pair_s1/`。
- 北京时间 **23:47:09** 两组训练、548 图完整 val960、固定操作点 recall 及自动
  验收完成；两 run 均为连续 1–30 epoch、训练值有限。尺度分层和统一阈值学生
  日志累计耗时分别为 6567.53 和 5080.26 秒，训练 tile 分别为 11310 和 11095。
  实际参数在同 seed 内配对，相对各自 seed0 仅学生 seed 和 run 名变化。
- seed1 差值为“尺度分层 − 统一 0.50”：

| 指标 | 尺度分层 / % | 统一 0.50 / % | 尺度−统一 / pp |
|---|---:|---:|---:|
| mAP50 | 39.5056 | 38.9834 | +0.5222 |
| mAP50-95 | 23.6126 | 23.3706 | +0.2420 |
| Small recall | 36.1403 | 35.4969 | +0.6434 |
| Medium recall | 77.6428 | 77.3639 | +0.2789 |
| Large recall | 88.9410 | 89.5033 | −0.5623 |

- small 匹配数为 9605 对 9434，多匹配 171 个小目标。seed1 逐类 AP50-95
  在统一阈值下为 2 升/8 降，仅 car、truck 上升；medium recall 的相对方向与
  seed0 不同，large recall 在两个 seed 中均为统一阈值更高，不作全面改善表述。
- seed0/1 的 mAP50-95 和 small recall 均支持尺度分层。种子内“尺度−统一”
  的平均配对增益±样本 SD（ddof=1）分别为 **+0.4299±0.2657 pp** 和
  **+0.4929±0.2128 pp**；两方案各自的均值与 SD 及全部单次值见表 4-S3。
- `scripts/audit_p1_scale_seed_pair.py` 独立验收 **36 项检查**通过，重新核对
  33,047 个冻结文件、教师/伪标签/损失、best/last 和完整 val 协议，并重放两组
  各 548 图保存预测进行 GT 匹配。验收与两组权重 SHA256 见
  `outputs/p1_scale_threshold_pair_s1/{acceptance_audit,independent_audit,comparison}.json`
  和论文证据索引；执行脚本与方法源码在训练期间未改变。
- 论文表生成器已接入表 4-S3，实验第 4.11 节及证据/复现路径同步更新。
  最终 `build_paper_tables.py --check` 核对 59 个来源，6 组 recall 共用 GT 计数；
  `python -m pytest --basetemp outputs/pytest_scale_seed_pair_final_recheck` 36/36 通过。
  这不是烟测；结论限于单 split、固定教师下两个学生 seed，仍是 Ultralytics
  开发评测，不等同于 AP_small、统计显著或跨 split 泛化，不改变原 Gate 3。

### 3.20 B2 学生 seed1 与主对照案例图（2026-10-30，验收完成）

- 用户授权补 B2 学生 seed1 并整理案例，完成后验收、更新论文及本地 Git 提交。
  固定 split seed0、原 B1 教师、B2 的全局 0.50 伪标签和标准损失；30 epoch、
  640、batch8、SGD、COCO 初始化及 `close_mosaic=10` 沿用已有 B2 pilot。
  P1 seed1 已完成，直接复用；不训练 B1 seed1，不扩大正式 100 epoch 矩阵。
- 10 月 28 日 09:27 的原进程 PID 42232 已退出，日志停在第 3 epoch，CSV 仅有
  连续 1–2 两行，无异常栈，原因尚不能由日志确定。10 月 30 日核实无对应训练
  进程后，将原 run 与流水线目录的 10 个文件、来源脚本/配置和哈希归档至
  `outputs/b2_seed1_interrupted_20261030_091251/`。中断权重不作为论文结果。
- 10 月 30 日 **09:13:39** 以未修改的执行脚本和原初始化重新启动，PID 31944；
  37,695 个冻结输入再次通过预检。`outputs/b2_pilot30_s1/recovery.json` 链接归档；
  新运行不拼接旧 epoch，不覆盖已完成 run。
- 10 月 30 日 **10:10** 心跳发现 PID 31944 再次消失：日志最后更新于 10:07:29，
  停在第 16 epoch，CSV 完成 1–15，stderr 为空；操作系统未重启，也没有能确认
  训练退出原因的应用异常记录。将该次 12 个文件、原脚本和配置/哈希另行归档至
  `outputs/b2_seed1_interrupted_20261030_101330/`，原档案链均保留。
- **10:15:05** 改由本机 `Win32_Process.Create` 启动 yolo 环境的 `pythonw.exe`，
  PID 14304，父进程为 WMI 服务进程而非 Codex 子进程。仅启动和日志包装不同，
  仍执行原 `scripts/run_b2_seed1.py`、相同配置/源码、同一初始化，37,695 文件
  预检再次通过；从 epoch1 重跑连续 30 epoch。`launch_receipt.json` 保存启动包装
  文件及哈希，独立验收递归复核两次归档链。该次运行已按用户要求停止并归档，转交终端手动训练；当时尚未验收，后续完整结果见下文。
- 流水线自动完成 548 图完整 val960、固定操作点 recall，以及 B2/P1 seed0/1
  四组各 548 图保存预测的 GT 匹配重放。随后运行 `scripts/audit_b2_seed1.py`，
  核对连续 1–30 epoch/有限值、实际参数配对、教师/伪标签/损失、best/last 和评估
  哈希，再生成表 4-S4 的单次值、各方法均值±样本 SD 及种子内配对差均值±SD。
- 已在新增 B2 seed1 结果出现前，从冻结 seed0 验证预测选定四类案例：P1 新检出、
  P1 丢失 B2 已检出、共同漏检、P1 未匹配预测。选图按原验证顺序及固定事件顺序，
  取首个合格目标，并排除重复文件名前缀和相同图像内容；验证集本身不变。
  总览为 1776×1910、300 dpi PNG，输入/输出哈希及裁剪坐标均记录在 manifest；
  详见 `docs/paper/case_studies.md`。图像、日志和权重均留在忽略目录。
- seed0 的 26,577 个 small GT 中，两者均检出 8,151、仅 P1 1,335、仅 B2 836、
  均漏检 16,255，净多匹配 499。案例解释覆盖成功与失败，不据此推断因果或
  官方错误分类；主结果继续以完整验证集指标为准。原 Gate 3 不变。

- **10:27** 按用户要求停止后台 PID14304，原 run 的 13 个文件及来源链保存在
  `outputs/b2_seed1_interrupted_20261030_102720/`；前两次归档也完整保留，心跳暂停。
  **10:33:25** 用户终端直接执行原入口，从同一 COCO 初始化开始，最终 CSV
  连续 1–30 epoch 且全部数值有限，累计训练日志耗时 5845.51 秒。
- **12:11:53** 完整训练、548 图 val960、recall 和自动验收完成。新增 seed1：

| 指标 | B2 seed1 / % | P1 seed1 / % | P1−B2 / pp |
|---|---:|---:|---:|
| mAP50 | 38.9843 | 39.5056 | +0.5214 |
| mAP50-95 | 23.3688 | 23.6126 | +0.2438 |
| Small recall | 34.4471 | 36.1403 | +1.6932 |
| Medium recall | 76.3383 | 77.6428 | +1.3045 |
| Large recall | 89.5970 | 88.9410 | -0.6560 |

- P1 seed1 多匹配 450 个 small（9605 对 9155）、145 个 medium（8630 对 8485），
  少匹配 7 个 large（949 对 956）。seed0/1 的 mAP50、mAP50-95、small/medium
  recall 均为 P1 更高，large recall 则一升一降；seed1 逐类 AP50-95 为 6 升/4 降。
- 两 seed 种子内 P1−B2 的 mAP50-95 平均差±样本 SD 为
  **+0.2893±0.0644 pp**，small recall 为 **+1.7854±0.1304 pp**（ddof=1）。
  B2/P1 各自 mAP50-95 均值为 23.4111±0.0598% / 23.7005±0.1243%；small recall
  为 34.1310±0.4470% / 35.9164±0.3166%。全部单次值/指标见表 4-S4，不选报。
- 独立验收 **116 项检查**通过：37,695 个冻结文件、四 run 连续 30 epoch/有限值、
  实际参数及原初始化配对、同教师/既有伪标签/损失、best/last 与评估哈希、完整
  548 图/38,759 GT 和固定 recall 操作点、四组保存预测 GT 重放、独立配对统计；
  三层归档的 35 个文件和执行脚本/配置快照均核对。中断日志不拼接、不入论文。
- `build_detection_cases.py --check` 再核对四类案例、1,106 输入哈希及图像输出。
  论文表 4-S4、第 4.12–4.13 节、证据和复现入口均已同步；表格核对 123 个来源、
  7 组不同 recall 目录；最终 `python -m pytest --basetemp outputs/pytest_b2_seed1_final_20261030_1308`
  **39/39 通过**，`git diff --check` 通过。B1 未重复，原主表仍为 seed0；仅是单 split/固定教师
  下两个学生 seed 的开发复验，不等同于独立教师、多 split、显著性或官方 AP_small，
  不完成正式 100 epoch 主矩阵，也不改变原 Gate 3；不自动启动新 seed/消融。

### 3.21 三篇文章、学位论文与树莓派部署目标（2026-10-30，仅规划）

- 用户希望围绕相关成果规划三篇文章，最高按北核层级，最终模型需要轻量化并部署树莓派。
  用户补充论文按录用计算，专利按“申请通过”；该词暂不擅自解释成受理或授权。
  学校/专业、考核截止日期、作者/单位及专利类型/认可节点待核实。
- 新增 `docs/publication_and_edge_deployment_plan.md`：A 局部标注半监督检测、
  B 漏检感知可靠补全、C 蒸馏压缩与树莓派部署；三者须有独立问题与新增证据，
  不将已有阈值/权重/ignore 三模块拆成三篇。实际顺序 A→C→B，先保障毕业必需
  成果和硬件落地；暂按两篇录用论文+学位论文，或一篇录用论文+两件认可专利+
  学位论文理解，最终仍以学院要求为准。
- 硬件未回复，排期暂按树莓派5/8GB/纯CPU/图像检测，明确是待确认假设。
  先验证未剪枝模型/原生小模型的导出与上板，再选择结构化剪枝和蒸馏；FP32、
  后端支持的量化、CPU/NPU结果分开报告。完整原图精度、端到端时延、P95、内存
  和热稳定性均纳入验收，不能用GPU速度或单tile速度代替树莓派整图结果。
- 另给三篇文章边界、学位论文章节和约20周相对研发排期，审稿/录用等待不作固定
  时长承诺；专利先核对公开内容与申请顺序。LoRA/DINOv3/CLIP均为候选工具，
  不直接假定有效或同时接入，不覆盖原Gate3判定，不自动启动训练或购买硬件。

### 3.22 第一篇论文实验缺口与低成本排序（2026-10-30）

- 用户决定先推进第一篇方法稿，要求先梳理缺口、低成本优先，只有出现成果后再投入；
  需要训练时由用户运行。已新增 `docs/paper1_experiment_gaps.md`。
- 首项零训练漏检诊断完成。工具 `scripts/audit_paper1_gaps.py` 对 6,471 个未标注 tile 的
  38,245 个 hidden GT 精确重放既有 P1 正伪标签及 25,669 个真匹配；核对 11,319 个输入哈希。
  small 未匹配 hidden GT 8,984 个，其中同类别 ignore IoU≥0.50 3,652、异类 IoU≥0.50 1,644、
  同类别 IoU 0.10–0.50 1,243、无保存候选 IoU≥0.10 1,597。类别为描述性证据，不能证明
  ignore 可独立转正或原因归于 NMS；候选文件已是 NMS 后结果。
- 计划依次做：零训练“可恢复增量/候选覆盖”检查；仅在信号充分时运行一次 5 epoch 配对筛选；
  筛选过线后才补完整 30 epoch、seed0/1 确认。B0 和新 split 均是更昂贵、按投稿需要决定的
  后续验证，不是当前应立即启动的任务。软权重全量状态先审计，不能把短程筛选写成正式消融。
- 本轮未启动训练、推理或伪标签生成；原 Gate3 和已有正负结果不变。

### 3.23 Ignore 可恢复增量分析完成（2026-10-30，零训练）

- 用户明确先做 ignore 可恢复增量，不训练；只有明确改进方向后再交训练命令。
  新增 `scripts/audit_ignore_recovery.py` 与 `docs/paper/ignore_recovery_analysis.md`，
  用类别一致/IoU≥0.50 的旧 positive 匹配锁定基线，ignore 仅与剩余 GT 联合一对一匹配。
- 当前 4,839 入训 tile 的 52,117 个 ignore 框覆盖 5,340 个剩余 GT；普通贪心新增
  匹配 5,303，锁定旧匹配后的最大基数容量上界 5,315；贪心新增 small 为 3,627。
  3,490 个 ignore 框只覆盖旧匹配 GT；新增独立匹配/新增框仅 10.18%。
- 仅按预测 small 且 conf≥0.25 保留原 ignore，额外 2,548 框新增独立匹配 691，
  其中 small 为 672、medium 为19，新增匹配比例 27.12%。0.20/0.15/0.10/0.05
  对应比例 22.85/18.76/14.60/9.55%。不能将其当作总体 precision 或学生 mAP。
- 排除 tile 的 4,152 ignore 框单列，普通新增187、条件上界188，不能计入当前训练。
  最大匹配按总量求解，各尺寸计数不代表单独最大化各尺寸的上界；所有 oracle 仅诊断，
  隐藏GT不用于训练标签。旧正框可重分配的 union IoU 贪心结果另存，不混用。
- 存在可恢复空间，但尚无不依赖 GT 的可靠转正规则；当前不启动扩训。下一步可以
  考虑固定小样本的跨视图支持核验，仍先测伪标签精度—覆盖，不直接训练学生。
  软权重全量30epoch训练/val/recall/派发均核实不存在，缺口保留，不重复中性短程筛查。
- 工具复核 11,322 输入哈希，完整6,471 tile计数；新匹配函数经过全部2×3二部图
  的穷举容量对比和重复/空输入等测试。Pyright初始化修正前的脚本/输出和来源已归档，
  计数算法不变，两份重生成 analysis.json 已与旧产物逐字节核对一致。全套 pytest 46/46通过、
  两诊断脚本 Pyright 0错误/0警告，既有论文表123来源/7组recall仍通过。当前训练方法和已验收指标未改。

### 3.24 跨视图与候选保留筛查后停止扩训（2026-10-30）

- 已完成 256 tile 翻转探索及另取 128 tile 的 NMS/max-det 检查；后者排除前者
  全部下标，原入训 89/排除 39 tile 分列，冻结教师和原配置在全部128 tile重现。
- NMS IoU 0.70→0.90：原入训全候选1647→3438、GT匹配617→628；正框650→860、
  匹配500→508，precision76.92%→59.07%。新增210正框在锁定基线匹配后仅新增9个
  small GT，新增匹配比例4.29%；按原正框同类IoU≥0.70过滤后仅余2框/1新增匹配。
  max-det500→1000无变化，原配置单tile最多213候选。
- 原初分析误将与任意原候选重叠者全部排除，口径过于保守；保留初始源码/预测/manifest，
  v2复用推理，区分新增预测身份、未过滤恢复容量和正框重叠过滤，不重新推理。
  原始和修订来源均保存于忽略目录，详见 `docs/paper/candidate_retention_analysis.md`。
- 两方案均未达到探索性继续条件，故不训练、不输出新训练标签。只读重放入口为
  `python scripts/audit_candidate_retention.py --check`，成本和结果在
  `outputs/candidate_retention_128_20261030_v2/`；不改变原Gate3。
- 全套pytest52/52通过，359个输入哈希及全部128 tile重放通过；既有论文表
  123个来源与7组recall计数核对通过，原训练配置和正式结果未改。
- 用户最新授权替代先前“只由用户手动启动”安排：可靠信号出现后可自主进行至多
  30 epoch有界训练，启动时建临时进度心跳，完成/失败/需处理时通知、结束清理；
  更长训练仍由结果决定。每轮修改中文提交。本轮无训练，无需创建进度定时器。

### 3.25 跨尺度支持确认与总安排更新（2026-10-30）

- 用户指定`docs/publication_and_edge_deployment_plan.md`为总安排依据；新增第10节
  当前工作队列、职责和完成定义，替换旧的全部训练由用户手动启动规则；
  按用户最新收束要求，当前目标只做A，C/B为远期规划，不并行执行。
  Codex负责可靠信号后的至多30epoch有界小试与临时监控，用户负责有证据后的更大规模训练。
- 新增GT-free候选支持函数，保持原正框、原ignore坐标与置信度，按同类/同尺度候选
  的640视图支持与正框去重决定是否可转正；隐藏GT只用于离线计数，不输出训练标签。
- 原256 tile探索中，floor0.25/support0.30新增22框、11个small匹配，达到探索继续线。
  固定该唯一候选后，在排除既有384 tile的另256 tile确认；192个原入训tile新增23框、
  9个small匹配，新增匹配比例39.13%，未达继续线，不改选其他规则。
- 本轮未启动学生训练；保留探索和确认的预测/manifest/成本，入口为
  `scripts/audit_scale_support.py`，报告见`docs/paper/scale_support_analysis.md`。
  当前仍未找到可交给用户扩大训练的新方向；下一项先核对A的必要消融、基线和
  正式训练预算缺口，不继续横向扩展新教师或B类补全机制。
- 全套pytest55/55通过，探索722/确认719个输入哈希与各256tile只读重放通过。

### 3.26 A的软权重单因素补证据（2026-10-30，已验收）

- 收束到A后，优先补现有稿件软权重开关的完整30epoch消融；旧1024图5epoch中性
  筛查不追认为过线。用户最新至多30epoch自主小试授权用于这一组必要机制检验，
  不是启动新教师、补全论文或大规模矩阵。
- 原入口/配置预检通过23781冻结文件、11310 tile；只关闭soft_weights，保留
  原教师、数据、增强、初始化与ignore。预算一组30epoch，复用已完成完整P1参考。
- 新增后台包装和独立验收：保存父/子PID与退出码，检查连续epoch、实际参数、
  评估/权重哈希，并重放两组548图预测。全套pytest56/56通过。
- 预算/预先判据见`docs/paper1_experiment_gaps.md`第7节。北京时间2026-10-30
  14:30:20实际启动，包装PID37236、训练PID34612，命令与创建时间见
  `outputs/p1_no_soft_weights_pilot30_s0/launch_receipt.json`。
  临时心跳`a-30`每10分钟核查进程身份、CSV和阶段，只在完成、失败或需处理时通知，
  完成/失败处置后删除；不自动重启或扩训。本组于16:28:18正常退出，exit_code=0。
- 连续30epoch、完整val960/固定操作点recall、实际参数配对、23783个冻结输入及
  两组各548图预测重放通过16项独立验收。CSV累计训练7016.44秒；
  入口`python scripts/audit_p1_no_soft_weights.py`，证据同目录`independent_audit.json`。
- 去软权重−完整P1：mAP50−0.0585pp、mAP50-95−0.1623pp，
  small recall+0.9143pp（+243），medium+0.6658pp（+74），large−0.3749pp（−4）；
  逐类别AP50-95为3升7降。整体mAP与召回有取舍，不满足预先记录的后续线，
  不启动100epoch或追加训练；保留完整P1主配置，不改写旧5epoch中性筛查或原Gate3。
- 表4-S5、实验4.14和证据/复现记录已更新；表格核对137个来源/8组recall，
  全套pytest56/56通过。单split/固定教师/学生seed0，不作显著性或官方AP_small主张。
- 本轮训练和验收均结束，临时心跳`a-30`已删除；未启动新训练或长期监控。

### 3.27 A的关键词文献调研（2026-10-30）

- 使用paper-search MCP围绕漏标补全、定位不确定性、损失和基础模型查询原论文。
  7份PDF及SHA256清单位于忽略目录；另外5篇目前仅核对元数据/摘要，阅读范围
  和固定版本逐项见`docs/paper/a_keyword_research.md`，不虚构全文阅读。
- 优先候选是“固定现有正伪框，仅检验跨视图定位稳定性能否预测定位质量”，
  先做离线诊断，尚未实现或训练。明确与Soft Teacher、RUPL、UTv2重叠，
  NWD只列为独立备选，不先验宣称创新或收益。
- 第3.26节软权重消融已完成，未达到扩大训练条件；LoRA/DINOv3/CLIP未下载模型或运行，
  蒸馏后结构剪枝留在C。下一步可检验既有正伪框定位稳定性，当前仅文献假设，
  尚未执行该诊断或启动第二项训练。

### 3.28 已有正伪框定位可靠性（2026-10-30，诊断完成）

- 下一项只复用两个不重叠256tile缓存，检验跨尺度定位稳定性在控制置信度、
  类别组成后是否区分已有正框的定位质量，不增加标签、推理或训练。
- GT-free匹配、固定分层及继续线见`configs/positive_localization_diagnostic.yaml`和
  `docs/paper/positive_localization_analysis.md`；协议由`2a42fa5`先提交，再执行诊断。
- 已完成2710个正框、1442来源哈希的逐框重放。探索/确认small正框906/641，
  稳定822/570、不稳定各6、无匹配50/51；没有足够分层可比样本，未通过继续线。
  稳定组同类GT IoU≥0.75仅52.7981%/49.4737%，不能由同教师多尺度稳定推出正确。
- 无匹配组仍有20.0000%/23.5294%原一对一匹配，不直接剔除；
  不合并无匹配组重试、不改组阈值，不实现或训练新回归加权。
  全套pytest61/61通过；只读复核`python scripts/audit_positive_localization.py --check`。

### 3.29 原图上下文类别纠正（2026-10-30，探索失败后停止）

- 跨尺度定位诊断停止后，下一项只检查新的观察范围：同一B1教师的原图960预测
  映射回既有未标注tile，能否纠正低置信度small正伪框的类别。
  不增加正框、不改几何/置信度/ignore，不引入新教师、模块或训练。
- 协议`docs/paper/context_rectification_analysis.md`及
  `configs/context_rectification_diagnostic.yaml`固定128原图探索、过线后另128图确认；
  排除历史640尺度与NMS样本。类别冲突/匹配选择无隐藏GT输入；
  隐藏GT只离线计数，未导出训练标签。与CrossRectify/密集裁片SSOD重叠已说明。
- 入口`python scripts/audit_context_rectification.py`；`--check`只读验收；
  确认组`--confirm`必须先重放并通过探索继续线。协议和代码先提交再推理。
  协议/代码由`ef8f565`先提交，实际推理commit为`f1c997e`。
- 唯一128图探索完成：98个已接受tile、799正框，只触发1次类别纠正，
  将正确van改为car；总体匹配562→561、small302→301，继续线失败。
  推理含warmup4.3874秒、新增训练0epoch；不执行确认或更改阈值追试。
  372来源/全部决策与GT统计只读重放通过，确认入口拒绝失败探索已实际验证。
  结果与限制见`docs/paper/context_rectification_results.md`。
- 预检修复历史`data/`忽略规则屏蔽四个数据源码的问题；原文件内容不改，
  Git归档核对源码齐全且不含生成数据，全套pytest74/74通过。
- Zotero已按用户要求调整为“我的文库”下并列的`YOLO-RegionSSOD`和
  `YOLO-VisDrone Research`；前者28篇均有PDF，S³OD附件为注明的2023早期预印本。
  文献清单见`docs/paper/zotero_literature_library.md`；扩展关键词归档不代表启动B/C。

### 3.30 开放区域拟合的置信度校准（2026-10-30，探索未过线）

- 基于已读Calibrated Teacher方法页，单独检验分类置信度可靠性；不复现其在线EMA、
  IoU队列或FIoU，不宣称校准概念新颖。执行前协议
  `docs/paper/confidence_calibration_protocol.md`和固定配置先提交，再推理。
- 512个开放标注tile拟合全局分数段/类别×尺寸单调收缩映射；拟合仅用已有
  `labeled/labels`，与探索/确认源图不重叠，不用隐藏GT拟合或调整分数段。
  探索/确认各256个未标注tile，仅复用已有正伪框；类别/几何/原分数/ignore不改。
- 八项继续线同时要求分组Brier优于原分数和全局简化对照、small同向、
  足够样本/覆盖且错误权重占比不明显恶化。探索失败不执行确认，不调规则追试。
  入口`python scripts/audit_confidence_calibration.py`，`--check`只读重放拟合和诊断。
  执行前由`f3d77cd`提交；全套84/84tests通过。
- 已完成512开放tile推理（8.1703秒）和7548正框拟合；不重叠256tile探索中
  194接受tile/1421正框（small816），分组映射覆盖96.4110%。
  分组Brier整体0.160310，高于原分数0.158864；虽small由0.183221降至0.176334，
  仍不满足预先整体继续线，未运行确认或学生训练，也不改成small-only追认通过。
- 拟合1045来源/7548正框、探索474来源/1421正框只读重放一致，
  失败探索的确认阻断已实测；结果见`docs/paper/confidence_calibration_results.md`。
  当前两项新规则均不支持扩训，后续须有新的可验证信息，不在同批隐藏GT上调参追试。

### 3.31 小框混合NWD回归筛查（2026-10-30，完成后停止）

- 在先前两项规则失败后，单独检验几何损失；原NWD论文式7–9及Sec.3.3已核对，
  其已包含VisDrone及回归/NMS/分配应用，不把接入NWD称为原创。
  固定协议`docs/paper/small_nwd_protocol.md`先提交，再估计C和执行。
- 仅增强后640像素空间面积<32²的目标框回归项改为0.5CIoU+0.5NWD；
  C来自同一1024子集开放GT resize后的平均small边长，不读隐藏GT或val调参。
  新增独立模块，不修改原P1/runner；TAL、DFL、分类、ignore及标签集合保持固定。
- 复用已验收1024源图/1765tile的完整P1独立5epoch参考，只新增一组5epoch，
  配置`configs/screen_p1_small_nwd_5ep.yaml`。继续线为同预算val960
  mAP50-95≥+0.30pp且small recall≥0；未过线停止，不自动30/100epoch。
- 当前全套pytest94/94；stride/像素换算、零/非重叠/包含梯度、AMP类型、
  空前景、DFL与分类不变、关闭恢复P1及继续线测试通过。
  已通过23794个冻结输入预检；1024开放tile的7519个resize后small框得到
  C=20.22804306训练像素。真实YOLO11s、2×3×640×640、AMP前反传通过，
  255个梯度张量均有限，优化器更新0步；证据
  `outputs/screens/p1_small_nwd_1024_e5/{preflight,geometry_gpu_check}.json`。
  2026-10-30 18:16:07实际启动唯一5epoch，包装PID32304、训练PID28860，
  记录`outputs/screens/p1_small_nwd_1024_e5/launch_receipt.json`；
  命令/创建时间经Get-CimInstance核实，CSV实际增长。18:19核查时已完成2epoch，
  stderr为空，仍在原训练，不将中途640指标当作最终val960结果。
  包装于18:22:23正常退出，exit0；连续5epoch累计299.989秒、完整val960/recall完成。
  相对同子集P1：mAP50-95−0.1746pp、small recall+1.5464pp、
  medium+0.0630pp、large−0.7498pp；未达继续线，不扩训或调参追试。
  独立核验23794来源、源码/参数/产物哈希与两组各548图预测匹配通过；
  详细结果与限制见`docs/paper/small_nwd_results.md`。
  临时心跳`a-nwd`收尾后删除，不将本组结果写入正式主表。

### 3.32 至少30epoch判断性能与NWD全量确认（2026-10-30）

- 用户指出短程可能不能体现最终性能，并明确至少30epoch后再判断改进。
  该要求覆盖此前“5epoch筛查未过即停”的后续安排，历史结果与门槛不改写。
  当前自主单候选固定30epoch；更短仅工程检查，更长仍交用户执行。
  用户另授权中文Git提交后push，替代此前不push偏好。
- 缓存548图离线重放：固定conf0.25/IoU0.50，NWD小目标+411、总FP+888；
  等FP后small recall仍+0.6321/+0.7713pp（P1 conf0.25/0.50两档预算）。
  IoU0.75为+0.2521/+0.3951pp，IoU0.90为−0.0489/−0.0564pp。
  10类AP为5升5降；仅事后描述，缓存conf≥0.25，不生成部署阈值或重建完整AP。
  详见`paper/small_nwd_results.md`，重放入口
  `python scripts/audit_small_nwd_tradeoff.py --check`。
- 独立登记`paper/small_nwd30_protocol.md`，只新增完整11310tile的NWD学生30epoch，
  与已验收完整P1 seed0/30epoch配对；从原COCO权重初始化，
  close_mosaic10，不续接五轮。原NWD源码及C=20.22804306/mix0.5/面积边界均冻结。
  新入口`python scripts/run_p1_small_nwd.py --check-only`、后台包装与独立验收已实现；
  当前尚未启动，不提前声称性能改善。
- 后续线在30epoch未知时登记：mAP50-95≥+0.30pp且small recall非负，
  或small recall≥+1.00pp且mAP50-95≥−0.10pp。采用与既有30epoch
  软权重消融相同的两路线；过线才准备更大训练交接，不自动扩训。
  工程检查108/108 tests；新增五个Python模块Pyright零错误/警告。
  缓存诊断1111来源SHA256及两组预测精确重放通过，论文表137来源/8组recall核对通过。

## 4. 固定实验协议

### 4.1 训练与评估

| 项目 | 固定值 |
|---|---|
| 检测器 | YOLO11s |
| 初始化 | COCO 预训练 `D:\Python\YOLO\yolo11s.pt` |
| 训练输入 | 640 |
| 主评估输入 | 完整原图，960 |
| 优化器 | SGD，`lr0=0.01`，`lrf=0.01` |
| 训练轮数 | 主实验 100 epoch |
| batch | 8；OOM 时所有对照统一降低 |
| 其他 | `close_mosaic=10`、`patience=0`、`workers=0` |
| 数据 | VisDrone train/val |

主实验阶段不修改网络结构、损失权重和增强策略。任何额外模块必须先通过
30 epoch pilot，再进入 100 epoch 主矩阵。

### 4.2 区域标注协议

- 5%/10%/20% 表示标注面积比例，不是图片数量比例。
- 每张训练图选择一个目标中心的 square tile，允许最多 tile 边长 20% 的抖动。
- 框与 tile 的可见面积比例低于 0.35 时丢弃。
- 每张图额外采样一个低重叠未标注 tile，其隐藏 GT 只用于分析，不能进入训练。
- `manifest.jsonl` 必须保留 tile 坐标、保留框数、隐藏 GT 数和随机种子。
- 标注 split seed 与训练 seed 必须成对记录。

### 4.3 主指标

- 主指标：mAP50-95、mAP50、AP_small、AR500。
- 分析指标：按尺寸的伪标签 precision/recall/覆盖率、每类 AP、隐藏 GT 漏检数、
  参数/FLOPs/FPS。
- 报告均值 ± 标准差；同一 split 做配对比较。
- 预注册主结论：10% 标注面积下，P1 相对 B2 至少 `+1.0 mAP50-95` 或
  `+1.5 AP_small`，且至少两个 seed 方向一致。
- Ultralytics mAP 只能作为开发指标；最终表必须补 VisDrone 官方 AP 或明确说明
  评测工具差异。

## 5. 方法矩阵

| 名称 | 训练数据 | 论文作用 |
|---|---|---|
| B0 Full-sup | 完整 VisDrone 标签 | 性能上界，不作为方法变量 |
| B1 Partial-sup | 仅标注 tile | 回答“直接丢弃未标注区域损失多少” |
| B2 Naive self-training | 标注 tile + 全局阈值 0.50 伪标签 tile | 判断半监督信号是否存在 |
| P1 Scale-aware | 标注 tile + 尺度分层阈值 + 置信度软权重 + uncertain-ignore | 论文主方法候选 |

P1 第一轮只做一次离线自训练。只有 P1 明确优于 B2，才考虑第二轮或在线 EMA
教师；不得用不断增加模块来掩盖 pilot 失败。

## 6. 8 周执行计划

| 周次 | 日期建议 | 核心任务 | 必须交付 | 进入下一周的条件 |
|---|---|---|---|---|
| 第 1 周 | 10/26–11/01 | 正式 5/10/20% seed0 数据；B0/B1 启动；论文骨架 | `summary.json`、split/manifest、5 epoch 曲线、B1 seed0 训练已启动、引言/相关工作骨架 | 数据审计通过；20 图与正式 split 链路均可复现 |
| 第 2 周 | 11/02–11/08 | 10% B1/B2 30 epoch pilot；伪标签尺寸诊断 | pilot 结果表、伪标签 precision/recall/覆盖率图、失败案例、方法章节初稿 | B2 相对 B1 不低于 `-0.2 mAP50-95`；否则先诊断漏检 |
| 第 3 周 | 11/09–11/15 | 实现 P1 尺度阈值、软权重和 uncertain-ignore | 单元测试、P1 配置、30 epoch 对照、方法流程图 | P1 相对 B2 至少 `+0.5 mAP50-95`，AP_small 或小目标 recall 同向改善 |
| 第 4 周 | 11/16–11/22 | 10% 标签完整 100 epoch，seed 0；启动 seed 1 | B1/B2/P1 主结果、吞吐和显存记录、实验章节表格模板 | P1 未出现数据泄漏或明显的训练不稳定 |
| 第 5 周 | 11/23–11/29 | 10% seed 1/2；5%/20% 预算实验 | 完整主矩阵初稿、均值 ± 标准差、标注预算曲线 | 主方法在两个 seed 上方向一致 |
| 第 6 周 | 11/30–12/06 | 消融、阈值敏感性、第二轮自训练 | 尺度阈值、ignore、软权重、轮数消融；阈值曲线 | 不增加新模块；只回答核心问题 |
| 第 7 周 | 12/07–12/13 | 冻结版本；官方/外部确认；补 AP_small 和效率 | 最终表、权重 SHA256、评测日志、冻结配置、案例图 | 所有数字都来自冻结版本 |
| 第 8 周 | 12/14–12/20 | 图表、论文全文、复现实验包、内部审阅 | 论文初稿、图表源文件、README、配置归档、答辩材料 | 他人能按 README 复现最小实验 |

如果 GPU 占用或训练排队导致延期，优先保持“数据→pilot→主方法→消融→写作”的
顺序，不要为了赶日期同时启动未筛选的模型。

## 7. 实验矩阵与运行顺序

主矩阵固定 YOLO11s 640/100 epoch：

| 方法 | 5% | 10% | 20% |
|---|---:|---:|---:|
| B0 Full-sup | 参考 | 参考 | 参考 |
| B1 Partial-sup | seed 0,1 | seed 0,1,2 | seed 0,1 |
| B2 Naive self-training | seed 0,1 | seed 0,1,2 | seed 0,1 |
| P1 Scale-aware | seed 0,1 | seed 0,1,2 | seed 0,1 |

建议运行顺序：

1. 第 1 周启动 B0 和 10% B1 seed0。
2. 第 2 周完成 10% B1/B2 30 epoch pilot。
3. 第 3 周只训练 P1 pilot，不扩到全矩阵。
4. 第 4 周完成 10% seed0 的 B1/B2/P1 100 epoch。
5. 第 5 周按“已有结果价值”排序，补 seed 和 5%/20% 预算。
6. 第 6 周只跑预先登记过的消融。

完整矩阵约需 150–220 GPU 小时；单张 RTX 4070 Ti 连续运行约 2–3 周。每个
100 epoch run 预计 6–10 小时，实际以本机测得时间为准。

## 8. 论文写作计划

| 论文章节 | 主要证据 | 最晚完成时间 |
|---|---|---|
| 摘要、引言 | 标注成本、未标注不等于背景、研究问题和三点贡献 | 第 4 周初稿，第 8 周定稿 |
| 相关工作 | SSOD、稀疏/部分标注、航拍小目标、伪标签 | 第 1 周骨架，第 3 周补全 |
| 方法 | tile-aware 流程、尺度阈值、软权重、uncertain-ignore | 第 3 周完整初稿 |
| 实验设置 | 数据协议、B0–P1、固定参数、评测工具 | 第 2 周模板，第 7 周冻结 |
| 主结果 | 10% 主表、5%/10%/20% 曲线、均值±标准差 | 第 5 周 |
| 消融与机制 | 四组消融、阈值曲线、伪标签精度/召回/漏检 | 第 6 周 |
| 讨论与局限 | 可见性阈值、合成区域协议、官方评测差异、外部集限制 | 第 7 周 |
| 结论与复现包 | 结果边界、失败案例、README、SHA256 | 第 8 周 |

### 8.1 必做图表

- 图 1：原图、区域标注 tile、未标注 tile 和隐藏 GT 的协议示意图。
- 图 2：B1 教师、B2/P1 伪标签筛选和学生训练的流程图。
- 图 3：5%/10%/20% 标注面积与 mAP/AP_small 的预算曲线。
- 图 4：按尺寸的伪标签 precision、recall、coverage 和漏检率。
- 图 5：成功/失败案例与 gate/uncertain-ignore 可视化。
- 表 1：数据与隐藏 GT 统计。
- 表 2：B0/B1/B2/P1 主结果。
- 表 3：尺度阈值、ignore、软权重、轮数消融。
- 表 4：参数量、FLOPs、FPS 和显存。

### 8.2 投稿前的最低完成标准

- [ ] 10% 标签下 B1/B2/P1 的 100 epoch 主结果和至少两个 seed。
- [ ] 5%/20% 预算曲线，或者明确说明未完成的原因和结论边界。
- [ ] 至少三组核心消融：尺度阈值、uncertain-ignore、置信度软权重。
- [ ] AP_small、伪标签质量和隐藏 GT 漏检分析。
- [ ] 官方 AP 或明确的 Ultralytics/官方评测差异说明。
- [ ] 所有表可由冻结配置、权重哈希和日志追溯。
- [ ] 论文、图表源文件、README 和复现实验包归档。

## 9. Go/No-Go 门槛

### Gate 1：工程可复现

20 张图的 split 生成、5 epoch 训练、完整 val 推理均成功；同 seed 生成结果一致；
9 项以上单元测试通过。

### Gate 2：半监督信号存在

10% 标签、30 epoch pilot 中，B2 相对 B1 不得低于 `-0.2 mAP50-95`。若下降，
先检查教师伪标签 recall、未标注 tile 密度和阈值，而不是继续加模块。

### Gate 3：主方法有独立收益

30 epoch pilot 中，P1 相对 B2 至少 `+0.5 mAP50-95`，且 AP_small 或小目标 recall
同向改善。未达到时停止扩训，优先改单一机制或转诊断分析。

### Gate 4：完整训练可成立

100 epoch 主实验在 10% 标签、两个 seed 上均值相对 B2 达到预注册效应量。若未
达到，不把 0.1–0.3 个点写成显著提升；改为区域标注失效分析或更换单一伪标签机制。

## 10. 进度日志与后续会话交接

每次工作结束必须在本节追加一条记录，至少包含：日期、commit、执行的命令/
实验、产物位置、关键结果、是否通过 gate、下一步。不要把“启动训练”写成
“完成训练”，也不要把烟测写成正式结果。

| 日期 | Commit | 工作或实验 | 产物/命令 | 结果 | Gate/下一步 |
|---|---|---|---|---|---|
| 2026-10-26 | `d3503c9` | 初始化仓、数据/B1/B2 工具链；20 图烟测 | `data/smoke_area10`、`runs/smoke/*` | 9/9 tests；B2 13/20 伪标签图、38 框、33 训练图；完整 val 成功 | 仅过 Gate 1；正式数据未生成 |
| 2026-10-26 | working tree | 建立论文完成计划、新会话入口和文件地图 | `docs/paper_completion_plan.md`、`AGENTS.md`、README 链接 | 8 周计划、证据边界和交接规则已落地；测试仍为 9/9 | 等待审阅后提交 |
| 2026-10-26 | `23fdcd5` + working tree | 修正 Git Bash 环境命令；生成并审计正式 10% seed0 数据；明确 20 图烟测的数据路径 | Git Bash 加载 `conda.sh` 后 `conda activate yolo`；`data/visdrone_area10/{summary,audit}.json`；README/baseline 命令 | 6471 张训练图、105058 个监督框、38245 个可见隐藏 GT；完整 val 548 张；12/12 基础审计通过；未改源码，未运行训练 | 数据构造完成；Gate 1 尚缺同 seed 一致性和 5 epoch/640、完整 val960 验收；之后启动 B1 |
| 2026-10-26 | `8501b73` + working tree | 审核用户完成的 5 epoch 烟测；补完整 val960、同 seed 重生成检查和测试 | `runs/smoke/partial10_sup_smoke5/gate1_audit.json`；`runs/val/partial10_sup_smoke5_val960/metrics.json`；`python -m pytest` | 20 图/640/batch8 训练完成；完整 val 548 图；开发 mAP50-95=0.080882；60 个数据文件哈希一致；9/9 tests；正式数据关键哈希未变 | Gate 1 通过；仅烟测，无正式方法结果；建议启动正式 10% seed0 的 B1 100 epoch，本次未启动 |
| 2026-10-26 | `14eaf2c` + working tree | 验收正式 B1 100 epoch，补 val960/权重哈希；修复全量伪标签批处理 OOM；生成/诊断/合并 B2 数据 | `runs/partial10_sup_s0/{audit.json,SHA256SUMS.txt,validation/metrics.json}`；`data/visdrone_area10/pseudo/naive_t050/`；`python -m pytest --basetemp outputs/pytest_pseudo_batch_20261026` | B1 正式开发 mAP50-95=0.248835；4624 个伪标签 tile、27667 框，合并 11095 图；11/11 tests；教师哈希固定，隐藏 GT 仅离线分析 | B1 完成，B2 数据就绪；Gate 2 未测，两个 30 epoch 学生均未启动；P1 尚未实现 |
| 2026-10-27 | `f7b711c` + working tree | 验收用户完成的 B1/B2 30 epoch pilot，分别补完整原图 val960 和权重/输入审计 | 两个 pilot run 下的 `audit.json`、`SHA256SUMS.txt`、`validation/metrics.json`；`outputs/gate2_20261027/comparison.json` | B1 mAP50-95=0.229114，B2=0.234534，差 +0.5421 个百分点；30 epoch 及实际训练参数配对，11106 个伪标签包文件哈希未变 | Gate 2 通过；仅单 seed 开发 pilot，非最终方法结论；下一步实现 P1 并测 Gate 3，本次未改源码或启动新训练 |
| 2026-10-27 | `f7b711c` + working tree | 实现 P1 三项机制、固定分尺寸 recall；单元测试/GPU 烟测、全量生成和哈希预检；10:33 启动 30 epoch pilot | `python -m pytest --basetemp outputs/pytest_p1_20261027`；`configs/{pseudo_p1_10,train_p1_10}.yaml`；`outputs/gate3_20261027/{preflight,status}.json`；`scripts/run_p1_pilot.py` | 28/28 tests；37 tile/1 epoch 烟测通过；全量 4839 伪标签 tile、35067 正框、52117 ignore 框，合并 11310 图；B2 学生固定小目标 recall=33.815%；详见 3.6 | 工程实现完成，正式 pilot 仅已启动；Gate 3 未测，待 val960 和小目标 recall 同时达标；不扩训 |
| 2026-10-27 11:04 | `f7b711c` + working tree | 核查 P1 训练进程、CSV、权重文件、实际配对参数和评估产物；更新进度记录 | `outputs/gate3_20261027/progress_check_20261027_110453.json`；stdout/stderr、`results.csv`、`args.yaml` | 已完成 6/30 epoch，第 7 epoch 到 1344/1414 batch；CSV 连续且全部有限，best/last 存在，stderr 为空；与 B2 实际参数配对一致 | 训练未完成，最终 val960/小目标 recall/比较文件尚未生成；Gate 3 尚未测量，继续原后台流程；本次未改源码或重启训练 |
| 2026-10-27 | `8f02449` + working tree | 增加 P1 本机实时进度查看器并打开浏览器，方便用户持续查看和完成后通知验收 | `python scripts/watch_p1.py`；地址见 `outputs/p1_monitor/server.json`；`python -m pytest --basetemp outputs/pytest_p1_monitor_20261027` | 页面每 3 秒读取日志/CSV/阶段；显示 epoch/批次、估计剩余训练时间、最近日志和评估完成提示；28/28 tests，HTTP 页面及完成/评估中状态检查通过；核查时完成 17 epoch、正在 18 epoch | 仅增加只读查看工具，训练继续，Gate 3 未测；全部完成后验收结果 |
| 2026-10-27 12:44 | `8f02449` + working tree | 验收 P1 30 epoch、完整 val960、小目标 recall，独立核对冻结输入/产物哈希与 Gate 3；按用户目标保留正向阶段成果 | `outputs/audit_gate3_acceptance_20261027.py`；`outputs/gate3_20261027/acceptance_audit.json`；run 下 `audit.json` | 19/19 验收检查通过；mAP50-95=23.7883%，比 B2 +0.3349 个百分点；小目标 recall=35.6925%，+1.8776 个百分点；23773 冻结输入哈希匹配 | 原 Gate 3 未达 +0.5 个百分点门槛，小目标条件通过；用户决定认可初步收益，优先整理方法、结果和分析；后续验证按论文需要安排 |
| 2026-10-27 | `8f02449` + working tree | 整理 B1/B2/P1 论文表格，起草方法与实验分析；补齐 B1 分尺寸 recall，增加表格生成核对命令 | `docs/paper/`；`python scripts/build_paper_tables.py --check`；`runs/size_recall/b1_pilot30_s0/metrics.json` | 七组结果表及两章中文初稿；B1 small recall=34.379%，P1 对比 B2 的 AP 为 7 类升/3 类降；27 个来源哈希/表格核对通过，28/28 tests | 写作已启动，主表为配对 30 epoch 单 seed 开发结果；后续审阅与整合正文 |
| 2026-10-27 | `8f02449` + working tree | 检索并核对 SSOD、稀疏标注与航拍小目标参考论文；新增相关工作及 BibTeX；调整方法/实验引用边界 | `docs/paper/{related_work,references}.md`、`docs/paper/references.bib`；第 3.10 节 | 16 项核心文献；识别 S³OD 为最近邻工作，明确其尺度阈值与教师负学习重叠；正文不宣称这些概念首次提出 | 后续精读核心文献、按首次引用顺序统一编号、写引言；比较性实验按现有结果需要安排 |

| 2026-10-27 | `9dd79f7` + working tree | 按用户“下一步怎么做”请求复核进度、协议、论文缺口及现有 Gate 3 结果，给出后续优先级建议 | `python scripts/build_paper_tables.py --check`（已有 yolo 环境）；`outputs/gate3_20261027/{comparison,acceptance_audit}.json` | 表格核对通过，27 个来源哈希及三组 recall 的 GT 计数核对通过；现有正文缺引言、案例图，核心单因素消融与重复 seed 尚未完成；本次未改源码、未启动训练 | 建议先精读 S³OD 并明确方法差异、补引言和图；随后优先做去掉 ignore、去掉软权重、全局阈值三组 30 epoch 消融，再按证据需要安排重复 seed 与正式训练；这是建议，尚未实施或改变原 Gate 判定 |

| 2026-10-27 | `9dd79f7` + working tree | 完成 S³OD 方法差异对照和第 1 章引言；核对正式发表版本、早期预印本与公开代码；修正 Soft Teacher 权重对象 | `docs/paper/{introduction,s3od_comparison}.md`；`references.md`、BibTeX 与论文入口/证据索引；第 3.11 节 | 第 1–4 章初稿齐备；S³OD 更新为 ISPRS Journal 2025 期刊文献；预印本百分位与公开代码 GMM 阈值分开描述；表格 27 来源及文稿链接/文献/引言数字核对通过；仅文稿，无新训练 | 原 Gate 3 判定不变；后续补案例图、核心消融和重复 seed，期刊全文细节与最终引用排序待补 |

| 2026-10-27 | `9dd79f7` + working tree | 准备第①组 no-ignore 配置及训练→val960→分尺寸 recall→完整 P1 对比脚本，提供用户执行命令 | `configs/train_p1_no_ignore_10.yaml`；`scripts/run_p1_no_ignore.py --check-only`；README、baseline 第 4.4 节 | 23781 个冻结文件核对通过；11310 tile 与完整 P1 条件一致，仅关闭 uncertain-ignore；28/28 tests | 仅准备和预检完成，训练尚未启动；用户执行 `python -u scripts/run_p1_no_ignore.py` 后自动评估对比；原 Gate 3 不变 |

| 2026-10-27 | `1e2ef5a` + working tree | 验收用户完成的 no-ignore 30 epoch、val960 和分尺寸 recall；独立核对输入/输出哈希、参数与保存预测匹配；记录首组消融结果 | `outputs/p1_no_ignore_pilot30_s0/{comparison,acceptance_audit}.json`；`outputs/audit_p1_no_ignore_acceptance_20261027.py`；第 3.13 节 | 23795 冻结/产物哈希和 19 个 run 输入匹配；两组各 548 图 recall 重算一致；去掉 ignore 的 mAP50-95 +0.0217 pp、small recall −0.1392 pp（−37）；逐类 4 升/6 降 | 第一组正式配对消融已完成，非烟测；整体 mAP 近似持平，不支持 ignore 独立提高整体 mAP；继续软权重/尺度消融及重复 seed，本次未启动新训练 |

| 2026-10-27 17:44 | `4fe664d` + working tree | 按用户低成本/候选出现后停止约束，启动软权重 1024 源图×两组各 5 epoch 配对筛选，准备全量确认指令但不启动 | `scripts/screen_p1_soft_weights.py`，PID 5784；`outputs/screens/p1_soft_weights_1024_e5/`；README/baseline 第 4.5 节 | 全量配置 23781 文件预检通过；31/31 tests；当前 1765 tile 配对筛选已启动、尚无最终结果；Gitee main 已核实为 4fe664d | 等待筛选完成及验收，发现值得确认的方向后用中文提交并推送，给训练指令并停止；不自动扩训 |

| 2026-10-27 17:53 | `4fe664d` + working tree | 完成软权重 1024 源图×两组各 5 epoch 配对筛选、val960、分尺寸 recall 和独立审计；按预算约束停止全量确认 | `outputs/screens/p1_soft_weights_1024_e5/{comparison,acceptance_audit}.json`；第 3.15 节 | 27 项审计通过；去掉软权重 mAP50-95 +0.0884 pp、small recall +0.0790 pp、mAP50 −0.2249 pp，低于 0.3 pp 筛选线；总学生耗时 417.319 秒 | 中性、不进论文表、不启动 30 epoch；本轮代码与记录用中文提交并推送 Gitee；继续下一项低成本机制筛选 |

| 2026-10-27 18:01 | `919d71e` + working tree | 完成尺度阈值 1024 图单臂 5 epoch 筛选和独立审计；准备全量统一阈值数据与训练入口，按成本约束不启动 30 epoch | `outputs/screens/p1_global_t050_1024_v2_e5/{comparison,acceptance_audit}.json`；`data/.../p1_global_t050_from_candidates_20261027`；`scripts/run_p1_global_t050.py --prepare-only` | 新组 1737 tile/238.414 s；mAP50-95 +0.2237 pp 但 small recall −3.1230 pp；全量数据 11095 tile、4624 伪标签 tile、27667 正框、57219 ignore，预检通过 | 自动候选线未触发但小目标信号值得确认；30 epoch 尚未启动，向用户给命令后停止；本轮代码中文提交并推送 |

| 2026-10-27 20:21 | `ad298ce` + working tree | 验收用户完成的统一 0.50 阈值全量 30 epoch、val960、分尺寸 recall，并接入论文表 4-S2 | `outputs/p1_global_t050_pilot30_s0/{comparison,acceptance_audit}.json`；`outputs/audit_p1_global_t050_acceptance_20261027.py`；`scripts/build_paper_tables.py`；第 3.18 节 | 24 项哈希/参数检查及 548 图预测重放通过；mAP50-95 −0.6177 pp、small recall −0.3424 pp（−91），逐类 1 升/9 降；best hash `94571a…28a4a`；表格 36 个来源及 31/31 tests 通过 | 全量确认支持尺度分层更有利于总体和小目标覆盖；仍为单 seed 开发结果，原 Gate 3 不变；中文提交并推送，不自动扩训 |
| 2026-10-27 20:31 | `5d0d514` + working tree | 按用户授权启动尺度阈值配对学生 seed1；固定 split seed0/教师/两套伪标签与损失，顺序运行两组各 30 epoch | `python -u scripts/run_p1_scale_seed_pair.py`；PID 29248；`outputs/p1_scale_threshold_pair_s1/`；第 3.19 节 | 33,047 冻结文件与 metadata 预检通过；36/36 tests；已有论文表 36 来源核对通过；第一组运行中，尚无 seed1 结果 | 完成两组训练/val960/recall/保存预测重放后更新论文两 seed 配对统计并提交 Git；原 Gate 3 不变 |
| 2026-10-27 23:47（后续收尾见本次提交） | `5d0d514` + working tree | 完成尺度阈值配对学生 seed1 两组 30 epoch、548 图 val960/recall 与独立验收；更新论文表 4-S3、第 4.11 节、证据及复现路径，并纠正短程筛选与全量总体 mAP 方向不同的表述 | `scripts/audit_p1_scale_seed_pair.py`；`outputs/p1_scale_threshold_pair_s1/{comparison,acceptance_audit,independent_audit}.json`；`python scripts/build_paper_tables.py --check` | seed1 尺度−统一：mAP50-95 +0.2420 pp、small recall +0.6434 pp（+171 匹配）；两 seed 平均配对增益 +0.4299±0.2657/+0.4929±0.2128 pp；36 项独立验收、33,047 冻结文件、两组预测重放、59 来源表格与 36/36 tests 通过 | 尺度阈值总体与小目标方向在两学生 seed 保持，支持保留；medium 方向不一致，large 均为统一阈值更高；单 split/固定教师，无显著性或官方 AP_small 主张，原 Gate 3 不变；按用户要求本地 Git 提交 |

| 2026-10-30 09:13（进行中） | `9295c3e` + working tree | 跟进 B2 seed1：原训练第 3 epoch 中断且进程已退出；归档后同配置从初始化重跑完整 30 epoch；整理 seed0 四类案例 | `outputs/b2_seed1_interrupted_20261030_091251/recovery.json`；`outputs/b2_pilot30_s1/`；PID 31944；`docs/paper/case_studies.md` | 37,695 冻结文件核对通过；四类案例和来源 manifest 已生成；尚无 B2 seed1 完整结果 | 等待 30 epoch/完整 val960/recall/四组预测重放，独立验收后更新论文两 seed 统计、测试并本地提交；不推送，原 Gate 3 不变 |

| 2026-10-30 10:15（进行中） | `9295c3e` + working tree | 心跳发现第二次进程中断：日志停于第16epoch且PID31944消失；归档12个文件后用WMI独立启动原入口，避免依赖Codex子进程存续 | `outputs/b2_seed1_interrupted_20261030_101330/recovery.json`；`outputs/b2_pilot30_s1/launch_receipt.json`；PID14304 | 37,695冻结文件再次核对通过；原训练脚本/配置未改；独立验收已支持两级归档来源链及包装哈希 | 当前从原初始化重跑完整30epoch；不使用两次中断指标；完成后按原标准验收、更新论文并本地提交，原Gate3不变 |

| 2026-10-30 10:27（手动交接） | `9295c3e` + working tree | 用户要求由自己终端执行训练；核实PID14304身份后停止后台进程，完整归档当前run和来源链，暂停自动跟进 | `outputs/b2_seed1_interrupted_20261030_102720/`；`python -u scripts/run_b2_seed1.py` | 37,695冻结输入复核通过；13个文件及脚本/配置来源已保留；标准输出/错误由用户终端记录 | 等待用户手动完成连续30epoch及自动评估后，继续独立验收、更新论文和本地Git提交；不得自动重启训练 |

| 2026-10-30 13:05（验收收尾） | `9295c3e` + 本次提交 | 验收用户终端完成的 B2 seed1 连续30epoch及完整评估，复用P1 seed1；完成两seed主对照统计、四类案例和论文/复现记录 | `outputs/b2_pilot30_s1/{comparison,acceptance_audit,independent_audit}.json`；表4-S4；实验4.12–4.13；`outputs/paper_cases_b2_p1_s0/` | seed1 P1−B2：mAP50-95 +0.2438 pp、small recall +1.6932 pp（+450匹配），large −0.6560 pp；两seed配对差 +0.2893±0.0644/+1.7854±0.1304 pp；116检查/37,695冻结文件/四组预测重放、案例1,106输入核验、论文表123来源及7组recall、pytest39/39均通过；中断35文件归档不入统计 | 总体/小目标两seed同向，large方向不同；单split/固定教师/B1未重复，不作显著性或AP_small主张；原Gate3不变；按授权本地Git提交、不push，清理已暂停心跳，不自动扩训 |

| 2026-10-30 13:21（研究安排） | `80e9f2b` + working tree | 按用户三篇文章、毕业成果组合和树莓派轻量化要求制定分阶段计划；论文录用/专利“申请通过”口径保留待校方核定 | `docs/publication_and_edge_deployment_plan.md`；第3.21节 | A方法、B补全、C端侧压缩三项独立问题；执行A→C→B；单独规划学位论文及专利备选；CPU/NPU、分辨率和完整图像延迟分开验收 | 仅规划，无新训练/硬件购买/投稿/专利申请；硬件、毕业期限及认可细则待补；原Gate3不变 |

| 2026-10-30 13:31（第一篇规划） | `main` + working tree | 按用户低成本优先要求审计第一篇缺口，完成冻结候选漏检归因与下一步排序 | `scripts/audit_paper1_gaps.py`；`outputs/paper1_gap_audit_20261030/`；`docs/paper1_experiment_gaps.md`；第3.22节 | 6471 tile/38245 hidden GT/35067正伪标签/25669匹配精确重放；8,984个未匹配小目标的描述性证据类别已分开记录；3/3新测试通过 | 不做因果外推；先L0候选覆盖检查，再决定是否需要用户执行5epoch筛选；尚无新训练，原Gate3不变 |

| 2026-10-30 13:43（ignore增量） | `80e9f2b` + working tree | 按用户指令只做ignore可恢复增量：锁定旧匹配、全尺度联合贪心与最大基数匹配，单列排除tile、重复与低阈值代价 | `scripts/audit_ignore_recovery.py`；`outputs/paper1_ignore_recovery_20261030/`；`docs/paper/ignore_recovery_analysis.md`；第3.23节 | 入训52117 ignore→普通新增5303/条件上界5315，其中普通新增small3627；全纳入新增匹配比例10.18%，small conf≥0.25为27.12%；11322来源哈希与完整6471tile重放 | 有几何恢复空间，尚无足够可靠的无GT转正规则，不下发扩训；隐藏GT只诊断、不是学生AP；先考虑小样本可靠性验证；原Gate3不变 |

| 2026-10-30（跨视图离线验证） | 本次提交 | 固定随机256 tile进行同教师水平翻转推理，重放少量候选的精度与覆盖 | `scripts/audit_cross_view_flip.py`；`docs/paper/cross_view_flip_analysis.md`；`outputs/cross_view_flip_20261030/` | 原入训196 tile；small conf≥0.25新增97框/30匹配，翻转conf≥0.25及IoU≥0.50新增63框/22匹配；新增匹配比例30.93%→34.92%，small匹配增量29→22；pytest 48/48 | 探索性小样本，组合精度低于原基线；不训练、不改变Gate 3，不下发训练命令 |

| 2026-10-30（候选保留检查） | 本次提交 | 128个独立抽样tile比较原NMS/max-det与分别放宽；保留初版和v2重放来源 | `scripts/audit_candidate_retention.py --check`；`outputs/candidate_retention_128_20261030_v2/`；第3.24节 | NMS放宽新增210正框/9独立small匹配，precision76.92%→59.07%；max-det无变化；未达到继续条件；52/52tests、359输入哈希/128tile重放、论文表123来源/7组recall通过 | 不训练、不新增训练标签；更新最多30epoch自主小试/临时监控/中文提交规则；原Gate3不变 |

| 2026-10-30（跨尺度确认与范围收束） | 本次提交 | 固定教师640视图支持960 uncertain候选；固定唯一规则后用不重叠256tile确认；更新总计划，当前只做A | `scripts/audit_scale_support.py --check`及`--confirm --check`；`docs/publication_and_edge_deployment_plan.md`第10节 | 探索22框/11small匹配，确认23框/9small匹配，未达继续线；55/55tests及两样本重放通过 | 不启动该规则训练；先A的必要证据和稿件，不并行B/C；Codex有界小试/临时监控/中文提交，用户执行有证据的大规模训练；A未完成，原Gate3不变 |

| 2026-10-30（A软权重补证据准备） | 本次提交 | 准备唯一一组去软权重30epoch正式消融、后台包装与独立验收，固定预算和继续条件 | `scripts/run_p1_no_soft_weights.py --check-only`；`scripts/audit_p1_no_soft_weights.py` | 23781冻结文件/11310tile预检通过；56/56tests；尚未启动 | 仅A，旧5epoch中性结论不变；启动后创建临时监控，不自动扩训，原Gate3不变 |

| 2026-10-30 14:30（A训练启动与文献调研） | `d218720` + 本次文档提交 | 实际启动唯一一组去软权重30epoch并配置临时监控；检索六类用户关键词，明确A单机制后续假设 | `outputs/p1_no_soft_weights_pilot30_s0/launch_receipt.json`；PID34612/包装37236；心跳a-30；`docs/paper/a_keyword_research.md` | 训练进程和CSV增长核实正常，尚无最终学生指标；7份PDF与哈希，5篇摘要级来源；本次仅改文档，沿用启动前56/56测试 | 等完整训练/val960/recall及独立验收后决定下一项；不并行新损失/基础模型/蒸馏剪枝，不推送，原Gate3不变 |

| 2026-10-30 16:37（A软权重验收收尾） | `dbff977` + 本次提交 | 完成去分类软权重30epoch及独立验收，新增自动表4-S5、实验4.14和来源记录 | `outputs/p1_no_soft_weights_pilot30_s0/{comparison,independent_audit,launch_receipt}.json`；第3.26节 | exit0；16项检查/23783冻结文件/两组548图预测重放通过；去软权重mAP50-95−0.1623pp、small recall+0.9143pp（+243），3类升7类降；137来源/8组recall，pytest56/56 | 体现取舍、未达后续线，保留完整P1主配置；不扩训、不push，临时监控验收后清理；下一项仅为定位可靠性待验证假设，原Gate3不变 |

| 2026-10-30（正伪框定位诊断协议） | 本次提交 | 固定已有正伪框跨尺度匹配、置信度/类别分层和两样本继续线，再开始读取本项GT统计 | `configs/positive_localization_diagnostic.yaml`；`docs/paper/positive_localization_analysis.md`；第3.28节 | 仅协议，无该项结果；两组各256tile缓存已存在且不重叠 | 下一步实现GT-free匹配与只读离线诊断；不改现有训练输入、不新增推理或训练，原Gate3不变 |

| 2026-10-30（正伪框定位诊断完成） | `2a42fa5` + 本次提交 | 实现GT-free一对一稳定性特征，复用缓存完成探索/确认，按预先继续线停止 | `scripts/audit_positive_localization.py --check`；`outputs/positive_localization_20261030/`；第3.28节 | 2710正框/1442来源重放，pytest61/61；small不稳定两组各6框，分层证据不足；稳定组严格定位质量比例52.7981%/49.4737% | 不增推理/训练，不改分组重试、不直接删除无匹配框；下一项需新信息来源，原Gate3不变；中文本地提交、不push |

| 2026-10-30（原图上下文纠错协议与Zotero归档） | 本次提交 | 固定128原图探索、过线后独立确认的单规则；实现GT-free映射/类别纠错及只读验收；按用户要求将Zotero分类调整为并列顶层 | `scripts/audit_context_rectification.py`；第3.29节；`docs/paper/zotero_literature_library.md` | pytest73/73；28论文均有PDF；尚无本项推理/纠错效果 | 先提交再推理；探索未过线则不做确认、不训练；当前仅A，不push，原Gate3不变 |

| 2026-10-30（原图上下文探索完成） | `f1c997e` + 本次提交 | 固定128原图上下文类别纠正探索、只读重放及失败确认阻断；修复历史数据源码误忽略 | `scripts/audit_context_rectification.py --check`；`docs/paper/context_rectification_results.md` | 98接受tile/799正框，仅改1框且改错；总匹配及small各−1；372来源一致，推理4.3874秒/训练0epoch；pytest74/74，Git归档源码齐全 | 继续线失败，不运行确认、不调阈值或训练；当前未找到值得扩训的新机制，原Gate3不变；中文提交、不push |

| 2026-10-30（置信度校准协议与代码） | 本次提交 | 固定开放区域拟合、来源不重叠探索/确认、全局简化对照和八项继续线；实现单调收缩校准/验收 | `scripts/audit_confidence_calibration.py`；第3.30节 | pytest84/84；尚未推理或拟合，不改既有训练输入 | 先提交再运行512开放tile推理；隐藏GT只验收，不绕过简化对照或启动训练 |

| 2026-10-30（置信度校准探索完成） | `f3d77cd` + 本次提交 | 512开放tile拟合、来源不重叠256tile探索、全局简化对照及失败确认阻断 | `scripts/audit_confidence_calibration.py --check`；第3.30节 | 拟合7548正框；探索1421正框；整体分组Brier0.160310高于原0.158864，small0.183221→0.176334；1045+474来源重放；84tests，表格137来源/8recall核验 | 整体继续线失败；不重拟合/改分组追试，不运行确认或学生训练，不发长期命令，原Gate3不变；中文提交、不push |

| 2026-10-30（小框NWD协议与工程实现） | 本次提交 | 固定开放GT像素常数、小框混合回归及单组5epoch缓存配对筛查；实现GPU检查、冻结输入、验收和后台包装 | `scripts/screen_p1_small_nwd.py --check-only`；第3.31节 | pytest94/94；尚未估计C/做GPU检查或训练 | 先提交协议再预检；预检通过才启动唯一5epoch并创建临时监控；不自动30/100epoch |

| 2026-10-30（小框NWD预检完成） | `626fa9e` + 本次提交 | 完成23794冻结输入预检和真实YOLO11s GPU AMP前反传；只用开放GT估计常数 | `python scripts/screen_p1_small_nwd.py --geometry-only` | C=20.22804306像素，1024开放tile/7519小框；255梯度张量有限，0优化器步；94tests | 工程可执行，不代表有效；下一步仅新增单组5epoch及临时监控，不自动扩训 |

| 2026-10-30 18:16（小框NWD启动） | `d1da33a` + 本次提交 | 实际启动唯一五epoch子集筛查，复用缓存P1；创建临时监控a-nwd | `outputs/screens/p1_small_nwd_1024_e5/launch_receipt.json`；包装32304/训练28860 | WMI隐藏启动返回0；命令/创建时间验证，CSV增长；18:19已完成2epoch、stderr为空；尚无最终val960/recall | 继续原进程，完成后独立验收；未判定效果，不自动扩训，当前仅A；结束清理监控，中文提交、不push |

| 2026-10-30 18:22（小框NWD验收） | `78b92ae` + 本次提交 | 完成唯一5epoch、val960/recall及独立验收，按固定继续线停止 | `python scripts/audit_small_nwd_screen.py`；`docs/paper/small_nwd_results.md` | exit0/连续5epoch/299.989秒；mAP50-95−0.1746pp、small+1.5464pp、large−0.7498pp；23794输入与两组548图重放 | 继续线失败；不改C/mix/阈值重试、不扩训；清理a-nwd，原Gate3不变；中文提交、不push |

| 2026-10-30（用户更新30epoch性能判断预算） | 本次提交 | 完成五轮缓存等FP/严格定位诊断；按至少30epoch要求准备唯一全量NWD确认，并接受中文提交后push授权 | `scripts/audit_small_nwd_tradeoff.py --check`；`scripts/run_p1_small_nwd.py --check-only`；第3.32节 | 两组548图/1111来源精确重放；等FP时IoU0.50小目标+0.6321/+0.7713pp，IoU0.90略降；108tests/Pyright零错误/论文表核验通过；全量30epoch尚未启动 | 五轮保留为历史探索，不再据此永久淘汰；完整P1缓存复用，仅新增30epoch，不自动扩训；启动后配置临时监控 |

### 10.1 新会话开始流程

2026-10-30原图上下文诊断执行前的Git预检发现，历史`.gitignore`的`data/`
也屏蔽了`src/regionssod/data`四个源码文件，导致代码无法从Git完整复现。
本次将生成数据忽略限定为根目录`/data/`，保留原源码内容并补入版本控制；
新增忽略规则回归测试。两次预检均在GPU/输出创建前拒绝执行，未消耗推理或训练预算。
此项是复现缺陷修复，不更改split、裁片算法或已冻结的输入文件。

1. 进入 `D:\Python\YOLO-RegionSSOD`，运行 `git status --short --branch`。
2. 按顺序阅读：`AGENTS.md`、本文件、`docs/experiment_plan_8weeks.md`、
   `docs/baseline_protocol.md`、当前周对应的配置和日志。
3. 核对 `data/`、`runs/` 是否只是本地忽略产物，不能从 Git 状态推断实验已经完成。
4. 核对实际权重、`experiment.json`、`results.csv`、`metrics.json` 后再引用指标。
5. 修改前确认当前脏文件是否来自用户或其他会话；不要回滚未授权的改动。
6. 结束时更新本文件的进度日志，并在最终回复中明确“已完成/未完成/仅烟测”。

### 10.2 结果引用规则

- `data/`、`runs/`、`weights/` 被 `.gitignore` 排除，不代表它们不存在。
- 20 图、1 epoch、320 分辨率的数值只能作为链路证据。
- 只有固定 split、固定配置、完整 val 和可追溯权重的结果才能进入论文表。
- Ultralytics mAP 与 VisDrone 官方 AP 不能无说明混用。
- 任何外部确认集的结论都必须先完成版本审计、类别映射和固定评估命令。

## 11. 风险与退出条件

| 风险 | 触发信号 | 处理 |
|---|---|---|
| B2 无收益 | B2 pilot 低于 B1 | 检查隐藏 GT 密度、教师 recall、阈值和伪标签覆盖；不继续加模块 |
| P1 只增加伪标签数量 | 框数增加但 precision 下降 | 报告 precision/coverage，按尺寸和类别拆分，不宣称质量提升 |
| 主方法只在 10% 有效 | 5%/20% 无同向收益 | 如实报告预算敏感性，检查协议差异，不扩大结论 |
| 显存不足 | 640/batch 8 OOM | 所有对照统一降低 batch；不要只给主方法加梯度累积 |
| 指标不可比 | Ultralytics 与官方 AP 不一致 | 明确评测工具；正式表尽量用官方工具，保留差异说明 |
| 时间不够 | 第 6 周仍未过 Gate 4 | 停止扩训，转诊断论文或缩小到单一机制，不继续堆模块 |

## 12. 关键文件地图

| 文件 | 用途 |
|---|---|
| `README.md` | 环境、快速开始和 baseline 命令 |
| `AGENTS.md` | 后续会话的入口规则和当前范围 |
| `docs/experiment_plan_8weeks.md` | 科学问题、固定协议和完整实验设计 |
| `docs/baseline_protocol.md` | B0/B1/B2 的可执行命令和审计规则 |
| `docs/paper/README.md` | 方法、实验分析初稿及论文表格的写作入口 |
| `configs/train_full.yaml` | B0 完整监督配置 |
| `configs/train_partial_05.yaml` | 5% 部分监督配置 |
| `configs/train_partial_10.yaml` | 10% 主实验配置 |
| `configs/train_partial_20.yaml` | 20% 部分监督配置 |
| `configs/train_naive_10.yaml` | B2 自训练配置 |
| `configs/pseudo_p1_10.yaml`、`configs/train_p1_10.yaml` | P1 阈值与 30 epoch pilot 配置 |
| `src/regionssod/data/partial.py` | tile、隐藏 GT 和 manifest 生成 |
| `src/regionssod/pseudo/generator.py` | 伪标签生成和列表合并 |
| `src/regionssod/training/runner.py` | 训练、seed 和元数据记录 |
| `src/regionssod/training/p1.py` | 元信息增强传递、置信度分类权重和 uncertain-ignore 损失 |
| `src/regionssod/pseudo/scale.py` | 按原 tile 面积分层、候选/置信度/ignore 导出 |
| `src/regionssod/evaluation/runner.py` | 完整原图 val 和 `metrics.json` |
| `src/regionssod/evaluation/size_recall.py` | 固定操作点完整 val 分尺寸 recall，非官方 AP_small |
| `scripts/run_p1_pilot.py` | 本机已冻结输入的 pilot 训练/验证/Gate 3 顺序执行脚本 |
| `scripts/watch_p1.py` | 仅监听本机的实时进度页，3 秒刷新；不控制训练 |
| `scripts/build_paper_tables.py` | 从冻结评估产物生成论文结果表，`--check` 核对表格数字 |
| `tests/` | 数据、配置、伪标签和路径回归测试 |

## 13. 非本论文内容

`docs/RGB_IR_Reliability_Experiment_Design.md` 记录的是 M3FD/LLVIP 的 RGB-IR
局部可靠性融合方案。它目前不是本论文的实验部分，也不应该在本仓库中复用
VisDrone split 或 RegionSSOD 权重。未来若要启动该方向，应另建独立仓库和独立
数据/结果目录。
