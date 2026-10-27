# 论文完成计划与项目交接

> 最后更新：2026-10-27<br>
> 当前阶段：第 1–4 章初稿及 S³OD 对照已完成；第①组 no-ignore 消融已验收，整体 mAP 近似持平、固定操作点召回下降；下一阶段补软权重/尺度阈值消融、案例图和重复 seed<br>
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

以下状态以 2026-10-27 为快照。后续会话必须先运行 `git status` 并核对实际产物，
不能只依赖本文档的旧描述。

| 项目 | 状态 | 证据或位置 | 下一步 |
|---|---|---|---|
| Git 仓库 | P1 实现已提交为 `8f02449`，本次在此基础上验收结果 | `main...origin/main`；进度文档/README 及查看器仍有本地改动 | 保留既有改动，按阶段整理 |
| 数据构造 | 已实现；10% seed0 已生成 | `src/regionssod/data/partial.py` | 生成正式 5%/20% split |
| B0 完整监督 | 配置已加入，未正式训练 | `configs/train_full.yaml` | 复用旧结果或重训 seed 0 |
| B1 部分监督 | 正式 100 epoch 和独立 30 epoch pilot 均完成；val960 mAP50-95 分别为 0.248835、0.229114 | `runs/partial10_sup_s0/`、`runs/partial10_sup_pilot30_s0/` | 冻结为正式参考与 pilot 对照 |
| B2 朴素自训练 | 30 epoch pilot 完成；val960 mAP50-95=0.234534；100 epoch 尚未运行 | `runs/partial10_naive_t050_pilot30_s0/` | 保留为冻结的配对对照 |
| P1 尺度感知方法 | 30 epoch 和完整评估已验收；mAP50-95=0.237883、小目标 recall=0.356925，均优于 B2 | `outputs/gate3_20261027/{comparison,acceptance_audit}.json`；第 3.8 节 | 保留积极的单 seed 初步结果，整理方法/结果/分析；原 Gate 3 未达预设门槛 |
| 第①组 no-ignore 消融 | 30 epoch、val960、固定操作点 recall 及独立验收完成；相对完整 P1 的 mAP50-95 +0.0217 pp、small recall −0.1392 pp | `outputs/p1_no_ignore_pilot30_s0/{comparison,acceptance_audit}.json`；第 3.13 节 | 不能声称 ignore 独立提高整体 mAP；继续拆分软权重与尺度阈值，补配对 seed |
| 软权重低成本筛选 | 1024 源图、两组各 5 epoch 和 val960 完成；mAP50-95 +0.0884 pp、small recall +0.0790 pp，低于 0.3 pp 候选线 | `outputs/screens/p1_soft_weights_1024_e5/{comparison,acceptance_audit}.json`；第 3.15 节 | 中性，不启动 30 epoch 去软权重确认；继续下一项低成本机制筛选 |
| 正式 5/10/20% 数据 | 10% seed0 已生成；20 图同 seed 检查通过；5%/20% 未生成 | `data/visdrone_area10/summary.json`、`audit.json` | 生成 5%/20% split |
| 正式 100 epoch 主矩阵 | 已完成 B1 10% seed0；B2/P1 未完成 | `runs/partial10_sup_s0`；已有 P1 30 epoch 正向初步结果 | 先整理已有成果，后续按论文需要决定补充验证规模 |
| 论文正文 | 第 1–4 章及七组结果表已有初稿；S³OD 对照完成，参考文献与 BibTeX 已更新正式发表信息 | `docs/paper/`；第 3.9–3.11 节 | 补案例图、三组核心消融及重复 seed；最终排版时统一引用编号 |

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
- `results_tables.md` 包含表 4-1 至 4-6 和表 4-S1，共七组：主结果/差值、
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

### 10.1 新会话开始流程

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
