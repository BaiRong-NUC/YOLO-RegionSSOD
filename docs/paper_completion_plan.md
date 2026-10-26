# 论文完成计划与项目交接

> 最后更新：2026-10-26  
> 当前阶段：B1 seed0 的 100 epoch 及完整 val960 已完成，B2 伪标签与训练清单就绪，待执行 Gate 2 配对 pilot<br>
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

以下状态以 2026-10-26 为快照。后续会话必须先运行 `git status` 并核对实际产物，
不能只依赖本文档的旧描述。

| 项目 | 状态 | 证据或位置 | 下一步 |
|---|---|---|---|
| Git 仓库 | 已初始化并推送 Gitee，本次会话基于 `14eaf2c` | `main...origin/main`；本次源码/测试/文档修改未提交 | 每个阶段单独提交 |
| 数据构造 | 已实现；10% seed0 已生成 | `src/regionssod/data/partial.py` | 生成正式 5%/20% split |
| B0 完整监督 | 配置已加入，未正式训练 | `configs/train_full.yaml` | 复用旧结果或重训 seed 0 |
| B1 部分监督 | 正式 10% seed0、100 epoch 已完成；完整 val960 mAP50-95=0.248835 | `runs/partial10_sup_s0/audit.json`、`validation/metrics.json` | 补独立 30 epoch B1 学生，与 B2 配对 pilot |
| B2 朴素自训练 | 正式伪标签与合并清单就绪；学生未训练 | `data/visdrone_area10/pseudo/naive_t050/{summary,audit,quality_audit}.json` | 30 epoch 配对 pilot |
| P1 尺度感知方法 | 仅完成实验设计，尚未实现 | `docs/experiment_plan_8weeks.md` | 第 3 周实现并测试 |
| 正式 5/10/20% 数据 | 10% seed0 已生成；20 图同 seed 检查通过；5%/20% 未生成 | `data/visdrone_area10/summary.json`、`audit.json` | 生成 5%/20% split |
| 正式 100 epoch 主矩阵 | 已完成 B1 10% seed0；B2/P1 未完成 | `runs/partial10_sup_s0`；单 seed 开发结果 | 先测 Gate 2，再继续方法与矩阵 |
| 论文正文 | 未开始 | 无 | 第 1 周建立骨架 |

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

**下一步：**按 baseline 协议第 4.1 节运行 B1/B2 各 30 epoch 的独立学生并完成
完整 val960；两者均从 COCO 预训练开始，B2 和后续 P1 pilot 固定使用上述 B1
100 epoch 教师。学生训练预算配对，教师训练成本另记。不得将 B2 30 epoch 与
B1 100 epoch、或其第 30 行 CSV 直接作 Gate 2 比较。Gate 2 尚未测量；P1 尚未
实现。当前 B1 为单 seed 的正式开发结果，官方 AP、AP_small 和多 seed 仍待补。

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
| `configs/train_full.yaml` | B0 完整监督配置 |
| `configs/train_partial_05.yaml` | 5% 部分监督配置 |
| `configs/train_partial_10.yaml` | 10% 主实验配置 |
| `configs/train_partial_20.yaml` | 20% 部分监督配置 |
| `configs/train_naive_10.yaml` | B2 自训练配置 |
| `src/regionssod/data/partial.py` | tile、隐藏 GT 和 manifest 生成 |
| `src/regionssod/pseudo/generator.py` | 伪标签生成和列表合并 |
| `src/regionssod/training/runner.py` | 训练、seed 和元数据记录 |
| `src/regionssod/evaluation/runner.py` | 完整原图 val 和 `metrics.json` |
| `tests/` | 数据、配置、伪标签和路径回归测试 |

## 13. 非本论文内容

`docs/RGB_IR_Reliability_Experiment_Design.md` 记录的是 M3FD/LLVIP 的 RGB-IR
局部可靠性融合方案。它目前不是本论文的实验部分，也不应该在本仓库中复用
VisDrone split 或 RegionSSOD 权重。未来若要启动该方向，应另建独立仓库和独立
数据/结果目录。
