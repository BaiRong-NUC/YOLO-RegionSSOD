# YOLO-RegionSSOD

面向高分辨率航拍图像的区域局部标注小目标半监督检测实验仓库。

本仓库对应一个明确的论文问题：每张高分辨率图像只标注少量区域，未标注区域
不能被视为纯背景。目标是复用 VisDrone、YOLO11 和航拍小目标经验，在单张
RTX 4070 Ti 12 GB 上完成毕业/中文核心所需的完整实验。

当前状态（2026-10-30）：B2/P1 固定 split/教师、学生 seed0/1 的独立 30 epoch
配对复验已完成验收。P1 相对 B2 的 mAP50-95 两 seed 均提高，平均配对增益为
**+0.2893±0.0644 个百分点**；small recall 为 **+1.7854±0.1304 个百分点**
（样本 SD，ddof=1）。seed1 分别提高 0.2438/1.6932 个百分点，多匹配 450 个
小目标，但 large recall 降低 0.6560 个百分点。116 项独立检查通过。
表 4-1 至 4-6 保留 seed0 主对照，新增表 4-S4 单列两 seed 统计；B1 未重复，
这不是多 split、显著性或官方 AP_small 证据。原 Gate 3 未达门槛的判定不变，
详见 `docs/paper_completion_plan.md` 第 3.8、3.20 节。

论文初稿入口：[论文写作](docs/paper/README.md)。已有
[引言初稿](docs/paper/introduction.md)和 [S³OD 方法差异对照](docs/paper/s3od_comparison.md)，以及
[论文结果表格](docs/paper/results_tables.md)、[方法章节](docs/paper/methods.md)
和[实验分析章节](docs/paper/experiments.md)，B1/B2/P1 的同操作点分尺寸 recall
也已补齐；[相关工作](docs/paper/related_work.md)与[参考文献](docs/paper/references.md)
已整理，含 BibTeX；S³OD 已更新为 2025 年期刊版引用。用 `python scripts/build_paper_tables.py --check` 核对表格。

实时查看 P1：在 `yolo` 环境、仓库根目录运行 `python scripts/watch_p1.py`，打开
终端输出的本机地址。页面每 3 秒刷新 epoch、批次、估计剩余训练时间及最近日志，
并区分训练、val960、小目标 recall 和全部完成。看到“训练和自动评估均已完成”后
即可通知助手验收。当前查看地址保存在 `outputs/p1_monitor/server.json`；
查看器只读取现有日志与产物，关闭页面不影响训练。

## 研究问题

- 每张训练图只保留一个目标中心 tile 的标签。
- 5%、10%、20% 表示标注面积比例，不是随机抽取图片的比例。
- 标注 tile 之外的真实目标仍存在，但对监督分支不可见。
- 验证集使用完整原图和完整标签，不能只验证标注 tile。
- 首轮比较固定 YOLO11s、640 训练分辨率、100 epoch，避免把网络改动混入方法。

## 环境

建议直接使用已有 `yolo` Conda 环境：

```powershell
conda activate yolo
Set-Location D:\Python\YOLO-RegionSSOD
python -m pip install -e . --no-deps
python -m pytest
```

如果终端提示包含 `MINGW64`，当前使用的是 Git Bash。本机 Conda 安装在
`D:/MiniConda`，先在当前终端加载它，再激活已有环境：

```bash
source /d/MiniConda/etc/profile.d/conda.sh
conda activate yolo
cd /d/Python/YOLO-RegionSSOD
regionssod --help
```

若激活后仍找不到 `regionssod`，在仓库目录执行
`python -m pip install -e . --no-deps`。Git Bash 使用 `cd` 切换目录，命令参数中的
Windows 路径使用正斜杠，例如 `D:/Python/YOLO/data/visdrone/dataset.yaml`。
本文其余命令块使用 PowerShell 语法；Git Bash 下可将命令写成单行，例如：

```bash
regionssod make-partial --source-data D:/Python/YOLO/data/visdrone/dataset.yaml --output data/visdrone_area10 --ratio 0.10 --seed 0 --unlabeled-per-image 1
```

数据默认使用：

```text
D:\Python\YOLO\data\visdrone\dataset.yaml
```

预训练权重默认使用：

```text
D:\Python\YOLO\yolo11s.pt
```

## 快速开始

先用 20 张图像做链路烟测：

```powershell
regionssod make-partial `
  --source-data D:\Python\YOLO\data\visdrone\dataset.yaml `
  --output data\smoke_area10 `
  --ratio 0.10 `
  --seed 0 `
  --unlabeled-per-image 1 `
  --limit 20
```

生成正式 10% 区域标注数据集：

```powershell
regionssod make-partial `
  --source-data D:\Python\YOLO\data\visdrone\dataset.yaml `
  --output data\visdrone_area10 `
  --ratio 0.10 `
  --seed 0 `
  --unlabeled-per-image 1
```

在上述 20 张图的数据上进行 5 个 epoch 的 baseline 烟测：

```powershell
regionssod train `
  --config configs\train_partial_10.yaml `
  --data data\smoke_area10\dataset.yaml `
  --epochs 5 `
  --project runs\smoke `
  --name partial10_sup_smoke5
```

烟测权重也必须在完整原图上进行 960 验证：

```powershell
regionssod validate `
  --weights runs\smoke\partial10_sup_smoke5\weights\best.pt `
  --data data\smoke_area10\dataset.yaml `
  --imgsz 960 `
  --name partial10_sup_smoke5_val960
```

正式部分监督 baseline：

```powershell
regionssod train `
  --config configs\train_partial_10.yaml `
  --data data\visdrone_area10\dataset.yaml `
  --name partial10_sup_s0
```

Git Bash 的正式 B1 训练命令如下，配置固定 100 epoch，并从 COCO 预训练
`D:/Python/YOLO/yolo11s.pt` 初始化：

```bash
regionssod train --config configs/train_partial_10.yaml --data data/visdrone_area10/dataset.yaml --name partial10_sup_s0
```

用完整原图在 960 分辨率评估：

```powershell
regionssod validate `
  --weights runs\partial10_sup_s0\weights\best.pt `
  --data data\visdrone_area10\dataset.yaml `
  --imgsz 960 `
  --name partial10_sup_s0_val960
```

完整监督参考 B0：

```powershell
regionssod train `
  --config configs\train_full.yaml `
  --name full_sup_s0
```

## 朴素自训练 baseline

先生成全局阈值 0.50 的伪标签：

```powershell
regionssod pseudo-label `
  --weights runs\partial10_sup_s0\weights\best.pt `
  --images-list data\visdrone_area10\splits\unlabeled_train.txt `
  --output data\visdrone_area10\pseudo\naive_t050 `
  --threshold 0.50 `
  --conf-floor 0.05 `
  --imgsz 960 `
  --batch 8
```

Git Bash 对应命令：

```bash
regionssod pseudo-label --weights runs/partial10_sup_s0/weights/best.pt --images-list data/visdrone_area10/splits/unlabeled_train.txt --output data/visdrone_area10/pseudo/naive_t050 --threshold 0.50 --conf-floor 0.05 --imgsz 960 --batch 8 --device 0
```

推理输入会保存在 `prediction_images.txt`，按 `--batch` 分批读取；输出目录必须
为空，已有成功产物可直接复用。

合并标注 tile 和伪标签 tile：

```powershell
regionssod combine-lists `
  --inputs `
    data\visdrone_area10\splits\labeled_train.txt `
    data\visdrone_area10\pseudo\naive_t050\pseudo_train.txt `
  --output data\visdrone_area10\combined_naive_t050.txt `
  --dataset-yaml data\visdrone_area10\dataset_naive_t050.yaml `
  --source-data D:\Python\YOLO\data\visdrone\dataset.yaml
```

Git Bash 对应命令：

```bash
regionssod combine-lists --inputs data/visdrone_area10/splits/labeled_train.txt data/visdrone_area10/pseudo/naive_t050/pseudo_train.txt --output data/visdrone_area10/combined_naive_t050.txt --dataset-yaml data/visdrone_area10/dataset_naive_t050.yaml --source-data D:/Python/YOLO/data/visdrone/dataset.yaml
```

先执行 30 epoch 的 B1/B2 配对 pilot，检查半监督信号。两个学生均从同一 COCO
预训练初始化；B2 及后续 P1 的 pilot 固定使用上面的 B1 100 epoch 教师：

```bash
regionssod train --config configs/train_partial_10.yaml --data data/visdrone_area10/dataset.yaml --epochs 30 --name partial10_sup_pilot30_s0
regionssod train --config configs/train_naive_10.yaml --data data/visdrone_area10/dataset_naive_t050.yaml --epochs 30 --name partial10_naive_t050_pilot30_s0
```

两个 pilot 均需完整原图 960 验证，再按 Gate 2 比较；完整验证命令见
`docs/baseline_protocol.md` 第 4.1 节。B2 的 30 epoch 结果不能直接与 B1 的
100 epoch 结果作 gate 比较。

完成 pilot 后，100 epoch 的正式自训练 baseline 命令是：

```powershell
regionssod train `
  --config configs\train_naive_10.yaml `
  --data data\visdrone_area10\dataset_naive_t050.yaml `
  --name partial10_naive_t050_s0
```

## P1：尺度阈值、软权重与 uncertain-ignore

固定参数见 `configs/pseudo_p1_10.yaml`：原始 tile 框面积 small<32²、
medium<96²，其余为 large；正阈值依次为 0.30/0.50/0.50，下限为 0.05。
这是一组待检验的 pilot 参数，不是已证明最优的阈值。正框保留教师置信度，
其余候选作为 ignore；没有正框的 tile 被排除，候选仍写入离线记录。

以下为本轮冻结的复现命令，已有训练及评估产物可直接复用：

```bash
regionssod pseudo-label-p1 --weights runs/partial10_sup_s0/weights/best.pt --images-list data/visdrone_area10/splits/unlabeled_train.txt --output data/visdrone_area10/pseudo/p1_s030_m050_l050 --policy configs/pseudo_p1_10.yaml --imgsz 960 --batch 8 --device 0
regionssod combine-lists --inputs data/visdrone_area10/splits/labeled_train.txt data/visdrone_area10/pseudo/p1_s030_m050_l050/pseudo_train.txt --output data/visdrone_area10/combined_p1.txt --dataset-yaml data/visdrone_area10/dataset_p1.yaml --source-data D:/Python/YOLO/data/visdrone/dataset.yaml --p1-metadata data/visdrone_area10/pseudo/p1_s030_m050_l050/metadata.jsonl
regionssod train --config configs/train_p1_10.yaml --name partial10_p1_pilot30_s0
regionssod validate --weights runs/partial10_p1_pilot30_s0/weights/best.pt --data data/visdrone_area10/dataset_p1.yaml --imgsz 960 --name partial10_p1_pilot30_s0_val960
regionssod size-recall --weights runs/partial10_naive_t050_pilot30_s0/weights/best.pt --data D:/Python/YOLO/data/visdrone/dataset.yaml --output runs/size_recall/b2_pilot30_s0 --imgsz 960 --conf 0.25 --match-iou 0.50
regionssod size-recall --weights runs/partial10_p1_pilot30_s0/weights/best.pt --data D:/Python/YOLO/data/visdrone/dataset.yaml --output runs/size_recall/p1_pilot30_s0 --imgsz 960 --conf 0.25 --match-iou 0.50
```

训练器通过相同数据增强同步变换正框、置信度和 ignore 框；分类损失按匹配正框的
教师置信度加权，人工标签权重为 1，分母保持原 TAL 分母。ignore 只屏蔽锚点中心
落入候选框内的**负**分类项，已匹配正样本优先；box/DFL 不变。实现细节、消融
开关和审计见 baseline 协议第 4.3 节。

Gate 3 要求 P1 完整原图 val960 mAP50-95 至少 0.239534294688062，且相对 B2
小目标 recall 同向改善。这里的 `size-recall` 固定 conf=0.25、IoU=0.50、max_det=500，
按完整原图 GT 面积分组，在所有尺度 GT 上联合一对一匹配；这是开发诊断 recall，
不是官方 AP_small 或 AR500，也不是伪标签的隐藏 GT recall。

## 第①组消融：关闭 uncertain-ignore

2026-10-27 已完成训练和独立验收。相对完整 P1，去掉 ignore 的 mAP50-95 为
23.8100%（+0.0217 个百分点），固定操作点 small recall 为 35.5533%
（−0.1392 个百分点）。已有产物可复用；结果范围和审计见进度文档第 3.13 节。

独立配置为 `configs/train_p1_no_ignore_10.yaml`，仅将 `p1.uncertain_ignore` 设为
`false`，保留尺度阈值、分类软权重和完整 P1 的其余条件：同一数据/伪标签、
COCO 初始化、640、30 epoch、batch8、SGD、seed0。元信息与增强链路仍保留，
只关闭损失侧负分类项屏蔽，不重新生成伪标签。

在已有 `yolo` 环境、仓库根目录运行以下一条命令，自动完成训练、完整原图
val960、固定操作点分尺寸 recall，以及与已完成的完整 P1 的对比：

```bash
python -u scripts/run_p1_no_ignore.py
```

只核对配置与冻结输入、不启动训练或 GPU 评估：

```bash
python scripts/run_p1_no_ignore.py --check-only
```

训练目录为 `runs/partial10_p1_no_ignore_pilot30_s0`。阶段记录在
`outputs/p1_no_ignore_pilot30_s0/status.json`；完成后的对比表及完整数值分别为
同目录 `comparison.md`、`comparison.json`。差值定义为“去掉 ignore − 完整 P1”，
单位为百分点。脚本检查冻结输入/参考产物 SHA256、实际参数、连续 30 epoch、
评估权重及相同验证列表/操作点；已有训练目录或派发记录时拒绝重复启动。
本组用于单 seed 机制分析，不重新判定原 Gate 3。

## 低成本筛选：置信度软权重

按 2026-10-27 的最新执行约束，新增训练先做有上限的配对筛选，出现值得确认的
方向后提供正式训练指令并停止自动扩训。软权重筛选入口为：

```bash
python -u scripts/screen_p1_soft_weights.py
```

该筛选固定抽取 1024 张训练原图（选择 seed=20261027），保留对应的 1024 个
人工标注 tile 和 741 个已接受伪标签 tile，共 1765 个训练 tile。完整 P1 和
去掉软权重两组均从 COCO 初始化，独立训练 5 epoch、batch8、640、SGD、seed0，
两组均保留 ignore；短程对照统一设 `close_mosaic=0`。完成后分别评估完整原图
val960 和相同操作点 recall。运行与日志位于 `runs/screen/`、`outputs/screens/`。

这些数值仅用于探索性筛选，不进入论文正式结果表，也不与已有 30 epoch 数字
直接比较。先行筛选标准为：两组 mAP50-95 差的绝对值至少 0.3 个百分点，且
小目标 recall 变化同向；它不代表统计显著性或完整训练结论。

2026-10-27 17:53 本轮筛选完成，独立验收通过。去掉软权重相对保留软权重：

| 子集 5 epoch 结果 | 保留软权重 | 去掉软权重 | 差值 / pp |
|---|---:|---:|---:|
| mAP50 | 26.8181% | 26.5932% | −0.2249 |
| mAP50-95 | 15.3751% | 15.4635% | +0.0884 |
| Small recall | 28.0129% | 28.0920% | +0.0790 |

mAP50-95 与小目标 recall 的方向为正，但都低于预先设定的 0.3 个百分点
筛选线，mAP50 反而下降。结论为**中性、未达到继续全量训练的标准**；不启动
30 epoch 去软权重实验。完整记录见 `outputs/screens/p1_soft_weights_1024_e5/`。

## 尺度阈值筛选与全量确认

同一 1024 张训练原图上，将全部伪标签阈值改为 0.50、仍保留软权重和 ignore，
运行：

```bash
python -u scripts/screen_p1_scale_threshold.py
```

2026-10-27 18:00 已完成，独立审计通过。统一 0.50 相对尺度分层：mAP50-95
从 15.3751% 升至 15.5988%（+0.2237 个百分点），但 small recall 从
28.0129% 降至 24.8899%（−3.1230 个百分点），medium/large recall 则分别
提高 1.1066/1.7807 个百分点。自动候选规则因方向不一致而未触发。

由于小目标 recall 的下降与本文核心问题直接相关，且幅度明显大于总体 mAP
变化，已使用全量数据完成一次 30 epoch 确认。确认数据从冻结教师候选中生成并
核对：4624 个伪标签 tile、27667 个正框，合并后 11095 个训练 tile；未重跑教师。
2026-10-27 20:00 完成训练、完整 val960、分尺寸 recall 和与完整 P1 的对比，
20:21 独立验收通过。统一 0.50 相对完整 P1：mAP50-95 从 23.7883% 降至
23.1706%（−0.6177 个百分点），small recall 从 35.6925% 降至 35.3501%
（−0.3424 个百分点）；medium/large recall 仅分别提高 0.4139/0.0937 个百分点。
small 匹配数少 91 个，逐类别 AP50-95 为 1 类升、9 类降。该结果在全量 30 epoch
下支持尺度分层阈值更有利于整体检测和固定操作点小目标覆盖。

复现入口为：

```bash
python -u scripts/run_p1_global_t050.py
```

只复核数据、不启动 GPU 训练：

```bash
python scripts/run_p1_global_t050.py --prepare-only
```

训练已写入 `runs/partial10_p1_global_t050_pilot30_s0/`，比较表及验收记录位于
`outputs/p1_global_t050_pilot30_s0/{comparison.md,comparison.json,acceptance_audit.json}`。
入口因已有 run 而拒绝重复启动；完整结果与边界见进度文档第 3.18 节。

### 尺度阈值配对学生 seed 复验

2026-10-27 23:47 已完成学生 seed1 的两组独立 30 epoch 对照及自动验收，后续独立复核通过。
固定已有 split seed0、B1 教师、两套伪标签与损失机制；仅学生训练 seed 改为 1。
这不生成新的标注 split，也不重新训练教师或生成伪标签。

```bash
python scripts/run_p1_scale_seed_pair.py --check-only
python -u scripts/run_p1_scale_seed_pair.py
```

入口顺序运行尺度分层与统一 0.50 两组，每组训练结束后自动进行完整 548 图
val960 和固定操作点分尺寸 recall，最终重放两组保存预测进行 GT 匹配验收。
日志、状态、比较和验收记录位于 `outputs/p1_scale_threshold_pair_s1/`。
训练目录分别为 `runs/partial10_p1_pilot30_s1` 和
`runs/partial10_p1_global_t050_pilot30_s1`。已有训练或派发记录时拒绝重复启动。
仅重做结果收集与验收可用 `--collect-only`；该选项不训练或推理。

论文表生成器已加入表 4-S3：报告 seed0/1 的单次结果、各方案
均值±样本标准差及种子内配对差的均值±样本标准差（ddof=1）。结果仍限于一个
split、同一教师和两个学生 seed，不作显著性结论，不改变原 Gate 3。

seed1 中尺度分层相对统一 0.50：mAP50-95 +0.2420 pp、small recall +0.6434 pp
（多匹配 171 个小目标），与 seed0 的两项收益方向一致。两学生 seed 的配对增益
均值±样本 SD 分别为 +0.4299±0.2657 pp 和 +0.4929±0.2128 pp。
独立验收命令为 `python scripts/audit_p1_scale_seed_pair.py`：36 项检查、33,047
冻结文件及两组各 548 图预测重放通过。结果、证据和限制见论文第 4.11 节。

全量 30 epoch 的去软权重确认配置和入口已准备，但不由筛选脚本自动启动：

```bash
python -u scripts/run_p1_no_soft_weights.py
```

它只关闭 `p1.soft_weights`，保留 `p1.uncertain_ignore=true` 和完整 P1 的其他条件。

## B2 学生 seed1 与主对照案例图

2026-10-30 12:11，用户终端完成 B2 seed1 的连续 30 epoch、548 图完整 val960 和
固定操作点 recall，复用已有 P1 seed1 后，两 seed 均支持 P1 的总体检测和小目标
召回增益；large recall 在两 seed 中方向不同，完整统计见表 4-S4 和实验第 4.12 节。
固定 split seed0、原 B1 教师及各自既有伪标签，640、batch8、SGD 和原增强不变。

已有结果直接核验：

```bash
python scripts/audit_b2_seed1.py
python scripts/build_detection_cases.py --check
python scripts/build_paper_tables.py --check
```

完整数值、自动/独立验收和日志保存在 `outputs/b2_pilot30_s1/`，训练为
`runs/partial10_naive_t050_pilot30_s1`。`--collect-only` 只重做完整已有评估的验收/
汇总，不训练或推理；重做后须再运行独立验收及表格生成/核对。入口有防重复保护，
在具有相同冻结输入且尚无本组 run/派发记录的环境中，训练复现命令为：

```bash
python scripts/run_b2_seed1.py --check-only
python -u scripts/run_b2_seed1.py
```

最终完整运行前的两次异常中断及一次手动交接，均已连同权重、日志、脚本/配置
快照和哈希保留在 `outputs/b2_seed1_interrupted_20261030_{091251,101330,102720}/`。
独立验收复核三层来源链，中断 epoch 不拼接或进入论文。用户手动训练已完成，
本组完整运行已冻结，不自动扩大训练或启动新 seed。

四类 seed0 成功与失败案例在新增 B2 seed1 结果出现前选定，图注、选图规则及
边界见[案例图](docs/paper/case_studies.md)。生成器只重放保存预测，在没有输出的
环境可用 `python scripts/build_detection_cases.py` 生成；已有输出时用 `--check`。
300 dpi PNG、单行图和 manifest 位于忽略目录 `outputs/paper_cases_b2_p1_s0/`；
Git 保存配置、脚本、测试与文稿，数据/日志/权重/图像均保留于忽略目录。

## 第一篇的低成本诊断

实验缺口和投入顺序见 [第一篇清单](docs/paper1_experiment_gaps.md)，已完成的
[ignore 可恢复增量](docs/paper/ignore_recovery_analysis.md) 仅重放保存候选，不训练或推理。
原始数据与诊断产物均在忽略目录；已有结果核验：

```bash
python scripts/audit_paper1_gaps.py --check
python scripts/audit_ignore_recovery.py --check
```

当前结论是有条件恢复空间、但直接降低阈值的新增独立匹配比例较低，尚不据此扩训。
有可靠筛选信号后再准备单因素有界训练命令，由用户自行运行。

2026-10-30 后续已完成翻转跨视图筛选及 128 tile 的 NMS/max-det
[候选覆盖检查](docs/paper/candidate_retention_analysis.md)，两项暂未给出值得学生训练的信号。
提高 max-det 没有新增候选，放宽 NMS 的新增正框独立匹配比例仅 4.29%；本轮停止该路线。
已有检查可在 `yolo` 环境只读重放：

```powershell
python scripts/audit_candidate_retention.py --check
```

用户最新授权允许有可靠信号时自主做至多 30 epoch 的小规模训练，启动时创建临时
进度定时任务，完成/失败后清理；更长训练仍须结果支持。该授权替代前述手动启动
安排，不表示已有新训练启动，每轮代码和文档修改使用中文 Git 提交。

跨输入尺度候选支持已完成探索和不重叠样本确认，固定候选未通过确认线，
见[跨尺度分析](docs/paper/scale_support_analysis.md)。只读复核：

```powershell
python scripts/audit_scale_support.py --check
python scripts/audit_scale_support.py --confirm --check
```

当前统一执行安排见[三篇文章与部署总计划第10节](docs/publication_and_edge_deployment_plan.md#10-当前可执行安排与职责)：
当前只推进A第一篇，C/B留作远期规划；有界小试由Codex执行和临时监控，
值得扩大训练后准备代码与命令交给用户。

按最新A范围和30epoch小试授权，本轮只补软权重正式单因素消融，
预算与后续判据见[第一篇清单第7节](docs/paper1_experiment_gaps.md#7-本轮唯一训练软权重正式单因素消融)。
旧5epoch中性结论不改写为过线；已有原始训练入口执行后自动完成val960和recall。
本组已于北京时间2026-10-30 16:28完成训练及自动评估，并通过独立验收；
去软权重mAP50-95−0.1623pp、small recall+0.9143pp（+243小目标），
存在精度/覆盖取舍，未达预先记录的扩大训练线，不启动长训练。请勿重复启动。
表4-S5、实验4.14已同步；六类关键词的论文依据、阅读范围与后续假设见
[A关键词调研](docs/paper/a_keyword_research.md)，未另行启动模型微调或新损失训练。
完成后的独立复核：

```powershell
python scripts/audit_p1_no_soft_weights.py
```

已有正伪框的跨尺度定位稳定性也已完成零训练诊断：两组不稳定small各仅6框，
未形成足够的置信度/类别分层比较，停止当前规则，不实现回归加权。
详见[定位可靠性诊断](docs/paper/positive_localization_analysis.md)；只读复核：

```powershell
python scripts/audit_positive_localization.py --check
```

原图上下文类别纠正的执行前协议、固定继续线和版本依据见
`docs/paper/context_rectification_analysis.md`，只检验已有small正伪框，不训练。
首次执行只做128张原图推理；已有结果只读重放：

```bash
python scripts/audit_context_rectification.py --check
```

探索通过继续线后，`--confirm`才允许另128张不重叠原图确认；未过线时拒绝执行。
128图探索已完成，只触发1次类别纠正且改错，总体/small匹配各−1；
未过线，不运行确认或训练。结果和成本见`docs/paper/context_rectification_results.md`。
论文归档在Zotero顶层
`YOLO-RegionSSOD`，清单与PDF版本说明见`docs/paper/zotero_literature_library.md`。

开放区域置信度校准的固定协议见`docs/paper/confidence_calibration_protocol.md`。
只拟合512个已有开放标注tile，再用来源不重叠的256个未标注tile检查；
若已有结果，只读核对命令为：

```bash
python scripts/audit_confidence_calibration.py --check
```

首次生成去掉`--check`，只新增512tile推理、不训练；`--confirm`仅在探索过线后
允许另256tile确认且不再推理/拟合。校准器不得使用隐藏GT或val标签拟合，
不更改已有正伪标签集合。通过诊断不等同于学生效果或值得长期训练。
2026-10-30探索已完成：整体Brier未改善，虽small改善，仍未过固定继续线；
不执行确认或训练，完整计数/成本/局限见`docs/paper/confidence_calibration_results.md`。

小框NWD回归的单因素筛查见`docs/paper/small_nwd_protocol.md`。
先核对冻结输入，只估计开放GT常数、不训练：

```bash
python scripts/screen_p1_small_nwd.py --check-only
python scripts/screen_p1_small_nwd.py --geometry-only
```

第二条额外做真实YOLO11s的GPU AMP前反传，零优化器步骤。源码/协议需先提交；
通过后首次去掉这两个选项只新增一组5epoch，复用已验收的同子集P1参考，
随后完整val960和固定操作点recall。已有训练/阶段记录时拒绝重复启动。
验收为`python scripts/audit_small_nwd_screen.py`，必须包装进程exit0且全部阶段完成；
此筛查不自动派发30/100epoch，也不进入正式论文结果表。
2026-10-30已完成独立验收：mAP50-95−0.1746pp、small recall+1.5464pp，
原固定继续线失败、验收当时停止扩训；详细结果和限制见`docs/paper/small_nwd_results.md`。

用户随后明确至少30epoch才判断性能，旧5epoch仅保留探索记录。
当前新增唯一全量NWD30确认，固定原C/mix/面积边界与完整P1同预算配对，
协议`docs/paper/small_nwd30_protocol.md`。短程不再作为性能淘汰依据，
当前自主单候选固定30epoch，更多训练仍由用户执行。中文提交后按最新授权push。

```bash
python scripts/run_p1_small_nwd.py --check-only
python scripts/audit_small_nwd_tradeoff.py --check
```

第一条仅预检；首次由后台包装`scripts/run_small_nwd30_background.py`运行唯一
30epoch，再做548原图val960/recall。状态、启动记录与日志位于
`outputs/p1_small_nwd_pilot30_s0/`；已有状态/dispatch/产物时拒绝重启。
完成后运行`python scripts/audit_p1_small_nwd.py`独立验收。
训练启动时创建临时监控，终态处置后删除，不并行第二组或自动扩训。
该唯一训练于2026-10-30 18:54启动，20:11发现中断，20:14二次核实：
包装/训练进程均不存在，只完成13/30epoch，日志停在第14轮181/1414batch。
Windows记录原包装进程停止响应并关闭；退出码缺失，根因尚未确定。
本次未完成30轮及最终评估，不能判断性能或是否值得扩训。
证据与处理见[中断记录](docs/paper/small_nwd30_interruption.md)；
临时监控`a-nwd`在中断处置后删除，不重复运行启动入口，不拼接或自动重训。

2026-10-31用户确认上次为手动中断，改由本人终端重新运行完整30轮。
旧输出及run已保留到`outputs/p1_small_nwd_manual_handoff_20261031/`，
原中断快照仍保留；新训练从COCO初始化，配置/损失/继续线不变。
在Git Bash执行：

```bash
source /d/MiniConda/etc/profile.d/conda.sh
conda activate yolo
cd /d/Python/YOLO-RegionSSOD
python -u scripts/run_small_nwd30_background.py
```

虽然入口文件名带background，这条命令会在当前终端等待全部阶段结束；
训练进度写入`outputs/p1_small_nwd_pilot30_s0/stdout.log`。
可在另一Git Bash窗口运行
`tail -f /d/Python/YOLO-RegionSSOD/outputs/p1_small_nwd_pilot30_s0/stdout.log`查看。
30轮后自动完成548原图val960、分尺寸recall和对比；训练完由用户通知验收。
上段为当时交接命令，现已执行完成，不要再次启动。
2026-10-31用户完成完整30轮、val960/recall，exit0及独立验收通过。
NWD相对完整P1的mAP50-95−0.2123pp、small recall−0.6698pp，
两条预登记后续线均未达到，保留原P1，不扩大本变体训练。
完整类别/尺寸取舍与证据见[全量NWD结果](docs/paper/small_nwd30_results.md)；
仅需复核时运行`python scripts/audit_p1_small_nwd.py`，不消耗新训练或推理。

NWD之后的小框分配诊断见`docs/paper/assignment_geometry_results.md`：
初版减少框外点却丢失已有小目标分配，不训练该初版。
带整图覆盖保护的新规则已完成128个新基础tile确认，目标保留100%，
但框外监督质量只降低48.10%，未达固定50%条件；两种分配候选均不训练。
见`docs/paper/assignment_coverage_results.md`。这仍是零训练诊断，
不是新的学生性能结果，也不据此改写原P1主对照。

## Baseline 定义

| 名称 | 训练数据 | 作用 |
|---|---|---|
| B0 Full-sup reference | 完整 VisDrone 标签 | 上界参考，不参与方法变量 |
| B1 Partial-sup crop | 仅标注 tile | 区域局部监督主 baseline |
| B2 Naive self-training | 标注 tile + 全局阈值伪标签 tile | 半监督基线 |
| P1 Scale-aware | 尺度分层阈值 + 置信度软权重 + uncertain-ignore | 已验收，有正向初步收益；原 Gate 3 未达门槛 |

## 数据目录约定

每次 `make-partial` 生成一个新目录：

```text
data/visdrone_area10/
  labeled/images/train/
  labeled/labels/train/
  unlabeled/images/train/
  splits/labeled_train.txt
  splits/unlabeled_train.txt
  splits/val.txt
  manifest.jsonl
  summary.json
  dataset.yaml
```

`manifest.jsonl` 会逐图记录 tile 坐标、被保留标签数和未标注 tile 中实际可见的
隐藏 GT 数。这个字段用于分析伪标签漏检，不允许作为训练输入。

## 重要限制

- Ultralytics 的 mAP 不是 VisDrone 官方 AP；论文最终表仍需官方工具或服务器。
- 现有 VisDrone test-dev 已被旧项目多次查看，不能继续当作盲测集。
- 表 4-1 至 4-6 仍为 seed0 主对照，表 4-S4 是固定 split/教师下 B2/P1 的两学生 seed 复验；原 Gate 3 不变，见进度文档第 3.8、3.20 节。
- 首轮不要把网络结构、P2、注意力和推理分辨率一起修改，否则无法归因。

详细实验计划见 `docs/experiment_plan_8weeks.md`，命令协议见
`docs/baseline_protocol.md`。

论文完成计划、当前进度和后续会话交接见
`docs/paper_completion_plan.md`；新的 Codex 会话应先阅读根目录 `AGENTS.md`。
