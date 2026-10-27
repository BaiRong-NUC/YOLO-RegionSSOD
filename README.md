# YOLO-RegionSSOD

面向高分辨率航拍图像的区域局部标注小目标半监督检测实验仓库。

本仓库对应一个明确的论文问题：每张高分辨率图像只标注少量区域，未标注区域
不能被视为纯背景。目标是复用 VisDrone、YOLO11 和航拍小目标经验，在单张
RTX 4070 Ti 12 GB 上完成毕业/中文核心所需的完整实验。

当前状态（2026-10-27）：P1 的 30 epoch、完整原图 val960 和小目标 recall 已完成
验收，19/19 检查通过。相对 B2，mAP50-95 从 23.4534% 升至 23.7883%
（+0.3349 个百分点），小目标 recall 从 33.8150% 升至 35.6925%
（+1.8776 个百分点）。原 Gate 3 的 +0.5 个百分点门槛未达到；按用户当前目标，
保留本轮积极的单 seed 初步成果，优先整理方法、结果和分析。验收及证据见
`docs/paper_completion_plan.md` 第 3.8 节。

论文初稿入口：[方法与实验写作](docs/paper/README.md)。已有
[论文结果表格](docs/paper/results_tables.md)、[方法章节](docs/paper/methods.md)
和[实验分析章节](docs/paper/experiments.md)，B1/B2/P1 的同操作点分尺寸 recall
也已补齐。用 `python scripts/build_paper_tables.py --check` 核对表格与冻结结果。

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
- P1 的 30 epoch 配对结果可作为单 seed 初步成果；原 Gate 3 的判定和结果后的工作决策见进度文档第 3.8 节。
- 首轮不要把网络结构、P2、注意力和推理分辨率一起修改，否则无法归因。

详细实验计划见 `docs/experiment_plan_8weeks.md`，命令协议见
`docs/baseline_protocol.md`。

论文完成计划、当前进度和后续会话交接见
`docs/paper_completion_plan.md`；新的 Codex 会话应先阅读根目录 `AGENTS.md`。
