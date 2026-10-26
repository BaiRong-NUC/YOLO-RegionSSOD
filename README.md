# YOLO-RegionSSOD

面向高分辨率航拍图像的区域局部标注小目标半监督检测实验仓库。

本仓库对应一个明确的论文问题：每张高分辨率图像只标注少量区域，未标注区域
不能被视为纯背景。目标是复用 VisDrone、YOLO11 和航拍小目标经验，在单张
RTX 4070 Ti 12 GB 上完成毕业/中文核心所需的完整实验。

当前状态：数据构造、部分监督 baseline、全局阈值自训练 baseline 和评估入口
已落地；尺度感知伪标签、低置信度 ignore 和最终方法实现留给第 3–6 周。

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

5 个 epoch 的 baseline 烟测：

```powershell
regionssod train `
  --config configs\train_partial_10.yaml `
  --epochs 5 `
  --name partial10_sup_smoke
```

正式部分监督 baseline：

```powershell
regionssod train `
  --config configs\train_partial_10.yaml `
  --data data\visdrone_area10\dataset.yaml `
  --name partial10_sup_s0
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
  --imgsz 960
```

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

训练自训练 baseline：

```powershell
regionssod train `
  --config configs\train_naive_10.yaml `
  --data data\visdrone_area10\dataset_naive_t050.yaml `
  --name partial10_naive_t050_s0
```

## Baseline 定义

| 名称 | 训练数据 | 作用 |
|---|---|---|
| B0 Full-sup reference | 完整 VisDrone 标签 | 上界参考，不参与方法变量 |
| B1 Partial-sup crop | 仅标注 tile | 区域局部监督主 baseline |
| B2 Naive self-training | 标注 tile + 全局阈值伪标签 tile | 半监督基线 |
| P1 Scale-aware | 尺度分层阈值 + 低置信度 ignore | 后续主方法候选 |

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
- 当前版本不实现尺度感知阈值和 uncertain-ignore，只提供可复现 baseline。
- 首轮不要把网络结构、P2、注意力和推理分辨率一起修改，否则无法归因。

详细实验计划见 `docs/experiment_plan_8weeks.md`，命令协议见
`docs/baseline_protocol.md`。
