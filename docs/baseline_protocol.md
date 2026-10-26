# Baseline 执行协议

本文的 `powershell` 命令块适用于 PowerShell。Git Bash（提示包含 `MINGW64`）
需要先按 `README.md` 的环境章节加载 `/d/MiniConda/etc/profile.d/conda.sh` 并激活
`yolo`；使用 `cd /d/Python/YOLO-RegionSSOD` 切换目录，参数路径使用正斜杠。
例如，正式 10% 数据的 Git Bash 命令是：

```bash
regionssod make-partial --source-data D:/Python/YOLO/data/visdrone/dataset.yaml --output data/visdrone_area10 --ratio 0.10 --seed 0 --unlabeled-per-image 1
```

## 1. 生成 5/10/20% 数据

每个比例使用独立输出目录，不要覆盖已有目录：

```powershell
regionssod make-partial --source-data D:\Python\YOLO\data\visdrone\dataset.yaml --output data\visdrone_area05 --ratio 0.05 --seed 0 --unlabeled-per-image 1
regionssod make-partial --source-data D:\Python\YOLO\data\visdrone\dataset.yaml --output data\visdrone_area10 --ratio 0.10 --seed 0 --unlabeled-per-image 1
regionssod make-partial --source-data D:\Python\YOLO\data\visdrone\dataset.yaml --output data\visdrone_area20 --ratio 0.20 --seed 0 --unlabeled-per-image 1
```

seed 1/2 必须使用不同输出目录，例如 `visdrone_area10_seed1`。

检查 `summary.json`：

- `images_used` 应等于 6471。
- `labeled_tiles` 应等于 `images_used`。
- `visible_gt_boxes_in_unlabeled_tiles` 应大于零，否则未标注区域没有隐藏目标，
  该 split 不能用于半监督失效分析。

## 2. B0 Full-supervised reference

完整 VisDrone 标签训练，只作为性能上界，不参与方法变量：

```powershell
regionssod train --config configs\train_full.yaml --name full_sup_s0
```

训练完成后仍使用完整原图 960 验证：

```powershell
regionssod validate --weights runs\full_sup_s0\weights\best.pt --data D:\Python\YOLO\data\visdrone\dataset.yaml --imgsz 960 --name full_sup_s0_val960
```

## 3. B1 Partial-supervised

烟测（先按 README 生成 `data/smoke_area10` 的 20 张图数据）：

```powershell
regionssod train --config configs\train_partial_10.yaml --data data\smoke_area10\dataset.yaml --epochs 5 --project runs\smoke --name partial10_sup_smoke5
regionssod validate --weights runs\smoke\partial10_sup_smoke5\weights\best.pt --data data\smoke_area10\dataset.yaml --imgsz 960 --name partial10_sup_smoke5_val960
```

正式：

```powershell
regionssod train --config configs\train_partial_10.yaml --data data\visdrone_area10\dataset.yaml --name partial10_sup_s0
```

完整原图 960 验证：

```powershell
regionssod validate --weights runs\partial10_sup_s0\weights\best.pt --data data\visdrone_area10\dataset.yaml --imgsz 960 --name partial10_sup_s0_val960
```

Git Bash 下，正式训练及完成后的验证命令是：

```bash
regionssod train --config configs/train_partial_10.yaml --data data/visdrone_area10/dataset.yaml --name partial10_sup_s0
regionssod validate --weights runs/partial10_sup_s0/weights/best.pt --data data/visdrone_area10/dataset.yaml --imgsz 960 --name partial10_sup_s0_val960
```

正式 B1 使用 6471 张标注 tile，按配置从 COCO 预训练重新开始，固定 100 epoch。
20 图烟测权重仅用于工程验收。

## 4. B2 Naive self-training

先固定 teacher。使用 B1 的 `best.pt`，不要使用 last.pt 或 epoch 中途权重。

```powershell
regionssod pseudo-label --weights runs\partial10_sup_s0\weights\best.pt --images-list data\visdrone_area10\splits\unlabeled_train.txt --output data\visdrone_area10\pseudo\naive_t050 --threshold 0.50 --conf-floor 0.05 --imgsz 960 --batch 8 --device 0
```

生成器将选中的输入保存为 `prediction_images.txt` 并使用文件清单按 batch 推理，
避免 Python 图片路径列表被 Ultralytics 当作一个大批次。`summary.json` 记录
实际请求的 batch、device 和 max_det。Git Bash 的生成和合并命令见 README。

检查 `summary.json`：

- `accepted_images` 不能接近 0。
- `empty_predictions` 高时，先分析教师漏检。
- 记录 `per_class`，避免伪标签只覆盖 car/pedestrian。

合并：

```powershell
regionssod combine-lists --inputs data\visdrone_area10\splits\labeled_train.txt data\visdrone_area10\pseudo\naive_t050\pseudo_train.txt --output data\visdrone_area10\combined_naive_t050.txt --dataset-yaml data\visdrone_area10\dataset_naive_t050.yaml --source-data D:\Python\YOLO\data\visdrone\dataset.yaml
```

### 4.1 先执行 Gate 2 的 30 epoch 配对 pilot

B1 的 100 epoch 正式结果用于主表及固定教师；Gate 2 另外比较两个从同一 COCO
预训练初始化、各训练 30 epoch 的学生。B2 pilot 使用上面固定的 B1 100 epoch
`best.pt` 生成的伪标签；后续 P1 pilot 必须复用同一教师。学生阶段的训练预算
一致，教师阶段的 100 epoch 成本另行记录。

Git Bash 按顺序执行以下命令，每个训练结束后都对其 `best.pt` 做完整原图 960
验证。两个训练都保留原配置的 640、batch 8、SGD、seed 0：

```bash
regionssod train --config configs/train_partial_10.yaml --data data/visdrone_area10/dataset.yaml --epochs 30 --name partial10_sup_pilot30_s0
regionssod validate --weights runs/partial10_sup_pilot30_s0/weights/best.pt --data data/visdrone_area10/dataset.yaml --imgsz 960 --name partial10_sup_pilot30_s0_val960
regionssod train --config configs/train_naive_10.yaml --data data/visdrone_area10/dataset_naive_t050.yaml --epochs 30 --name partial10_naive_t050_pilot30_s0
regionssod validate --weights runs/partial10_naive_t050_pilot30_s0/weights/best.pt --data data/visdrone_area10/dataset_naive_t050.yaml --imgsz 960 --name partial10_naive_t050_pilot30_s0_val960
```

Gate 2 比较上述两个独立 30 epoch run 的 mAP50-95，要求 B2 不低于 B1 超过
0.2 个百分点（0–1 指标差不低于 -0.002）。不要将 B2 30 epoch 与 B1 100 epoch
直接比较，也不要将 100 epoch run 的第 30 行当作独立 30 epoch 实验：学习率
调度和 `close_mosaic` 时点不同。通过该 gate 后才实现 P1。

### 4.2 后续 100 epoch 正式 B2

完成 pilot 和对应数据诊断后，正式 B2 的训练命令是：

```powershell
regionssod train --config configs\train_naive_10.yaml --data data\visdrone_area10\dataset_naive_t050.yaml --name partial10_naive_t050_s0
```

```bash
regionssod train --config configs/train_naive_10.yaml --data data/visdrone_area10/dataset_naive_t050.yaml --name partial10_naive_t050_s0
regionssod validate --weights runs/partial10_naive_t050_s0/weights/best.pt --data data/visdrone_area10/dataset_naive_t050.yaml --imgsz 960 --name partial10_naive_t050_s0_val960
```

## 5. 对照规则

- B1 与 B2 必须从同一 `yolo11s.pt` 初始化。
- 同一实验组的 epoch、batch、imgsz、增强和 seed 完全一致。
- B2 的学生训练集包含 B1 标注 tile 与伪标签 tile；评估仍使用完整 val 原图。
- 第一轮只做一次离线自训练，不使用多轮迭代。
- 只根据 val 调阈值；最终确认前不得反复查看 test-dev。

## 6. 输出审计

每个正式 run 至少保存：

```text
weights/best.pt
weights/last.pt
args.yaml
results.csv
experiment.json
validation/metrics.json
SHA256SUMS.txt
```

运行结束后在仓库外或忽略目录中计算权重和关键配置的 SHA256，不要提交权重。
