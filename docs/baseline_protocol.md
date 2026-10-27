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
`best.pt` 生成的伪标签；后续 P1 pilot 必须复用同一教师。学生训练轮数一致；
B2 数据量和每轮批次数更多，因此实际计算量、耗时及教师 100 epoch 成本须分别
记录，不能将相同 epoch 表述为相同计算预算。

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

2026-10-27 的 seed0 配对结果已测量：完整原图 val960 的 B1/B2 mAP50-95 分别为
0.229114/0.234534，差 +0.5421 个百分点，Gate 2 通过。已完成的 run 可直接复用；
下一步实现 P1 并做 Gate 3 的 30 epoch 对照，结果和边界见
`docs/paper_completion_plan.md` 第 3.5 节。

### 4.2 后续 100 epoch 正式 B2

完成 pilot 和对应数据诊断后，正式 B2 的训练命令是：

```powershell
regionssod train --config configs\train_naive_10.yaml --data data\visdrone_area10\dataset_naive_t050.yaml --name partial10_naive_t050_s0
```

```bash
regionssod train --config configs/train_naive_10.yaml --data data/visdrone_area10/dataset_naive_t050.yaml --name partial10_naive_t050_s0
regionssod validate --weights runs/partial10_naive_t050_s0/weights/best.pt --data data/visdrone_area10/dataset_naive_t050.yaml --imgsz 960 --name partial10_naive_t050_s0_val960
```

### 4.3 P1 的实现与 Gate 3 pilot

Gate 2 已通过，P1 已实现。完整生成、合并、30 epoch 训练、val960 和分尺寸 recall
命令见 README 的 P1 章节。固定复用 B2 的 B1 100 epoch 教师：
`58f40e1e7b397c3f0b873bbc9cbd9e50660b41c48c9cadb1e7ef3d0f0b0ebf54`。
学生仍从同一 COCO 权重初始化，640、batch8、SGD、seed0、增强和原损失 gains 不变。

第一轮预设参数与语义：

- `configs/pseudo_p1_10.yaml`：按原 tile 中框面积 <32²、[32²,96²)、>=96²
  分 small/medium/large，正阈值为 0.30/0.50/0.50，包含阈值边界。教师预测仍
  使用 960、NMS IoU0.7、max_det500、batch8，与 B2 一致。
- conf>=对应阈值为正框，0.05<=conf<阈值为 uncertain-ignore；更低候选不进入
  方法。尺度和阈值在增强前确定，增强后不重新分层。仅有正框的 tile 才入训。
- 五列 YOLO 标签仍只含正框；`metadata.jsonl` 保存教师置信度和 ignore 框，
  `candidates.jsonl` 保留所有 tile 的下限以上、NMS 后候选（含被排除 tile）。
  `summary.json` 中的 per_scale 与 ignore_boxes 只统计入训 tile。
- 数据增强内使用整数 object ID 携带元信息，统一经过 mosaic、透视、缩放、
  翻转和裁剪；collate 前解码并分离正框/ignore，ID 不进入 TAL 或模型类别。
  ignore 框沿用 Ultralytics 的几何裁剪/最小框过滤，过小或被裁掉的框会被移除。
- TAL 分配只使用正框；匹配伪标签的 anchor 的整个 BCE 类别向量乘教师 conf，
  人工标签乘 1；未加权 TAL target-score 总和作分母，避免权重被归一化抵消。
  box/DFL、目标分配和原分类 target 不变。
- ignore 作用于中心落入 uncertain 框的负 anchor，屏蔽其所有类别 BCE；正 anchor
  优先，不会因与 uncertain 框重叠而失去训练。框外背景项仍按原 YOLO 损失处理。
- `train_p1_10.yaml` 中 `p1.soft_weights`、`p1.uncertain_ignore` 为布尔消融开关。
  尺度消融需用三阈值均为 0.50 的新 policy、新输出目录重新生成，不覆盖原产物。
  P1 数据必须配 `method: p1`，训练器拒绝缺失元信息或与标签不对应的伪标签。

实现绑定 Ultralytics 8.4.172；不能升级依赖后直接复用本适配器。训练开始前在
run 的 `experiment.json` 冻结配置、数据 YAML、元信息、源代码 SHA256、方法
开关和教师摘要；`args.yaml` 仍保存实际 Ultralytics 参数。正式启动前另做完整
产物 SHA256 清单，并记录输入规模、每轮批次数、耗时和教师成本。

Gate 3 的主门槛为 B2 val960 mAP50-95 +0.005，即至少 **0.239534294688062**。
小目标补充门槛在查看 P1 结果前固定为：同一 548 图完整 val，conf=0.25，类别
匹配、IoU>=0.50、max_det=500、原图 GT 面积<32² 的 recall 严格高于 B2。
所有尺寸 GT 联合按预测置信度降序一对一匹配，再统计尺寸；不能以 P1 伪标签
的隐藏 GT recall 代替学生完整 val recall。本工具是 YOLO 开发诊断，不等于官方
AP_small/AR500。最终论文仍需官方评测。原计划以两个 Gate 3 条件均满足作为扩训门槛。

2026-10-27 验收结果：P1 mAP50-95=0.23788331849556607，比 B2 提升
0.3349 个百分点；小目标 recall=0.35692516085336945，提升 1.8776 个百分点。
原 Gate 3 的整体 mAP 条件未达到、小目标条件达到，总判定为未通过。
用户随后明确将“保留有提升的阶段成果”作为当前目标，后续优先整理方法和已有
结果，补充验证规模按论文需要决定。数值门槛、原判定及这一结果后的决策分别
记录在 `docs/paper_completion_plan.md` 第 3.8 节；冻结数据和命令仍可复现本轮。

## 5. 对照规则

- B1 与 B2 必须从同一 `yolo11s.pt` 初始化。
- 同一实验组的 epoch、batch、imgsz、增强和 seed 完全一致。
- B2 的学生训练集包含 B1 标注 tile 与伪标签 tile；评估仍使用完整 val 原图。
- 第一轮只做一次离线自训练，不使用多轮迭代。
- 只根据 val 调阈值；最终确认前不得反复查看 test-dev。

## 6. 输出审计

论文主表与方法/实验初稿见 `docs/paper/README.md`。数字核对命令为
`python scripts/build_paper_tables.py --check`，重新生成表格使用
`python scripts/build_paper_tables.py`。本次为补齐三组对照，追加 B1 固定操作点
评估（已有产物直接复用）：

```bash
regionssod size-recall --weights runs/partial10_sup_pilot30_s0/weights/best.pt --data D:/Python/YOLO/data/visdrone/dataset.yaml --output runs/size_recall/b1_pilot30_s0 --imgsz 960 --batch 8 --device 0 --conf 0.25 --match-iou 0.50
```

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
