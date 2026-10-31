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

### 4.4 第①组：P1 去掉 uncertain-ignore

本组已于 2026-10-27 完成并通过独立验收，现有 run 和评估可直接复用。去掉
ignore 相对完整 P1 的 mAP50-95 +0.0217 个百分点、固定操作点 small recall
−0.1392 个百分点；不支持将整体 mAP 收益归因于 ignore，详见进度文档第 3.13 节。

使用独立配置 `configs/train_p1_no_ignore_10.yaml`：

```yaml
p1:
  soft_weights: true
  uncertain_ignore: false
```

除 run 名称与上述 ignore 开关外，配置与完整 P1 冻结 `experiment.json` 一致。
复用 `data/visdrone_area10/dataset_p1.yaml` 和全部伪标签/元信息，保留相同增强、
尺度阈值及 COCO 初始化。学生为 640、30 epoch、batch8、SGD、seed0；教师不重训。
ignore 框仍经过相同数据加载和增强，只关闭负分类项的屏蔽。

在 `yolo` 环境、仓库根目录执行：

```bash
python -u scripts/run_p1_no_ignore.py
```

脚本先核对原 P1 的冻结输入与参考产物，再串行执行训练、完整 val960、分尺寸
recall 和结果对比；可先用 `--check-only` 仅做预检。比较对象固定为
`runs/partial10_p1_pilot30_s0/weights/best.pt`；分尺寸操作点仍为 conf=0.25、
匹配 IoU=0.50、NMS IoU=0.70、max_det=500，548 张完整验证原图。

单独的基础训练/评估命令如下；自动脚本已经串行执行这些步骤，选择一种入口即可：

```bash
regionssod train --config configs/train_p1_no_ignore_10.yaml
regionssod validate --weights runs/partial10_p1_no_ignore_pilot30_s0/weights/best.pt --data data/visdrone_area10/dataset_p1.yaml --imgsz 960 --batch 8 --device 0 --name partial10_p1_no_ignore_pilot30_s0_val960
regionssod size-recall --weights runs/partial10_p1_no_ignore_pilot30_s0/weights/best.pt --data D:/Python/YOLO/data/visdrone/dataset.yaml --output runs/size_recall/p1_no_ignore_pilot30_s0 --imgsz 960 --batch 8 --device 0 --conf 0.25 --match-iou 0.50
```

训练为 `runs/partial10_p1_no_ignore_pilot30_s0`，评估为
`runs/val/partial10_p1_no_ignore_pilot30_s0_val960`、
`runs/size_recall/p1_no_ignore_pilot30_s0`。自动对比输出为
`outputs/p1_no_ignore_pilot30_s0/{comparison.md,comparison.json}`，正差值表示去掉
ignore 后指标更高；报告不预设哪一组获胜。训练时间取 CSV 最后一行累计时间，
不解释为独立 ignore 模块的精确耗时；指标仍属于 Ultralytics 开发评测。

### 4.5 软权重的低成本配对筛选与全量确认

用户最新执行要求是：每次验证实验成本受控，发现值得后续训练的方向后提供
训练指令并停止；每个完成的实验用中文提交信息提交并推送 Gitee。

本次先运行 `python -u scripts/screen_p1_soft_weights.py`：从固定 10% 区域数据中
随机选取 1024 张源训练图（选择 seed=20261027），保留其人工标注 tile 与已接受
伪标签 tile；不使用隐藏 GT 的计数或框进行选择。两组共用同一份 1765-tile
列表与裁减后的置信度/ignore 元信息，仅 `soft_weights` 开关不同，ignore 均开启。

配对配置为 `configs/screen_p1_soft_weights_5ep.yaml` 和
`configs/screen_p1_no_soft_weights_5ep.yaml`。两组均为 COCO 初始化、5 epoch、
640/batch8/SGD/seed0，统一 `close_mosaic=0`，其余超参数与完整 P1 相同；
训练后各自以 best.pt 在相同 548 图完整原图 val960 和固定操作点评估。
默认 warmup 仍为 3 epoch，短程结果受学习率调度和子集影响。

提前固定的筛选信号为：mAP50-95 变化绝对值≥0.003（0.3 个百分点），小目标
recall 同向变化。仅用于判断是否值得全量确认；不得放入论文正式表，也不得
与旧 30 epoch 模型直接比较。结果为
`outputs/screens/p1_soft_weights_1024_e5/comparison.json`，总训练预算固定为两组
各 5 epoch，不自动扩大数据量、轮数或启动全量确认。

本轮已完成并独立验收。去掉软权重相对保留软权重的 mAP50-95 为
15.4635% 对 15.3751%（+0.0884 个百分点），small recall 为 28.0920% 对
28.0129%（+0.0790 个百分点），mAP50 为 26.5932% 对 26.8181%
（−0.2249 个百分点）。变化未达到 0.3 个百分点候选线，结论为中性，
**不启动全量去软权重 30 epoch 训练**。验收记录为同目录 `acceptance_audit.json`。

全量确认入口 `python -u scripts/run_p1_no_soft_weights.py` 使用
`configs/train_p1_no_soft_weights_10.yaml`，只关闭软权重，保留 ignore，与完整 P1
一样使用全部 11310 个 tile、30 epoch、同一教师和 COCO 初始化，并自动完成
val960、分尺寸 recall 和与完整 P1 的对比。当前未出现达到筛选线的信号，因此
保留该入口但不执行；只有改变证据条件后才应重新评估是否使用。

**2026-10-30执行更新：**用户随后允许Codex自主进行至多30epoch小试，并指定当前
仅推进A。为填补正式机制证据缺口，本轮仅执行上述去软权重30epoch单组对照，
预算和停止条件见`docs/paper1_experiment_gaps.md`第7节；不把旧中性筛查追认为过线。
训练仍使用原入口/配置，不改教师、伪标签或网络；完成后用
`python scripts/audit_p1_no_soft_weights.py`核验全量冻结输入和两组保存预测。
本组于北京时间2026-10-30 14:30启动、16:28正常完成，临时心跳`a-30`
按10分钟间隔监控，验收后清理。16项独立检查、23783冻结输入和两组548图
预测重放通过；去软权重mAP50-95−0.1623pp、small recall+0.9143pp（+243），
未达扩大训练条件，当前不扩训；表4-S5/实验4.14保留取舍。请勿重复启动。
更大规模训练仍交给用户。文献与下一项待验证问题见
`docs/paper/a_keyword_research.md`，当前不并行改损失或启动新模型训练。

### 4.6 尺度阈值低成本筛选与全量确认

固定沿用软权重筛选的 1024 张训练原图。尺度分层 5 epoch 结果直接复用；新训练
一组统一阈值 0.50，软权重和 ignore 均保留。新组从冻结的
`p1_s030_m050_l050/candidates.jsonl` 重划正/ignore 状态，不重跑教师在 1024 张
原图上的推理，也不读取隐藏 GT。全量训练确认的数据也从同一冻结教师候选生成。

低成本筛选入口：

```bash
python -u scripts/screen_p1_scale_threshold.py
```

新组配置为 `configs/screen_p1_global_t050_5ep.yaml`：1024 图内接受 713 个伪标签
tile，合并后 1737 个训练 tile，5 epoch、640、batch8、SGD、seed0、
`close_mosaic=0`。完整 val960 与 recall 操作点与尺度分层组相同。结果差值为
“统一 0.50 − 尺度分层”：mAP50-95 +0.2237 个百分点、small recall
−3.1230 个百分点、medium/large recall +1.1066/+1.7807 个百分点。

自动候选规则未触发（总体增益小于 0.3 个百分点，且小目标与总体方向不一致）。
基于本文小目标覆盖目标，这里将 3.1230 个百分点的固定操作点召回下降视为值得
做一次全量确认的机制信号，而不是统计显著或正式方法结论。

全量确认已于 2026-10-27 20:00 完成，20:21 独立审计通过。复现入口：

```bash
python -u scripts/run_p1_global_t050.py
```

脚本从冻结候选生成新数据，不覆盖尺度分层产物。已生成并验证：

- `data/visdrone_area10/pseudo/p1_global_t050_from_candidates_20261027/`
- `data/visdrone_area10/combined_p1_global_t050.txt`
- `data/visdrone_area10/dataset_p1_global_t050.yaml`

新数据有 4624 个伪标签 tile、27667 个正框和 57219 个 ignore 框，合并 11095 个
训练 tile。确认组保留 `soft_weights=true`、`uncertain_ignore=true`，仅将三个尺度
阈值统一为 0.50，仍为 30 epoch、同一 COCO 初始化，实际训练超参数、完整 val960
和 recall 协议与完整 P1 相同。阈值变化会改变正/ignore 划分、入训 tile 数和每轮
batch 数，因此比较的是完整机制方案，而不是同数据量下的等计算量对照。

差值为“统一 0.50 − 完整 P1”：mAP50 −0.7998 个百分点、mAP50-95 −0.6177
个百分点、small recall −0.3424 个百分点，medium/large recall 分别提高
0.4139/0.0937 个百分点。small 固定操作点匹配数由 9486 降至 9395，逐类别 AP50-95
为 1 类升、9 类降。全量结果支持尺度分层阈值更有利于总体检测和小目标覆盖。
best/last 权重 SHA256 分别为
`94571aae799119becc033fa6a7296eb0326dbf7fb83e44f650ef7ab488728a4a` 与
`cd96237d62d121ec8132cdfd30dbce94aefe39afd887bfde6a39759ebae3b5ff`。
比较与验收记录在 `outputs/p1_global_t050_pilot30_s0/`。

### 4.7 尺度阈值配对学生 seed 复验

2026-10-27 用户授权在当前全量配置下补学生 seed1。split 仍为 seed0，教师、
候选与两套伪标签均复用；不把本次复验写成新增 split 或独立教师实验。
配置为 `configs/train_p1_10_seed1.yaml` 和
`configs/train_p1_global_t050_10_seed1.yaml`，分别只对对应 seed0 配置修改
学生 seed 和 run 名，保留 COCO 初始化、640、30 epoch、batch8、SGD、
`close_mosaic=10`、软权重与 ignore。两组顺序运行：

```bash
python scripts/run_p1_scale_seed_pair.py --check-only
python -u scripts/run_p1_scale_seed_pair.py
```

预检核对冻结输入和 seed0 产物 SHA256、统一阈值 metadata/标签对应关系。
每组连续完成 30 epoch 后使用 best.pt 做 548 图完整 val960、conf=0.25、
匹配 IoU=0.50、NMS IoU=0.70、max_det=500 的 recall，保存预测并重放 GT 匹配。
最终复查运行前后输入哈希、实际参数配对、教师/损失和验证列表；产物在
`outputs/p1_scale_threshold_pair_s1/`，日志为 `stdout.log`、`stderr.log`。
不得重复派发；`--collect-only` 仅从已有完整评估重做验收和汇总。

两 seed 汇总采用样本标准差（ddof=1），配对差先按每个学生 seed 计算
“统一 0.50 − 尺度分层”再汇总。两 seed 不作显著性主张，不替代 AP_small
或正式 100 epoch 主矩阵；本组尺度消融不重新判定原 Gate 3。

2026-10-27 23:47 两组完成自动验收，独立复核命令为：

```bash
python scripts/audit_p1_scale_seed_pair.py
python scripts/build_paper_tables.py
python scripts/build_paper_tables.py --check
```

36 项独立检查通过，复核 33,047 个冻结文件及两组各 548 图保存预测的 GT 匹配。
seed1 中尺度分层相对统一 0.50：mAP50-95 +0.2420 pp，small recall +0.6434 pp
（9605 对 9434 个 small 匹配），与 seed0 的两项方向一致。两个学生 seed 的配对
增益均值±样本 SD 为 +0.4299±0.2657 pp、+0.4929±0.2128 pp。
medium recall 的方向在两 seed 间发生变化，large recall 均是统一阈值更高；
完整数值见表 4-S3，不声称所有尺寸或类别全面获益。独立验收记录为
`outputs/p1_scale_threshold_pair_s1/independent_audit.json`。

### 4.8 B2 学生 seed1、主对照配对统计与案例图

2026-10-30 12:11 完成用户终端训练及完整评估，随后 116 项独立验收通过。固定原
split seed0、B1 教师和 B2 全局 0.50 伪标签，只改学生 seed/name；新配置为
`configs/train_naive_10_seed1.yaml`，独立 30 epoch、640、batch8、SGD、原初始化
与增强不变。P1 seed1 直接复用；B2 仍用标准损失，不使用 P1 元信息或损失。

仅在具有相同冻结输入、没有本组 run/派发记录的环境中运行：

```bash
python scripts/run_b2_seed1.py --check-only
python -u scripts/run_b2_seed1.py
```

入口核对 37,695 个冻结文件，拒绝重复启动或覆盖。完成后使用各自 best.pt 在
548 图完整原图上做 val960；recall 使用 conf=0.25、匹配 IoU=0.50、NMS IoU=0.70、
max_det=500，重放 B2/P1 seed0/1 四组保存预测，原图尺寸 GT 共 38,759。
日志、比较和验收保存在 `outputs/b2_pilot30_s1/`。

已有完整结果的独立验收、案例核验及论文生成：

```bash
python scripts/audit_b2_seed1.py
python scripts/build_detection_cases.py --check
python scripts/build_paper_tables.py
python scripts/build_paper_tables.py --check
```

`--collect-only` 仅从已有完整评估重做汇总，不重训或推理；之后须重新运行独立
验收和表格生成/核验，使来源哈希同步。此前两次异常中断、一次手动交接的 run、
日志和脚本/配置哈希完整归档于
`outputs/b2_seed1_interrupted_20261030_{091251,101330,102720}/`；不拼接 epoch。
最终运行于 10:33 用户终端从原初始化开始，连续完成 30 epoch。

表 4-S4 的差值为 P1−B2。seed1 的 mAP50-95 +0.2438 pp、small recall +1.6932 pp
（多匹配 450）；两 seed 均值±样本 SD 为 +0.2893±0.0644 pp、+1.7854±0.1304 pp。
large recall 方向不一致，seed1 为 −0.6560 pp。各方法单次/均值±SD 和全部指标均
在表中报告，ddof=1；B1 未重复，原 seed0 主表不变，不作多 split/显著性或官方
AP_small 主张，不改变原 Gate 3，也不替代正式 100 epoch 主矩阵。

案例图从事先冻结的 seed0 预测生成，成功/失败均保留，详见
`docs/paper/case_studies.md`。没有输出时可用 `python scripts/build_detection_cases.py`
生成；已有输出用 `--check` 核验，PNG 与 manifest 均只保留于忽略目录。

### 4.9 候选保留的有限推理检查

2026-10-30 已完成同一冻结教师、conf≥0.05、960/batch8 的 128 tile
NMS/max-det 检查，样本排除先前 256 个翻转验证下标。原 NMS IoU=0.70、
max_det=500 对照分别与 NMS IoU=0.90 和 max_det=1000 比较，其他因素相同。
入训 89 tile 与排除 39 tile 分列，隐藏 GT 只用于离线诊断，不输出训练标签。
具体结果、原始/修订分析来源及停止条件见 `docs/paper/candidate_retention_analysis.md`。

```powershell
python scripts/audit_candidate_retention.py --check
```

该命令核对输入/输出哈希并重放已有推理结果，不训练或推理。没有输出的新环境可用
`python scripts/audit_candidate_retention.py` 首次生成；输出已存在则拒绝覆盖。
本机 v2 从保留的初次推理生成，来源链由 manifest 固定。

本轮两方案未达到继续条件，暂不进入学生筛选。后续如有可靠候选，可按用户
最新授权做至多 30 epoch 的有界实验；必须固定配对数据/配置与完整 val960，
明确单组/双组成本，不把短程结果写成正式矩阵。启动训练时创建临时心跳观察，
完成、失败或需要用户处理时通知，结束后清理，不自动扩大到更长训练。

### 4.10 跨输入尺度支持的探索与确认

固定原B1教师，对256tile的原960候选补充640输入预测。只支持原small uncertain
候选；正框不变，新增候选保持原坐标与置信度，先做同类正框去重，再做同类跨视图
一对一IoU支持。隐藏GT只评估新增独立匹配，不进入选择函数或训练标签。

```powershell
python scripts/audit_scale_support.py --check
python scripts/audit_scale_support.py --confirm --check
```

两条均只读重放；新环境首次生成移除`--check`，拒绝覆盖已有输出。探索中固定
floor0.25/support0.30候选后，另取不与既有384tile重叠的256tile确认，未达继续线，
不启动学生训练。选择函数见`src/regionssod/pseudo/consistency.py`，计数与边界见
`docs/paper/scale_support_analysis.md`。总执行安排以
`docs/publication_and_edge_deployment_plan.md`第10节为准。

### 4.11 原图上下文辅助类别纠正（零训练诊断）

冻结协议见`docs/paper/context_rectification_analysis.md`。先提交协议/代码，
再用同一B1教师、960/conf0.05/NMS0.70/max_det500/batch2预测128张固定随机原图；
仅将原已接受的small/conf<0.50正伪框与原图conf≥0.50支持匹配后纠正冲突类别。
所有原正框先占用独立支持；保留原几何、置信度和正框数。隐藏GT只用于诊断。

```bash
python -u scripts/audit_context_rectification.py
python scripts/audit_context_rectification.py --check
```

首次运行拒绝覆盖输出；`--check`不推理，只重放纠错/质量/覆盖统计并核对SHA256。
探索通过固定六项继续线后，`--confirm`才允许另128张独立原图确认，
`--confirm --check`只读验收两组。两组即使过线，也不能代替学生val960效果，
需单独固定学生预算。2026-10-30唯一128图探索已完成、继续线失败：
只纠正1框且改错，总体及small匹配各−1。只读验收372来源通过，
确认入口已验证拒绝，无确认组/训练；完整结果见
`docs/paper/context_rectification_results.md`。

### 4.12 开放区域拟合的分类置信度校准（诊断）

```bash
python -u scripts/audit_confidence_calibration.py
python scripts/audit_confidence_calibration.py --check
```

固定规则见`docs/paper/confidence_calibration_protocol.md`及
`configs/confidence_calibration_diagnostic.yaml`。先提交再运行：512开放tile用原B1
教师960推理，拟合只读`labeled/labels`；来源不重叠256tile探索及过线后另256tile
确认均复用既有正框，不推理或训练。全局校准是简化对照，分组映射不能仅优于
原分数就进训练。`--check`核对所有来源和输出并从开放标签重拟合重放；
`--confirm`必须先重放且通过探索，不允许在缺少探索时启动新的拟合。
输出位于`outputs/confidence_calibration_20261030/{fit,discovery,confirmation}/`。
2026-10-30拟合和探索已完成：整体分组Brier0.160310高于原分数0.158864，
继续线失败；未生成确认目录或训练，完整记录见`docs/paper/confidence_calibration_results.md`。

### 4.13 小框NWD回归单组5epoch筛查

协议`docs/paper/small_nwd_protocol.md`先提交；复用已验收的
`runs/screen/p1_full_area10_subset1024_e5_s0`，不重新训练参考。
同1024源图/1765tile/640/batch8/SGD/seed0/close_mosaic0，
仅面积<32²训练像素的小框回归替换一半CIoU为NWD；DFL、TAL及分类不变。
C从该子集的开放标注估计，不使用隐藏GT或val选择。

```bash
python scripts/screen_p1_small_nwd.py --check-only
python scripts/screen_p1_small_nwd.py --geometry-only
```

两条均不训练，后者GPU前反传无优化器更新。首次运行
`python -u scripts/screen_p1_small_nwd.py`只新增5epoch及完整548原图val960/recall。
后台包装`scripts/run_small_nwd_background.py`记录PID/创建时间/命令、
stdout/stderr与exit_code；启动训练后创建临时监控，结束/失败处理后删除。
独立验收`python scripts/audit_small_nwd_screen.py`只读重放、核对参数和SHA256。
继续线为相对同子集P1的mAP50-95≥+0.30pp且small recall≥0；
过线也只意味着值得单独登记全量30epoch确认，不自动扩训或交长期命令。
本组2026-10-30正常结束并独立验收通过，但mAP50-95下降0.1746pp，
虽small recall提高1.5464pp，仍未过固定继续线；不开展全量确认。
结果及成本见`docs/paper/small_nwd_results.md`。

### 4.14 用户更新预算后的NWD全量30epoch

上节“不开展全量确认”为五轮验收当时的决定。用户随后明确至少30epoch
才判断性能；此要求覆盖后续预算安排，旧失败记录不变，短程不再用于淘汰机制。
独立协议`docs/paper/small_nwd30_protocol.md`固定唯一新增30epoch学生，
复用完整P1参考、11310tile、seed0、COCO初始化、640/batch8/SGD/close_mosaic10。
不续接五轮，原C=20.22804306/mix0.5/面积边界及损失源码保持不变。

```bash
python scripts/run_p1_small_nwd.py --check-only
python scripts/audit_p1_small_nwd.py
```

第一条只预检；第二条仅在正常完整终态后独立验收，不会启动训练。
实际启动入口为`scripts/run_small_nwd30_background.py`，首次独占输出目录；
日志/进程身份/状态在`outputs/p1_small_nwd_pilot30_s0/`，run为
`runs/partial10_p1_small_nwd_pilot30_s0`，val960和recall依次自动执行。
启动后创建临时定时监控，结束或失败处置后删除；不因超时自动重启。
后续线在本轮未知时登记：AP≥+0.30pp且small recall非负，
或small recall≥+1.00pp且AP≥−0.10pp；过线只准备大规模训练交接。
中文提交并push由用户最新授权；当前仅A。
2026-10-30 18:54启动的唯一训练已中断：20:11发现、20:14再次核实，
包装30248/训练46428均不存在；CSV连续13轮，stdout停在14/30轮181/1414batch。
回执无exit_code，status仍为旧training；不能按完整成功流程验收。
Windows应用挂起事件对应原包装PID、创建时间和可执行路径，但根因未明。
原始文件与快照保留，见`paper/small_nwd30_interruption.md`。
本次未达到至少30轮判断要求，性能结论和后续线均未评定；
不运行续训、不清理原目录重跑、不自动启动其他候选，临时监控处置后删除。
上述预检命令会因已有状态拒绝执行，这是防止重复启动的正常保护。

2026-10-31手动重跑交接：用户确认原训练由本人中断，要求给命令自行训练，
完成后通知。旧输出/run在核对13个原文件SHA256、确认无原进程后，
移入`outputs/p1_small_nwd_manual_handoff_20261031/{output_original,run_original}`，
`handoff.json`保存路径映射；原中断证据快照未动。该操作允许原入口重新独占
输出位置，不修改任何训练源码、配置或冻结协议，不读取旧last.pt续训。

```bash
python -u scripts/run_small_nwd30_background.py
```

在仓库根目录和yolo环境执行一次；本次从原COCO初始化完整训练30轮，
自动接续548原图val960、固定操作点recall和比较，包装回执保存真实退出码。
当前终端会等待完成，详细进度在`outputs/p1_small_nwd_pilot30_s0/stdout.log`。
用户通知完成后再运行`python scripts/audit_p1_small_nwd.py`独立验收。
本次交接不启动训练，也不创建新的定时监控，按用户要求由其完成后通知。
预检和启动只在本次准备后的空状态有效；启动后不能再次预检或重复启动。

2026-10-31用户终端09:02:42重新启动，11:18:53包装正常退出（exit0），
连续30epoch、完整val960/recall与比较已结束；独立验收入口通过。
mAP50-95−0.2123pp、small recall−0.6698pp，两条后续线均未通过，
不自动扩训或按val调C/mix/面积边界。完整结果见`paper/small_nwd30_results.md`。
上面的启动命令已完成，现只能运行独立验收命令复核，不能重复训练。

### 4.15 伪框回归置信度单因素30轮

缓存诊断先提交后统计并通过，范围见`paper/regression_confidence_diagnostic.md`。
独立协议`paper/regression_confidence30_protocol.md`固定唯一11310tile学生，
原COCO初始化、seed0、640/batch8/SGD/close_mosaic10、30epoch；
仅对box/DFL乘原教师c，人工框1、TAL分母原样，其他P1机制不变。

```bash
python scripts/run_p1_regression_confidence.py --check-only
python scripts/run_p1_regression_confidence.py --engineering-only
python -u scripts/run_regression_confidence_background.py
python scripts/audit_p1_regression_confidence.py
```

前两条是预检/真实AMP零优化步工程检查，第三条只启动一次并自动完整val960/recall，
末条仅在exit0、连续30轮及全部阶段结束且原进程停止后独立验收。
预检不能在已有dispatch/status/run后再执行，以免混淆来源或重复训练。
输出`outputs/p1_regression_confidence_pilot30_s0/`保留日志/PID/创建时间/退出码，
启动时创建临时监控，正常静默，终态处置后删除；不重启、续训或拼接中断轮次。
预登记继续线为AP≥+0.30pp且small recall不降，或small recall≥+1.00pp且AP≥−0.10pp；
通过也只准备更大训练交接，不自动100轮。缓存质量不等同于学生性能。
本项于2026-10-31 12:31启动，23807来源与实际args配对核对通过，
但12:37首轮验证因损失函数错误要求`pseudo_weights`而失败（exit1）。
包装21156/训练10196均已停止，无CSV或checkpoint，不存在完整30轮结果；
不执行成功验收、不自动重试。临时监控按失败终态清理。
后续应先修复验证批次兼容并补充实际验证路径工程检查，
另行冻结独立运行后再给用户命令；当前上述启动入口不可重复执行。
详见`paper/regression_confidence30_failure.md`。

### 4.16 回归置信度验证兼容修正版：独立30轮入口

科学变量、配对参考和继续线沿用4.15；只修复验证路径，不改变训练损失。
旧失败源码/配置/协议六个文件已逐一与旧preflight核对后保存到
`outputs/p1_regression_confidence_pilot30_s0/frozen_source_snapshot/`。
新配置为`configs/train_p1_regression_confidence_valfix_10.yaml`，
新协议为`paper/regression_confidence30_valfix_protocol.md`。
修复、完整测试及零优化器步骤工程检查已完成。
2026-10-31用户明确“小训练由助手启动，大训练找用户”，
本次由助手在归档已停止的首轮中断证据后，独立启动一次完整30轮。

```powershell
python scripts/run_p1_regression_confidence.py --validation-fix --check-only
python scripts/run_p1_regression_confidence.py --validation-fix --engineering-only
```

工程检查增加实际标准验证加载器的2张原图、EMA缓存训练criterion、
DetectionValidator训练期调用、原P1损失精确对照及训练元数据保护。
其结果绑定源码哈希；检查通过后从同一COCO初始化独立训练30轮：

```powershell
python -u scripts/run_regression_confidence_background.py --validation-fix
```

run为`runs/partial10_p1_regression_confidence_valfix_pilot30_s0`，
输出为`outputs/p1_regression_confidence_valfix_pilot30_s0/`。
已有run/status/dispatch/回执时不覆盖，不允许接续旧失败进度。
完成val960/recall、exit0且原进程停止后：

```powershell
python scripts/audit_p1_regression_confidence.py --validation-fix
```

至少完整30轮后按原两条继续线判断。过线也不自动扩训；
缓存质量、工程检查及单seed结果不等同于显著性或创新性证明。
2026-10-31实际工程检查已通过：真实YOLO11s AMP前反传255梯度张量有限，
标准验证2图/1batch及原P1损失精确对照通过，训练缺元数据仍拒绝，
优化器步数0；工程产物加入后预检23809来源通过。
修复代码/协议提交`84817b2`，124tests/五文件Pyright通过。
13:28启动于首轮265/1414batch中断，包装20616/训练1540均已停止；
原回执无exit_code、status仍training，未伪造成功/失败退出码。
9个原始文件与六份源码快照保存至
`outputs/p1_regression_confidence_valfix_interrupted_20261031_133202/`。
当前按新授权准备独立启动一次，不续训/拼接，之后只监控原新进程。
实际已于2026-10-31 13:36通过Win32_Process.Create隐藏独立启动，
包装22020/训练22512命令和创建时间与回执匹配；临时监控`a`已创建。
启动预检23809来源通过，实际args除name/save_dir外与完整P1相同，
元数据/工程来源哈希匹配。13:37首轮346/1414batch，stderr空，
首轮尚未结束时无CSV属正常，不据此判断中断或性能。
运行中不重复启动、改冻结来源或并行候选，完整终态后执行上面的valfix独立验收。

终态更新（2026-10-31）：15:41包装exit0/status completed，连续1–30轮、
完整548原图val960/recall均结束；15:48原进程停止后独立验收通过，
23809冻结来源、实际args/权重/产物及两组548图预测重放一致。
mAP50-95−0.1755pp、small recall+0.9444pp（+251），类别1升9降，
两条预登记继续线均未达到，不扩训，也不按结果调权追试。
完整结果见`paper/regression_confidence30_results.md`；上述启动命令仅为复现历史，
已有产物时不要再次执行。临时监控`a`在收尾后删除，持续目标保持用户设置的paused。

### 4.4 覆盖保护分配的探索性30轮（2026-10-31新增）

用户重新激活目标并授权降低后续探索门槛；原覆盖保护诊断48.10%未达50%
的判定保留，现单独投入完整学生确认。只替换P1训练分配器，其余数据/配置、
初始化和30轮参考保持配对，不叠加NWD或回归置信度。

```powershell
python scripts/run_p1_coverage_guard.py --check-only
python scripts/run_p1_coverage_guard.py --engineering-only
python -u scripts/run_coverage_guard_background.py
python scripts/audit_p1_coverage_guard.py
```

真实GPU AMP前反传和两图训练期验证须先通过，零优化器步骤。
小训练由助手单次启动，已有run/dispatch拒绝重复；启动即临时监控，
终态后独立核对连续30轮、exit0、冻结来源、实际参数和两组完整原图预测。
探索交接条件AP≥+0.10pp且small不降，或small≥+0.50pp且AP不降；
不追改原Gate3或历史实验结论。详见`paper/assignment_coverage30_protocol.md`。

正式100轮配对预备入口为`scripts/run_coverage_guard_full100.py`；
当前仅`--plan-only`可查看计划，不做训练。只有上述完整30轮正向独立验收后，
才按`paper/coverage_guard_full100_protocol.md`由用户顺序运行两臂，
各自COCO初始化、固定100轮及完整评估，不从pilot续训。

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
