# B0、B1、B2 与 P1 执行协议

在仓库根目录激活 `yolo` 环境。[实验设计](experiment_plan_8weeks.md)说明面积预算与隐藏标签的用途。[正式 seed 0](paper/paper1_full100_protocol.md)和[正式 seed 1](paper/paper1_seed1_full100_protocol.md)分别给出训练、验证、恢复及验收入口。

## 数据与教师

以下命令使用 PowerShell。正式数据集使用完整 6471 张训练图。5% 和 20% 的数据准备可相应替换独立输出目录和 `--ratio`。烟测使用单独目录和 `--limit 20`。

```powershell
$sourceData = 'D:/Python/YOLO/data/visdrone/dataset.yaml'
regionssod make-partial --source-data $sourceData --output data/visdrone_area10 --ratio 0.10 --seed 0 --unlabeled-per-image 1
regionssod train --config configs/train_full.yaml --name full_sup_s0
regionssod train --config configs/train_partial_10.yaml --data data/visdrone_area10/dataset.yaml --name partial10_sup_s0
regionssod validate --weights runs/partial10_sup_s0/weights/best.pt --data $sourceData --imgsz 960 --name partial10_sup_s0_val960
```

B0 使用完整标注作为上界参考。B1 仅用人工标注裁片训练，固定其 `best.pt` 为 B2/P1 教师。学生从共同的 COCO 权重初始化。

## 生成 B2 和 P1 学生数据

```powershell
$teacher = 'runs/partial10_sup_s0/weights/best.pt'
$unlabeled = 'data/visdrone_area10/splits/unlabeled_train.txt'
$labeled = 'data/visdrone_area10/splits/labeled_train.txt'
regionssod pseudo-label --weights $teacher --images-list $unlabeled --output data/visdrone_area10/pseudo/naive_t050 --threshold 0.50 --conf-floor 0.05 --imgsz 960 --batch 8 --device 0
regionssod combine-lists --inputs $labeled data/visdrone_area10/pseudo/naive_t050/pseudo_train.txt --output data/visdrone_area10/combined_naive_t050.txt --dataset-yaml data/visdrone_area10/dataset_naive_t050.yaml --source-data $sourceData
regionssod pseudo-label-p1 --weights $teacher --images-list $unlabeled --output data/visdrone_area10/pseudo/p1_s030_m050_l050 --policy configs/pseudo_p1_10.yaml --imgsz 960 --batch 8 --device 0
regionssod combine-lists --inputs $labeled data/visdrone_area10/pseudo/p1_s030_m050_l050/pseudo_train.txt --output data/visdrone_area10/combined_p1.txt --dataset-yaml data/visdrone_area10/dataset_p1.yaml --source-data $sourceData --p1-metadata data/visdrone_area10/pseudo/p1_s030_m050_l050/metadata.jsonl
```

## 正式训练与证据复核

两份正式协议绑定当时的教师、伪标签、pilot 证据和配置。已有正式运行的机器可从[两 seed 验收报告](paper/results/p1_b2_seed1_full100_results.md)和[恢复记录](paper/paper1_p1_seed1_recovery.md)检查既有产物。表格复核使用 `python scripts/build_paper_tables.py --check`，工程测试使用 `python -m pytest`。公式、指标口径和局限见[方法](paper/manuscript/methods.md)与[正式结果](paper/manuscript/main_results.md)。
