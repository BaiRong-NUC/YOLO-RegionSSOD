# B2学生seed1正式100轮验收

2026-11-01，用户启动的B2学生seed1正式训练已完成并通过独立验收。
这是第二个学生随机种子的基线结果；同seed1的P1尚未运行，
不能据此更新P1方法增益或两seed正式100轮配对统计。
原seed0正式配对和30轮结果分别保留，不跨轮数混合。

## 协议与完整性

- 固定10%区域标注面积split0、既有B1教师、全局0.50离线伪标签和标准检测损失。
  YOLO11s从COCO权重独立初始化，学生seed1，11095个训练tile，
  640/SGD/batch8/workers0/close_mosaic10，连续100轮、未中断或恢复。
- 2026-11-01 17:45:25开始，23:04:51包装真实exit_code=0。
  原包装PID10476、工作PID31640按创建时间和命令跟踪，验收时均已停止；
  CSV连续完整1–100轮且有限，stderr为空。
- 23:18运行`python scripts/run_paper1_seed1_full100.py --arm b2 --audit`，
  accepted=true。核对37803个冻结来源、实际args、权重和产物哈希，
  并重放548张完整原图的保存预测。训练源码和协议保持冻结。
- CSV累计学生耗时19111.9秒（5.3089小时），包装总时段另含验证及I/O；
  教师、伪标签生成和既有实验不计入这个学生训练时间。

## 完整原图结果

| 指标 | B2学生seed1 / % |
|---|---:|
| mAP50 | 42.5503 |
| mAP50-95 | 25.6683 |
| small recall | 39.3611 |
| medium recall | 79.1273 |
| large recall | 89.1284 |

| 尺寸 | 匹配GT | GT总数 | recall / % |
|---|---:|---:|---:|
| small | 10461 | 26577 | 39.3611 |
| medium | 8795 | 11115 | 79.1273 |
| large | 951 | 1067 | 89.1284 |

mAP为Ultralytics 8.4.172在548张完整原图val960上的结果。
尺寸召回使用conf0.25、匹配IoU0.50、NMS IoU0.70、max_det500，
按原图GT面积分组（small<32²），全部尺寸共同进行按置信度排序的分类别贪心匹配。
这是固定操作点诊断，不是VisDrone官方AP_small或AR500。
不能把训练期tile验证CSV中的mAP替代本页原图结果。

## 证据路径

- `outputs/paper1_b2_formal100_s1/`：
  `preflight.json`、`launch_receipt.json`、`status.json`、`metrics.json`、
  `artifact_sha256.json`、`independent_audit.json`、`stdout.log`、`stderr.log`。
- `runs/partial10_b2_formal100_s1/`：`results.csv`、`args.yaml`、
  `experiment.json`、`weights/{best,last}.pt`。
- `runs/val/partial10_b2_formal100_s1_val960/metrics.json`。
- `runs/size_recall/b2_formal100_s1/`：
  `metrics.json`、`images.txt`、`predictions.jsonl`。
- 来源提交：`f7ced05ec7724c7a04765e3e0a64922b51988cbc`；
  best SHA256：
  `cf72dd9b533b6fd07874dde725525f50e2b860bdc8444ba8d322e2a2e9039d69`；
  last SHA256：
  `98496415afa4fe892795f02b45ba41d35da6dcc1d6b838fbaabd337f8fb628b6`。

原始证据在ignored目录保留，本页新表独立核对，不加入原30轮表格生成器统计。
本次只修改论文、状态和复现文档，不更改冻结训练源码、配置或协议。
8行新增表格与原始JSON逐项一致，best/last权重哈希复算通过；
原表检查137来源/8组recall通过，回执保存为本次output下
`documentation_check_20261101.json`。

## 下一步

B2已验收，不重复训练或恢复。用户在已激活yolo的Git Bash执行：

```bash
cd /d/Python/YOLO-RegionSSOD
python scripts/run_paper1_seed1_full100.py --arm p1 --check-only
python -u scripts/run_paper1_seed1_full100.py --arm p1 --train
```

P1由用户启动，入口会先验收同seed1 B2；助手不代启动。
按seed0实测P1学生约6.73小时，含最终评估等建议预留7–8小时，
实际受负载影响，不能保证完成时间。
本次P1只读预检回执另存
`outputs/paper1_b2_formal100_s1/p1_handoff_preflight_20261101.json`。
23:21实测预检exit0、37803来源通过；P1的output/train/val/recall目录
前后均不存在，没有启动训练或推理。
P1完成后再按同seed配对检查小目标收益、误检/类别退化及成本，
仅以两个学生seed、固定split和教师报告重复性，不作跨split或显著性主张；
原Gate3保持，不自动增加训练或恢复持续研究目标。
