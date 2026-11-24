# YOLO-RegionSSOD

**研究问题**　在 VisDrone 航拍图像中，每张训练图只标注约 10% 的面积时，让更多可信的小目标参与 YOLO11s 学生训练，能否比统一置信度阈值找回更多小目标？本项目研究水平框图像检测。

## 按五步读仓库

| 步骤 | 在这个项目里看什么 |
|---|---|
| [01 复现最接近任务的工作](docs/paper/01-论文复现.md) | 从 S³OD 的两张架构图入手，看作者验证了什么，以及本机复现到哪一步。 |
| [02 按功能拆解基线](docs/paper/02-基线拆解.md) | 从一张训练图走到 B2 的检测结果，定位数据、教师选框、学生训练和整图评测的代码。 |
| [03 为同一功能找不同实现](docs/paper/03-方法设计.md) | 对照 B2 和 P1 的选框、分类加权与背景处理，找到每项改动对应的配置和实现。 |
| [04 先验证能用，再检验为什么有效](docs/paper/04-实验验证.md) | 查正式 100 轮结果、30 轮单因素消融、误检代价和未保留的候选。 |
| [05 按证据写论文故事](docs/paper/05-论文写作.md) | 阅读中文初稿，逐项核对结论、证据和仍待完成的对照。 |

**当前结果**　在同一 10% 面积划分、同一教师下，B2 和 P1 各训练两个学生 seed、每次 100 轮。在 548 张完整验证图上，P1 相对 B2 平均提高 **2.2726 个百分点的小目标召回率**（conf 0.25，匹配 IoU 0.50），Ultralytics mAP50–95 平均提高 **0.1939 个百分点**。两个 seed 的全尺寸未匹配预测分别增加 1221 和 1435 个。[正式配对报告](docs/paper/p1_b2_seed1_full100_results.md)

## 运行与复核

在仓库根目录打开终端，使用现有的 `yolo` 环境：

```powershell
conda activate yolo
python -m pip install -e . --no-deps
regionssod --help
```

生成数据和运行 B0/B1/B2/P1 的命令见[执行协议](docs/baseline_protocol.md)。正式实验分别有[学生 seed 0 协议](docs/paper/paper1_full100_protocol.md)和[学生 seed 1 协议](docs/paper/paper1_seed1_full100_protocol.md)。已有权重与日志的机器可按[恢复与审计记录](docs/paper/paper1_p1_seed1_recovery.md)核查 seed 1；数据、权重和运行产物存放在本地。

查早期脚本用法可看[历史命令](EXPERIMENT_COMMANDS.md)，查实验决策可看[项目进度日志](docs/paper_completion_plan.md)。[八周实验设计](docs/experiment_plan_8weeks.md)记录最初的协议。
