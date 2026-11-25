# 论文复现结果总览

| 论文复现卡 | 基线结构图 | 论文方法结构图 | 本机复现进度 |
|---|---|---|---|
| [S³OD](s3od/README.md) | [SSOD baseline](s3od/ssod-baseline-architecture.svg) | [S³OD](s3od/s3od-architecture.svg) | VisDrone SAT 双种子离线评测已完成，DOTA-v1.5 原协议待测 |
| [ESOD](esod/README.md) | [YOLOv5m](esod/yolov5m-baseline-architecture.svg) | [ESOD-YOLOv5m](esod/esod-architecture.svg) | **论文主要效果未稳定复现**。32 图双种子训练、548 图同机评测，整图处理提速，前向变慢，小目标 AP 差值 −0.285 / −0.336 点。全量训练待测。 |
| [SAHI](sahi/README.md) | [整图推理](sahi/full-inference-architecture.svg) | [切片推理](sahi/sahi-architecture.svg) | **切片推理的精度收益在两份裁片训练的 YOLO11s 权重上复现**。VisDrone val 全 548 张，小目标 AP 增加 11.88 / 11.53 点，单图耗时上升。切片微调待验证。 |

每篇论文在 `docs/reproduce/<论文简称>/README.md` 记录评测协议和论文报告值。判断收益时，以本机相同数据、尺寸和评测入口下的基线为对照，逐训练种子报告方法与基线的差值；固定权重的重复推理用于核验评测重复性。S³OD 和 ESOD 的复现卡直接预览同目录的两张 SVG，两图沿用相同的画布、层位置和编号。作者代码、适配代码和绘图脚本位于 `reproduce/<论文简称>/`。
