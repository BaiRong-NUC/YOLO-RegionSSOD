# 论文复现结果总览

| 论文复现卡 | 基线结构图 | 论文方法结构图 | 本机复现进度 |
|---|---|---|---|
| [S³OD](s3od/README.md) | [SSOD baseline](s3od/ssod-baseline-architecture.svg) | [S³OD](s3od/s3od-architecture.svg) | VisDrone SAT 双种子离线评测已完成，DOTA-v1.5 原协议待测 |
| [ESOD](esod/README.md) | [YOLOv5m](esod/yolov5m-baseline-architecture.svg) | [ESOD-YOLOv5m](esod/esod-architecture.svg) | VisDrone val 全量权重重评，长边 1536 / 1920 的小目标 AP 为 25.97 / 27.42；独立训练待测 |

每篇论文在 `docs/reproduce/<论文简称>/README.md` 记录评测协议、论文报告值、本机复现值和差异。S³OD 和 ESOD 的复现卡直接预览同目录的两张 SVG，两图沿用相同的画布、层位置和编号。作者代码、适配代码和绘图脚本位于 `reproduce/<论文简称>/`。
