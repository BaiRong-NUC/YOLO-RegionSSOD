# 外部论文复现

每篇论文使用一个独立子目录。`upstream/` 保存固定版本的作者代码，同篇目录保存适配实现和绘图源码。项目编写的训练与评测入口放在 [scripts/reproduce](../scripts/reproduce/)，对应测试放在 [tests/reproduce](../tests/reproduce/)，基线图和方法图放在 [docs/reproduce](../docs/reproduce/)。各处使用相同的论文简称。

| 论文 | 作者代码 | 复现进度 |
|---|---|---|
| [S³OD](s3od/) | [Rotated Faster R-CNN / DOTA-v1.5](s3od/upstream/) | 已在 VisDrone 上完成两个种子的伪框筛选离线评测，DOTA-v1.5 完整检测指标待测。[复现卡](../docs/reproduce/s3od/) |
| [ESOD](esod/) | [YOLOv5m / VisDrone](esod/upstream/) | 已重评作者权重，并与本机 YOLOv5m 做了 32 图双种子训练对照。整图耗时缩短，小目标 AP 两次均下降。[复现卡](../docs/reproduce/esod/) |
| [SAHI](sahi/) | [切片推理 / VisDrone](sahi/upstream/) | 同图同权重比较整图与切片推理，两份 YOLO11s 权重的小目标 AP 分别提高 11.88、11.53 点，整图耗时增加。[复现卡](../docs/reproduce/sahi/) |
| [YOLC](yolc/) | [CenterNet / VisDrone](yolc/upstream/) | 作者权重、548 图本机对照，局部放大相对整图的小目标 AP 提高 2.81 点，平均单图耗时增加。[复现卡](../docs/reproduce/yolc/) |

新克隆仓库后，在仓库根目录执行 `git submodule update --init --recursive` 即可获取登记的作者版本。新增论文时，在同级目录登记作者来源、精确版本、权重来源、评测协议、论文报告值、本机值和差异。
