# 外部论文复现

每篇论文使用一个独立子目录。`upstream/` 保存固定版本的作者代码，同篇目录保存适配实现和绘图源码。项目编写的训练与评测入口放在 [scripts/reproduce](../scripts/reproduce/)，对应测试放在 [tests/reproduce](../tests/reproduce/)，基线图和方法图放在 [docs/reproduce](../docs/reproduce/)。各处使用相同的论文简称。

| 论文 | 作者代码 | 复现进度 |
|---|---|---|
| [S³OD](s3od/) | [Rotated Faster R-CNN / DOTA-v1.5](s3od/upstream/) | 已在 VisDrone 上完成两个种子的伪框筛选离线评测，DOTA-v1.5 完整检测指标待测。[复现卡](../docs/reproduce/s3od/) |
| [ESOD](esod/) | [YOLOv5m / VisDrone](esod/upstream/) | 已重评作者权重，并与本机 YOLOv5m 做了 32 图双种子训练对照。整图耗时缩短，小目标 AP 两次均下降。[复现卡](../docs/reproduce/esod/) |
| [SAHI](sahi/) | [切片推理 / VisDrone](sahi/upstream/) | 两份 YOLO11s 权重的小目标 AP 分别提高 11.88、11.53 点，整图耗时增加。已追踪一辆整图漏检、单块找回的小车，批量归因待做。[单车输入对照](../analyzes/reproduce/sahi/) · [复现卡](../docs/reproduce/sahi/) |
| [YOLC](yolc/) | [CenterNet / VisDrone](yolc/upstream/) | 548 图本机对照，作者权重的小目标 AP 提高 2.81 点。16 图双种子独立训练 30 轮，小目标 AP 变化 −0.01685 / +0.00013 点，召回率略升，单图耗时增加。[复现卡](../docs/reproduce/yolc/) |

| 新增论文与作者源码 | 本机对照 | 记录 |
|---|---|---|
| [SNIP / SNIPER](snip/upstream/) | 三尺度等前向预算，val 与 test-dev 两个划分、两份权重的 AP_small 均升 | [卡片](../docs/reproduce/snip/) |
| [Soft-NMS](softnms/upstream/) | 同一批候选改用高斯降分，两个划分两份权重的小目标 AP 均升 | [卡片](../docs/reproduce/softnms/) |
| [WBF](wbf/upstream/) | 相同双视图，坐标融合使两个种子的小目标 AP 均降 | [卡片](../docs/reproduce/wbf/) |
| [RemDet](remdet/upstream/) | 作者 Tiny 权重已重评；双种子 64 图独立训练的骨干替换均下降 | [卡片](../docs/reproduce/remdet/) |
| [NWD](nwd/upstream/) | 同一 YOLOv8s 迁移回归损失，双种子结果一升一降 | [卡片](../docs/reproduce/nwd/) |
| [Focal Loss / Detectron](focal/upstream/) | RetinaNet 表 1b 的两套分类损失，64 图双种子小目标 AP 均略降且接近零 | [卡片](../docs/reproduce/focal/) |

新克隆仓库后，在仓库根目录执行 `git submodule update --init --recursive` 即可获取登记的作者版本。新增论文时，在同级目录登记作者来源、精确版本、权重来源、评测协议、论文报告值、本机值和差异。
