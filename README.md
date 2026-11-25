# 航拍目标检测论文复现

当前关注无人机航拍水平框小目标的检测精度与整图推理耗时。[ESOD](docs/reproduce/esod/) 的论文主要效果在本机小样本初筛中未稳定复现。与普通 YOLOv5m 相比，ESOD 的整图处理耗时缩短，模型前向变慢，两个训练种子的小目标 AP 分别低 0.285、0.336 点。[S³OD](docs/reproduce/s3od/) 已完成 VisDrone 的 SAT 伪框离线评测，DOTA-v1.5 原协议待测。

[SAHI](docs/reproduce/sahi/) 的切片推理在两份裁片训练的 YOLO11s 权重上提高了小目标 AP（+11.88、+11.53 点），代价是整图推理时间增加。复现卡收录了本机对照及 RemDet、YOLC 的公开复现条件。

[YOLC](docs/reproduce/yolc/) 在 VisDrone 验证集全部 548 张图上使用同一份作者权重，额外放大最多两个密集区后，小目标 AP 提高 2.81 点，平均单图耗时增加。

| 内容 | 目录 |
|---|---|
| 论文复现卡、基线图和方法图 | [docs/reproduce](docs/reproduce/) |
| 作者源码、项目适配与绘图源码 | [reproduce](reproduce/) |
| 本项目编写的运行和评测入口 | [scripts/reproduce](scripts/reproduce/) |
| 本项目编写的论文专用测试 | [tests/reproduce](tests/reproduce/) |

新克隆仓库后，在根目录执行 `git submodule update --init --recursive` 获取固定版本的作者代码。每新增一篇论文，在对应目录使用同一论文简称，分别记录作者版本、权重、数据划分、评测命令、论文值、本机值及差异。
