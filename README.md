# 航拍目标检测论文复现

**当前进度**　[第一步 · 复现结果总览](docs/reproduce/README.md)已筛出三项在两份权重上同向提高小目标 AP 的推理改动。下一步按[第二步 · 按功能拆解基线](docs/baseline-analysis.md)追踪 VisDrone 小目标从输入到最终检测框的去向。

当前关注无人机航拍水平框小目标的检测精度与整图推理耗时。[ESOD](docs/reproduce/esod/) 的论文主要效果在本机小样本初筛中未稳定复现。与普通 YOLOv5m 相比，ESOD 的整图处理耗时缩短，模型前向变慢，两个训练种子的小目标 AP 分别低 0.285、0.336 点。[S³OD](docs/reproduce/s3od/) 已完成 VisDrone 的 SAT 伪框离线评测，DOTA-v1.5 原协议待测。

[SAHI](docs/reproduce/sahi/) 的切片推理在两份裁片训练的 YOLO11s 权重上提高了小目标 AP（+11.88、+11.53 点），代价是整图推理时间增加。复现卡收录了本机对照及 RemDet、YOLC 的公开复现条件。

[YOLC](docs/reproduce/yolc/) 在 VisDrone 验证集全部 548 张图上，作者权重开启局部放大后小目标 AP 提高 2.81 点。用 16 张原图独立训练 30 轮后，两个种子的小目标 AP 分别变化 −0.01685、+0.00013 点，精度收益未稳定复现。两组召回率略升，每张原图都多花约 140 毫秒。

在两份 YOLO11s 权重上，本机 VisDrone val 的小目标 AP 同向提升出现在三个推理改动上：SAHI 切片、[SNIP 三尺度筛框](docs/reproduce/snip/)（+0.606 / +0.614 点）、[Soft-NMS 重叠降分](docs/reproduce/softnms/)（+0.154 / +0.123 点）。保持设置换到 test-dev，SNIP 得到 +0.471 / +0.261 点，Soft-NMS 得到 +0.083 / +0.079 点。其余对照中，[WBF](docs/reproduce/wbf/) 双视图融合与 [RemDet](docs/reproduce/remdet/) 小样本骨干替换两次都下降，[NWD](docs/reproduce/nwd/) 回归损失一升一降。按阶段与耗时解读详见[结果总览](docs/reproduce/README.md)。

[Focal Loss](docs/reproduce/focal/) 也已完成 RetinaNet 的 64 图、30 轮双种子配对。两臂小目标 AP 都接近零，方法比平衡交叉熵低 0.00645 / 0.00581 点。

| 内容 | 目录 |
|---|---|
| 论文复现卡、基线图和方法图 | [docs/reproduce](docs/reproduce/) |
| 作者源码、项目适配与绘图源码 | [reproduce](reproduce/) |
| 本项目编写的运行和评测入口 | [scripts/reproduce](scripts/reproduce/) |
| 本项目编写的论文专用测试 | [tests/reproduce](tests/reproduce/) |

新克隆仓库后，在根目录执行 `git submodule update --init --recursive` 获取固定版本的作者代码。每新增一篇论文，在对应目录使用同一论文简称，分别记录作者版本、权重、数据划分、评测命令、论文值、本机值及差异。
