# 航拍目标检测论文复现

当前复现 [S³OD](docs/reproduce/s3od/)。已固定作者实现版本并完成 VisDrone 上的 SAT 伪框离线评测。DOTA-v1.5 原协议下的作者权重评测和独立训练仍待完成。

| 内容 | 目录 |
|---|---|
| 论文复现卡、基线图和方法图 | [docs/reproduce](docs/reproduce/) |
| 作者源码、项目适配与绘图源码 | [reproduce](reproduce/) |
| 本项目编写的运行和评测入口 | [scripts/reproduce](scripts/reproduce/) |
| 本项目编写的论文专用测试 | [tests/reproduce](tests/reproduce/) |

新克隆仓库后，在根目录执行 `git submodule update --init --recursive` 获取固定版本的作者代码。每新增一篇论文，在对应目录使用同一论文简称，分别记录作者版本、权重、数据划分、评测命令、论文值、本机值及差异。
