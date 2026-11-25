# 外部论文复现脚本

按 `scripts/reproduce/<论文简称>/` 存放本项目为该论文编写的数据准备、训练启动和评测入口。简称与[论文实现目录](../../reproduce/)一致。

[S³OD 离线评测](s3od/evaluate_sat_visdrone.py)和 [ESOD 验证集评测](esod/evaluate_visdrone.py)分别使用各自的作者设置。ESOD 另提供[作者评测器入口](esod/author_val.py)核对内部 AP。

[SAHI 评测](sahi/evaluate_visdrone.py)比较 YOLO11s 的整图与切片推理。[YOLC 小样本训练](yolc/train_pilot.py)使用固定训练子集和指定种子。[YOLC 评测](yolc/evaluate.py)可载入作者权重或本机训练权重，在 VisDrone 验证集比较整图与密集区局部重检。

[SNIP](snip/evaluate.py)、[Soft-NMS](softnms/evaluate.py)、[WBF](wbf/evaluate.py) 使用同一套[配对计时与评测](paired_inference.py)，默认完整 val，`--split test` 对应有标签的 test-dev。[RemDet](remdet/pair.py)、[NWD](nwd/pair.py) 和 [Focal Loss](focal/pair.py) 使用同一 64 张训练图及 548 张验证图。各篇复现卡列出所需环境、权重与命令。
