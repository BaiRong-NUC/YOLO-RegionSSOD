# 项目脚本

| 功能 | 入口 |
|---|---|
| 外部论文复现 | [按论文查找本项目编写的运行脚本](reproduce/) |
| 运行监控 | [P1 进度查看器、后台启动器](monitoring/) |
| 生成论文表格 | [表格生成与复核](reporting/build_paper_tables.py) |
| 成果保存与重放 | [保存、校验和冻结评测重放](preservation/) |
| 数据与伪标签 | [构造局部标注](make_partial_dataset.py) · [生成伪标签](pseudo_label.py) · [合并图像清单](combine_lists.py) |
| 训练与消融 | [通用训练](train.py) · [P1 pilot](run_p1_pilot.py) · [正式 seed 0](run_paper1_full100.py) · [正式 seed 1](run_paper1_seed1_full100.py) · [中断恢复](resume_paper1_p1_seed1.py) |
| 评测与诊断 | [通用评估](validate.py) · [小目标 PR](evaluate_paper1_small_pr.py) · [操作点分析](evaluate_scaled_operating_point.py) · [检测案例](build_detection_cases.py) |

顶层保留冻结实验回执引用的源码、它们直接导入的模块，以及历史报告使用的命令入口。外部论文的作者实现与适配代码见[论文实现目录](../reproduce/)。
