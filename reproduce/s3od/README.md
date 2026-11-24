# S³OD

作者代码固定在 Git 提交 `d7806b78947cdc403d525e67a28193a88a4f8598`。从[作者说明](upstream/README.md)、[DOTA-v1.5 配置](upstream/configs/dota1.5/)和[训练启动脚本](upstream/tools/dist_train_partially.sh)可追踪数据划分、模型和训练入口。

| 复现项 | 当前记录 |
|---|---|
| 权重与数据 | 本项目尚缺 S³OD 检查点和 DOTA-v1.5 验证数据。 |
| 预印本表 III | 1% 有标注裁片、单划分，验证集旋转框 mAP@0.50，SSOD baseline 36.0%，S³OD 41.9%。 |
| 预印本表 I | 10% 有标注裁片、五划分均值，SSOD baseline 57.08%，S³OD 60.24%。 |
| 本机值及差异 | 待按对应划分评测。 |

[复现卡与结构图](../../docs/reproduce/s3od/)并排展示基线和方法，记录预印本、期刊版本与公开实现的差异。[绘图源码](figures/)中 `build_s3od_pair_once.py` 将双图 SVG 写入 `docs/reproduce/s3od/`，`draw_s3od_reference_layout.py` 是较早的单图版本。绘图脚本的工作目录均为项目根目录。
