# P1 主线与配对消融

| 功能 | 实现 | 验证 |
|---|---|---|
| 同一 B1 教师生成 B2 对照 | [统一阈值伪标签](../../../src/regionssod/pseudo/generator.py) | [正式 100 轮对照](../results/p1_b2_seed1_full100_results.md) |
| P1 按尺度选取伪框 | [选框实现](../../../src/regionssod/pseudo/scale.py) · [阈值配置](../../../configs/pseudo_p1_10.yaml) | [全局与尺度阈值的配对结果](../results_tables.md) |
| 学生分类采用置信度软权重、屏蔽不确定背景 | [损失实现](../../../src/regionssod/training/p1.py) | [no-soft 和 no-ignore 消融](../results_tables.md) |
| 解释精度与召回的取舍 | [固定点召回](../../../src/regionssod/evaluation/size_recall.py) | [等精度召回](../small_pr_results.md) · [诊断](../high_precision_diagnostic_results.md) |

后续停止的候选与各自强控制的对比保存在[归档记录](../../paper_completion_plan.md)。正式入口导入或由历史回执绑定的共享脚本继续保留原路径。
