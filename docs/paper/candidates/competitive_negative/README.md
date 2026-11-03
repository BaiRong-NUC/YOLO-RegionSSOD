# P1互补候选：竞争类别负项衰减

日期：2026-11-03。状态：**探索诊断及隔离实现已完成，学生训练未启动。**

当前按用户明确的“基线、P1、现有改进”执行[新主对照协议](main_comparison_protocol.md)：
默认B2/原P1/P1＋竞争负项各30轮，主表给绝对值和三项增量；强度控制另作可选第四组。
新入口`run_p1_main_comparison.py`替代旧三臂机制入口作为当前建议。
旧学生合同/源码及预检证据保持原版本。

- [先固定的缓存诊断协议](diagnostic_protocol.md)
- [实际零训练结果（含反对证据）](diagnostic_results.md)
- [三臂30轮方案、精确命令与停止条件](student_protocol.md)
- [文献差异与实际阅读范围](literature_review.md)
- [最多三个方向及论文证据投入排序](direction_review.md)

新候选保持原P1正类目标，只衰减歧义top2负项；不是旧完整类别分数目标的重命名。
原`training/p1.py`、`training/class_ambiguity.py`、原配置、合同和原始结果均保留。
新实现`training/competitive_negative.py`复用已验证的对象身份运输/数据集，
通过包装原P1的BCE输出插入负项权重，原分配、目标和回归计算继续调用原代码。
配置独立在`configs/candidates/competitive_negative/`。

本轮只运行只读缓存重放、预检和CPU工程测试；预检不是实际CUDA学生训练验收。
任何新性能结论必须等待三臂各完整30轮与审计，不把本候选写入已完成贡献。
