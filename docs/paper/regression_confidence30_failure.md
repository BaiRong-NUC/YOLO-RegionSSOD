# 伪框回归置信度30轮确认：工程失败记录

2026-10-31。该次运行未完成30轮，不能用于判断方法性能。
不将此次工程失败计为负向消融，也不修改预登记继续线。

## 真实终态

- 源码提交：`5f47671e04ae5706945e888ebf21175e048c20e9`。
- 北京时间12:31:07启动，12:37:02包装退出，真实exit_code=1、status=failed。
- 包装PID21156、训练PID10196均已停止；启动时命令和创建时间已核实。
- 第1轮训练1414/1414 batch结束，随后验证在0/35时报错。
- 无results.csv，无best/last checkpoint；完整训练加验证epoch数为0。
- 未执行成功验收、未自动重启、续训、拼接或启动替代候选。

## 原因与工程检查限制

报错为：

```text
ValueError: confidence regression requires augmented per-object provenance
```

`P1Trainer.build_dataset`只对训练使用带伪框元数据的P1Dataset；
验证走标准全标注数据集，没有训练专用的`pseudo_weights`字段。
新增`RegressionConfidenceLoss`在所有调用中都要求该字段，
因此首轮验证计算loss时异常退出。

此前122项测试及真实YOLO11s AMP前反传通过，仅覆盖带元数据训练输入，
没有覆盖标准验证批次通过同一损失计算的路径。预检通过也不能证明完整
训练/验证流程正确。这是实现和工程检查的漏项，不能解释为收敛或性能问题。

后续应先修复验证兼容，同时保留训练输入缺少必要元数据时的异常保护，
补充实际验证调用路径检查。修复后需要另行冻结独立运行，
由用户执行完整30轮；不能复用该次输出目录或接续本次未保存的进度。
本次终态处置只更新记录，没有修改被冻结的源码、配置或协议。

## 证据

原位保留`outputs/p1_regression_confidence_pilot30_s0/`中的
preflight、engineering_check、launch_receipt、status、stdout/stderr和dispatch，
以及`runs/partial10_p1_regression_confidence_pilot30_s0/`中的
args.yaml与experiment.json。

`outputs/p1_regression_confidence_pilot30_s0/failure_report.json`
记录真实退出码、进程停止核实、缺失产物、原因及10个原始文件的SHA256。
该目录属于忽略产物，不加入Git；本文提供版本化的终态说明。

临时监控a在失败处置后删除。当前没有运行中的该候选训练；
持续目标尚未达成，不声称已找到值得扩大训练的学生结果。
