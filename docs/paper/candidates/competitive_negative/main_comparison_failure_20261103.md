# B2／原P1／竞争负项主对照：工程失败终态（2026-11-03）

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent run（终态核对）
- Verification Status: ENGINEERING_FAILED；学生性能 NOT_EVALUATED
- Version Label: main_comparison30_v2_terminal_failure
- 原冻结源码基点：`b75e58cb6271f204a2b1193940ff3b0c1b7b56f2`

## 实际状态

用户10:04启动的默认主三臂在candidate首批失败，包装器已停止，未自动重试。
2026-11-03约14:08核对，无命令包含本入口的Python进程；原包装PID8452已退出。

| 方法 | 完整epoch | worker退出码 | CSV训练累计秒 | 状态 |
|---|---:|---:|---:|---|
| B2 baseline | 连续1–30 | 0 | 6697.56 | 已训练，最终统一评估未执行 |
| 原P1 reference | 连续1–30 | 0 | 7078.94 | 已训练，最终统一评估未执行 |
| P1＋竞争负项 candidate | 0 | 1 | 无 | 首批损失前向失败；无results.csv或权重 |

上述时间为各自results.csv末行累计值，不是完整主对照墙钟成本。B2约11:57退出，
原P1约13:55:43退出，candidate原日志最后修改时间13:56:32。
训练器自带验证日志不能替代冻结的548图val960／双IoU完整PR统一评估。
`training_audit.json`、`evaluation/`、`evaluation_audit.json`、
`main_comparison.{json,md}`、`decision.json`均不存在。

## 原始错误与定位边界

candidate日志显示第1轮`0/1414`后抛出：

```text
ValueError: original hard pseudo-positive TAL target required
```

调用路径为`p1.py:167`的BCE → `competitive_negative.py:94`的
`negative_multipliers` → 同文件第43行断言。发生于首批损失前向，未走到该批反向／优化。
该断言在pseudo前景上要求目标最大类别等于教师top1、目标质量大于0、且恰有一个
非零类别；任一条件失败都会报同一错误。原日志没有张量快照，**尚未确定哪项触发**。
AMP目标转换、分配索引与教师类别绑定均只能列为后续工程排查项，不能当成已证实原因。
此前283项工程测试和check-only未覆盖此次真实CUDA链路中的失败条件。

## 保留与验证

独立回执及17个原文件快照（逐份复制后SHA256一致）：

`outputs/p1_competitive_negative_main30_s0/terminal_failure_20261103_140708_a233c4/`

- `failure_receipt.json`记录三臂连续轮次／退出码、进程查询、四份权重哈希及缺失终态产物。
- 39,828项冻结输入重新哈希：缺失0、不一致0；dispatch绑定原preflight哈希一致。
- 原日志、退出文件、preflight、dispatch、status、已有CSV／args／experiment独立复制；
  原目录及权重保留原位，没有覆盖、移动、恢复或删除。
- B2 best SHA256：`76b678af53eb340860273d434a7404bd6edc4545d6d156adea92341c1ee30bcb`。
- P1 best SHA256：`6cc444decbcdcd67b4be7d9ed373e070d82aaefb0e522c6bd09debf1bb6e70a4`。
- 本轮仅只读核对及新增报告／快照，没有改源码、冻结合同、配置或训练输入，未追加推理。
  因三臂完整训练／自动评估前置条件未满足，没有执行成功终态`--audit`。

## 结论与后续边界

当前不能给出有效的主三臂绝对性能表或P1−B2／改进−P1／改进−B2三项最终差值。
七项性能护栏均为NOT_EVALUATED，不能标为性能FAIL；无strength，机制为NOT_TESTED。
这是一项工程失败，不能据0轮判定竞争负项有效或无效，原P1历史成果仍独立保留。

按监控授权，本轮不自动修复、重训、恢复、补评估或扩大100轮。
后续应先在独立修订中定位断言条件，保留原硬正目标及数据绑定，补有针对性的
集成验证并运行完整pytest；若重新执行须单独冻结、采用新输出目录，不覆盖本失败证据。
已完成B2／P1权重保留，可再设计明确绑定的复用／补评估方案，但本次未执行。
终态记录中文commit并push，之后删除临时监控p1。
