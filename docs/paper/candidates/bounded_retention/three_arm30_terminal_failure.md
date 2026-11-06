# 连续集合三臂训练：全局点求解容量故障终态

2026-11-06。状态：**STOP_ENGINEERING_POINT_RESOURCE_LIMIT**。
本轮确实启动了学生训练，但未完成三臂各30轮：原P1完成30轮，候选完成9轮，
第10轮异常退出，强控制未派发。没有最终548图full-image val960比较，
不能作性能改善、性能退化或机制成立判断。当前实现的训练可执行性门失败，
不等于连续相容集合在数学上无效。

## 冻结、启动与实际完成量

沿用[三臂科学合同](three_arm30_contract.md)和
[Windows启动修订](three_arm30_launch_addendum.md)。源代码/配置/数值限额没有在训练中改动。
初始工程资格见[零优化联合验证](continuous_amp_results.md)：2042 tests通过、
真实来源/完整增强流/AMP通过；这些检查未证明任意后续模型分布下都能在容量内求解。

实际执行命令（仅历史记录，**禁止再次执行**）：

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' -u scripts/run_bounded_retention_continuous_v2.py --train --spec outputs/bounded_retention_continuous_freeze_20261105_v2/freeze.json
```

调度PID30104于北京时间2026-11-05 18:21:51启动。训练输出：
`outputs/bounded_retention_continuous_three30_s0/`。
各臂的process/claim/exit、实际args、CSV、权重和日志均保留。

| 臂 | PID | 完整轮数 | 退出码 | worker墙钟时间 | 结论 |
|---|---:|---:|---:|---:|---|
| 原P1参考 | 45600 | 连续1–30 | 0 | 6920.531秒（1.922小时） | 单臂训练完成、单臂封印/配置审计通过 |
| 连续集合候选 | 51256 | 连续1–9 | 1 | 74543.078秒（20.706小时） | 第10轮全局点容量故障 |
| 同界最强连续点DFL控制 | 未派发 | 0 | 无 | 0 | 先行候选失败，未启动 |

候选最后进度记录为第10/30轮89/1414批次，随后抛出异常；该显示值不是失败batch的
精确序号证书。由启动时间与worker耗时推算退出约为北京时间2026-11-06 17:00:05；
只读监控17:00:16已记录父进程与所有worker结束。本会话下一次约30分钟复查时发现故障。
没有观察到训练中断后重启、checkpoint恢复或自动重试；这次是首个已记录的异常退出。

## 故障证据与解释边界

日志异常：

```text
NumericalCertificationError: global point resource cap
```

调用链为`ContinuousNativeELoss → NativeEAssigner.get_pos_mask → continuous_support
→ optimize_point`，停在`bounded_retention_continuous_numerics.py:710`。
冻结函数默认`max_cells=20000`、`tolerance=1e-5`：全局点的上下界间隙尚未满足精度条件，
而继续二分将超过单元限额时抛出此异常。代码门为`calls + 2 > max_cells`；
不能把上限20000误报为日志已保存的实际访问单元数。

这是候选和强控制共享的连续点求解路径，发生在TAL相容几何生成阶段；
不能归因于候选的multi-U后验采样，也不能声称控制若启动必然在同一步失败。
当前日志没有持久化此异常附带的对象身份、U维数、实际cells、最终gap或logits；
不能从前9轮统计反推失败对象，亦未重放训练来补造现场。

该退出不是CUDA OOM、显卡缺席、非有限loss告警或人工杀进程。它证明当前求解实现
在固定容量内不能完成这一次请求；不证明相容集合为空，也不证明全局点不存在。
不提高容量、放松误差、降级坐标或丢对象绕过，不恢复此失败实验。

## 计算成本与启动决定复盘

原P1前9轮CSV累计时间均值230.248秒/轮；候选前9轮为8181.711秒/轮（2.273小时），
同前9轮累计时间比为35.534。候选完整轮耗时范围1.900–3.116小时。
两worker累计墙钟22.629小时；这些数值不是GPU独占时间或能耗，系统负载未被控制，
也不是严格性能剖析，不能把全部差值归给某一个函数。

只读设备检查确认候选PID51256使用RTX4070Ti，进度日志GPU_mem约3.29G。
源码包含逐对象/逐候选锚点全局求解、精确有理数边界运算、采样及CPU/GPU同步。
这支持额外数值计算是重要开销来源，但未测得各部分时间占比。
本次故障是算法单元上限，换4090不会直接改变该条件；没有证据表明本次需要更多显存。

启动依据是合理机制假设、工程检查通过和用户预先授权的30轮比较，**不是已观察到
性能改善**。希望来自对合法未知边避免强行指定可见边界，并利用同界集合保留多解。
零优化梯度差异只证明目标可区分。启动前没有充分做代表性耗时/长尾评估与完整预算说明，
把有限工程覆盖误当成足够的投入依据，是本轮执行决策的不足。
30轮是预先承诺的性能判断长度，不能代替可执行性和成本判断。
任何未来另行授权的实现，需先解决求解复杂度、异常现场持久化和成本预测；
本报告不创建新实验、不批准重试，也不追改本轮冻结门槛。

## 只读终态审计与保留证据

终态命令已执行，exit0：

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' scripts/run_bounded_retention_continuous_v2.py --audit
```

返回`accepted=false, terminal_evidence_verified=true, automatic_retry=false`。
含义是失败证据完整性通过，**不是实验通过**。61个封印产物及24004冻结输入重放通过；
另对参考臂调用原`audit_arm`，连续30轮、实际args、来源与权重封印通过。
候选实际args也与冻结配置相同，两臂已记录CSV全部为有限数。

独立只读收尾产物位于`outputs/bounded_retention_continuous_terminal_20261106/`，
不在原封印目录内；`review.json`保存完成量、耗时、权重哈希、诊断汇总及进程终态。
关键绑定：

| 文件 | SHA256 |
|---|---|
| 最终freeze.json | `a04ec4f117472fc2e11f988a61534d3de5898c9c9bff5d7f7a592928a6a57901` |
| artifact_sha256.json | `16eded019744b93db09b632012647da257d158caed1d1893c5aa1face3f04b35` |
| candidate_stdout.log | `a104a3c3e3da280b751500fe94cd3ba17b05c277fe3243d52651d6ed8a70e5a6` |
| 原P1源码p1.py | `5aadf46f534449a28ca7313763d3cd163663355d2004cd104ad9251ecee18efb` |
| 参考best.pt | `5fd3125da11dc079c54ced4c3a506b63811587ba78a07cc2b24bddafb013c0fd` |
| 候选last.pt | `8b27ba426c82315c62d1120accdb05921a2d8af0427012a84ef277a963a8b56e` |

候选保留9份完整epoch诊断，共12726批次。记录partial对象增强实例136962个，
按U维数1/2/3分别130414/6544/4；posterior事件63008个；
`effective_anchors=1233682`仅为partial前景回归锚点计数，未记录正TAL权重总量。
这些是重复增强暴露/计算事件，不是独立真实截断对象数量，不替代机制证据；
未汇总entropy/面积趋势作提前性能筛选，也未把9轮与P1的30轮配成结果。

最终548图val960评估目录不存在，因此AP、AP_small、召回、同K、同FP、重复框及定位
三臂比较均**不可用**。原生训练期验证日志保留，但不作为本合同最终比较。
本轮收尾没有新增推理、优化步骤或隐藏训练GT读取。仅改报告/入口文档，未改源码，
因此不重复执行训练前已通过的2042项测试；终态采用文件/配置/封印的只读审计。

父进程30104、参考45600、候选51256及监控18428均已自然结束，无终止进程操作。
临时`monitor.py`在保留内容副本和哈希后删除，health/checks/stdout/stderr保留。
没有创建应用自动化定时任务，故无此类任务待删除；本会话的等待也已结束。
原P1、所有旧失败、冻结合同/源码及本次失败权重均保留。当前状态是停止本次实现，
不作方法性能结论，不自动继续控制臂或开启新方向。
