# 三臂启动核验修订：Windows路径身份

2026-11-05，学生训练前。[科学合同](three_arm30_contract.md)的目标、控制、训练参数、
评估和停止规则不变。本文件只替换启动入口/输入绑定；V2是启动器版本，不是新loss。

首次只读`--check`因输入map只规范化一侧的Windows路径而失败：
`D:/Python/YOLO/yolo11s.pt`与`D:\Python\YOLO\yolo11s.pt`被误判不同。
`outputs/bounded_retention_continuous_freeze_20261105/`的原freeze、源码副本与失败记录
全部保留；未派发任何学生进程，也没有恢复/重试训练。独立
`scripts/run_bounded_retention_continuous_v2.py`仅修正两侧路径规范化，仍检查每个SHA、
缺失输入与冲突别名；不放宽来源、AMP或完整流证据。完整`python -m pytest`现为
**2042 passed/126.04秒**，绑定源码的回执为`.../suite_launch_03/result.json`。

最终绑定替换为`outputs/bounded_retention_continuous_freeze_20261105_v2/freeze.json`。
先用同目录草案完成只读核验，再独占创建此最终文件。实际入口：

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' scripts/run_bounded_retention_continuous_v2.py --check --spec outputs/bounded_retention_continuous_freeze_20261105_v2/freeze.json
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' -u scripts/run_bounded_retention_continuous_v2.py --train --spec outputs/bounded_retention_continuous_freeze_20261105_v2/freeze.json
```

输出仍为`outputs/bounded_retention_continuous_three30_s0/`；P1→候选→强控制，各30轮。
原V6工程及AMP通过回执保持不变；不重新前向、不重新审计数据流、不改变训练AMP配置。

诊断范围明确：`effective_anchors`是实际选中进行回归的partial前景anchor计数，
没有逐对象记录TAL正权重总量，不能把它或native候选池大小冒称有效正权重样本数。
multi-U posterior事件证明计算发生，不单凭事件数证明正权重效应；若冻结产物无法
认证此层证据，机制结论必须保留“未证实”。零优化完整loss梯度差已证明该压力batch
中候选/控制的实际监督可区分；这不替代完整30轮的性能及机制分析。
