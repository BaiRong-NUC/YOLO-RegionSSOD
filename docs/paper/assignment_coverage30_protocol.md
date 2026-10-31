# 覆盖保护分配：探索性完整30轮确认

2026-10-31，学生结果未知时登记。用户再次授权自主30轮，并明确“门槛不要设计的太高”。
这是对旧诊断停止安排的显式更新，不改写旧确认未通过50%代理线的事实。

## 依据与范围

旧128基础tile诊断保留全部原有效目标，小框框外质量降低48.0992%，
小框总质量保留95.6121%。旧50%诊断线并非学生精度阈值；
本次按用户新授权投入一次完整学生训练，不重采样、不调整原分配规则。
这属于已有探索证据驱动的研究，不能将旧缓存称作新独立确认。

唯一改动是用既有CoverageGuardedAssigner替换P1的TAL：
原框含候选点时优先框内点，否则保留原扩框池；
如任何原本目标分数大于0的目标失去有效分配，整张图退回原TAL。
不修改原assignment.py、伪框、阈值、权重、网络、DFL或ignore规则。
固定同split/教师/11310tile/学生seed0/COCO初始化，640、batch8、
SGD、close_mosaic10、完整30epoch，参考复用完整P1的独立30轮。
不接续NWD或回归加权权重，不用隐藏GT/val拟合训练规则。

FCOS的框内点与非负ltrb回归提供已有几何依据；本方案不是首次框内分配，
也不是FCOS或S³OD复现。可能的新价值在区域局部标注下的监督覆盖保护，
是否构成论文贡献须由学生结果与必要消融验证。
S³OD已有尺度重平衡分配，差异按现有s3od_comparison.md处理。

## 预算与前置检查

```powershell
python scripts/run_p1_coverage_guard.py --check-only
python scripts/run_p1_coverage_guard.py --engineering-only
python -u scripts/run_coverage_guard_background.py
python scripts/audit_p1_coverage_guard.py
```

先中文提交push，冻结配置/源码/协议、既有输入及参考哈希，再做真实
YOLO11s AMP前反传和标准两图验证，零优化器步骤；通过后单次启动。
复制旧工程检查结果不算通过。训练启动即创建临时监控，核实进程身份，
正常静默，完成或失败后独立验收、记录并清理监控。中断不拼接轮次。
双TAL增加训练计算，预计约2–3小时，仅为预算估计；完整val960和recall另计。
不并行训练新候选，运行期间不改冻结源码。

## 本轮探索性交接条件

相对完整P1，完整548原图val960满足任一：

- mAP50-95至少+0.10pp，且small recall不下降；
- small recall至少+0.50pp，且mAP50-95不下降。

这是较低的工程筛选门槛，不是显著性检验、论文成功或原Gate3通过；
旧回归/NWD判定不因这次新规则而改变。满足后只准备公平100轮
P1/候选配对、必要单因素消融与用户命令，由用户运行大训练。
不满足则记录取舍，不能按val调本候选规则追试。
若结果仅有很小正向变化，交接明确单seed探索和重复验证风险。
recall固定conf0.25/IoU0.50/NMS0.70/max_det500，不等于官方AP_small。

输出：outputs/p1_coverage_guard_pilot30_s0；
run：runs/partial10_p1_coverage_guard_pilot30_s0。
独立验收要求连续30轮、exit0、完整评估、实际参数与两组预测重放一致。
