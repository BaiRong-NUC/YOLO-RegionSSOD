# 原P1＋类别分数分类监督

日期2026-11-02。诊断已通过、全量分数已导出；学生训练尚未启动。
这是在原P1上的单项叠加，原P1仍是已完成主结果，独立保存点和历史产物保留。

教师竞争类别比值在控制top1分数、预测面积和类别后仍提供额外错类信息：
确认样本AUROC 0.7916→0.8214，log-loss降低约6.8%，通过预定的诊断门槛。
这是教师训练域内信息诊断，不能写成新学生精度提升。

- [诊断协议](diagnostic_protocol.md)：在读取诊断结果前固定样本及继续条件。
- [诊断结果](diagnostic_results.md)：全部数字、原图bootstrap、失败修正和限制。
- [学生协议](student_protocol.md)：唯一机制、两臂各30轮、固定继续线。
- [工程验证](validation.md)：代码等价性、实际验证路径及执行状态。

原P1对伪正anchor的分类目标为 `m e_y`；新候选为 `m*s/sum(s)`。
保留原尺度筛选、35067正伪框、top1类别用于TAL、原c权重、ignore、回归和网络。
归一化sigmoid只是类别目标分配，不是校准后验；目标质量守恒不代表梯度守恒。
人工标签不软化，背景/ignore规则不变。没有margin阈值、温度或额外loss系数。

| 内容 | 独立位置 |
|---|---|
| 原P1 | `src/regionssod/training/p1.py`，原配置及已验收100轮产物 |
| 教师提取 | `src/regionssod/pseudo/class_scores.py`；`scripts/export_p1_class_scores.py` |
| 候选训练实现 | `src/regionssod/training/class_ambiguity.py` |
| 显式配置 | `configs/train_p1_class_ambiguity_10.yaml` |
| 顺序两臂30轮入口 | `scripts/run_p1_class_ambiguity.py` |
| 完整AP_small/PR评估 | `scripts/evaluate_class_ambiguity_pair.py` |
| 侧文件 | `outputs/p1_class_scores_20261102/` |
| 配对实验产物 | `outputs/p1_class_ambiguity_pair30_s0/` |

在仓库根目录、已激活yolo环境：

```bash
python scripts/run_p1_class_ambiguity.py --check-only
python -u scripts/run_p1_class_ambiguity.py --train
```

第二条由用户执行，会顺序训练全新reference和candidate各30轮，
再自动进行548原图960验证、AP_small/完整PR/等精度召回与数值重放验收。
每臂日志在上述输出目录 `reference_stdout.log`、`candidate_stdout.log`。
进度见 `status.json`。失败不自动重试，既有运行拒绝覆盖。
完成后只读审计：`python scripts/run_p1_class_ambiguity.py --audit`。

P90下small recall至少+0.30pp且四项保留能力护栏全部满足，才建议考虑100轮。
失败也保留原P1成果与本轮诊断，不按validation追调参数，不自动扩训。
Humble Teacher已有软标签先例，S³OD已有尺度/负学习机制；本实现未确立首创性。
