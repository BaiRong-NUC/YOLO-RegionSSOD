# 伪框回归置信度：唯一全量30轮学生确认

2026-10-31，在训练结果未知时固定。零推理缓存诊断已按
`regression_confidence_protocol.md`通过：两组small加权同类IoU提高0.0472/0.0526，
高质量比例提高7.3397/6.7066pp，类别组成控制后仍提高0.0411/0.0470。
这是已使用开发缓存上的监督分配依据，不是学生性能或独立盲测。

## 1. 单因素与配对

唯一新增`partial10_p1_regression_confidence_pilot30_s0`，
11310tile、固定10%面积split/教师/伪标签、学生seed0、原COCO独立初始化、
640、batch8、SGD、close_mosaic10、完整30epoch，不早停、不接续历史权重。
参考复用已验收完整P1 seed0 30轮，不重训参考或混用五轮子集。

只将匹配正anchor的box和DFL权重乘原教师置信度c；人工框经现有元信息运输为1。
保留原未加权TAL分母，分类软权重/ignore/分配/框坐标/标签数量/模型结构均不变。
不加入NWD或任何分配新规则，不读取隐藏GT或val拟合训练权重。
新模块隔离实现，冻结原P1源码；单元测试核对c=1等价、c仅缩放回归、
混合人工/伪框及分类/分配保持，真实AMP前反传只作工程检查。

该改动也降低伪框监督总强度；本轮不区分逐框排序与常数衰减的独立贡献。
不将置信度加权称为首次或复现RUPL/UTv2/Soft Teacher定位可靠性机制。

## 2. 命令与实际预算

```bash
python scripts/run_p1_regression_confidence.py --check-only
python -u scripts/run_regression_confidence_background.py
python scripts/audit_p1_regression_confidence.py
```

先中文提交协议/配置/代码并push，通过输入预检与工程检查后仅启动一次。
首条只预检，第二条训练/完整评估，第三条仅在原进程停止且全部阶段结束后独立验收。
运行目录已有或已有dispatch/status时拒绝启动；原进程身份、日志、退出码保留。
不因观察超时重启，不拼接中断轮次、不并行其他候选。
学生耗时参考原P1约7287秒，加上完整评估；这是估计，不是时限或速度承诺。
启动时创建临时监控，正常静默，完成/失败处置后删除；中文提交并push。

输出为`outputs/p1_regression_confidence_pilot30_s0/`及
`runs/partial10_p1_regression_confidence_pilot30_s0/`。
30轮后best.pt做548完整原图val960和固定recall
（conf0.25/匹配IoU0.50/NMS0.70/max_det500），保存预测及SHA256。
验收核对输入/源码/实际参数、连续30轮、两组预测重放和全部终态记录。

## 3. 预登记性能继续线

沿用既有30轮单因素协议两路线，不按本轮结果改线：

- mAP50-95至少+0.30pp，且small recall不下降；
- 或small recall至少+1.00pp，且mAP50-95不低于−0.10pp。

均不满足则保留负面/取舍，不按val调整权重形式或追试参数。
满足其一也只准备冻结更大训练的配置、预检/验收代码及用户命令，
由用户执行，不自动扩训。至少完整30轮才判断性能，单seed不证明显著性、
跨split泛化或已完全收敛；recall不是官方AP_small/AR500，原Gate3不改写。
目前仅A，不做B/C、基础模型、蒸馏剪枝或部署。
