# 原置信度回归权重的全量框级监督份额核对

2026-10-31，在首次全量统计前登记。当前valfix训练尚未启动，
已交用户独立30轮命令，本项不启动或修改训练。

## 问题与输入

已有开发缓存中，small平均置信度约0.58–0.59，medium约0.78–0.80。
因此原c用于回归不仅改变同一组内可靠性排序，还可能改变尺度/类别组成。
本项只检查固定训练清单中已接受正伪框的框级权重分布。
读取`data/visdrone_area10/dataset_p1.yaml`、其训练清单、p1_metadata和summary，
核对元数据全部正伪框图均在训练清单内，正框数与原生成summary一致，
输入哈希与已验收全量NWD预检中的同一P1输入相符，
与valfix使用的输入相同；不依赖启动时会重写的valfix预检回执。
不读取人工标签、隐藏GT、验证GT或预测。

使用元数据中的原始tile坐标尺度，不重新推理、增强、分配或改变标签。
全量、small/medium/large、10类及每类每尺度分别报告：
框数N、sum(c)、mean(c)、置信度固定五数概括、Kish有效样本量；
按N计数的组份额与按sum(c)计数的组份额及差值。
空组显式报告为0和null；不筛掉低置信度/类别/尺度组。

## 解释边界

mean(c)=sum(c)/N只是将每框原权重视为1时的框级保留比例，
不是实际训练回归loss、梯度或正anchor质量的保留率。
实际box/DFL还取决于增强后框、分配、TAL目标分数、框误差和DFL误差，
不能由框级sum(c)推出实际损失或梯度降幅。

按sum(c)归一化后的small份额变化只描述框级组成，不等于small recall变化。
本项不使用质量GT，不证明c具有独立定位可靠性，也不作性能继续/停止决策；
至少完整30epoch后才按原继续线判断学生。
不从本项拟合常数、指数、面积补偿、类别阈值或新评分；
正式归因如需常数衰减对照，应另外预登记并解释匹配的具体量。
仅匹配框级平均c不能宣称实际box/DFL监督量完全相同。

## 可重放执行

首次执行前中文提交并push脚本/协议。运行：

```powershell
python scripts/audit_regression_confidence_mass.py
python scripts/audit_regression_confidence_mass.py --check
```

输出`outputs/regression_confidence_mass_20261031/`，记录完整输入/代码/协议SHA256。
首次执行拒绝已有输出；--check逐项核对哈希并精确重算结果。
不改已有训练源码、配置、协议、工程产物和旧失败证据。
