# 原图上下文辅助类别纠错：执行前协议

2026-10-30。已有正伪框的同教师跨尺度几何稳定性诊断因分层样本不足停止，
不继续改其阈值。下一项检验区域裁剪丢失上下文时，原图预测能否提供可用的
类别纠错信号。当前只做A，不加入新教师/网络模块。

## 1. 原论文依据和差异

- *CrossRectify: Leveraging Disagreement for Semi-supervised Object Detection*，
  arXiv:2201.10734v2；paper-search MCP下载，阅读PDF pp.3–4、12–13。
  Sec.4.1使用不同初始化的两检测器，匹配候选并比较分数进行交叉纠错。
  本项目将只测试固定同一教师的不同观察范围，不是原方法复现；
  同模型错误可能相关，两个视图分数也未必校准一致，必须验证而不能假定更大上下文更准。
- *Density Crop-guided Semi-supervised Object Detection in Aerial Images*，
  arXiv:2308.05032v1；MCP下载，阅读PDF pp.3–4、Sec.III-A/B与Fig.2。
  已有全图/裁片互补、密集区域放大和伪标签生成；原方法还检测密集区域类别，
  本项目不增加这种类别或网络，不将“全图加裁片”本身主张为首次提出。
- 原文固定版本：<https://arxiv.org/abs/2201.10734v2>、
  <https://arxiv.org/abs/2308.05032v1>。PDF及SHA256保存在既有忽略文献目录，
  本轮只读上述方法相关页，未复现论文数值或精读全部实验。

## 2. 单规则与成本

固定配置`configs/context_rectification_diagnostic.yaml`，先提交再推理：

1. 用固定seed从未用于前述640尺度/NMS诊断的源图索引中随机取128图探索。
   同一B1教师只对完整原图做一次960推理，batch2、conf0.05、NMS0.70、max_det500。
   不使用全标注教师或基础模型，不训练；保存原图预测和实际成本。
2. 原图预测按已知tile坐标裁回未标注tile，使用既有0.35可见比例；
   只对P1原已接受入训tile分析，未接受的tile计入排除数。
3. 所有原正框先与conf≥0.50的原图映射框按IoU≥0.50降序贪心一对一匹配，
   不限制类别。仅对原small且conf<0.50的冲突框替换类别；保留原框几何、
   原分类权重、数量、顺序和其余正框。无支持的正框保持不变。
4. 隐藏GT只离线统计：纠错框更改前/后同类IoU≥0.50比例；
   全部正框重新一对一匹配后的净匹配、small新增/丢失GT数量。
   不用隐藏GT决定哪框改类别，不把诊断结果写成训练标签。
5. 探索至少改20框，改后质量比例≥70%、较改前提高≥20pp，
   总体净匹配和small净匹配各至少+10，small丢失最多3个，才执行另128图确认。
   确认排除探索及历史样本，规则不改且必须同样通过。未通过不改阈值追试。

这只是伪标签诊断继续线，不是学生改善或显著性结论。最多新增256张原图推理，
探索失败只消耗128张；当前不生成训练数据、不进行学生训练或临时训练监控。
两组即使通过，也需单独预登记学生预算并通过val960/recall后才交长训练命令。

## 3. 当前状态

执行前准备，尚无原图上下文预测或纠错结果。GT-free映射和类别纠正、
高置信度框占用支持、不可变性与继续线测试已实现，全套pytest73项通过。
执行入口拒绝未提交的协议/代码、已有输出目录、教师或环境版本变化；
确认组必须先只读重放探索结果且通过继续线，不能绕过。

在已有`yolo`环境、仓库根目录运行：

```bash
python -u scripts/audit_context_rectification.py
python scripts/audit_context_rectification.py --check
```

第一条只运行一次128原图推理和诊断，第二条只核对输入/输出哈希并重放全部
纠错决策与GT统计，不做推理。探索过线时才可执行
`python -u scripts/audit_context_rectification.py --confirm`；
对应只读验收为`python scripts/audit_context_rectification.py --confirm --check`。
输出在`outputs/context_rectification_20261030/{discovery,confirmation}/`，
包括协议/commit、原始预测、逐框诊断、计数、耗时和输入/输出SHA256。
协议文档本身也冻结，因此结果另在下节主计划中记录，保留本执行前协议不变。
