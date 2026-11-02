# 原P1＋类别分数分配：唯一30轮确认协议

冻结日期2026-11-02，任何学生训练及对应30轮新PR评估之前制定。
诊断通过仅准许准备本实验；未观察学生效果。原Gate3/4不变。

## 单一变量

原P1正anchor分类目标为 `m e_y`。对伪正anchor，将其改为
`m q`，其中 `q_k=s_k/sum(s)`，s为原B1教师同一保留框的完整sigmoid分数。
q是归一化分配，不是经过校准的后验概率。仍以原top1类别、框进行TAL。
保留原c对整条BCE向量的乘法、未加权TAL分母、原box/DFL、人工目标、
negative/ignore规则。每anchor目标质量及c加权目标质量守恒，不宣称损失/梯度守恒。
只允许无类别权重或单位权重。没有温度、margin阈值或新loss系数。

全量导出必须对应原6471教师推理tile的原batch顺序，验证全部旧候选身份，
覆盖4839伪图/35067原正框。独立sidecar不修改原标签或原metadata。
训练时标量object token同步运输向量/伪标签身份，增强后解码。
仅推理导出分数，不读取隐藏GT。

## 配对及执行预算

使用**两臂各30完整epoch的新配对**，split0、教师、原P1伪框、COCO初始化、
seed0、640、batch8、SGD、原增强及close_mosaic10均相同。
reference调用原P1训练器，candidate调用隔离扩展；顺序执行，无并行训练。
这样不依赖历史30轮运行环境一致性的假定。原历史模型均保留。
原100轮成果不参加30轮定量差值，也不重新训练。

用户显式运行 `python scripts/run_p1_class_ambiguity.py --train` 后才启动。
`--check-only`只预检；训练完成后自动全图548张val960、
conf≥.001/NMS.7/maxDet500缓存与COCO-style AP/完整PR分析。
实际训练args须逐项配对；每臂必须连续1–30轮且哈希审计通过。
输出独立目录；检测到既有运行拒绝重启，保留失败记录。

## 固定继续线（全部满足）

candidate−reference：

1. 总体Ultralytics AP50–95 ≥ −0.10pp；
2. COCO-style AP_small ≥ 0pp；
3. IoU0.50、全尺寸P≥90%下small recall ≥ +0.30pp；
4. 同口径P≥70%下small recall ≥ 0pp；
5. 同一val缓存conf.25下small recall ≥ −0.20pp。

P90为主要候选目标，其余为保留现有能力的护栏。任一操作点不可达、非有限、
轮数不足或配对审计失败均不能通过。此为投入决策，不是显著性检验或录用标准；
单seed开发集反复探索仍有选择偏差。通过后也仅建议用户考虑100轮配对确认，
不会自动扩训；失败不按val调温度/阈值追试。

## 必报证据

总体AP50/AP50–95、COCO AP_small/逐类AP_small、IoU.50/.75完整PR、
P50/60/70/80/90下small recall、conf.25召回、每图500上限命中数。
两臂训练时间、教师额外分数推理成本单列。保留所有负向类别/操作点。
分类目标质量、c加权质量、BCE及score梯度诊断为机制检查；
目标质量守恒不能排除隐式正则化/梯度强度影响。即使通过，也不能仅据该配对
声称“保留类别信息”是唯一因果；后续若作因果主张需相应匹配的监督控制实验。

验证集操作点由当前验证集估计，是经验PR比较，不是独立部署阈值。
COCO-style使用当前YOLO标签/maxDet500，不冒充VisDrone官方AP。
soft labels已有Humble Teacher先例；本实验未建立可独立发表的新颖性。
