# 回归置信度30轮：验证兼容修正版协议

2026-10-31，修正版学生性能未知时登记。沿用
`regression_confidence30_protocol.md`的科学变量、参考和性能继续线；
旧运行首轮验证失败，不能判断性能，见`regression_confidence30_failure.md`。
旧回执、日志、配置、源码和协议快照全部保留，不覆盖旧输出或拼接轮次。

## 修复范围

训练模式继续使用原教师置信度c乘伪框box/DFL权重，人工GT为1，
保留未加权TAL分母、分类软权重和ignore；训练缺少元数据仍抛出异常。
验证模式走原P1损失，正常全标注验证批次不要求伪框元数据；
即使EMA已经缓存训练损失，也必须使用验证分支，不替换原训练criterion。
训练后再次切回train时，仍执行置信度加权和必要字段保护。
不关闭逐轮验证、不跳过验证损失、不改模型结构/标签/分配或学习率。

独立命名为`partial10_p1_regression_confidence_valfix_pilot30_s0`，
从同一COCO预训练独立初始化，11310tile、相同10%面积split/教师/伪标签、
seed0、640、batch8、SGD、close_mosaic10、完整30epoch，无早停。
完整P1 seed0 30轮仍为参考；不重新训练参考。

## 工程检查与命令

使用yolo环境，在仓库根目录运行。原入口必须显式选择`--validation-fix`，
否则拒绝启动；新的run、dispatch或status已有时也拒绝执行。

```powershell
python scripts/run_p1_regression_confidence.py --validation-fix --check-only
python scripts/run_p1_regression_confidence.py --validation-fix --engineering-only
python -u scripts/run_regression_confidence_background.py --validation-fix
```

前两条只预检和工程检查：真实YOLO11s CUDA AMP前反传，
再用标准验证加载器的前2张原图调用训练期DetectionValidator，
包含EMA缓存训练criterion、eval前向、验证损失及NMS/统计路径。
验证损失必须有限，并与原P1对同一预测的结果精确一致。
再核对train缺元数据仍拒绝。工程过程0优化器步骤，不保存学生权重，
不将这2张图的指标用于方法判断、阈值选择或论文结果。
工程产物绑定配置/入口/训练源码哈希，训练启动必须再次核对。

第三条才执行唯一完整30轮、548完整原图val960和固定操作点recall。
此前修复阶段仅执行前两条；2026-10-31用户进一步明确
“你的小训练由你开启，需要大训练找我”，替代此前等待用户启动本次30轮的安排。
助手按该授权执行第三条一次，不扩展预算；更长训练仍只交用户执行。
启动后创建临时监控，正常静默，完整结束或失败处置后删除。
不因观察超时重启、不自动续训、不拼接轮次、不并行新实验。

13:28的一次启动在首轮265/1414batch以KeyboardInterrupt中断，
原包装20616/训练1540均已停止，原回执没有exit_code，原status仍为training。
中断来源未知，不推断为方法失败或伪造退出码；9个原始文件及六份冻结源码快照
已保存在`outputs/p1_regression_confidence_valfix_interrupted_20261031_133202/`。
本次按新授权从原COCO权重独立开始完整30轮，不接续这段中断进度。

原进程停止且exit0/status completed后，运行：

```powershell
python scripts/audit_p1_regression_confidence.py --validation-fix
```

核对实际args/输入/源码/权重/产物SHA256、连续1–30轮，
548图val960及两组固定操作点预测重放。
recall口径conf0.25、match IoU0.50、NMS0.70、max_det500，
不是官方AP_small/AR500。
新证据目录为`outputs/p1_regression_confidence_valfix_pilot30_s0/`，
训练目录为`runs/partial10_p1_regression_confidence_valfix_pilot30_s0/`。

## 不变的性能继续线

- mAP50-95至少+0.30pp且small recall不下降；
- 或small recall至少+1.00pp且mAP50-95不低于−0.10pp。

至少完整30轮后判断性能。工程失败不视为方法失败，不追认缓存诊断为学生增益；
即使过线也仅准备更大训练冻结配置/预检/验收代码和用户指令，不自动扩训。
单seed不证明显著性、泛化或充分收敛；常数衰减对照仍待后续决定。
仅A，不做B/C、基础模型、蒸馏剪枝或部署。
