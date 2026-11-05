# 正目标重排三臂负结果：事后缓存分析协议

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate + deep-research synthesis
- Origin Date: 2026-11-05，Asia/Shanghai
- Verification Status: PROTOCOL；在新增统计前保存。原三臂结论已知，本分析完全是探索性分析。

## 问题与边界

判断固定正锚点几何重排的失败更符合分数操作点变化、可用检测覆盖/定位损失，还是已有数据不足以区分；据此判断下一项方法研究是否值得投入。
不重新判定旧11门槛，不选最佳类别、阈值、损失系数或训练轮数，不以分析代替新学生验证。
不训练、模型forward、新真实batch、新val推理、隐藏GT读取或修改冻结源码/合同/封印目录。
原P1正式100轮与本次新配对30轮分别报告。终态独立--audit已在本聊天完成一次，本轮不重复。

## 唯一数据来源与绑定

- `outputs/positive_quality_transport_v2_three30_s0/`：三个完整30轮学生的终态、比较、决定、三臂seal/exit、统一evaluation缓存。
- 使用其`artifact_sha256.json`校验全部封印产物及run，evaluation manifest/analysis的weight/prediction绑定；读取已有548图`ground_truth.json`与后NMS预测缓存。
- GT来自反复查看的开发val，不是新盲集；不读取原始训练隐藏标签。只读既有缓存不能消除选择偏差。
- 新产物只写新`outputs/positive_quality_negative_review_20261105/`，目录存在即拒绝覆盖；独立诊断脚本只作numpy/JSON缓存分析。原目录仅哈希读取。

## 在计算前固定的分析

1. 完整复述reference/candidate/control绝对指标、三种差值、全部11门槛及每类AP_small，不混Ultralytics AP、COCO-style AP_small和micro PR。
2. IoU .50/.75：保持整组同分预测，候选与控制分别受reference在conf.25的FP预算及预测数量K约束；取预算内最长预测前缀，不拆分同分组，不插值。点不存在报告不可达。另列原固定.25/P70/P90，不把GT预算阈值当推理方法。
3. 相同GT总体内，small在固定.25及各臂P70/P90的共同命中/新增/丢失/均漏；比较全缓存conf>=.001的同类一对一可匹配覆盖（受后NMS、maxDet500、.001限制，不是原始proposal容量）。
4. 同时可匹配small的IoU/中心/边界误差与分数变化，附条件选择限制；不能因共同GT变好就忽略失去的GT，也不声称训练期几何因果。
5. conf.25双IoU的全部FP类别分解；不只挑wrong-class。检查固定操作点cap尾分数是否可直接截断该点，不能排除NMS/pre-NMS/.001限制。
6. 全10类AP_small保留；预先仅给去掉单类的9类差值范围作敏感性描述，不据其调类别名单或重判失败。
7. 记录三臂实际训练秒数/worker墙钟和完整验证成本；只给未来方法的预算级别，不自动分配GPU任务。

## 解释规则与停止条件

同FP/K的下降可反对“只因固定conf不合适”，不能唯一归因为排序/回归；控制不匹配熵、目标集合、改变锚点数、最终/参数梯度或独立轨迹累计强度。
候选与控制均P90升高只能说明几何特异性未建立，不证明两者等效或所有质量监督无用。
最多比较三个实质不同方向；每项列训练可得信号、相对P1新增信息、旧失败的排除范围、近邻重叠、反证诊断、控制及完整成本。
没有未被充分利用的新信息和可识别控制时，不写新loss、不提供新训练命令；允许结论为只值得一个有期限的零训练可行性审查。
ARS统计11类谬误逐项检查，不生成显著性/p值或把GT个数当学生seed样本量。
