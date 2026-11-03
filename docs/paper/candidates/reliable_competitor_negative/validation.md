# 人工正锚点竞争负监督：工程验证记录

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate
- Origin Date: 2026-11-03 (Asia/Shanghai)
- Verification Status: VERIFIED（张量与工程不变量）；学生性能NOT_TESTED；机制NOT_TESTED
- Version Label: reliable_competitor_negative_v1

## 实现与证据范围

先保存`engineering_protocol.md`，再实现源码、配置和三臂入口。
原P1、旧V2及旧候选合同／权重／失败证据未改。没有启动学生训练循环、optimizer，
没有隐藏GT读取、val推理、教师推理、阈值搜索、重采样或监控创建；未引入新论文。

`SourceDataset`沿P1标量对象token路径运输类别、置信度、ignore和显式human标记，
每个人工对象分配独立token；伪框即使置信度1仍为伪来源。mosaic／裁剪／翻转后
解码，人工／伪类别与token绑定校验，解码前拒绝非整数／越界token及非法lookup行。
解码ignore前也验证来源，避免损坏ignore隐藏对象；loss中先查全batch身份再处理数值质量。

`ReliableCompetitorLoss`直接调用原`P1DetectionLoss`；包装其TAL仅捕获输出，
包装BCE仅添加固定候选／控制乘数。实际对象类别来自TAL对象ID索引和真实标签，
从不从全零目标argmax恢复。正类乘数恒1；人工有效正质量>0才作用，
伪框／背景／ignore不动作。所有类权重要求原配置的unit权重，P1因子／分母／回归原样。
EMA可能缓存训练criterion，标准验证的`model.loss`明确回到原P1；缺训练来源仍拒绝。

非真类学生概率使用detach后的fp64 sigmoid计算，最高值并列固定取最小类别ID。
控制用最大值缩放的负类概率和计算相同比例，防止极小分母0/0；返回fp32乘数。
所有概率都下溢为0时控制不附加权重，候选乘2对应的新增梯度仍为0。
这仅匹配当前锚点局部负logit梯度L1；不匹配BCE值、参数梯度或后续优化动态。

## 实际执行

环境：指定yolo Python，PyTorch2.10.0+cu128／Ultralytics8.4.172／RTX4070Ti。
旧合法缓存SHA`8a1aa278e75756ac5e5315248235e0c60b33f6ab943d76cf57ba01700fb41bb6`，
1988个前景锚点，其中1322个人工有效正锚点，17个人工零质量锚点。
既有缓存只有前景张量，无法检查新来源运输与完整损失，故使用授权的**一个真实batch**。

真实batch8、640、开启原训练增强，来自11310tile训练集；增强后193人工对象／51伪对象。
仅一次CUDA FP16 AMP前向，保存`real_batch_cache.pt`约72.2MB；后续全用该缓存。
没有创建优化器，且SGD／Adam／AdamW.step、BaseTrainer.train、Tensor.backward均设禁止调用。
合成测试独立允许autograd，未对真实模型反传。

| 不变量 | 实际结果 |
|---|---|
| 新参考与原P1完整损失 | 逐位一致 |
| 三臂TAL输出／box／DFL | 同预测逐位一致，分母和分配调用均委托原P1 |
| 正目标／正类乘数 | 目标完全一致，正类改变数0 |
| 候选动作支持集 | 恰为1341个人工有效正锚点；每锚点仅1个非真类 |
| 伪框保护 | 497个伪正锚点完全不动作 |
| 人工零有效质量 | 26个不动作；另有合成FP16下溢测试 |
| 新增负logit梯度L1 | 真实缓存逐锚点最大绝对误差8.67683e-8；旧缓存4.23019e-8 |
| 误差容差 | 缓存解析梯度rtol2e-6／atol1e-10；实际autograd另有float32／AMP合成测试 |
| 参数哈希 | 前后同为160ed4742f2a980e8d00231249e2fd116622b66a3d2168ca535c3be81f0602bf |
| 训练／反传／优化步／val推理 | 0／真实模型0／0／0 |

一次train-mode前向可更新临时模型BN buffer；“参数不变”特指所有可训练参数，
临时模型未保存为权重／未用于学生训练。工程loss值不放入性能表。
首次真实工程检查13.422秒；加强损坏来源拒绝后的同缓存重放0.625秒，不计测试及预检。

初版检查后，负向审查发现ignore标志.bool和TAL ID.long可掩盖非法原值，
已先查原值再转换。保留初版`validation.json`和`capture.json`，最终源码只读重放
写新`validation_final.json`，没有第二个真实batch。最终回执绑定当前源码和原缓存SHA，
原capture保留首次前向源码SHA，两个版本差异仅来源拒绝边界，缓存数值不变。

合成增强测试在相同随机种子下比较原P1与新传输路径的图像／类别／框／置信度／ignore，
mosaic开启和close_mosaic两条路径逐位相同，附带真实对象lookup复核。
完整损失／autograd测试覆盖背景、伪框、人标与空标签；新增负向测试覆盖损坏来源、
真实类别、ignore、非整数TAL索引；数值测试覆盖并列、全零、半精度最小正数及极端logit。
最终完整pytest和预检结果见本记录末尾验收条目。

## 限制与11项解释检查

工程通过不证明学生AP、错类FP或覆盖改善，也不证明选中最难类有独立语义收益。
三臂还未训练，三臂终态audit和实际训练／评估端到端路径只有工程测试，待真实执行确认。
新来源检查／fp64概率增加开销，实际训练时长未测；计划三臂合计约6–9小时+评估，
其增量开销无法从单batch估计。人标拥挤重叠可能强化冲突，未消除伪标签偏差或严格定位限制。

覆盖11／11：Simpson（无聚合性能外推）；生态谬误（锚点工程不推目标性能）；
Berkson（选定batch不代表训练总体）；碰撞偏差（未拟合条件因果模型）；基率忽略
（人工／伪／零质量数量明列）；向均值回归（无性能比较）；幸存者偏差（保留初版与旧失败）；
多重查找（不按指标挑测试／调参）；分析路径自由度（固定2倍与10项投入门槛，不追val）；
相关因果（局部L1匹配不称语义因果）；反向因果（先工程，不由诊断倒推唯一训练根因）。
无p值／显著性／独立泛化宣称。困难负监督及S³OD近邻已在旧文献材料中核验，
本轮未新增文献，未重复导入Zotero或改变阅读状态。

主回执：`outputs/reliable_competitor_negative_engineering/validation_final.json`；
原batch：同目录`real_batch_cache.pt`；所有产物在ignored目录，源码／协议版本化保存。

## 最终验收

- 完整`python -m pytest --basetemp outputs/reliable_competitor_pytest_complete`：
  **806 passed in 11.93s**；包含原318项及新数值／来源运输／编排合成测试。
  日志`outputs/reliable_competitor_pytest_complete.log`，零学生训练。
- `run_reliable_competitor_negative.py --check-only`真实exit0／accepted=true，冻结39,906项来源，
  11310图、105058个人工标签行／35067伪正框／52117 ignore；548图val标签仅作预检读取，
  无模型／训练／推理／训练目录创建。39,906包含当前候选报告／源码等文件，不是样本数。
  stdout回执`outputs/reliable_competitor_preflight_console.log`，启动时重新预检冻结当前字节。
- 原完整P1实际args与新reference的完整默认／覆盖参数比较，仅忽略name/save_dir，差异为空；
  本次仍选择全新reference，不把历史权重或指标作为新配对。
- 61项runner合成测试含十项门槛逐项失败、缺control、权重／来源漂移、非法worker绕过、
  已有输出拒绝、聚合label cache绑定、顺序停止及终态audit不写入；未伪造真实训练终态。
- `git diff --check`通过；冻结P1/V2源码无diff。更新论文计划第3.75／10节和README／baseline入口。
  本次只提交上述授权修改，实际commit与push结果由交付回复给出。
