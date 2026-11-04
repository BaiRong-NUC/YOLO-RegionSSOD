# 人工同对象几何正目标重排：工程合同

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent implementation + validate
- Origin Date: 2026-11-04，Asia/Shanghai
- Version Label: positive_quality_transport_v1
- Verification Status: PLANNED（实现前保存）；学生性能NOT_TESTED，机制NOT_ESTABLISHED

先行混合缓存与显式人工来源检查投入线均通过。初始混合缓存来源不足的结论保留，
补充来源仅8图便利样本，不作独立确认。普通IoU搬移到框外anchor是已知反证，必须报告。

## 唯一变量

在原P1 TAL既有分配的每图/每个增强人工对象内，取真类实际BCE dtype目标>0的正锚点。
对这些锚点原真类目标值按detach当前预测框与该增强人工框的普通IoU排序重排。
目标值由小到大对齐IoU由小到大；相同IoU按原目标值次序，最终并列按anchor索引。
零质量、单锚点、几何全平不动。没有新值/归一化/系数/阈值/类别名单。

只改变分类BCE接收到的真类目标，不回写assigner输出：原TAL/正集合/target_gt_idx/
分母/box/DFL及其权重均保持逐位原样；所有伪对象和非真类、负锚点及ignore不动。
不能从pseudo_weights==1推断人工；复用已冻结SourceDataset及validate_batch_provenance。
对象按增强后本图local GT index分组，重复mosaic token不跨出现位置合并。
损坏来源/类别/整数/几何必须在动作前拒绝；参考臂同样校验并且关闭动作。

## 必要控制

三个独立学生：reference原P1，candidate几何排列，control同一对象/同一有效正集合内
按固定anchor+object ID整数散列排列原目标值，再与原值凸插值，匹配candidate的目标
搬移L1。若散列排列的搬移量不足，使用原目标值反序排列（最大L1排列）作方向，
确保可插值；方向不读IoU排序，只有目标扰动量来自candidate，且不使用训练RNG。
candidate无动作的对象control也无动作；零质量/单锚点/平几何均保持原样。
reference/candidate保持目标值多重集合；control保持同对象总质量、有效支持、值域，
**不保持目标值多重集合**，也不保证改变锚点数相等。这是方向控制，仍非完美因果隔离。
三臂样本数量不变；不新增BCE权重。候选/控制匹配同对象目标搬移L1（float32容差），
等价于固定logit/分母/系数下相对原目标的附加logit梯度L1；不匹配最终绝对梯度L1、
参数梯度或后续优化。控制输入使用原实际BCE dtype目标的精确float32表示，避免二次FP16
量化破坏匹配；原P1/候选的目标支持与AMP下溢保护不变。
质量守恒仅推出同对象真类BCE**有符号**logit梯度和不变（数学实数精度），
不推出各梯度、绝对梯度、模型覆盖/召回或总BCE损失相同。

控制定义修订发生在任何学生运行之前：首轮封闭工程重放发现散列排列搬移L1=251.0567，
候选=111.5723，存在明显强度混杂，因此不用该初稿控制进入学生比较。
旧工程回执保存为`outputs/positive_quality_transport_engineering_unmatched_control/`，
不是候选学生失败或重新选择科学阈值；本次只重放相同缓存，无第二forward。

## 验证顺序与不变量

1. 合成：来源绑定、conf=1伪来源保护；零有效/半精度下溢/极端finite logit；
   多对象/重复token分组；平IoU/单锚点/空批；NaN/Inf及非法类/来源拒绝。
2. 在同一个旧初始化完整cache与本轮唯一训练后完整cache重放三臂loss，
   不再取batch或前向。参考完整loss和autograd与P1相同；原TAL输出/分母/box/DFL逐位一致；
   伪来源/非真类/未触发位置梯度相同；候选目标值集合、三臂实际FP16有效支持相同；
   控制总质量/值域和逐对象目标搬移L1相等，检验实际附加logit梯度L1容差。
3. CUDA FP16 AMP+BCE/极端值检查；保留相同loss归约顺序，说明重排求和舍入误差。
4. 来源增强运输直接复用冻结实现，不复制改写；既有mosaic/关闭mosaic等价性测试继续覆盖，
   不为重复这些检查再加载真实batch。旧验证证明运输等价，不证明新排序效果。
5. 完整 `python -m pytest`；零训练preflight必须验证新工程/诊断回执、全部运行输入和新配置。

任何新增真实batch/前向预算已用完：本轮额外forward=1且只复用已有8图像素，
新DataLoader batch=0。后续只能重放已有缓存；optimizer步骤与学生训练循环始终0。
产物只存ignored新目录；旧P1和所有候选源码/协议/失败证据不修改。
