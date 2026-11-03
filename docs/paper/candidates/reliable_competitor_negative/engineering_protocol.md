# 人工正锚点竞争负监督：先行工程验证合同

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate
- Origin Date: 2026-11-03 (Asia/Shanghai)
- Verification Status: PLANNED（本合同先于实现与验证保存）
- Version Label: reliable_competitor_negative_engineering_v1

仅实现本目录新候选，不修改冻结P1/V2。人工来源、真实类别和对象token显式绑定；
沿原P1标量类别token增强路径运输，解码前检查整数、范围，解码后保留来源与对象ID。
人工对象不能由置信度是否等于1推断；损坏来源或真实类别必须拒绝，先于零质量筛选。

原P1 TAL分配不变，使用分配的对象ID和类别，不用全零目标argmax。
只有人工GT且实际BCE正质量>0才动作。候选对最大非真类sigmoid概率对应的负BCE乘2，
并列取最小类别ID。控制对同一批锚点所有非真类乘1+p_k/sum(p_negative)，
detach概率，用稳定精度处理极端值；全零概率不附加梯度。只匹配局部logit梯度L1。

按以下次序验证，所有原始产物保存在ignored outputs下，训练和val推理次数均0：

1. 合成损失/梯度：来源/类别绑定，人工与置信度为1的伪框可区分；正项/正类权重不变；
   动作支持集、并列、无有效质量、FP16目标下溢、极端logit、非法来源/类别。
2. 使用同一预测与batch对照原P1：TAL输出、分母、box/DFL逐位一致；参考组完整损失与
   梯度一致；候选/控制非触发锚点梯度一致；新增负logit梯度L1在声明容差内相同。
3. 优先检查已有合法训练张量缓存。若不足，仅加载一个真实train batch进行CUDA AMP
   前向/损失工程检查；禁止训练循环、optimizer.step、隐藏GT读取或val推理。
   合成梯度允许反传；真实模型参数在检查前后哈希一致，统计optimizer步数为0。
4. 增强运输：固定随机种子比较原P1与新来源数据集mosaic/裁剪及关闭mosaic路径；
   图像、类别、框、置信度、ignore一致；token携带的真实类别/来源可追溯。
5. 完整python -m pytest，三臂预检（不创建训练输出），负向测试覆盖覆写/缺证据/漂移。

测试通过只记工程通过；学生性能与语义机制保持NOT_TESTED。合成测试不提供性能证据。
性能/机制门槛沿README独立新合同，不修改旧V2失败判定；不新增论文或新颖性主张。
