# 共享几何缓存生产接线与三臂30轮合同

2026-11-06；academic-research-suite / experiment-agent / validate。
状态：压力资格通过，生产接线及最终预检待验证。此文件本身不授予已完成工程资格。
用户要求自主完成验证、最终提供训练指令，本会话不执行训练。

## 先决证据与不变科学合同

继承three_arm30_contract.md的机制、最强同界连续点DFL、合法来源、增强运输、
训练/评估设置、全部指标、停止规则和解释边界，不追改历史失败或性能门槛。
仅替换为point_search_v2及同域精确几何缓存：相同点、gap、遍历、tie和全部锚点。
固定trace_01所有2590点/证书与旧trace_04一致；audit_01完成34用例/102完整loss及
全参数反向；qualification_02用修订后的独立checker完整核验通过，覆盖255个可训练参数张量。
初始和第9轮EMA均有同一真实3U实例的10个正TAL监督锚点，非10个独立对象。
qualification_01占位NaN比较误判及两份修订均保留。第9轮权重仅诊断，不作新训练初始化。

20000cells、1e-5、K256、16384proposals、256bits与诊断120秒不变。
诊断scale1024不进入训练超参数；生产仍为原生AMP/GradScaler。
公开保留事件为模拟GT衍生侧信息，不是自然标注保证；不宣称首创CABB/GFL已有思想。
两臂共用同界求解，分布目标不是任意联合相关性建模；改进是否存在必须等完整实验。

## 独立生产实现和零训练验证

新增objective_cached、training_cached模块。原V6及P1源码不修改。
新support/get_pos_mask使用冻结V6同一code object及私有globals绑定缓存求解，
禁止生产全局monkeypatch；候选与控制共用同一实现，DFL/后验/归一化沿用V6。
原P1参考直接实例化P1Trainer。新Trainer沿用已审计初始化顺序，先names/options/mode再load。

先合成/全库测试，再验证真实factory→setup_model→set_model_attributes→model.loss路径。
候选/控制均验证初始和第9轮权重的固定真实3U压力输入；原P1校准、完整loss、两head及
全参数反向与已通过压力路径核对，共享几何/TAL/CIoU、不同梯度、缓存实际命中及调用计数。
相同权重/RNG/predictions须逐位核对；若设备重放存在差异必须保留并诊断，不能临时放宽。
验证真实dataloader身份/来源接线，禁止optimizer.step、训练和验证推理。
验证全known/无retention元数据回到原P1及标准验证loss，EMA创建/复制/保存不包含缓存闭包。
每阶段仍120秒保护；恢复BN/参数，不改旧产物。缺证或不一致停止，不自动重试。

## 三臂及最终冻结

独立入口scripts/run_bounded_retention_cached.py，顺序reference→candidate→control，各30轮。
新配置configs/candidates/bounded_retention_cached/{reference,candidate,control}30.yaml，
完全继承positive_quality_transport_v2/reference30.yaml，仅project/name/本次mode变化。
COCO初始化D:/Python/YOLO/yolo11s.pt，seed0、640、batch8、SGD、workers0、close_mosaic10。
新训练目录outputs/bounded_retention_cached_three30_s0必须不存在；不恢复、不覆盖、不并行。

新launcher保留原V2全部合法输入、旧G0 STOP/AMP资格、预览/源码快照/基线、故障与seal守卫。
旧几何工程要求的文件集合由其不可变回执确定；新文件另由trace、pressure资格、实际生产接线
和最终全库pytest绑定，不能要求新源码在历史回执出现，也不能删旧检查逃避来源验证。
SHA集合合并冲突即拒绝。新增schema/token/类名/固定输出检查；最终freeze绑定全部新代码、
配置、本合同、合法数据和证据，--check只读。只有接线及完整pytest通过才写FINAL许可。

三臂一致禁止OOM减batch重试、NaN回载、自动恢复/重试或杀用户worker；原生AMP跳步允许。
训练派发含进程身份与日志；30分钟临时监控仅记录存活/进展，终态退出清理，不控制worker。
最后统一548图full-image val960，保留AP/AP_small/召回、同K/同FP、重复框、定位和机制诊断。
完整30轮前不作性能判断；三臂完成后才按原合同分别判断工程、性能和机制，不挑子集。

## 成本披露

共享点求解在固定失败批从约101–104秒降到约43秒；这不是整轮提速证明。
第9轮固定16open/16closed批机械外推约75小时纯计算量，包含诊断反向而非原生训练耗时，
也不包括加载/优化器/验证/I/O，不是ETA保证。用户已表示时间可接受；不因此放宽资格。
训练结果可能无改进。最终只交付经只读预检的命令，用户自行启动。
