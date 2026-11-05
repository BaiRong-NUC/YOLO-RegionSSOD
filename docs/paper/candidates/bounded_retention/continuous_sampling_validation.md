# V5独立数值与真实集成执行清单

2026-11-05，正式posterior_capacity_01执行前固定。原容量参数勘误另见
continuous_capacity_erratum.md，不修改旧失败receipt。

完整3U来源/运输/anchor沿用原理想容量例；显式分别使用tau=35/100和
binary64(.35)−binary64(1e−12)。每种tau检查两种logits：全部0，以及
第k个bin的三边logits分别为−.7k、.5k、.8k（binary64输入）。固定seed0、K256。

第二种tau仅是公开阈值/容差的理想实数差，**不是生产SOURCE_OUTER的逐对象
effective_retention**；后者还扣除了导出运算及比较舍入的认证预算，依赖公开可见
面积下界。容量脚本中的public_effective命名只标识该理想参数，不认证真实来源。
实际来源外集仍必须在真实双阶段增强流中另行验证。
每例都要求全局point-DFL含数值预算gap≤1e−5、后验完整覆盖资格、认证接受率下界
≥1/8、收满256个样本、q均值误差≤1e−10及有限loss一阶梯度。先保存完整源码与
合同快照，只导入快照；失败保存所在case及诊断，绝不换logits/seed挑通过例。

独立单测另以解析截面+一维数值积分比较完整3U体积/48个q矩：理想tau下固定
seed0..7共2048样本，逐分量最大误差≤.07；Hoeffding并合上界<2e−7。
该统计界依赖独立理想随机样本，PCG64实现假设必须披露。数值参照本身不冒称
向外认证。其他检查包括非零logits面积耦合、薄分支、跨DFL knot/饱和、资源失败、
私有RNG、同界点控制、完整loss和全known原P1值/梯度。

数值及完整loss门通过后才执行真实集成入口：
`python scripts/check_bounded_retention_sampling.py --scan-all-batches`。
其含合法源绑定、真实Trainer/model初始化、11310 train/548 val loader、默认增强流
及原生close_mosaic转换后的第二条完整增强流逐对象运输/支持检查。只执行一个
实际检测器batch前向，预先确定选择规则为两条流中首次达到最大活跃U维数的batch，
仅依赖公开观测元数据进行工程压力检查，不涉及验证集或性能筛选。
同一真实batch AMP前后向、原P1/候选/控制比较、实际head
logits独立算术audit、参数hash不变、零optimizer；不执行验证推理。
全库`python -m pytest`和源hash一致性仍为放行必要条件。
