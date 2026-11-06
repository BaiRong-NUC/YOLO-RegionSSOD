# 定位记录器修订：只保存纯数据

2026-11-06，trace_01结束后、trace_02之前冻结。
trace_01的P1完整loss/assignment逐位校准通过；候选在第一次点调用的观测器内
pickle.dumps(domain)失败，域实例包含不可pickle的实例级lru_cache方法。
异常发生在真正optimize_point之前，不是数值故障。控制按合同未运行，原目录完整保留。

仅修改独立观测器：不序列化行为对象，改保存box/canvas/mask/source key/affine/clip/bounds
的纯数据JSON及摘要；同一域只记录一次。新增真实NativeEDomain缓存实例的记录回归测试。
保留同一输入、原P1逐位校准、候选/控制顺序、全部算法和120秒绝对期限。
不是调整数值参数重试，不接续旧run，不改旧terminal。全库pytest通过后才在trace_02执行。
trace_02若数值失败/超时，按主合同交付定位证据，不直接延时或开训。
