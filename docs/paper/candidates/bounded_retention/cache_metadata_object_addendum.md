# 元数据验收修订：对象数组补充

2026-11-06；academic-research-suite / experiment-agent / validate；新验收器执行前冻结。

4c256ca的cache_metadata_audit_revision.md中“拒绝object dtype”过宽。
独立只读审查确认，合法obs.physical原本就是object数组，保存PhysicalBoundaryContract或None。
拒绝整类object数组会拒绝未改变的真实输入；原检查器此前按对象内容相等比较，并非按指针字节。

只修订这一点：saved_tree_equal对非object NumPy数组要求dtype/shape/C顺序bytes一致；
对object NumPy数组保持原np.array_equal内容比较，并显式要求dtype和shape一致。
不得对object数组使用tobytes判相等，也不跳过physical；测试相同内容的独立对象、内容变更、
形状变更及None组合。object内容比较不因NaN而普遍放行，原拒绝行为继续保留。

其它冻结要求不变。新验收器额外绑定两份修订与旧停止报告；数值压力及旧检查器均不改。
原失败报告SHA仍为2a72884d41d1c19275a16d6756b218538d2c31edc0b8aeebeb06139bd4b54ea4。
补充原因和先前过宽规则均保留公开记录；当前尚未实施新资格审计或启动训练。
