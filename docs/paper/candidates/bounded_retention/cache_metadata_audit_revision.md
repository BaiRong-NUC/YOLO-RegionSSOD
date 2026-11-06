# 缓存压力证据：占位元数据逐字节核验修订

- Origin Skill: academic-research-suite / experiment-agent / validate。
- 2026-11-06；实施前冻结；仅修订独立只读验收器，不修改数值执行或科学机制。
- 原audit_01完成34用例/102次完整loss及全参数反向，terminal=PASSED。
  qualification_01因`request metadata changed: obs`停止，保留全部失败证据。
  原停止报告SHA256：2a72884d41d1c19275a16d6756b218538d2c31edc0b8aeebeb06139bd4b54ea4。

## 已定位原因

原BoundedRetentionDataset为可选original_area_cap初始化float64 NaN占位；
连续域所用合法保留事件及几何字段不靠该可选字段补证。
初始压力request与pressure_batch的obs.original_area_cap均为shape(63,)数组，
63个NaN，dtype、形状、原始字节完全相同；np.array_equal默认仍返回False。
这是不可变元数据比较器的误判，不是loss/梯度非有限，也不证明其余证据已经通过。

## 唯一允许修订

新增独立audit_bounded_retention_cached_pressure_v2.py，保留原验收器、执行器及全部来源。
仅将saved_tree_equal中NumPy数组的比较改为相同dtype、shape和按C顺序的逐字节比较；
拒绝object dtype，避免以指针地址代替内容比较。不同NaN载荷、位置、有限值、符号零、
形状或dtype均拒绝。其它类型、张量和所有有限性/时限/来源/102臂/状态/3U暴露规则不变。
新版验收器额外绑定本修订和qualification_01停止报告，不将新增审计文件伪装成
audit_01数值执行时的输入，不重跑或拼接数值证据。

## 验证顺序

1. 合成测试覆盖相同NaN占位、NaN位置/载荷变更、有限值变更、shape/dtype/object数组。
   AST对比保证其它验收函数、历史绑定和规则原样保留。
2. 全库python -m pytest通过后，使用新qualification_02核验原audit_01全部证据。
   任一新的不一致仍停止并定位，禁止跳过字段或把存在NaN当作普遍放行条件。
3. 仅完整资格通过后实施生产接线；最终生产资格仍需另做真实初始化/loss接线检查。

本修订不改变120秒、20000cells、1e-5、K256、16384proposals、256bits或scale1024；
不宣称性能改善，不启动训练，不修改旧失败结果。
