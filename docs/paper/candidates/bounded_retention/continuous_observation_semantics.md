# 连续候选的已知边与来源观测语义

2026-11-05，真实V5集成检查前冻结；来源几何独立审查补充。该澄清不改变
V4/V5公式、SOURCE_OUTER、运输算法或原P1，不增加删对象/降级E的许可。

记O为按合法来源hash和行号绑定的原始tokens，A为实际发生的增强操作。
最终名义已知边满足 `E_nominal = F_native(O, A)_E`。原生标签计算读取O，
不读取相容集合oracle临时选择的latent物理E。给定O、A后，最终E并不是对
latent E的第二次独立测量。故不得额外要求latent物理E的精确实数仿射值，
或其一次float量化值，等于经历完整原生数据流的E_nominal。

实现采用明示的混合表示：E固定为原生监督坐标并使用原P1 DFL；U为连续
latent坐标；保留事件的面积耦合和来源xywh区间由同一个SOURCE_OUTER物理
见证满足。原始已知坐标的来源区间及相关约束仍在见证中，未主动移除。
`C = projection_U(transport_outer(SOURCE_OUTER)) intersect D(E_nominal)`。
不同U边的运输区间参数独立选择是已声明的保守放宽。原U经确定性clip解析
后在来源消元中保持自由，不能把它误替换为新观测。

这不承诺拼出的四维名义框恰等于某个源框的完整物理运输结果；
`full_box_transport_inverse=False`本身不代表丢失已知坐标。也**不声称所有
SOURCE_OUTER见证经重新导出都会生成原tokens**：该集合只认证必要外包，
不是历史导出的精确逆像。两臂使用完全相同的混合集合、已知信息和连续点控制。

必要工程约束仍是：原tokens的float32解析与真实加载标签一致；不同tokens
碰撞成同一float重复行的歧义必须拒绝；真实变换的同一次矩阵/筛选同步作用于
原生框与sidecar；source_object/occurrence及最终原生E全程绑定。绑定失效
属于实现错误，不得用额外容差、隐藏物理E、删除对象或E→U绕过。

代码依据：`bounded_retention_training.py`的合法行绑定，
`bounded_retention_transport.py`的矩阵/筛选运输，
`bounded_retention_objective_v4.py`的身份对齐与E原值保留。
独立审查只读源码，没有读取隐藏GT或真实样本，没有代替真实集成门。
