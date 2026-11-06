# 共享精确几何缓存：等价性与真实压力验证合同

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；Mode: validate。
- Date: 2026-11-06；Version: geometry_cache_v1；Verification Status: UNVERIFIED。
- 起点7f9babf。用户授权继续处理阻塞、完成资格验证；沿用先给训练指令、不启动训练。

## 科学与工程不变项

这是实现成本修复，不是新机制或性能证据。保留原P1、V6连续相容集合/名义精确边、原DFL插值、
面积耦合、后验目标及最强同界连续点控制。候选和控制共用同一缓存求解器。
不删除对象/锚点、不降级已知坐标，不读取隐藏GT/计数/质量缓存，不改TAL、CIoU和归一化。
point_search_v2的搜索遍历、tie规则、误差1e-5和20000cells不变；K256、16384proposals、256bits不变。
仍用每次完整loss/反向120秒保护，不以延长期限来通过旧故障；历史失败及所有源码保护。

## 独立实现

新增point_geometry_cache模块，只缓存同一NativeEDomain及精确Fraction cell输入的
contract_bounds返回值（tuple或None）。每域LRU最多1024条，域销毁后可回收，不跨臂/进程共享热状态。
几何快照字段发生变化则明确拒绝使用旧缓存；缓存不含logits/分数/随机采样或梯度。
使用原v2求解函数的同一code object、私有globals绑定缓存函数；原模块globals不修改，
所有计数/返回diagnostics和异常保留。单未知边直接用原函数。缓存清理只影响成本。
不预设此优化足以消除全部耗时；任何后续扩大缓存范围须先另记修订。

## 固定验证顺序与停止条件

1. 单元/合成验证：不同维数、翻转、区间运输、同分、空域、容量异常及重复查询；
   与原v2完整point/gap/diagnostics逐项相同，验证域隔离/有界回收/结构改变拒绝、原globals未改。
   全库python -m pytest通过后才能真实数值检查。
2. 固定trace_04保存预测重放：原P1校准→新候选→新控制顺序，保持原输入/RNG/autocast、
   每臂冷缓存和120秒绝对期限；所有2590点输入及已记录point/gap/cells对旧trace逐项相同。
   如完整候选不通过，停止压力门，不为凑训练放宽期限或弱化控制。
   带记录器耗时仅作描述性同输入比较；不得用不同阶段时间混加，也不推翻旧2倍速度门失败。
3. 通过后新独立pressure入口仅替换共享求解器/输出root，继续point_pressure_protocol及两个补充：
   真实Trainer初始化、两条完整1414批增强运输重放，固定closed1316；初始和第9轮EMA压力case，
   再第9轮EMA固定open00..15/closed00..15。34次显式前向、102次完整loss及全参数反向，
   scale1024不搜索，参数/BN恢复，真实3U正TAL权重消费和不同梯度，候选/控制同几何/TAL/CIoU。
   旧第9轮EMA只用于诊断，绝不恢复旧训练或用作新训练初始化。
4. 独立只读checker核验全部保存证据；任何不完整/超时/非有限/证书差异停止，不自动重试。
   包装错误须公开保留、先冻结针对性修订，不把失败拼接为通过。原冻结生产文件不改。
5. 全部资格通过才新增独立训练Model/Trainer/launcher和三臂30轮配置，按原548图full-image
   val960协议及全部AP/AP_small/召回/同K/同FP/重复框/定位指标冻结；新增接线再验证初始化和pytest。
   仅执行只读--check并提供顺序三臂命令，不执行--train。诊断成功不代表未来无故障或性能改善。

新输出仅outputs/bounded_retention_cache_20261106/，全新子目录，保留所有失败。
先提交合同，再实现与执行；最终更新第10节和入口文档，中文commit/push。
