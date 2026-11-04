# 正锚点质量排序：零学生训练诊断结果

## Material Passport

- Origin Skill / Mode: academic-research-suite / deep-research full + experiment-agent validate
- Origin Date: 2026-11-04，Asia/Shanghai
- Version Label: positive_quality_transport_diagnostic_v1
- Verification Status: VERIFIED（来源绑定/确定性缓存重放）；ANALYZED（探索信号）
- 学生性能NOT_TESTED，机制NOT_ESTABLISHED

## 判断

信号足以支持一个封闭最小工程实现，并准备设置明确召回护栏的配对实验；
**不足以证明此方案会增加AP、小目标召回或论文可辩护贡献**。
初始混合缓存按先定线全部通过但缺人工来源，保留SIGNAL_INSUFFICIENT；
随后显式来源单batch和实际CUDA AMP完整loss重放补齐局部人工信号，不把旧结论覆盖为通过。

选择[人工同对象正目标几何重排](../../candidates/positive_quality_transport/README.md)，
没有继续加减负监督系数，也未恢复旧NWD/回归/分配失败合同。
它保留监督质量总量，而非保证检测覆盖。所有数据都已被反复探索，不能称独立确认。

## 先行协议与来源

原[protocol.md](protocol.md)先于新统计保存：32批训练后训练增强缓存，
另16批原覆盖保护缓存，均为冻结原P1 30轮best、640原训练标签、FP32预测。
缓存SHA、receipt、原源码/运行库/权重/原preflight文件绑定核验，72项来源，
新objects/summary精确重放；零模型前向。绑定历史receipt不是重新重跑旧数据生成过程。
缓存没有增强对象人工/伪来源，confidence=1未用于推断来源。

来源不足后另存[source_check_protocol.md](source_check_protocol.md)：复用旧工程缓存中的
8图像素、244增强对象和显式来源，只做**一次**原P1训练后best前向。
无DataLoader、无新抽样，BN eval/raw head/CUDA FP16 AMP；参数及buffer完整state SHA
前后相等。缓存输出保存后，全部后续检查只重放，无第二前向。
193个人工对象/51伪对象来自旧便利batch，不是随机独立样本；mosaic可重复来源。
实际读取范围仅开放训练标签/已有伪标签；隐藏GT读取0、val新增推理0、optimizer步骤0。

## 1. 混合来源训练后缓存

small按640增强空间面积<32²，与val原图small和原tile尺度分组不同。
逆序率分母为有效多锚点small对象；>.05仅作诊断大差统计，不筛训练动作。

| 量 | 原32批 | 后16批 |
|---|---:|---:|
| 增强训练标签对象 | 4152 | 2350 |
| 有效多锚点small对象 | 2095 | 1297 |
| 目标值/普通IoU严格逆序对象 | 1538（73.4129%） | 836（64.4564%） |
| 有IoU差>.05逆序对象 | 478（22.8162%） | 277（21.3570%） |
| small原目标质量 | 6963.8037 | 3502.8767 |
| 搬移质量（目标变化L1/2） | 441.2807（6.3368%） | 231.3900（6.6057%） |
| small框外目标质量：原→重排 | 426.3383→462.5860 | 352.9970→376.2303 |
| 普通IoU/CIoU严格排序冲突对 | 498/50884 | 259/27534 |

四条先定投入线（≥100对象、≥10%逆序、≥5%大差逆序、≥1%搬移）两份全部通过。
但“后16批”也早已查看、mosaic来源重叠，不再称新确认。
大量成对逆序有很小IoU差；对象逆序率不能换成同等比例的严重框错误。

## 2. 显式人工来源与实际AMP路径

补充来源脚本的初版统计是AMP预测转FP32后CPU分配的**代理**。
独立反方指出它不等于原P1的BCE dtype路径，故不把代理结果当实际训练动作。
封闭工程使用保存的完整head预测，在CUDA AMP中直接调用P1损失/分配与目标cast，
没有再前向。下表以实际BCE float16目标>0为准。

| 实际人工small量 | CUDA AMP完整loss路径 |
|---|---:|
| 增强人工small对象（来源缓存） | 131 |
| 有有效分配 / 多锚点对象 | 129 / 126 |
| 严格逆序对象 | 100/126（79.3651%） |
| 有IoU差>.05逆序对象 | 32/126（25.3968%） |
| 原目标质量 | 380.4486685 |
| 搬移质量 / 比例 | 33.2916540 / 8.7506% |
| small框外质量：原→候选 | 35.3207978→38.6970288 |
| 目标质量加权普通IoU均值：原→候选 | 约0.7470→0.7516 |
| 普通IoU/CIoU严格排序冲突对 | 41/2788 |

FP32代理的大差逆序33个，实际AMP为32个；这是路径差异，完整保留。
实际工程summary只枚举已有效的人工作用对象，不把其objects=129误当全部131。
来源代理对伪框也计算了假想搬移，**不是候选对伪框实施动作**；
真实候选/控制完整loss已证明伪来源目标/梯度不动。

## 3. 支持与反证同时成立

支持：原TAL已有s^.5×非负CIoU^6质量信号，但实际目标顺序仍与普通IoU有错配；
人工GT与预测都在训练时可得；不增加未知背景负监督，可保持原有效监督集合。
新增约束是同对象真类目标次序，不是新几何度量/首次软标签。

反证：重排按定义提高目标加权普通IoU，这个数本身不能支持学生增益。
人工small框外质量增长约9.56%，同时box/DFL权重不变，可能扩大分类/回归不一致。
普通IoU忽略部分中心/形状惩罚；即时较好IoU也未必代表可持续可学习性。
这些反证未被“总目标质量守恒”消除，不在本轮按结果新增仅框内动作或切CIoU。

本候选只作用人工正锚点；主要wrong-class FP和跨对象排序不被直接监督。
因此可能工程很稳但AP/P90无益。旧回归/分配/人工负监督结果都提醒：代理性质不等于检测收益。
不假定伪标签错误、负监督强度或任一单项是唯一根因。

## 4. 工程控制及残余混杂

初稿散列纯排列虽同目标值集合，但总目标搬移L1=251.0567，候选111.5723，
存在明显扰动强度混杂，不拿它进学生比较。
在学生训练前修订为同对象原值到非几何排列方向的凸插值，匹配candidate搬移L1与总质量。
方向只依赖anchor/token/原目标，不看IoU排序；仅scalar预算来自candidate。
最终实际训练后缓存候选/控制L1为111.572260618/111.572259753，
控制总质量最大误差3.57628e-7；原对象有效支持、伪/负类和回归保护均验证。

控制不保目标值集合，改变锚点数866/1167不同；只匹配局部**附加**logit梯度L1，
不匹配最终梯度L1/参数梯度/优化路径。胜过控制仍需同时胜过原P1、满足全部覆盖护栏，
不单凭控制较差宣布因果机制。详见[工程验证](../../candidates/positive_quality_transport/validation.md)。
旧初稿工程回执保存到`outputs/positive_quality_transport_engineering_unmatched_control/`，
其中原辅助geometry占位统计不用于任何结论；最终工程已使用实际points/CIoU替代。

## 5. 复核与解释检查

```powershell
Set-Location D:/Python/YOLO-RegionSSOD
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B scripts/diagnose_positive_quality_transport.py --audit
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B scripts/check_positive_quality_source.py --audit
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B scripts/run_positive_quality_transport.py --check-only
```

前两项只读、确定性数值重放；第三项仅预检，不训练/前向或创建训练输出。
诊断收集入口默认拒绝已有输出，**不要重跑无--audit的source脚本**。
工程验证脚本同样拒绝已有目录；已有回执由预检核验，不需要另一个真实batch。

按ARS解释检查覆盖11项：非独立对象/重复来源、选择偏差、多次探索、代理与终点、
相关与因果、缺失来源、指标口径、构造性改善、强度/样本混杂、停止线与事后改线、
外推与显著性均检查。均明确限制，不生成p值/置信区间，也不把检查称统计认证。
本轮仍缺真正学生性能和机制验证；拟定三臂各30轮见独立学生协议，本次未执行。
