# 有界保留事件监督：实现完成至零训练集成，精确边支撑阻塞

2026-11-05。起点2dcece9；使用academic-research-suite及独立来源、运输、Trainer复核。
**终态：STOP_BEFORE_STUDENT_TRAINING。三臂均未启动，完整epoch=0；性能NOT_TESTED，机制NOT_ESTABLISHED。**
不是30轮性能失败，也不能由此否定所有有界分布监督。当前固定E、原非负DFL支撑、原增强协议不能通过必要门。

## 1. 先行合同与独立实现

按顺序保存[工程合同](engineering_protocol.md)、[舍入补充](rounding_addendum.md)、
[消费者合同](consumer_contract.md)、[运输补充](transport_addendum.md)、
[覆盖检查补充](support_coverage_addendum.md)及[原生TAL事件修订](native_event_addendum.md)。
各次执行保存实际源码/合同快照；早期合同的PRE_IMPLEMENTATION是历史状态，不是本报告的当前状态。

- `bounded_retention_observation.py`：公开保留参数、`.10g`区间及IEEE误差推导，joint source witness外集认证。
- `bounded_retention_source.py`：钉住A/B报告、回执及source audit，合法人工标签逐行SHA、对象ID；不重新打开manifest。
- `bounded_retention_transport.py`：对象/增强实例身份、原精确/未知语义、仿射区间及独立累计增强裁剪域。
  Mosaic、RP、缩放、翻转、筛选、Format/collate与原安装版本真实变换图对齐；不支持空间变换显式拒绝。
- `bounded_retention_objective.py`与独立`_v2.py`：同一观测支撑供TAL/CIoU/DFL消费；E保留原DFL，U取sum或MAP。
  全known完整损失和logit梯度逐位恢复原P1；无新系数、backbone或附加head。
- `bounded_retention_training.py`与独立`_v2.py`：真实train/val Dataset/Trainer接入；正确lexical继承，eval损失走原P1。

版本1源码及integration_03停止证据保留。版本2只修订两项语义：使用原native TAL池；仅E检查名义距离非负/非饱和。
U仍枚举0..15并经过相同物理、面积、累计clip及画布约束，不作clamp。最终正分配不声称与原P1相同。

## 2. 来源与数值证据的准确范围

6471合法人工标签/105058原始行，1条完全相同五token的重复行按原生`np.unique`规则合并，
保留105057对象并记录两条来源alias；不同精确token碰撞到同float32时拒绝。
来源参数绑定为area=.1、seed0、visibility=.35、jitter=.2，`A_visible/A_input + 1e-12 >= .35`。
已绑定公开几何tile131..548、源图/tile线性比上界10.069343278363734，未读取原训练全图/未标注tile隐藏GT或隐藏计数。
接受的既有P1伪标签/ignore元信息仅作原训练输入；不使用旧逐框GT质量缓存。

`SOURCE_OUTER`是基于绑定导出代码、输入binary64定义、`.10g`及误差预算的保守外集，
不是唯一反演出历史完整框。原输入标注框不等于amodal物体。
名义E固定为原P1连续坐标，物理E只作同一来源的存在见证；不拼接独立不相容端点。
不声称每个名义输出框严格满足无误差的原始面积上界。

区间可能触边12408对象（11707单边、698双边、3三边），不是严格文本4145的重新解释，
更不是实际cut或有效训练样本数。公开保留事件含模拟生成GT衍生侧信息，不可外推为自然人工采集保证。

## 3. 实际检查轨迹（全部零优化）

独立输出根：`outputs/bounded_retention_engineering_20261105/`。

| 检查 | 结果与实际范围 |
|---|---|
| source_audit | 6471标签绑定、精确重放；后续来源模块补上有条件binary/outer认证 |
| integration_01 | 真实train loader发现原始行与native去重不一致，停止；保留证据后显式实现alias |
| integration_02 | 真实YOLO11s/train11310/val548初始化，首批21部分未知对象通过；1次显式AMP批forward/backward有限，0optimizer、0val inference；只是组件通过 |
| integration_03 | 扩大到首epoch增强流，第二batch在U-only负名义距/无可见内锚点停止；未完成整epoch，不能事后改PASS |
| native_event_integration_01 | 新协议下原03反例通过；第二batch遇到E负距实质阻塞，仍停止；0显式batch forward、0optimizer、0val inference |

上述forward计数不包含原生模型构造时的stride probe。没有运行完整Trainer优化setup或学生训练循环。
integration_02的真实AMP通过属于版本1；版本2只完成了合成完整loss/梯度及真实初始化/loader/几何检查，
不能偷换成版本2真实AMP已过。真实变换图测试覆盖安装环境默认路径；本机未实例化Albumentations的路径不作认证。

旧03对象115723仅top未知，原TAL(588,52)在版本2有合法bin0；实际累计clip起点45.2647排除了bin1(top44)。
这由真实数字回归测试确认，未用理想合成的“bin0/1都可行”冒充真实证据。

## 4. 最终阻塞与可重放证书

`native_event_integration_01/result.json`：PID2456，UTC04:57:56.864637至05:01:19.190132，202.3255秒，已自然退出。
Python入口按STOP返回2；宿主执行工具回报shell exit1。没有运行中的学生或工程进程需要终止。
第一batch21/21部分未知对象有支撑；第二batch检查28个，27个有支撑，第28个停止。
这些是增强实例检查数，不外推总体比例或实际正训练数。

阻塞为source object142895/occurrence51，第二batch第433行（0起始），图像索引4：

```text
合法标签：9999956_00000_d_0000029__lab_108_102_332.txt，第2行
4 0.9457831324 0.7951807227 0.1084337352 0.06626506089
名义输出框：[341.264892578125, 636.80810546875, 377.252685546875, 640]
U：仅right；top仍为E，新裁剪不制造未知边
原native TAL：7 anchors；y=636(stride8)或632(stride16)
精确top距离：(y_anchor−636.80810546875)/stride < 0
合法联合原子数：[0,0,0,0,0,0,0]
```

更强的反证：640画布全部8400个8/16/32网格锚点，最大y仅636，低于E-top **0.80810546875像素**。
所以不是候选池选窄，也不是U舍入容差：即使遍历全部锚点，原非负top距离支撑都不能实现该精确坐标。
不丢对象、不主动改E、不clamp伪造目标、不加overflow头，也不为开训继续放松合同。

`scripts/audit_bounded_retention_terminal.py`只重放当前终态：核对25输入/24源码合同快照、
该合法标签SHA及行号、来源物理区间、native候选、原子数和全网格反证；无图像/模型/隐藏标签读取。
结果`terminal_audit.json`及`--audit`字节一致重放通过。
终态result SHA：`ae24837415b55cb2465b42a0965c0b430ce270e1c368b9a3000a1b6c97a8eae7`。

已执行的最后零训练命令（不是新的训练指令，不重复使用现有输出名）：

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' -u scripts/check_bounded_retention_native_event.py --name native_event_integration_01 --scan-all-batches
```

可重复的只读终态复核入口：

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' scripts/audit_bounded_retention_terminal.py --audit
```

## 5. 控制、测试与未通过的门

同界MAP是当前最强匹配控制的工程实现：使用完全相同事件，选当前概率最大的合法联合原子，
没有缺少保留规则、面积信息或几何信息。合成完整loss确认相同TAL/CIoU/归一化，DFL及梯度不同。
但来源/工程全门未过，故没有冻结最终学生控制或三臂配置，也没有据此宣称胜过点/模式控制。
因子分布耦合loss不等于学到了联合相关性；仅收缩/变尖的解释仍待真实实验，CABB/GFL的新颖性边界不变。

完整pytest终态：**1285 passed in 62.14s**，包括最终精确边反例回归。
组件测试包括来源伪造拒绝、实际导出算术、联合oracle、增强/RNG/身份对齐、负距/空集、
全known严格还原、同界control梯度区别及真实Trainer初始化集成；不拿测试数当训练收益。

结论为必要工程门FAIL；首epoch几何流未扫完，三个30轮配置/训练/548图val960评估均NOT_STARTED。
不存在AP、AP_small、同FP/K、召回、重复框或定位的三臂结果，历史指标/门槛不改。
本轮已保存独立实现与阻塞证据；若未来改变精确坐标语义或表示支撑，属于另一个需预先审查的科学合同。
