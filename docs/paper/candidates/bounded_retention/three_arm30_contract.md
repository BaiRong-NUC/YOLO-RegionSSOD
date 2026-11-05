# V6连续集合监督：三臂各30轮冻结合同

冻结于2026-11-05，学生训练前，且在[联合零优化工程证据](continuous_amp_results.md)
通过之后。授权为顺序新跑原P1→候选→强控制，各完整30轮；不询问再次开训、不扩100轮、
不恢复旧实验、不并行实验。没有查看新验证预测或按性能选择方法。

## 方法及可作出的贡献

使用`ContinuousNativeETrainer/ContinuousNativeELoss`（V6）。保留原名义已知边E、
全部合法原生对象及原DFL两bin插值权重；原生距离编码clip与物理连续坐标分开。
全known严格恢复原P1的loss、权重与归一化。原source/occurrence身份、裁剪/过滤/缩放
运输、支持变换、空集、负距和有限支撑规则均继承已通过的合同，不新增标签降级。

候选为固定E下`(sum_E DFL + log Z_D - log Z_C)/4`；强控制为同一C上
`(sum_E DFL + inf_C continuous_point_DFL)/4`。两臂获得相同公开保留条件、面积界、
合法几何及来源外集；共享同一全局连续点见证用于TAL/CIoU，不能故意弱化控制。
既定全局点gap含算术界≤1e-5；1U解析，multi-U使用完整C上的精确目标后验拒绝采样，
K=256、最多20000cells/16384proposals/256prefixbits，私有RNG。不再调系数、容量或K。
显示loss采用既定NLL区间中点，不冒称精确NLL或精确Hessian；随机梯度有方差。

继承continuous_density_contract、continuous_numeric_addendum、continuous_sampling_contract、
continuous_observation_semantics、native_e_revision_contract及amp_audit_revision_contract。
V1–V5/旧V6失败及源码保持封存。本比较还含归一化差异，不称作纯sum/min消融。
CABB已有裁剪相容集、GFL已有分布定位；可以报告经验证的方法适配贡献，不要求新基础理论，
也不宣称首创以上思想。因子化head上的集合耦合不等于学得任意联合相关性。
保留事件来自模拟生成时GT衍生侧信息，不推广为自然人工标注普遍拥有的保证。

## 固定执行

- 参考臂使用原`P1Trainer`，不使用新loss的reference包装。
- 初始权重：`D:/Python/YOLO/yolo11s.pt`，SHA256
  `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`。
- 基础配置完全沿用`configs/candidates/positive_quality_transport_v2/reference30.yaml`，
  仅去掉其旧候选mode，另设本次project/name/mode。YOLO11s、640、batch8、SGD、seed0、
  workers0、deterministic、lr0/lrf=.01、momentum=.937、weight_decay=.0005、
  AMP、close_mosaic10、patience0；P1 soft_weights/uncertain_ignore均true。
- 三个新配置位于`configs/candidates/bounded_retention_continuous/`。
  同一area10 split/teacher/materialized P1标签，11310 train=6471人工+4839伪标签。
  不读取原全图/未标注tile隐藏GT、隐藏计数或旧逐框GT质量缓存。
- 训练输出仅`outputs/bounded_retention_continuous_three30_s0/`，目录必须全新。
  输入绑定为`outputs/bounded_retention_continuous_freeze_20261105/freeze.json`；
  它固定旧STOP、新AMP通过回执、完整pytest、预览、源码、合同、配置、初始权重和合法输入。
- 唯一启动入口：`python scripts/run_bounded_retention_continuous.py --train --spec
  outputs/bounded_retention_continuous_freeze_20261105/freeze.json`。
  `--check --spec ...`只核验；`--audit`仅终态重放。保留命令、PID、退出码和产物seal。

三臂一致禁止OOM自动减batch、NaN回载checkpoint、实验自动重试/恢复及自动杀worker。
每batch查未缩放loss，每epoch及保存前查实际model/EMA浮点状态；非法来源/运输/空集/
数值容量或非有限loss/状态直接失败，停止后续派发，保留现场，不以性能提前停训。
**原生GradScaler正常溢出跳步/降scale允许**，不把诊断1024写入训练超参数。
包装进程异常时保留仍活跃worker并报告，不能把其伪装成已结束。训练期间设临时进程
监控，正常进展静默；完成/失败/需处理才报告。终态处理后移除临时监控。

## 统一评估及预定解释

完整三臂30轮前不作性能判断。沿原基线用每臂best.pt（原生640验证选择）做最终统一
**548图full-image val960**。原生epoch曲线不作为最终比较；三臂均完整后各一次推理：
conf=.001、NMS IoU=.7、maxDet500、batch8、device0、workers0、rect=true、augment=false。

比较固定为P1→候选、P1→控制、控制→候选，全548图、全10类，无有利子集筛选。
主指标为同一预测缓存的COCO插值AP_all/AP_small（IoU.50:.95、maxDet500）；另报AP50、
AP75、全部尺寸/类别及原生Ultralytics指标。二者分开，不称作VisDrone官方AP，不拿COCO
差值重判历史Ultralytics Gate3；原P1对B2的旧门槛和失败结论保持。

同时报告conf.25下IoU.50/.75的precision/recall、small TP/recall、FP与重复框；
每对以前者conf.25为预算，做精确同预测数K（稳定排序并重新匹配）和不超同FP预算的
整score-group前缀（报告实际FP及slack）。固定保存域受conf floor/maxDet截断，单独披露
无截断证明是否可得；保存预测不足K则该项不可得，不补推理或伪造同K。
定位报告共同small GT的IoU、中心/边误差，并同时列新增、丢失、未匹配数；共同子集仅作
诊断，不能取代完整评估。阈值由验证标签求得仅用于描述预算比较，不回流训练或部署选择。

不预设“所有指标必须上涨”的额外训练门。完成后按未四舍五入结果分别判断：

1. 工程成功：三臂连续1–30轮、exit0、冻结输入/args/权重/诊断和完整评估审计通过。
2. 性能：分别报告候选对P1/强控制的AP_all与AP_small差值。两项均正才称相应比较的
   双主指标改善；一升一降写为权衡。仅赢P1而未赢强控制，不能声称分布监督有独立优势。
   单一seed/同split/teacher仅为配对30轮观察，无显著性或跨划分泛化证明。
3. 机制：先核对实际有效U及多U前景权重/暴露、采样/数值状态；再结合同FP、同K、重复框、
   定位和按U维度的entropy/decoded-area/shared-point-area诊断，检查是否仅减少预测、缩框
   或变尖。即使性能改善，也不自动宣称面积耦合因果机制或联合相关性已证实。

结果不理想也完整报告；本授权到三臂30轮及终态分析为止，不据结果追改合同、补有利子集、
追加其他控制或自行扩训。工程故障则明确失败条件，不为完成训练绕过。
