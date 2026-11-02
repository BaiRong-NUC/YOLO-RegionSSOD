# 高精度小目标保真的尺度条件操作点：零训练跨拟合诊断

2026-11-02。前置高精度诊断显示，IoU0.50、90%全尺寸precision下，P1相对B2丢失的
small GT绝大多数在完整缓存中已有同类合格候选，却低于P1选出的全局阈值；共同GT
几何未显示P1恶化。由此把H2训练假设收紧，临时检验更直接的H1必要条件：
尺度条件操作点能否在不低于高精度的约束下保留小目标覆盖。

本协议在读取新增跨拟合结果前固定。仍是反复使用val上的零训练回顾性诊断，不是新盲测、
论文正式主结果或部署政策。即使通过，也只能支持外部/新split确认，不能追认H2或Gate3/4。

## 1. 固定输入和预测分组

- 仅读取`outputs/paper1_small_pr_20261102/bound_v2/`四个冻结B2/P1学生seed0/1预测，
  以及`outputs/paper1_high_precision_diagnostic_20261102/`已验收的完整缓存匹配数组。
  不新推理、不训练、不改NMS/max_det/权重/split/教师。
- 匹配规则与前一诊断相同：类别一致、分数降序、全尺寸GT联合一对一，IoU0.50。
  small GT面积<32²。预测按原图像素框面积分组：pred-small<32²，其余进入non-small；
  **预测尺度不是GT尺度**，所有预测均与全部GT尺寸竞争。
- 每个预测只继承完整缓存匹配的GT，不在选阈值时重做NMS或重新定义TP。

## 2. 操作点策略

候选阈值固定为201个对数间隔值，从0.001到1.0（端点包含，依据指数为
`exp(log(1000)*i/200)/1000`）。不根据结果增加、删减或网格细化。

- **Global policy**：所有预测使用同一个阈值t。选择校准半集上small recall最大、
  且全尺寸micro precision≥目标的实际网格点；同small TP时选预测数更少者，再选更高阈值。
- **Scale-conditioned policy**：pred-small与non-small各用阈值ts、tn。枚举201×201，
  在校准半集上最大化small recall且全尺寸precision≥目标；同small TP时选总预测数更少，
  再选更高(tn,ts)。small阈值降低释放的候选可用提高non-small阈值满足整体精度。
- 主目标precision=0.90；补充预固定0.95也只全报，不与0.90互换主结论。
- 两臂各自校准自己的B2/P1阈值，不共享另一臂结果；类别不分阈值，避免高维过拟合。

## 3. 图像级跨拟合

固定split seeds：`20261102, 20261103, 20261104, 20261105, 20261106`。每个seed把
548张原图随机排列并等分274/274；两折交换校准/评估。这样每个split seed的全部548图
各作为评估图一次，阈值则总由不相交的另一半选择。两学生seed和五个图像split seed分别报告。

对每个模型/目标/fold记录实际校准阈值、同分阈值、校准与评估的实际precision、预测数、
small/medium/large TP及small recall。两折评估图合并，但阈值仍按折各自固定，再计算
split seed级结果；不把548图重新当完全同分布阈值样本。

## 4. 主比较与继续线

1. 同一学生seed内，Scale−Global的small recall变化，B2和P1分别报告。
2. P1−B2在Global与Scale下分别比较；核心量
   `Δgap = (P1−B2)_Scale − (P1−B2)_Global`，单位pp。
3. 全报评估半集实际precision；校准目标不等于评估保证。若评估precision低于0.88，
   标为精度约束未保持，不能称“相同高精度下”的提升。

预先固定“值得进一步确认”的全部条件（以目标0.90为主）：

- 两个学生seed的Δgap均值均至少+0.20pp；
- 每个学生seed在至少4/5图像split seeds的Δgap为正；
- 校准半集满足目标，评估半集平均实际precision≥0.88；
- P1的Scale−Global small收益大于B2的对应收益，两个学生seed同向。

未全部满足则停止该尺度条件阈值扩展，不通过改类别、改网格、调到0.85或另选split追试。
即使全部满足，只交付“支持在独立校准/评估划分上进一步验证”的结论；仍需外部或新split
确认后才能进入论文主张。阈值由GT校准，部署中须有标注校准集，不能使用隐藏GT。

## 5. 验收与输出

- 单元测试覆盖阈值网格、目标不可达、同分/并列、预测尺度与GT尺度分离、校准/评估不相交、
  两折覆盖548图、同一匹配序列不改变TP。
- `--run`输出`outputs/paper1_scale_operating_point_20261102/`；最终结果存在时拒绝覆盖。
  `--audit`只读输入/输出并精确重放，不推理/训练。源码与全部输入哈希、Python随机生成器
  实现/版本、退出码和失败回执保留。
- 最终文档必须保留所有split/fold、目标、负向结果、实际精度及低精度警告。ARS 11类统计
  谬误检查单独记录；不宣称显著性、跨数据集泛化或已实现可部署校准。

```bash
python scripts/evaluate_scaled_operating_point.py --run
python scripts/evaluate_scaled_operating_point.py --audit
```
