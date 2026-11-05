# V6：移除名义E的额外canvas支撑门

2026-11-05，先于新版本实现及真实集成。V5真实检查
`posterior_integration_01`在第21个batch停止，原receipt、源码快照和5780870代码
全部保留。该检查完成Trainer/权重/11310 train及548 val loader，未完成全流，
显式增强batch模型forward=0、optimizer=0、validation inference=0。

失败对象source181722/occurrence662，原left U经负scale翻转成为right U；
原生名义left E为-1/65536。V5的domain及normalizer额外要求所有E属于canvas，
与V4合同“只有U位于canvas、与固定E组成正面积框”不一致。独立见证修订检验得到
相容U约[249.36527259407916,640]，整个[300,301]认证inside。它不是空物理集合。

新版本只纠正该实现条件：

- E有限且保持原值；两边固定时仍须正宽高，不要求E落在canvas。
- U严格使用原canvas域。单U的积分区间为左U的[0,min(W,E_right)]或
  右U的[max(0,E_left),W]；这是U积分域的集合交，不把E写回边界。
- domain构造、point成员关系、cell分类和解析/Decimal积分必须一致。
- 旧DFL饱和、名义E的CIoU、来源见证、源区间、身份、过滤、面积耦合、K256、
  全局点控制、归一化和误差预算均不改变；不扩canvas、不加容差、不丢对象。

采用独立V6模块/入口，旧V4/V5生产文件不修改。先覆盖原失败见证、镜像、真实空域、
不同轴组合及非零logits的独立值/梯度；全known原P1严格保持。完成全库pytest后，
用新独立目录执行双阶段真实增强与同批AMP检查。新版本验证不是恢复旧失败运行，
不得更改旧receipt以显示通过。所有工程门通过才直接启动原P1/候选/强控制3×30。

贡献可定位为既有相容监督/分布定位在区域部分标注问题上的有依据适配；
不要求证明全新基础理论。已有方法来源、模拟保留侧信息和充分对照仍需如实说明。
