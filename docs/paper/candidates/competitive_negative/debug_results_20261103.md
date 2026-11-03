# 竞争负项首批错误定位：AMP目标下溢触发错误断言

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate（工程）
- Origin Date: 2026-11-03
- Verification Status: VERIFIED（同类错误的真实单批复现及缓存控制重放）；非性能验收
- 原失败记录：[主对照工程失败](main_comparison_failure_20261103.md)
- 执行前协议：[零优化步骤定位协议](debug_protocol_20261103.md)

## 定位结论

新候选把“前景锚点的BCE目标必须严格大于0”当作合法性条件，未考虑AMP的FP16下溢。
真实重建首批中，一个目标的原始TAL质量为`8.708838983295664e-09`（FP32），
在原P1的`target_scores.to(dtype)`转换后成为FP16的0。
新候选再对这个已经转换的目标调用`negative_multipliers`，于是触发第43行断言。

原目标为`m*e_9`；转换为全0后，`max`返回质量0且argmax默认类0，
所以同一个数值下溢还会制造表面的“教师top1与目标argmax不一致”。
真实教师top1和TAL分配类别均为9，不能把这个argmax现象误认成标签运输错误。

该对象top2/top1为`0.01424153707921505`，低于固定触发线.25；
本应不衰减它的竞争负项，但当前代码在判断触发条件前就做了过强断言。
这是实现对数值边界的误判，不能作为竞争负项方法有效／无效的性能证据。

## 实际检查结果

仅一个训练batch（8图）的CUDA AMP前向及损失计算，无训练循环、反向传播和参数更新。
初始化使用原COCO权重，原candidate参数仅改变输出目录／名称。
真实增强、对象ID运输、TAL分配与原损失链沿用已有代码。

| 检查 | 结果 |
|---|---:|
| 前景锚点 | 1988 |
| 其中伪标签前景 | 649 |
| 教师top1与TAL分配原类别不一致 | 0 |
| 原FP32 TAL质量为0的伪前景 | 0 |
| 原质量为正、仅转FP16后为0的伪前景 | 1 |
| 正质量目标类别错误／多非零类目标 | 0／0 |
| 原candidate报错 | 同一hard pseudo-positive TAL断言 |
| 相同预测／batch的原P1损失 | 有限，可正常计算 |
| 训练参数前后哈希 | 一致 |
| 监测的原源码／失败证据／已有数据缓存发生变化 | 0 |

异常在batch图像索引4、前景锚点列表索引79。捕获全部前景张量后做CPU控制重放：
固定logits、教师分数、来源标记，只替换传入检查的目标dtype。
semantic及strength两模式用BCE实际FP16目标都在图像索引4报同样错误；
用转换前FP32目标，两模式全部8图都不报错。
这项dtype干预支持数值下溢的工程原因，但不等于已完成修复或真实训练验收。

边界：历史失败没有保存首批张量，未建立与历史首批的逐位身份；这里是同配置真实单批
复现同类错误，不能断言历史恰好也是同一个锚点。
训练数据已被使用；没有独立性能确认、隐藏GT使用、阈值拟合或val调参。
setup扫描既有val标签缓存并执行框架AMP自检，但没有迭代val或进行其预测／指标评估。
前向中的BN缓冲只在诊断进程内变化，没有保存模型。

## 产物与复现

独立ignored目录：`outputs/p1_competitive_negative_debug_20261103_v1/`。

- `diagnosis.json`：各图统计、参数不变及监测文件哈希；
- `foreground_tensors.pt`：同批前景logits／原与转换后目标／教师／分配类别／来源标记；
- `controlled_replay.json`：单变量dtype重放及异常目标详情；
- 控制台：`outputs/p1_competitive_negative_debug_20261103_v1_console.log`。

前景张量SHA256：`8a1aa278e75756ac5e5315248235e0c60b33f6ab943d76cf57ba01700fb41bb6`。
定位入口新增于`scripts/diagnose_competitive_negative_failure.py`，不改任何原训练源码。
脚本硬性禁止`BaseTrainer.train`、`Tensor.backward`和`SGD.step`；只调用setup和单批前向。
需要独立重建时输出目录必须全新，且只能位于ignored outputs：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe scripts/diagnose_competitive_negative_failure.py --output outputs/p1_competitive_negative_debug_new
```

本次只有一次真实批次前向；之后新增CPU控制重放辅助函数，未再前向该batch。
测试新增真实下溢数值、来源类别错误和原始零质量的分离用例。
全套`python -m pytest`：**286 passed in 8.16s**；没有训练性能结论。
另对上一终态回执绑定的17份原证据及4份B2／P1权重重新哈希，21／21一致。

## 修复位置与范围

应在独立候选修订中，将**来源／类别合法性检查**与**BCE实际目标数值**分开：

1. 来源与类别从TAL返回的类别／对象ID及转换前目标验证，保留真错类／软目标的拒绝逻辑。
2. 原P1的BCE目标转换、分母、正项、分配和回归保持不变。
3. 对BCE目标已下溢为0的前景，明确保留原P1权重1、不执行新负项动作；
   strength控制也必须用分配类别保护原正类，不能凭全零目标将它当成普通负类。
4. 新版本补FP16边界与真实损失集成验证，再冻结新输出；不能在旧目录直接重跑。

单纯删除断言会掩盖真实绑定错误；只把检查输入改成FP32虽能通过本批，
仍需处理原始零质量前景及strength对正类的保护，不能据此宣布完整修复。
本次完成**定位与验证**，未替换训练实现、未重启训练／恢复／补评估。
已完成B2／原P1权重继续保留，后续执行方案须明确是否复用及新旧来源绑定。
