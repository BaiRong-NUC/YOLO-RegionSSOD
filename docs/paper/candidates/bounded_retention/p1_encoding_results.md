# 原P1编码语义复核：额外E合同已修正，字面U原子方向停止

## Material Passport

- academic-research-suite / experiment-agent run，2026-11-05，起点f87cf6f，main干净。
- 用户授权审查额外合同、合理修订后独立验证，通过则直接顺序三臂各30轮；没有新的许可缺口。
- 原P1、V1/V2源码、合同、停止回执及历史失败保持原样；新实现V3与新ignored目录隔离。
- 状态：**STOP_LITERAL_U_EVENT_DIRECTION / STUDENT_NOT_STARTED**。不把此结论扩大成所有保留事件或连续定位无效。

## 1. 对上一轮结论的纠正

**“E距离必须属于[0,14.99]”是我们额外施加的合同，不是原P1的要求。**
此前把DFL编码可表示性当成保持精确坐标的必要条件，解释过强。
旧V2停止仍正确反映旧合同的失败，但不足以单独停止整个研究机制。

安装Ultralytics8.4.172的实际调用链为：`P1DetectionLoss → BboxLoss → bbox2dist/DFLoss`。
`BboxLoss`用原连续目标框计算CIoU；DFL另行使用`q=clip(raw_distance,0,14.99)`编码。
TAL本来就扩大tiny候选池，所以可能产生负的E距离。这不要求移动原框或把E变成U。
源码SHA已随新执行绑定，不能由在线最新版或合成代数替代本机原P1事实。

因此先冻结[p1_encoding_contract.md](p1_encoding_contract.md)，再新增：

- `bounded_retention_objective_v3.py`：仅移除E可表示性否决；E调用原生DFLoss的克隆输入，raw距离/几何/来源不改。
- `bounded_retention_training_v3.py`：独立模型与Trainer路由，保留已验证lexical继承和eval原P1损失。
- `check_bounded_retention_p1_encoding.py`：新真实零步检查，记录forward尝试数及本机loss/TAL源码；
  几何覆盖通过后才以一次forward比较三种损失和梯度。本次未走到该阶段。
- `test_bounded_retention_p1_encoding.py`：独立原生编码、完整loss和既有真实witness回归。

修订只声称“固定供给E下的U相容质量，加原P1 E编码损失”。不声称完整四边预测框满足物理约束，
也不声称饱和编码恢复不可表示的E；原连续CIoU残差仍在。

## 2. 独立验证实际通过了什么

独立数值审查31项测试全过、零skip，包含CPU/CUDA AMP负E/上溢完整loss及梯度，
与原P1逐位一致；确认真实正锚点确实激活了编码饱和，避免只测没有负距的空泛退化情形。
部分未知E项与原生DFL的值/梯度一致；源对象、raw目标、名义E及U mask不被in-place编码改写。

旧阻塞object142895/occurrence51重放后，E-top仍为636.80810546875，U仍仅right，
7个原生候选合法原子数变为`[9,9,9,9,4,4,4]`。没有移动坐标或删对象，说明E合同修订有效。
独立反方认可这一修订的语义边界；旧STOP未被回写PASS。

同界MAP与mass仍可区分，两者共享E编码/TAL/CIoU/支撑/分母/增益。
单原子或并列最大值时局部梯度可以相同，不能据个别退化例停止整个方法；反之只看loss差也不够。
代码的`lp.max()`并列时分摊次梯度，未偷偷换成argmax单点反传。
MAP是同离散事件的强匹配控制，但不能先验称为所有点方法中的性能最强者。

完整`python -m pytest`：1316项通过。专项证据在
`outputs/bounded_retention_p1_encoding_review_20261105/independent_review_01.json`。
这些均不是学生性能结果。

## 3. 真实扩大检查的新阻塞

执行命令：

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' -u scripts/check_bounded_retention_p1_encoding.py --name p1_encoding_integration_01 --scan-all-batches
```

输出：`outputs/bounded_retention_p1_encoding_review_20261105/p1_encoding_integration_01/`。
PID49196，UTC05:16:50.048863至05:20:51.272837，241.2241秒，自然STOP。
入口返回2，宿主shell回报1；无学生进程，没有终止或自动重试用户进程。
真实train11310/val548 loader构造成功；前12个batch全部通过，第13个第三条U实例停止。
合计139个U实例通过，第140个失败；这不是有效正训练数或总体发生率。
显式增强batch forward尝试0、完成0、optimizer0、val inference0；构造器stride probe另行披露。

```text
object164630 / occurrence399，batch_index12 / row16 / image3
合法人工标签：9999984_00000_d_0000037__lab_488_113_383.txt，第1行
3 0.3067885117 0.9569190608 0.07571801671 0.08616187839
名义框：[429.80218505859375,594.7666015625,467.39654541015625,637.54638671875]
U仅bottom；32个native候选全部没有合法字面原子
所有E距离落在[0.0686817169,5.1541748047]，没有E编码支撑问题
```

物理边界经过来源与运输外包络后的区间为
`[637.5463731003373,637.5463731288917]`；输出画布上界640。
而所有检测层的字面bottom原子为`anchor_y + stride*k`：

| stride | 画布内最大原子 | 下一格 |
|---|---:|---:|
| 8 | 636 | 644 |
| 16 | 632 | 648 |
| 32 | 624 | 656 |

所以相容连续区间与全部字面原子均不相交，扩大TAL也无效。
**这不代表连续物理集合为空，或网络不能输出相容均值。**
独立joint oracle验证bottom638/639/640均OUTER_COMPATIBLE，保留比约.989507/.967137/.945756，超过.35。
这些是保守outer集见证，不是历史隐藏完整框重建。
anchor636、stride8的bin0/1按概率(.625,.375)即可解码出bottom639；但两个字面原子636/644都不在相容区间。

因此，新失败定位在当前**字面物理原子的集合质量定义**，不是E负距、面积条件无解、原P1定位无解或舍入epsilon不足。

## 4. 为什么没有把“编码union非空”直接当作救活

继续审查了一个更换事件定义的提议：`S_code={k: exists u in C, product q_k(u)>0}`，
其中q是原生DFL的clip加相邻bin插值kernel。只要C非空，编码支撑就非空；这是数学上可定义的
新部分标签事件，但**不是原物理概率质量的无损修复**。没有u上的概率测度，也不存在唯一的pushforward概率分布。

支撑投影丢掉插值权重及亚bin区间位置。例如C1=[.2,.4]、C2=[.6,.8]都映到{0,1}。
取p0=.882、p1=.098，其余bin总概率.02（下表U项未除4）：

| 目标 | C1 | C2 |
|---|---:|---:|
| 编码union mass | .020203 | .020203 |
| 编码MAP | .125563 | .125563 |
| 最优连续相容point-DFL | .565008 | 1.443898 |

两种集合质量目标并非彼此等价，而是都丢掉了强物理点控制能使用的信息。
退化到C={.3}仍只留下{0,1}，不能恢复原DFL的(.7,.3)权重。
某些跨边耦合也会在投影中消失：存在同一联合见证是必要条件，但不保证保留了有效面积耦合信号。

编码MAP是sum/max的严格匹配消融，却不能由此冒充最强连续物理点对照。
若研究更换为编码事件，主强控制需审查`inf_{u in C} DFL(p,u)`，返回同一物理集合内可核验见证与最优性/误差界；
TAL/CIoU不能把bin整数当物理坐标，也不能让两臂使用不同见证规则。
连续密度/去量化则还需新的概率测度假设与积分规则，不能从原生DFL自动推出。

本轮明确**不采用这些新事件来绕过既有空集**。它们没有获得原候选的训练资格，
也未完成新强控制认证；不能仅因“能使support非空”就放行三臂。
结论是停止当前字面离散质量候选，不是证明任何未来编码/连续方案都不合理。

## 5. 终态与重放

`audit_bounded_retention_p1_encoding.py`核对31输入、28快照、合法来源行、32候选与空原子，
重放旧E修订成功、连续outer见证、全网格间隙及编码union信息损失；`--audit`字节重放通过。
新result SHA：`01049b7b328b719c7f3c191ae2c4f99b3cf191688b07e36fe18fee15a29fee0e`。

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' scripts/audit_bounded_retention_p1_encoding.py --audit
```

三臂均0epoch；全epoch几何未扫完、V3真实AMP比较未执行、最终学生配置/强控制未冻结，
不存在AP/AP_small/同FP/K/召回/重复框/定位结果。不从短程工程检查判断性能，不追改历史门槛。
未读取隐藏训练GT或旧逐框质量缓存；公开选择事件的模拟GT衍生侧信息及SOURCE_OUTER局限不变。
原P1保留为主线，CABB/GFL新颖性未认证。更新主计划第3.83/第10节及入口，中文commit并push。
