# 原教师置信度回归加权：全量30轮确认结果

2026-10-31。**完整训练和独立验收通过，预登记的扩大训练条件未通过。**
本次为验证兼容修正版，科学变量和两条继续线沿用训练前冻结的
`regression_confidence30_protocol.md`及`regression_confidence30_valfix_protocol.md`。
旧版首轮验证工程失败和13:28用户主动中断均单独保留，不参与本次比较。

## 训练与验收

源码提交为`12b68866bf849586189e6c9deb8b76ee6099e17b`。
北京时间13:36:03独立启动，15:41:20包装结束，真实exit_code=0，
status=completed；验收时原包装22020和训练22512均已停止。
CSV连续1–30轮，11310tile、相同10%标注面积split、固定教师/伪标签、
学生seed0、YOLO11s、原COCO独立初始化、640、batch8、SGD、close_mosaic10。
参考复用已验收的完整P1独立30轮学生，不重训参考，不续接中断权重。

唯一科学改动为原教师置信度c同时乘已接受伪框的box/DFL正anchor权重，
人工框为1；保留原未加权TAL分母、分类软权重、ignore、分配、框坐标及数量。
验证兼容修复只使eval使用原P1损失，训练缺元数据仍拒绝。
没有叠加NWD、分配规则或结构修改，也未使用隐藏GT/val拟合训练权重。

```powershell
python scripts/audit_p1_regression_confidence.py --validation-fix
```

15:48独立验收通过：23809项冻结输入/源码/配置/协议哈希、实际args、
连续轮次及权重/产物哈希一致，两组各548张完整原图的保存预测匹配重放一致。
验收记录`outputs/p1_regression_confidence_valfix_pilot30_s0/independent_audit.json`
中accepted=true、worth_larger_training=false。证据验收通过不等于性能继续线通过。

## 完整原图960结果

| 指标 | 完整P1 / % | 原c回归加权 / % | 候选−P1 / pp |
|---|---:|---:|---:|
| mAP50 | 39.4990 | 39.1856 | −0.3134 |
| mAP50-95 | 23.7883 | 23.6129 | −0.1755 |
| Small recall | 35.6925 | 36.6369 | +0.9444 |
| Medium recall | 77.5709 | 77.4359 | −0.1350 |
| Large recall | 89.4096 | 89.2221 | −0.1874 |

固定recall口径为conf0.25、匹配IoU0.50、NMS0.70、max_det500，
GT为small26577、medium11115、large1067。
匹配数为small9486→9737（+251）、medium8622→8607（−15）、
large954→952（−2）。这是固定操作点recall，不是官方AP_small/AR500；
训练CSV的640逐轮指标不与此表的完整原图960结果混用。

本轮小目标多检出是学生固定操作点的观测，训练输入中的伪框坐标和数量未变，
不能写成“新增了251个训练伪标签”或修复了教师漏框。
整体AP下降与固定操作点recall上升并不矛盾；仅凭这两项汇总数字，
不能判定原因就是误检、框定位或某一损失项，需要额外匹配/PR证据。
本次不新增推理或按结果调参。

## 类别与训练开销

| 类别 | 完整P1 AP50-95 / % | 原c回归加权 AP50-95 / % | 差值 / pp |
|---|---:|---:|---:|
| pedestrian | 21.6761 | 21.5271 | −0.1490 |
| people | 14.7405 | 14.5479 | −0.1926 |
| bicycle | 8.1051 | 7.6020 | −0.5031 |
| car | 55.3456 | 55.1074 | −0.2382 |
| van | 32.6931 | 32.0359 | −0.6572 |
| truck | 24.1367 | 24.0228 | −0.1139 |
| tricycle | 12.6744 | 13.2620 | +0.5877 |
| awning-tricycle | 8.4236 | 8.3384 | −0.0853 |
| bus | 39.7864 | 39.7532 | −0.0332 |
| motor | 20.3018 | 19.9322 | −0.3696 |

10类中1升9降，tricycle上升0.5877pp，van、bicycle分别下降0.6572、0.5031pp；
不能把单类改善或small recall改善写成整体检测提升。
学生CSV累计耗时为参考7286.61秒、候选7450.65秒，多164.04秒（约2.25%）。
包装从启动至结束约2小时5分16秒，包含完整评估和流程开销。
运行负载和先后时间未控制，耗时差不是独立的模块速度基准。

## 固定继续线与决定

| 预登记路线 | 本次观测 | 判定 |
|---|---|---|
| mAP50-95≥+0.30pp，且small recall不下降 | AP−0.1755pp，small+0.9444pp | AP未达到 |
| small recall≥+1.00pp，且mAP50-95≥−0.10pp | small+0.9444pp，AP−0.1755pp | 两项均未达到 |

第二条路线的small增益距+1.00pp差0.0556pp，AP损失超过允许幅度0.0755pp。
不因接近阈值放宽继续线。保留这项召回/精度取舍，当前不准备本变体的更长训练，
不按val调整指数、常数或尺度补偿追试；完整P1继续作为既有主对照。

本轮满足至少完整30epoch才判断候选性能的要求，但仅覆盖单split、固定教师、
学生seed0及当前规则，不能证明100轮必定无效或已经充分收敛。
结果为Ultralytics开发评测，非盲测、官方VisDrone AP或统计显著性结论；
原Gate3和既有B2/P1的正向阶段证据保持。

缓存中c加权的定位质量均值改善未转化为本轮扩大训练条件。
本规则同时改变总伪回归监督强度、框排序及尺度/类别组成；
常数衰减对照未做，不能证明某一解释为因果机制。
此前small框级sum(c)份额减少7.4644pp只是代理统计，
既不能当作实际损失/梯度降幅，也不能据它解释本次recall变化。
原c不等于独立定位不确定性，不称RUPL/UTv2复现或新颖性证明。

## 证据与收尾边界

- 输出：`outputs/p1_regression_confidence_valfix_pilot30_s0/`中的
  preflight、launch_receipt、status、comparison、artifact_sha256、
  independent_audit及stdout/stderr。
- 训练：`runs/partial10_p1_regression_confidence_valfix_pilot30_s0/`，
  含30轮CSV、args、experiment、best/last。
- val：`runs/val/partial10_p1_regression_confidence_valfix_pilot30_s0_val960/metrics.json`。
- recall与保存预测：`runs/size_recall/p1_regression_confidence_valfix_pilot30_s0/`。
- best SHA256：`3899b907206b00440ec5a2aa51227afe0f0b4d1454d47484cf362b1c03c9a563`。
- last SHA256：`35386d2c882964efc4643ad1791d6b7f3571af7fb38b8f7b6808ea0bf929e18b`。

本页新增结果由上述独立验收入口核对；既有`build_paper_tables.py --check`
仍覆盖原主表/补充表的137来源及8组recall，不包含本页新增结果。
收尾检查131项pytest通过；本页5项总体/尺寸指标和10类AP共15行均与原始
JSON逐项核对，best/last哈希及23809冻结来源再次匹配。
检查记录为输出目录中的`documentation_check.json`，本次只修改Markdown文稿。
只做本次训练验收、记录、中文提交push及临时监控清理，不启动新研究或训练。
用户暂停的持续目标保持paused；若要继续后续研究，需要用户点击继续。
