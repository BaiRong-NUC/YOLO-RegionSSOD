# 小框混合NWD全量30轮确认结果

2026-10-31。**完整训练与独立验收通过，预登记的扩大训练条件未通过。**
执行前协议为`small_nwd30_protocol.md`，未按结果修改C、mix、面积边界或继续线。
本次由用户终端从COCO初始化重新训练，不拼接此前手动中断的13轮。

## 训练与验收范围

源码提交为`23c2656a3d162109d5a9a569a8e4b2e86c6e1501`。
北京时间09:02:42启动，11:18:53包装退出，exit_code=0，全部阶段completed。
包装16400/训练44996在验收时均已停止；CSV连续1–30轮，11310个训练tile。
相同10%面积split、固定教师/伪标签、seed0、YOLO11s、640、batch8、SGD、
close_mosaic10；参考为已验收的完整P1同预算学生。

仅对增强后640像素空间面积<32²的正anchor采用0.5CIoU+0.5NWD框回归，
C=20.228043063665424，其他框、DFL、TAL、分类软权重及ignore保持原设置。
冻结输入/源码/配置/协议共23793项哈希、实际参数、权重和评估产物一致；
两组各548图保存预测匹配重放通过。

```bash
python scripts/audit_p1_small_nwd.py
```

独立验收记录为`outputs/p1_small_nwd_pilot30_s0/independent_audit.json`，
其中accepted=true、worth_larger_training=false。工程及证据验收成功，
不等同于方法产生正向增益。

## 完整原图960评估

| 指标 | 完整P1 / % | 小框混合NWD / % | NWD−P1 / pp |
|---|---:|---:|---:|
| mAP50 | 39.4990 | 39.1453 | −0.3537 |
| mAP50-95 | 23.7883 | 23.5761 | −0.2123 |
| Small recall | 35.6925 | 35.0228 | −0.6698 |
| Medium recall | 77.5709 | 77.4809 | −0.0900 |
| Large recall | 89.4096 | 89.2221 | −0.1874 |

固定操作点为conf0.25、匹配IoU0.50、NMS0.70、max_det500，
GT分别为small26577、medium11115、large1067。
匹配数为small9486→9308（−178）、medium8622→8612（−10）、
large954→952（−2）。这是固定操作点recall，不是官方AP_small/AR500。
训练CSV的640指标不与本表的完整原图960指标混用。

## 类别与耗时取舍

| 类别 | 完整P1 AP50-95 / % | NWD AP50-95 / % | 差值 / pp |
|---|---:|---:|---:|
| pedestrian | 21.6761 | 21.2127 | −0.4634 |
| people | 14.7405 | 14.3412 | −0.3993 |
| bicycle | 8.1051 | 7.4263 | −0.6789 |
| car | 55.3456 | 55.0147 | −0.3309 |
| van | 32.6931 | 32.0251 | −0.6680 |
| truck | 24.1367 | 23.1635 | −0.9732 |
| tricycle | 12.6744 | 13.3005 | +0.6262 |
| awning-tricycle | 8.4236 | 8.8031 | +0.3794 |
| bus | 39.7864 | 40.1486 | +0.3623 |
| motor | 20.3018 | 20.3249 | +0.0231 |

10类中4升6降，truck退化最大；不能把局部类别收益写成整体改进。
CSV累计学生训练耗时为P1 7286.61秒、NWD 8092.09秒，
本次NWD多805.48秒，约11.05%；运行条件和负载未单独控制，不作为独立速度基准。

## 决策与限制

预登记条件为mAP50-95至少+0.30pp且small recall不下降，
或small recall至少+1.00pp且mAP50-95不低于−0.10pp。
本次两项核心指标均下降，两条路线均未通过，保持原P1作为既有主对照，
不准备本变体的100轮扩训，也不按验证结果调整C/mix/面积边界追试。

五轮子集探索曾出现small recall+1.5464pp；本轮全量30轮没有保留该方向。
两次同时存在数据规模和训练预算差异，不能把方向变化完全归因于训练轮数，
也不能将旧子集P1作为本轮全量参考。此前13轮中断产物不参与比较。

当前证据仅覆盖一个split、固定教师、学生seed0及本NWD变体，
不能证明所有NWD用法或100轮训练均无效，也不证明模型已完全收敛。
本轮已满足至少30轮后判断的预算要求；在当前成本约束下没有扩大训练的依据。
结果属于Ultralytics开发评测，不是独立盲测或统计显著性结论，原Gate3判定不变。

## 证据位置

- 输出：`outputs/p1_small_nwd_pilot30_s0/`中的preflight、launch_receipt、
  status、comparison、independent_audit及stdout/stderr。
- 训练：`runs/partial10_p1_small_nwd_pilot30_s0/`，含参数、元数据、
  30轮CSV、best/last、`SHA256SUMS.txt`。
- val：`runs/val/partial10_p1_small_nwd_pilot30_s0_val960/metrics.json`。
- recall及保存预测：`runs/size_recall/p1_small_nwd_pilot30_s0/`。
- best SHA256：`3a0357ae000a922441f2c472b8a07c03f18552dad77134a6809d24923dfc81e6`。
- last SHA256：`e50dcbd46c0aa7ece4345bfb75721433aba2df9300f2496da6a994e8cedf58c5`。

本页单独报告全量确认，不覆盖五轮探索记录，也不改写原B1/B2/P1主表。
本页由NWD独立验收入口核对来源；既有`build_paper_tables.py --check`
仍核对原主表/补充表的137来源和8组recall，不包含本页新增结果。
