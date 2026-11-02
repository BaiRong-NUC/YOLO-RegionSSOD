# 正式100轮权重的 AP_small、PR 与等精度召回补评估

登记日期：2026-11-02。用户授权补评估和文献调研，明确不启动新训练。
本协议在新增低置信度推理和读取其结果之前固定；这是既有验证集上的回顾性分析，
不是新的盲测、训练预注册或已确认的新方法。

## 1. 冻结输入与推理

- 仅使用已独立验收的 B2/P1、学生 seed0/1、各100轮的 best.pt。
  split0、原固定教师与伪标签保持不变；不重新选检查点。
- 548张完整 VisDrone val 原图及现有 YOLO 标签（38759 GT）。先记录图像、
  标签、权重、原验收/指标、协议和新增评估源码 SHA256；产物仅写独立忽略目录。
- 每臂只新增一次完整验证推理：Ultralytics 8.4.172，960、batch8、workers0、
  FP32、conf=0.001、NMS IoU=0.70、max_det=500、无TTA。
  使用与原完整验证相同的 DetectionValidator 路径，并保存未做小数截断的原图预测。
  不覆盖原验证/recall，不用新曲线回选训练参数、检查点或部署阈值。
- 原 conf0.25 缓存不足以计算完整PR/AP；新曲线覆盖保存的0.001至1置信度范围，
  仍受候选下限、NMS和每图500框截断影响，不宣称无限候选下的完整召回。
- 新推理的 Ultralytics mAP50/50-95 与原记录逐臂比较，绝对差超过1e-6时须明确
  排查并披露，不能无声替换旧表。旧固定点recall来自独立predict路径，另报兼容性。

## 2. COCO-style AP_small

使用 pycocotools 2.0.11 的标准 bbox 匹配、面积忽略、类别宏平均和101点插值。
GT由当前YOLO标签转换为原图像素框，area=w*h，iscrowd=0；不恢复已在YOLO转换时
排除的VisDrone区域/类别。因此这是本项目标签上的COCO-style补评估，**不是VisDrone
官方AP**，也不与旧Ultralytics mAP无说明混用。

- IoU=0.50:0.05:0.95，10类别，maxDets=[1,100,500]，主结果取500。
- 保留COCO默认面积范围及其边界处理：small=[0,32²]，medium=[32²,96²]，
  large=[96²,100000²]。标准工具对边界值的包含关系与旧诊断的small<32²不同，
  分别记录各组GT数；不私自修改工具以对齐旧计数。
- 报告AP_all、AP_small/medium/large、AP50_small、AP75_small及逐类别AP_small。
  不通过只保留small GT后把大目标预测全记FP的错误方式计算AP_small。
- 导出all/small、IoU0.50/0.75的类别宏平均插值PR，以及各类别曲线。

## 3. 实际置信阈值与 precision 约束

另用原诊断同类匹配规则：同类别、置信度降序、所有GT尺寸联合一对一贪心匹配，
再按原图GT面积统计small<32²等三组。每个预测仅匹配一次，重复框为未匹配预测。
全验证集按分数联合排序，同分预测整组纳入，不能以同分框内部排序构造不可实现阈值。

主比较为**全尺寸micro precision约束下的小目标recall**，不是COCO按类别宏平均PR，
也不是只对预测面积筛small的precision。分别在IoU0.50、0.75检查：

- 固定precision约束：0.50、0.60、0.70、0.80、0.90，全部报告，不只选正向档位。
- R_small@P≥p = 所有实际阈值中满足precision≥p的最大small recall。
  报告实际precision、阈值、TP、未匹配预测与small匹配数；达不到时写不可达。
  两臂实际precision可能略高于约束，不伪称完全相等。
- 补充每seed B2在conf0.25的precision作为锚点，比较P1能达到的small recall；
  不将这个验证集上回顾性选择的阈值当作独立部署策略。
- 输出全部实际置信阈值下的micro PR、small recall—precision曲线及0.25固定点。
  类别宏平均AP与micro操作点分别报告，避免常见类掩盖稀有类退化。

## 4. 统计、解释与验收

- 每seed内先计算P1−B2，再汇总两seed差值的均值与样本SD；不合并两次验证GT
  作为1096张独立图，不做两seed显著性宣称。
- AP50_small/AP75_small用于区分宽松/严格定位条件，不能单凭差值证明定位因果。
- 完整报告负向类、所有precision档位、成本及seed0 B2/seed1 P1的恢复事实。
  保留Gate3/4原判定；新增COCO-style AP_small不是预登记官方AP_small的自动替代。
- 有意义单元测试覆盖重复框/错类、所有尺寸联合匹配、同分阈值、不可达precision、
  COCO面积忽略与边界；源码修改后运行python -m pytest。
- 分析可由缓存完全重放；审计验证输入与产物哈希及重算数字。冻结模型再次训练的
  可复现性不在本轮测试范围内。新增推理不自动重试；失败保留回执后处理原因。

## 5. 命令与产物

在仓库根目录、已有yolo环境：

```bash
python scripts/evaluate_paper1_small_pr.py --predict
python scripts/evaluate_paper1_small_pr.py --analyze
python scripts/evaluate_paper1_small_pr.py --audit
```

`--predict`只推理四个冻结权重，拒绝覆盖已有预测；`--analyze`只读取缓存。
结果、JSON/CSV、PNG/PDF/SVG和运行回执位于
`outputs/paper1_small_pr_20261102/`。实际结果另写结果页，不修改本协议条件。
运行环境另需已有 pycocotools==2.0.11、matplotlib==3.10.8。
