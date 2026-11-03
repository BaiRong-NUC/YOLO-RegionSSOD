# V2主对照终态独立验证记录（2026-11-03）

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate
- Origin Date: 2026-11-03 (Asia/Shanghai)
- Verification Status: VERIFIED（冻结证据及缓存重放）；ANALYZED（单seed性能描述）
- Version Label: competitive_negative_v2_terminal_audit_v1

## 执行与结果

在现有yolo解释器、仓库根目录运行用户指定的只读终态命令：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe scripts/run_p1_main_comparison_repair.py --audit
```

stdout／stderr共同保存到新唯一文件
`outputs/p1_competitive_negative_main30_v2_s0/terminal_audit_20261103_171004_802028.log`，
进程exit0、`accepted=true`；训练／评估审计回执与自动结果精确一致。
新增独立`terminal_receipt_20261103_171656_958337.json`记录该日志哈希、原artifact清单哈希及核对结果；
没有覆盖训练日志、失败证据、原结果或原artifact清单。

| 核对项 | 结果及范围 |
|---|---|
| 完整训练 | B2、P1、V2各连续1–30轮，三worker均exit0；只新训练V2 |
| 冻结来源 | 原39828输入／失败快照及新39864绑定核对通过；preflight SHA为c6a6b3b2a345dc0ef5db649b1132b9f3e70d72582e3859f4308f0b7859d86f3d |
| 实际args | 30轮／640／batch8／SGD／AMP／seed0，P1与V2只差name／save_dir，resume=false |
| 机制工程快照 | 30份，正目标／分配正类权重变化均0、目标不变，负梯度移除量有限且非负 |
| 权重绑定 | 统一评估恰为预期三个best，B2／P1与复用冻结SHA一致 |
| 图像与预测 | 每组548张完整val960；预测／权重／manifest绑定，GT共38759（small26577） |
| 数值重放 | 从缓存重算COCO-style AP／AP_small／逐类AP_small、双IoU完整micro PR及固定／precision约束点；JSON精确一致，全部NPZ数组逐元素相等 |
| 主表与决定 | main_comparison.json／.md及decision.json重放一致；2／7护栏通过，性能条件false |
| 机制控制 | 无strength，NOT_TESTED；不建议追加控制／第二seed，不自动扩训 |
| 原失败证据 | 原candidate首批exit1、旧状态／日志／快照／B2和P1四权重保留，未恢复或覆盖 |

三best SHA256：

- B2：`76b678af53eb340860273d434a7404bd6edc4545d6d156adea92341c1ee30bcb`
- P1：`6cc444decbcdcd67b4be7d9ed373e070d82aaefb0e522c6bd09debf1bb6e70a4`
- V2：`2d9801748c90531c45f488ed938a7d5bd1879089f3526b55d23cbc9030ef7ed7`

Ultralytics总体AP为本次自动验证器输出，受到源码／权重／预测绑定和文件哈希保护，
主表数值重放使用该冻结输出；上述缓存数值重算针对COCO-style与micro PR，
**没有独立重算Ultralytics内部匹配／AP**，也未追加模型推理或重跑随机训练。
因此VERIFIED仅指冻结产物及确定性缓存计算，不表示训练随机性可复现或独立泛化成立。
本轮只改报告／执行状态文档，没有修改源码，未重复运行训练前已通过的315项pytest。

## 描述性解释与11项谬误检查

覆盖11／11；没有统计检验、p值、置信区间或多seed标准差。
完整效应值、全部10类正负结果及固定护栏见[results30.md](results30.md)。

| 检查 | 结果／限制 |
|---|---|
| Simpson悖论 | 分类结果5升5降，未把总体AP_small下降外推为每一类都下降；未作跨split分层因果推断 |
| 生态谬误 | 聚合PR不代表每图／每目标收益，P90约束是全尺寸micro而非small precision |
| Berkson选择偏差 | 冻结val和伪框筛选有条件性，不视为随机总体样本；多次开发val披露 |
| 碰撞偏差 | 未拟合控制模型；P90筛选后的结果仅作条件性能描述 |
| 基率忽略 | 报告26577个small GT、TP与FP，并披露类别不均衡；召回不冒充precision |
| 向均值回归 | 同次B2／P1作配对参考，没有据单个极值或最优类挑选候选；跨seed未检验 |
| 幸存者偏差 | 原失败candidate与负结果保留；V2完整30轮，未把失败的0轮计作性能样本 |
| 多重查找 | 全部7护栏和10类正负结果都报告；P90局部改善不冒充总体成功，不作显著性宣称 |
| 分析路径自由度 | 原合同阈值不变，开发val不独立；不调r／衰减追分，不事后换AP护栏口径 |
| 相关／因果混淆 | 无strength，语义与负监督强度未分离；工程快照不作语义独立因果证据 |
| 反向因果 | 干预先于训练结果，但信息收益归因仍受上述混杂限制；未以教师诊断反推学生获益 |

## 收尾

性能停止条件触发，保留原P1；无新增推理、训练或参数选择。
主计划第3.73／10节及当前执行说明同步，中文提交并push；完成后删除p1-v2临时监控。
历史协议／验证记录的“未启动”是当时状态，最新终态以本记录和results30.md为准。
