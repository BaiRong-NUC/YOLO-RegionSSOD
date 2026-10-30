# 小框NWD全量30epoch：执行前确认协议

2026-10-30。用户在看到5epoch精度/召回取舍后，明确要求“至少30epoch”
才判断性能。本项按该新授权独立登记，不把原5epoch失败改写成通过。
5epoch只保留探索性观测；未来短程只能排除工程故障，不作为性能淘汰依据。
Codex当前单候选自主预算固定30epoch，更长训练仍需证据和用户执行。

## 1. 单因素、配对和预算

只新增一个学生`partial10_p1_small_nwd_pilot30_s0`，从原COCO权重独立训练，
不是续接已有5epoch。与已验收`partial10_p1_pilot30_s0`公平配对：
相同11310tile、10%标注面积split、固定B1教师/伪标签/ignore、
seed0、640、SGD、batch8、close_mosaic10、30epoch、无早停。
固定参考不重复训练；不与5epoch的close_mosaic0子集结果直接比较。

沿用已实际运行的`src/regionssod/training/nwd.py`且核对其原SHA256：
增强后640训练像素中，GT面积<32²的正anchor，box项使用0.5CIoU+0.5NWD；
其他box、DFL、TAL、分类软权重、ignore、模型架构、采样和标签不变。
C仍为原1024开放tile/7519小框估计的20.228043063665424训练像素，
启动前从同一开放标签重算并要求精确一致。该开放子集属于同一完整训练集；
不重新估计C、不改变mix/面积边界，不用隐藏训练GT或val拟合。
原GPU AMP前反传和5epoch工程证据复用，不另外消耗工程训练轮数。

## 2. 预检、评估和验收

配置`configs/train_p1_small_nwd_10.yaml`。首先通过测试，中文提交并push配置/协议/代码，
再执行`python scripts/run_p1_small_nwd.py --check-only`，核对冻结输入与参考产物、
原损失实现/开放标签常数、数据规模及训练设置。
实际由`scripts/run_small_nwd30_background.py`隐藏后台启动一次。
运行目录存在或已有dispatch/status则拒绝重启；中断不拼接epoch，不自动重试。
保存包装/子进程PID、创建时间、命令、日志、CSV、退出码与最佳/末尾权重。
启动时创建临时定时监控，正常运行保持安静，完成/失败处置后删除。

30个连续epoch结束后，使用best.pt完成548原图val960及固定操作点分尺寸recall
（conf0.25/match IoU0.50/NMS0.7/max_det500），与原完整P1比较。
训练CSV的640值只作进度，最终结果不混用。独立验收
`python scripts/audit_p1_small_nwd.py`要求原进程停止、exit0、阶段完成、
输入/源码/产物哈希一致、实际args配对、两组548图预测重放。
单seed结果不是显著性证据或官方AP_small/AR500。

## 3. 后续线与停止条件

在未知本项30epoch结果前固定两条可选路线，与既有去软权重30epoch协议一致：

- 综合收益：mAP50-95至少+0.30个百分点，且small recall不下降；
- 召回收益：small recall至少+1.00个百分点，且mAP50-95下降不超过0.10个百分点。

二者满足其一才值得准备更大训练交接；不得把一点召回收益抵销任意AP损失。
阈值是控制成本的决策规则，不是显著性检验。类别/其他尺寸退化和资源开销
必须一并披露，不能只挑有利指标。未达到时如实记录，不调参数重试。
即使通过，也只准备冻结配置、预检/训练/验收代码与用户命令，不自动100epoch。
当前仅A；本协议不启动B/C、基础大模型、蒸馏、剪枝或部署。
