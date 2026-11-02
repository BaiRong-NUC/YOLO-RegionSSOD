# 类别分数叠加候选：工程验收与交接

日期2026-11-02。**实现及工程验收完成，学生训练未启动，候选效果未知。**
本轮沿用academic-research-suite的experiment-agent run/validate；
一个子代理负责独立训练模块/测试，主代理负责诊断、导出、配对入口、
评估与实际数据/验证路径验收。共同上下文检查，不是盲审或独立性能复验。

## 已执行

| 检查 | 实际结果 |
|---|---|
| `python -m pytest -q` | 269项全部通过 |
| `diagnose_p1_class_ambiguity.py --audit` | 原冻结事件及统计精确重放通过，无新推理 |
| 全量分数导出与`--audit` | 6471tile、91336候选、4839伪图、35067正框；框及最高分差均0 |
| `run_p1_class_ambiguity.py --check-only` | 22898来源哈希、11310训练tile及两臂30轮配置检查通过；未启动训练 |
| `check_class_ambiguity_engineering.py` | 完整训练集/sidecar加载；真实人工/伪标签混合输入640 AMP前反传；255项梯度有限 |
| 标准DetectionValidator路径 | 2张真实val图、缓存训练criterion的EMA实例、三项验证loss有限；与原P1验证loss精确一致 |
| 新配对评估器重放既有P1 seed0 | 548原图缓存的COCO-style AP、PR、P50–P90及conf.25统计与原结果精确一致 |
| 原论文`build_paper_tables.py --check` | 137来源及8组recall核对通过 |
| `preserve_paper1.py --audit` | 58707文件归档、4507658156bytes及源ZIP/证据ZIP哈希通过；该命令不重新比较当前原件 |
| `git diff --check` | 通过；原`training/p1.py`及原P1配置无修改 |

实际工程回执：
`outputs/p1_class_ambiguity_pair30_s0/engineering_check.json`；
完整pytest日志：同目录`pytest.log`。
工程验证共**0 epoch、0 optimizer step**，两图验证指标不作方法结果。
测试覆盖disabled/one-hot损失及梯度等价、非one-hot框/DFL及原TAL返回值一致、
人工/背景/ignore不变、质量守恒、confidence=1的显式伪标签身份、
sidecar缺失/陈旧/身份错位拒绝、mosaic/flip/close_mosaic及collate运输、
真实模型CPU/CUDA AMP前反传与普通验证回退、每轮首batch回调及原runner记录兼容、
性能继续线的全部护栏与不可达/非有限操作点拒绝。

## 工程发现与边界

- 首次真实数据工程probe使用随机增强，两张抽样图中的一种标注来源被增强裁掉，
  在模型前反传前因“未同时覆盖人工/伪标签”主动拒绝。
  原日志和脚本保存在`engineering_failed_provenance/`；第二次改用确定性原图
  标签运输以覆盖两类来源。训练仍使用原增强；增强运输由专门测试验证。
- 集成预检发现Ultralytics设置OMP线程数后，诊断logistic的float64系数出现
  约1.4e−16尾差。原诊断来源/结果保持不变；预检浮点绝对容差1e−12，
  样本事件、计数、门槛仍要求精确一致，容差与最大差值写入preflight。
- 真实AMP工程probe中总目标质量52.6228256、c加权质量47.4934731，
  新旧一致；BCE与梯度L1/L2有变化，不能宣称监督强度完全等价。
  该probe不是训练性能，也不是全数据梯度估计。
- 正式候选每epoch第一完成batch记录`classification_diagnostics.jsonl`，
  分人工正anchor、伪正anchor、有效负anchor、ignore负anchor及总体。
  配对验收要求30条完整有限记录且目标质量守恒；其余batch不开启该诊断。
  它是固定抽样的机制快照，不是epoch梯度总量或模型参数梯度。
- 当前未做优化器更新、未经历真实30轮。工程通过不能代替性能、训练终态或新颖性验收。

## 用户执行与结果判定

仓库根目录、yolo环境：

```bash
python -u scripts/run_p1_class_ambiguity.py --train
```

会先重验输入和工程源码绑定，再在两个新进程顺序运行reference/candidate各30轮，
保存每臂退出码与日志，自动全图960验证、AP_small、完整PR和固定继续线。
新训练目录分别为`runs/p1_class_ambiguity_reference30_s0/`与
`runs/p1_class_ambiguity_candidate30_s0/`。
进度/两臂日志及结果放在`outputs/p1_class_ambiguity_pair30_s0/`。
任一臂失败即停，不自动重试或扩训；已有目录拒绝覆盖。
完成后`python scripts/run_p1_class_ambiguity.py --audit`独立重放结果。

这一步验证的是“能否在原P1基础上改善高精度区间且保住已有能力”。
固定继续线见[学生协议](student_protocol.md)。原论文的正式100轮结果、
失败方向、保存点和Gate3/4均保留。
