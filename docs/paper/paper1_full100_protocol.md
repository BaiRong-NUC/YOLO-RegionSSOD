# 第一篇B2/P1正式100轮：用户运行协议

2026-10-31。依据用户最新授权进行综合盘点后，优先确认原P1主线。
研究决定见`training_investment_review_20261031.md`及
`training_investment_decision.json`。原Gate3仍未通过；该判断不改历史结果。
覆盖保护原条件式入口保持原用途，本入口不通过翻转其结果标志来启动。

## 固定实验与职责

- 仅B2/P1 seed0两组，各从相同COCO权重独立100轮，顺序运行。
- 固定10%区域面积split、既有B1教师/离线伪标签、YOLO11s、
  640/batch8/SGD/workers0/close_mosaic10；不叠加分配/NWD/回归c权重。
- B2 11095tile，P1 11310tile；不把相同epoch写成同计算量。
- 全部阶段自动执行：训练→完整548图val960→固定操作点recall→保存预测
  重放与来源/实际args/100行有限连续epoch验收。
- 助手只准备并实测预检，大训练由用户启动；没有默认自动训练选项。
- 两臂均使用新目录，不覆盖或续训任何pilot；P1实际启动前必须先通过B2正式验收。

## 用户执行顺序

以下命令在Git Bash执行，预检不消耗GPU训练或新增推理：

```bash
source /d/MiniConda/etc/profile.d/conda.sh
conda activate yolo
cd /d/Python/YOLO-RegionSSOD
python scripts/run_paper1_full100.py --arm b2 --check-only
python scripts/run_paper1_full100.py --arm p1 --check-only
python -u scripts/run_paper1_full100.py --arm b2 --train
```

确认上一条正常退出且打印accepted验收结果后，再执行：

```bash
python -u scripts/run_paper1_full100.py --arm p1 --train
```

不要用分号无条件串接两条训练命令。命令在当前终端等待，不需要后台启动。
标准输出/错误写入每臂output目录，可另开Git Bash查看：

```bash
tail -f outputs/paper1_b2_formal100_s0/stdout.log
# B2完成并启动P1后改为：
tail -f outputs/paper1_p1_formal100_s0/stdout.log
```

## 预算、位置与验收

按既有30轮两seed时间外推，学生总计约10.78–12.16小时，
建议另计完整评估/I/O而预留12–15小时；这不是保证。
不重新训练教师、不扩大split或其他标注预算。

| 项目 | B2 | P1 |
|---|---|---|
| output | outputs/paper1_b2_formal100_s0 | outputs/paper1_p1_formal100_s0 |
| train run | runs/partial10_b2_formal100_s0 | runs/partial10_p1_formal100_s0 |
| val | runs/val/partial10_b2_formal100_s0_val960 | runs/val/partial10_p1_formal100_s0_val960 |
| recall | runs/size_recall/b2_formal100_s0 | runs/size_recall/p1_formal100_s0 |

各臂终态记录包括启动PID/创建时间/命令、真实子进程exit_code、阶段、
源码/配置/来源/权重/产物哈希和独立验收。
`--train`正常结束时自动验收；以后只读复核：

```bash
python scripts/run_paper1_full100.py --arm b2 --audit
python scripts/run_paper1_full100.py --arm p1 --audit
```

P1验收时同时重新核对B2，生成
`outputs/paper1_p1_formal100_s0/paired_comparison.json`，
包含总体/各尺寸差值、类别AP和学生时间。
现有B1 100轮只作正式参考，不能用其第30轮替代独立pilot。

存在目录、来源变化、版本不符、另一个正式臂仍在运行、epoch不完整或非有限、
退出码不为0/缺失、预测重放不符时拒绝重复启动或拒绝验收。
中断/失败保留目录和日志，不自动续训/重试，不把两次训练拼接。
若控制终端被关闭导致控制进程未写入退出码，需要先核实实际工作进程，
不能只凭旧status判失败或再运行命令。

结果出来后按盘点文档综合决定是否补seed1；不预先承诺100轮会放大小试增益，
不把固定操作点recall/F1写成官方AP_small或VisDrone官方指标。
