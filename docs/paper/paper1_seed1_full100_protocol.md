# 正式100轮的学生seed1确认

2026-11-01，在seed0双方100轮独立验收及只读效果盘点后登记。
P1−B2的AP50-95+0.0528pp、small recall+2.1259pp（净+565），
同时未匹配预测+1221、precision−2.3690pp、F1+0.1800pp。
依据`p1_b2_full100_results.md`，值得做一次学生随机性确认；
不追改原Gate3，不把seed0微小AP差当成稳定领先。

## 唯一新增问题与预算

- 新增学生seed1的B2/P1各100轮，先B2后P1，总200epoch；
  已有seed0结果不重训、不续训到200/300轮。
- 固定10%区域标注面积split0、既有B1教师及全部离线伪标签；
  不重新生成split/教师/伪标签，不增加新标注或基础模型。
- 各自COCO独立初始化、YOLO11s、640/batch8/SGD/workers0/
  close_mosaic10，方法和所有训练超参保持seed0原规则，只改变学生seed与输出名。
- B2/P1训练tile数11095/11310，等epoch不等算力。
- seed0 B2在63轮后恢复的事实继续披露；
  seed1是另一个学生随机性重复，不是恢复效应的独立因果实验。
- 助手只准备代码、测试和预检；实际训练由用户启动。

## 用户顺序命令

在Git Bash执行：

```bash
source /d/MiniConda/etc/profile.d/conda.sh
conda activate yolo
cd /d/Python/YOLO-RegionSSOD
python scripts/run_paper1_seed1_full100.py --arm b2 --check-only
python scripts/run_paper1_seed1_full100.py --arm p1 --check-only
python -u scripts/run_paper1_seed1_full100.py --arm b2 --train
```

第一组正常exit0且验收通过后再执行：

```bash
python -u scripts/run_paper1_seed1_full100.py --arm p1 --train
```

开始前会重验seed0正式配对和冻结来源；禁止改冻结代码、
同时运行其他正式臂、覆盖已有输出或跳过B2验收。
这组P1实际训练必须先通过同seed1的B2验收，不能复用seed0 B2冒充seed1。
预检只读、不创建训练目录，不新增GPU训练或推理。

## 产物和观察

| 项目 | B2 | P1 |
|---|---|---|
| output | outputs/paper1_b2_formal100_s1 | outputs/paper1_p1_formal100_s1 |
| train | runs/partial10_b2_formal100_s1 | runs/partial10_p1_formal100_s1 |
| val | runs/val/partial10_b2_formal100_s1_val960 | runs/val/partial10_p1_formal100_s1_val960 |
| recall | runs/size_recall/b2_formal100_s1 | runs/size_recall/p1_formal100_s1 |

另一终端查看当前正在运行的臂：

```bash
tail -n 20 -F outputs/paper1_b2_formal100_s1/stdout.log
# 完成B2并启动P1之后改为：
tail -n 20 -F outputs/paper1_p1_formal100_s1/stdout.log
```

完成后自动full-val960、conf0.25/IoU0.50固定点recall与548图预测保存，
完整100轮/来源/实际args/权重/产物/预测重放独立核对，记录真实exit_code和进程身份。
只读复核命令：

```bash
python scripts/run_paper1_seed1_full100.py --arm b2 --audit
python scripts/run_paper1_seed1_full100.py --arm p1 --audit
```

P1完成后生成同seed1的paired_comparison.json与只含两学生seed、
各自独立100轮的two_seed_comparison.json；不混入30轮pilot。
两seed报告配对差均值和样本SD，不称为跨split泛化或显著性证据。
中断/失败保留原始证据，不自动重试、恢复或覆盖；如需恢复须单独核实与登记。

## 成本与完成后决定

按seed0实测，B2约5.93小时、P1约6.73小时，总约12.66小时学生训练，
另加预检/完整验证/I/O，可预留13–15小时，负载变化时可能更长。
不重新计入已有教师训练；完整成本仍须另列教师与伪标签生成。

完整报告AP50/50-95、尺寸召回/匹配、类别退化、误检/F1与成本。
重点确认small收益及其代价是否在正式预算另一学生seed中保留；
不强求AP涨幅超过某个任意数值，也不以任意正数自动认定论文可投。
若结果基本重复，优先收束稿件、同协议必要基线与机制分析；
若变化明显，先报告不确定性并回看缓存，不自动增加seed2或新模块。
