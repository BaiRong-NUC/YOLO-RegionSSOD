# B2基线、原P1、P1＋竞争负项：成果主对照

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent plan
- Origin Date: 2026-11-03 (Asia/Shanghai)
- Verification Status: UNVERIFIED（学生性能；待执行）
- Version Label: main_comparison30_v2

用户明确要求“基线、P1、现有改进”直观比较。此处基线是B2全局阈值自训练，
改进是刚准备的P1＋竞争类别负项衰减。B1仅标注区域监督、B0完整监督另列参考，
不冒充本次B2。已完成的完整类别分数变体仍是单独的历史负结果。

本协议是新版本；旧[三臂机制合同](student_protocol.md)、原代码和预检证据保留。
新训练尚未开始，先完成新B2/P1/改进的主对照，不覆盖或拼接历史结果。

## 1. 主表与主问题

| 主表方法 | 目的 |
|---|---|
| B2：全局阈值.50自训练 | 可比较的基线 |
| 原P1 | 原方法对基线的增量 |
| P1＋竞争负项 | 当前完整方案对基线的总增量及对P1的新增增量 |

必须同时给出各臂的绝对结果，以及**P1−B2、改进−P1、改进−B2**三项差值。
最后一项直接从同一新配对原数相减，不能把旧100轮P1增益加新30轮改进增益。
包括AP50/AP50–95、AP_small、conf.25小目标召回、P70/P90小目标召回、
严格IoU.75/P90、FP数、每臂耗时；保存全部精度网格/逐类和负向结果。
P70/P90为全尺寸micro precision约束，不是small专用precision或部署保证。
COCO-style不是VisDrone官方AP，结果全部来自已多次开发的val。

## 2. 配对与成本

三组**重新完整训练30轮**，共用10%面积split0、同一冻结B1教师、
COCO初始化、学生seed0、640/batch8/SGD、增强、close_mosaic10和548图val960。
B2复用原27667伪框/4624伪图（合计11095训练tile）；P1及改进复用原35067伪框
（合计11310训练tile）。这是各方法冻结的数据构成差异，同epoch不等于相同计算量。
不复用旧B2学生权重或历史数值作为本次配对。

顺序B2→P1→改进，无并行。按现有时长估计主三臂约5–8小时学生训练，
另加三臂评估/审计；本次没有实际新训练计时，不承诺收益或耗时。
新输出`outputs/p1_competitive_negative_main30_s0/`；新run均以`p1_maincmp_`命名。

## 3. 强度控制：独立机制对照

控制仍保留，作用/公式及原固定条件沿用[旧机制合同](student_protocol.md)。
默认主三臂用于量化总成果；没有控制时明确标记`NOT_TESTED`，不能声称语义方向
有独立因果收益。主三臂满足原性能条件后，只建议准备控制，不自动扩100轮。

若一开始就要同时完成机制检验，可加`--with-control`，顺序末尾增加第四组strength30，
总学生约7–11小时。它不混入主表/主曲线；另保留分析/曲线数组/机制决定。
不允许跑完主三臂后改变其冻结代码/输入，偷偷把新控制塞回原输出：
后续另跑须另行冻结和建立独立绑定，或事先选择四组模式。

## 4. 当前准确入口

在PowerShell：

```powershell
Set-Location D:\Python\YOLO-RegionSSOD
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe scripts/run_p1_main_comparison.py --check-only
```

预检保存独立回执，不训练/推理。用户启动主三臂：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe scripts/run_p1_main_comparison.py --train
```

若事先选择四组，分别在预检/训练命令末尾加`--with-control`。
上面新主对照入口取代旧`run_p1_competitive_negative.py --train`作为当前建议命令。
已确认旧入口尚未启动；不停止任何既有任务。

训练时建立临时完成监控，正常进度静默，失败/完成再通知；终态审计后删除。
包装器拒绝已有输出/运行、无自动重试或恢复。最终自动输出
`main_comparison.json`、`main_comparison.md`及三条主曲线，记录三项增量和全部护栏。
之后只读`python scripts/run_p1_main_comparison.py --audit`。

## 5. 判断与停止

候选相对原P1沿用原性能继续线：AP>=−.10pp、AP_small>=0、.50/P90>=+.30pp、
P70>=0、conf.25小目标recall>=−.20pp、.75/P90>=0、conf.25 FP<=原P1的102%。
同新B2的差值完整报告，不因总增量看起来大就忽略相对P1的退化。
任一性能线失败停止本候选，不在val调整r/衰减追分。全过也不自动100轮。
四组模式另须通过原semantic−strength的P90>=+.15pp、AP_small>=0；
主三组模式只能报告性能，不能追认为机制确认通过。

当前仅准备/工程预检，真实全量CUDA训练链路尚未执行；任何工程失败均先诊断，
不能把不足30完整epoch或短烟测当作性能结论。原正式100轮结果继续独立保留。
