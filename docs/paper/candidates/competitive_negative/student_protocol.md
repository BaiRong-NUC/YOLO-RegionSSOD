# 原P1＋竞争类别负项衰减：三臂30轮待执行方案

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent plan
- Origin Date: 2026-11-03 (Asia/Shanghai)
- Verification Status: UNVERIFIED（学生性能；仅准备实现/工程测试）
- Version Label: competitive_negative_three30_v1

本合同在任何本候选学生训练之前制定。诊断全为已查看数据上的探索，允许准备，
不证明学生收益。本次不训练；旧类别分数变体不扩到100轮，原Gate3/4不改。

## 1. 唯一方法变量

保留原`m*e_y`正类目标、正框/类别、尺度筛选、c乘数、未加权TAL分母、
TAL算法、人工标签、空间ignore及box/DFL。使用旧sidecar，不新推理。
仅在**匹配到伪正框的正anchor**、教师r=top2/top1>=.25时改变非top1负项。
此门槛沿用诊断旧分层；衰减1/2是一次性保守动作，不由val拟合，不扫描其取值。
同分按类别id稳定排序。没有温度、类别权重或新增模块。

| 臂 | 负类BCE乘数（原c之外） | 用途 |
|---|---|---|
| reference | 全为1 | 新训练原P1 |
| candidate / semantic | 触发anchor仅top2为.5，其他为1 | 定向减轻竞争真实类别的错误负监督 |
| strength | 触发anchor全部非top1类别为`1-.5*p_k/sum_non_top1(p)` | 匹配弱化总强度，检验方向信息 |

其中k为教师top2，p是**当前学生logit的sigmoid，权重整体detach**。
在相同logits、分配、c下，两个动作每anchor移除的负类logit梯度L1都为
`.5*c*p_k`（再乘相同loss gain/batch/分母）；正类梯度完全保留。
控制不要求目标质量重新归一化、不改变正目标，亦不匹配BCE值、逐类梯度、
参数梯度方向/范数或不同训练轨迹。不能把“强度匹配”夸大为全部混杂均被排除。
若softmax背景机制与本YOLO sigmoid不同，不能直接搬公式或以`1-max(s)`作背景概率。

这是单项损失位置变量加一个必要控制，不是三项叠加模块。控制臂也使用同一歧义
触发集合，因此专门区分“放松哪一类”与“整体放松多少”，不检验r选框本身的因果收益。

## 2. 配对、成本与入口

三臂均**全新、完整30epoch**，split0/教师/初始化/seed0/640/batch8/SGD、
增强与close_mosaic10同原P1；按reference→semantic→strength顺序，无并行。
同轮数不等于不同方法实际耗时相同。禁止复用历史30/100轮作为本轮对照。
三臂不包含B2；历史B2只能另列参考，不能称为本次三臂之一。

按最近原P1约2.04小时/30轮、类别分数扩展约2.26小时估计，本方案学生约6–9小时，
另加三臂548图val960/COCO-style AP_small/完整PR/数值审计，实际待计时。
是本机粗估，未实测新候选CUDA吞吐；不承诺耗时或提升。额外教师分数提取为0。

在PowerShell、仓库根目录：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe scripts/run_p1_competitive_negative.py --check-only
```

预检只核对冻结来源和配置，保存独立回执；不训练/推理。实际启动命令（交用户决定执行）：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe scripts/run_p1_competitive_negative.py --train
```

开始时建立临时完成监控，使用`outputs/p1_competitive_negative_three30_s0/status.json`
及三臂stdout；正常进度静默，异常/完成再通知，终态审计后删除。包装器保存状态/退出码，
无自动重试/恢复/扩训；监控不是包装器自动建立的Codex定时任务。
输出目录已存在即拒绝，任何失败保留原始记录，先诊断工程原因而不解读性能。

完成后只读复核：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe scripts/run_p1_competitive_negative.py --audit
```

本轮验证包括CPU合成损失/自动微分、分配和回归等价性、控制强度匹配，以及全套pytest；
尚未执行本候选真实全量CUDA训练链路。预检通过也不能称已完成真实训练验收。

## 3. 固定投入条件与停止规则

必须先通过三臂连续1–30轮/exit0、实际args配对、来源/权重/预测绑定、
30首batch机制快照及完整AP/PR数值重放；异常、不可达、非有限均不能通过。
IoU .50/.75、全尺寸micro precision，均使用同一val经验操作点，不是部署保证。

candidate−reference须全部满足：

1. Ultralytics AP50–95 >= −.10pp；
2. COCO-style AP_small >= 0pp；
3. IoU.50/P90 small recall >= +.30pp；
4. IoU.50/P70 small recall >= 0pp；
5. IoU.50/conf.25 small recall >= −.20pp；
6. IoU.75/P90 small recall >= 0pp；
7. IoU.50/conf.25的FP数不超过reference的102%。

此外candidate−strength：IoU.50/P90 small recall >= +.15pp，AP_small >=0pp。
此额外线检验语义定向是否值得投入；控制也有效时不能宣称收益全来自类别语义。
阈值是工程/科研预算决策线，不是显著性、最小可发表效果或投稿保证。
全网格、全部类别和负结果均保存，不按类别bus等val表现改loss。

任一线失败：停止当前r=.25/衰减.5方案，不在相同val调阈值、温度、权重追分；
只记录边界与负结果。全部通过：仅建议另行冻结第二学生seed的三臂30轮，
待该配对确认后再考虑100轮；不自动启动任何训练。

## 4. 新颖性与局限

最近邻不是只有Humble Teacher：Credible Teacher已有逐类别可靠/不确定柔性监督，
Soft Teacher与S³OD已有教师指导负学习，LabelMatch已有可靠/不确定监督分治。
本候选可验证的具体问题是：区域面积预算、固定离线教师、YOLO sigmoid/TAL下，
保留正类目标并定向削弱竞争负项，能否在强度匹配控制之上改善覆盖而不损失AP/误报。
尚未证明首创或独立算法贡献；跨split/teacher、类别与参数梯度因果仍缺。
