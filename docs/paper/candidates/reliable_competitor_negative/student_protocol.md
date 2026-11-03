# 可信竞争负监督：三臂30轮学生合同

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent implementation + validate
- Origin Date: 2026-11-03 (Asia/Shanghai)
- Verification Status: 工程编排准备；学生性能 NOT_TESTED
- Version Label: reliable_competitor_negative_three30_v1

## 固定设计

三组均从同一 COCO YOLO11s 初始化全新训练，严格依次 reference → candidate → control，
各连续30轮；不复用历史P1/B2学生，不重试、不恢复、不扩到100轮。
原 `configs/train_p1_10.yaml` 的内容哈希固定；每份新配置只改运行名称并增加
`reliable_competitor_negative: {mode: reference|candidate|control}`。
三臂均使用同一新来源运输实现，reference关闭附加负项；reference与原P1的等价性
由零步骤工程回执支持，而非断言所有随机训练轨迹逐位一致。

固定 split0、10%人工标注面积、同一teacher/伪框/ignore、COCO初始化、学生seed0、
640、batch8、SGD、AMP、原增强和close_mosaic10、11310训练tile。
全图548张val960沿用原统一评估器：conf≥.001、NMS IoU.7、maxDet500、rect=True。
训练内每轮默认val与本轮最终统一val960是不同阶段；最终比较只读统一评估结果。

候选只对人工来源且实际正质量>0的TAL正锚点加倍最大非真类负BCE；
control在相同支持集将全部非真类乘 `1+p_k/sum(p_negative)`，detach概率。
正项、分配、置信度、ignore、box/DFL不直接修改。控制仅匹配局部负logit梯度L1，
不能宣称参数梯度、后续TAL和优化动态相同。详细数值/来源约束沿工程合同与README。

## 入口与边界

在仓库根目录、现有yolo Conda环境：

```powershell
Set-Location D:/Python/YOLO-RegionSSOD
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B scripts/run_reliable_competitor_negative.py --check-only
# 仅在用户另行决定启动训练后执行：
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B -u scripts/run_reliable_competitor_negative.py --train
# 终态只读验收，包含CPU预测缓存重放，不重新val：
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B scripts/run_reliable_competitor_negative.py --audit
```

本次准备不执行train。`--check-only`不创建任何实验输出/训练目录，输出简短JSON；
`--train`重新预检后保存完整冻结preflight/dispatch，再顺序启动每臂独立worker。
内部worker须有匹配父PID、仅环境传递的随机凭据、当前臂状态与一次性独占claim；
前臂必须完整验收。claim失败不清理或重用，任何已有输出或run目录均拒绝。
这是防止误调用/意外绕过，不是抵御同账户恶意修改代码与回执的安全沙箱。

输出独占`outputs/reliable_competitor_negative_three30_s0/`，三run分别为
`runs/reliable_competitor_negative_{reference,candidate,control}30_s0/`。
一条`--train`顺序完成三臂，无单臂绕过／自动重试路径；本次没有执行。
按既有原P1约2小时／V2约2.3小时观察，先预留三学生6–9小时，统一评估约3分钟，
另留预检／缓存审计时间。新来源检查和fp64概率开销尚未测量，实际训练可能更长；
三臂时间／worker墙钟／评估秒数全部记录，不据单batch工程时长外推效率结论。
实际启动后再建立临时监控，保持真实进程与监控运行，终态处理后删除；本次不建空监控。

## 预检与来源冻结

工程入口固定为 `outputs/reliable_competitor_negative_engineering/validation_final.json`：
必须accepted、optimizer_steps=0、parameter_hash_unchanged=True，实际batch8/AMP/cuda:0，
runtime版本一致，完整source_sha256与artifact_sha256有效。原唯一真实batch及缓存重放
由工程流程管理；本runner预检/测试不做真实forward。

冻结全部项目src/scripts/tests源码、YAML配置、本候选协议、实际安装运行包Python源码、
Ultralytics默认YAML/损失/增强/TAL、相关编译扩展、Python可执行文件与包版本。
依赖包包括Ultralytics、torch、numpy、Pillow、PyYAML、pycocotools；不宣称哈希覆盖
操作系统、GPU驱动、所有CUDA DLL、硬件/调度或整个环境的每一个二进制依赖。
原P1预检哈希固定用于数据来源对照，不加载历史学生。

真实输入包括训练图/标签、来源元数据/划分清单、人工/伪来源身份、teacher和COCO权重、
完整val图/标签、源图和未标注tile图。检查类别整数0–9、有限合法框/置信度、正伪框标签
逐项绑定、重复/缺失来源、汇总计数和训练顺序。不会读取未标注tile的隐藏GT标签。
训练图像旁.npy缓存拒绝；已有YOLO聚合label cache按首标签目录定位，内容须与全部
11310图的实际标签/尺寸一致后冻结，不能按人工/伪标签目录拆成两个缓存。
训练期新生成的缓存属于衍生产物，不构成新的监督输入。

预检由实际Ultralytics DEFAULT_CFG_DICT与配置覆盖生成完整expected_args，
冻结task/mode/data/model/seed/AMP/增强/优化器/fraction/resume等，包括save_dir。
每臂完成需退出码0、连续1–30轮、所有CSV数值有限、实际args完全匹配，experiment的
implementation/mode/config和所有来源哈希绑定一致，best/last权重封印。
源码或配置改变必须重新准备独立实验，不能续用已dispatch目录。

## 统一评估与只读验收

三臂完成后复用 `evaluate_class_ambiguity_pair.evaluate/audit`，不更改其历史源码/旧产物。
完整COCO AP_small、两IoU PR曲线、各精度操作点、逐类AP_small、成本及所有正负差值保留。
manifest限定三臂各自封印best.pt，绑定权重/预测/manifest哈希、548图GT和所有输入；
audit原样重放COCO/PR数值与NPZ数组。Ultralytics AP是统一validator记录的结果并做
来源/哈希绑定，原通用audit不从像素/缓存独立精确重算该AP；不可夸大其重放范围。

原评估器生成的 `evaluation/comparison.json` **旧五项门槛不适用于本候选**；
`evaluation/LEGACY_NOT_THIS_CANDIDATE.json`明确标注，包括旧绘图标题。唯一判定是
实验根目录 `decision.json` 的新十项合同；根目录 `comparison.json` 给三臂绝对值、
候选减参考/控制、逐类正负差值与FP计数。保留旧比较仅为通用audit兼容。

wrong-class分解使用同一post-NMS预测缓存，固定conf≥.25/IoU.50，全尺寸联合匹配。
调用原 `assign_predictions` 与 `error_diagnosis.classify_false_predictions`；
分类优先级沿原实现：matched、duplicate_competition、wrong_class、localization_near、
near_other_class、unexplained_background。必须与conf.25 PR FP总数一致，逐图全部类型保留。
wrong-class不得从P90操作点取数或给FP继承邻近GT尺寸。

终态封印完整输出与训练产物，audit检查精确文件集合、哈希、训练来源、数值缓存、
FP分解和decision重算；audit无写文件/修复/新推理。失败终态仅验收证据，返回
accepted=False、performance=NOT_ACCEPTED，不伪造三臂性能。外部强杀未完成终态封印
时audit拒绝，需人工检查；不提供自动恢复路径。

## 训练前冻结的十项投入门槛

| 比较 | 指标 | 通过条件 |
|---|---|---|
| candidate−reference | Ultralytics AP50–95 | ≥+0.10pp |
| candidate−reference | COCO-style AP_small | ≥0pp |
| candidate−reference | conf.25/IoU.50 small recall | ≥−0.20pp |
| candidate−reference | IoU.50/P70 small recall | ≥0pp |
| candidate−reference | IoU.50/P90 small recall | ≥0pp |
| candidate−reference | IoU.75/P90 small recall | ≥0pp |
| candidate/reference | conf.25/IoU.50全尺寸FP | ≤97% |
| candidate/reference | 同口径wrong-class FP | ≤95% |
| candidate−control | Ultralytics AP50–95 | ≥+0.10pp |
| candidate−control | COCO-style AP_small | ≥0pp |

百分点按100×分数差，不将.10误作10个百分点。FP门槛比较绝对计数上限，reference为0时
要求candidate也为0，无除零比值。精度点不可达/缺失或非有限值均失败；任何一项失败
outcome=stop，全部结果保留。全过仅eligible_for_separate_confirmation_review，
不会自动训练第二seed或100轮。投入门槛不是显著性检验或因果证明；单seed、同split/teacher、
开发val上的各臂独立阈值不证明部署/外部泛化；原Gate3/4与旧V2失败判定均不改写。
