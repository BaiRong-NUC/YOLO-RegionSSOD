# 正目标几何重排：新三臂30轮冻结学生协议

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent plan + validate
- Origin Date: 2026-11-04，Asia/Shanghai
- Version Label: positive_quality_transport_three30_v1
- Verification Status: 准备手动验证；学生性能NOT_TESTED；机制NOT_ESTABLISHED

## 固定实验与控制

新原P1参考→几何候选→匹配扰动控制，严格顺序，各连续完整30轮，合计90学生轮。
不复用历史学生；同10%面积split0、固定B1教师/原P1伪框与ignore、COCO YOLO11s初始化、
学生seed0、640、batch8、SGD、AMP、原增强与close_mosaic10、11310训练tile。
不重训B2、改骨干/P2/attention或增加模块。评估统一548张完整原图val960，
conf=.001、NMS IoU=.7、maxDet500、rect=True，与原统一评估器相同。

三臂仅改变人工同对象真类BCE目标的处理规则：候选置换原值，
控制采用匹配目标扰动L1的凸组合；reference关闭附加动作。
candidate保存原目标值集合，control匹配同对象目标总量/有效支持/搬移L1，
但不保值集合/改变锚点数/最终绝对梯度/参数梯度。精确定义与float32插值见工程合同。
伪来源、负项、原TAL及分母、box/DFL均保持原实现。
控制凸插值存在软化/目标熵混杂，反序fallback非中性扰动；同状态局部L1匹配不等于
两个学生独立训练30轮的实际累计强度匹配。绝对值、candidate−reference/candidate−control
以及control−reference一并保存，避免只因控制退化就宣称信息有效。
三臂同样使用冻结SourceDataset，不声称随机训练将逐位复现历史参考轨迹。

## 执行命令（本轮只执行check-only，不执行train）

```powershell
Set-Location D:/Python/YOLO-RegionSSOD
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B scripts/run_positive_quality_transport.py --check-only
```

审阅后手动启动一次；这是未来执行命令，本轮未运行：

```powershell
Set-Location D:/Python/YOLO-RegionSSOD
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B -u scripts/run_positive_quality_transport.py --train
```

完成后自动统一评估与决定；另开进程只读终态audit，不重新推理：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B scripts/run_positive_quality_transport.py --audit
```

输出独占 `outputs/positive_quality_transport_three30_s0/`，学生run分别为
`runs/positive_quality_transport_{reference,candidate,control}30_s0/`。
只要其中任何已有目录存在即拒绝启动；没有恢复、自动重试、单臂绕过或100轮入口。
内部worker需要活父进程/环境随机凭据/匹配当前臂和一次性claim；前臂必须完整验收。
同账户可修改文件，这只是防误调用，不是安全沙箱。

## 预检、绑定与审计

check-only不建模型/数据集/训练输出，核对新诊断/来源/工程回执和实际AMP路径通过，
工程来源、P1配置、训练图/标签/来源清单/教师/初始化、完整val、安装运行包和源码。
不通过旧工程回执推断新实现就绪，不按置信度猜来源，不读取隐藏GT监督。
全部src/scripts/tests、配置/本合同、实际包/关键扩展、Python可执行文件等按新preflight封印；
不是整个OS/驱动/CUDA二进制全集认证。训练创建缓存按旧策略校验/绑定，图像旁.npy缓存拒绝。

各worker保存真实args/源码commit/实现/mode/配置、best/last/连续1–30 CSV及退出码。
最终统一评估的manifest限定三封印best.pt，并绑定预测/权重/GT/数组。
保存全部类别AP_small、双IoU完整PR与各精度操作点、全FP及wrong-class分解和全部正负差值。
总体AP为Ultralytics，AP_small为COCO-style；两者都不是VisDrone官方AP。
P70/P90是全尺寸micro精度约束下small recall，每模型各自取阈值，不是部署精度保证。

复用评估器内旧comparison/图名仅为audit兼容，标记LEGACY_NOT_THIS_CANDIDATE。
**唯一11项判定是新根目录decision.json**。audit只读，核验精确文件集合和哈希、
训练/配置/权重绑定，缓存重算COCO/双IoU PR/FP分解及判定；
Ultralytics内部AP仅绑定validator输出，不声称从预测缓存独立重算该AP。

## 训练前冻结的11项投入门槛

| 比较 | 指标 | 固定条件 |
|---|---|---|
| candidate−reference | Ultralytics AP50–95 | ≥+0.10pp |
| candidate−reference | COCO-style AP_small | ≥+0.10pp |
| candidate−reference | conf.25/IoU.50 small recall | ≥0pp |
| candidate−reference | IoU.50/P70 small recall | ≥0pp |
| candidate−reference | IoU.50/P90 small recall | ≥+0.30pp |
| candidate−reference | IoU.75/P90 small recall | ≥0pp |
| candidate/reference | conf.25/IoU.50全尺寸FP | ≤100%（计数不增） |
| candidate/reference | 同口径wrong-class FP | ≤100%（计数不增） |
| candidate−control | AP50–95 | ≥+0.10pp |
| candidate−control | AP_small | ≥0pp |
| candidate−control | IoU.50/P90 small recall | ≥0pp |

本合同要求所列固定操作点的小目标召回不下降、AP与AP_small同时提高；
不保证全部对象、类别或完整PR曲线均不退化。不以强行减少FP为目标，
故没有复用旧可信负监督的FP下降百分比。这是新合同投入线，不追改旧失败条件，
也不是显著性标准。不可达/缺失/非有限操作点一律失败；reference FP=0要求candidate也为0。
任一项失败outcome=stop，完整保留负结果，不在该val调参数、换IoU或锚点名单后重试。
全过仅建议另行审阅第二学生seed与独立区域split；仍不能称充分因果机制或自动100轮。

## 成本、停止与监控

预算上限三学生各30轮；最近同硬件三臂学生实测约7.97小时作为参照，
本实现逐对象排序的额外CPU/GPU同步开销未在真实训练测量，预留6–12小时并记录实际时间，
此区间不是超时保证。统一三权重评估约3–4分钟，预检/只读CPU重放/audit另计。
若耗时显著超过预计，由实际监控报告并由用户决定中断；不自动扩大预算/重试。
工程失败、非零exit或轮次缺失即停止后续臂并封印；失败audit仅验收证据，性能NOT_ACCEPTED。
外部强杀若没有终态封印，audit拒绝，不能假装完成或自动修复。

实际启动后才创建临时监控，终态验收后删除；本轮不创建空监控。
本轮只允许一个已缓存8图的训练后前向且已用完，后续预检和audit零forward/optimizer。
