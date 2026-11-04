# V2独立集成修复：三臂30轮执行合同

- Origin Skill / Mode: academic-research-suite / experiment-agent validate
- Origin Date: 2026-11-04，Asia/Shanghai
- Version Label: positive_quality_transport_trainerfix_v2
- Verification Status: 1088tests/独立工程检查/40267项绑定预检通过；未启动学生；NOT_TESTED / NOT_ESTABLISHED

科学定义/控制精确定义/历史反证沿用[V1冻结学生合同](../positive_quality_transport/student_protocol.md)
及其工程协议，本合同只另立修复后的执行路径。V1的0epoch初始化失败不计作学生性能结果，
V2不复用/恢复其reference，不改变原11门槛。

## 固定设置与必要控制

全新原P1参考→几何候选→匹配扰动控制，各连续完整30轮、总90学生轮，严格顺序。
同10%面积split0、固定教师/原伪框/ignore/初始化/学生seed0，YOLO11s、640、batch8、
SGD、AMP、close_mosaic10、11310训练tile；不重训B2、不扩100轮。
统一548张full-image val960，conf=.001/NMS IoU=.7/maxDet500/rect=True。
候选只置换人工同对象真类BCE目标值；reference关闭；control匹配同对象搬移L1/质量/支持，
方向固定hash及反序fallback，幅度来自候选。control不保值集合/改变锚点数/最终绝对梯度，
熵/softening混杂及独立训练累计强度差异仍存在。
原TAL、分母、box/DFL、伪来源、负项、ignore保持冻结模型/损失，
同时保存candidate−reference、candidate−control、control−reference，不能只靠控制退化判机制。

## 不变的11项联合门槛

所有百分比差值均为pp。candidate相对reference：AP50–95≥+0.10；AP_small≥+0.10；
conf.25 small recall≥0；IoU.50/P70 small recall≥0；IoU.50/P90≥+0.30；IoU.75/P90≥0；
全尺寸conf.25 FP≤reference；错类FP≤reference。
candidate相对control：AP50–95≥+0.10；AP_small≥0；IoU.50/P90 small recall≥0。
所有门槛必须同时通过，任一操作点不可达、缺失、非有限即失败。
不足30轮只判工程/执行失败，不做方法性能判定；不事后改门槛。
通过也仅允许审阅后讨论后续确认，不证明因果机制、统计显著性、跨split泛化。

## 手动命令

旧目录保留。新check-only不创建实验输出，不构造模型/数据集，不forward、不训练。
新receipt须包含三个mode真实dataloader合成路径检查和既有完整loss缓存，
同时绑定V1失败封印/独立只读audit回执与源数据/安装包。旧回执不能代替V2集成证据。

```powershell
Set-Location D:/Python/YOLO-RegionSSOD
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B scripts/run_positive_quality_transport_v2.py --check-only
if ($LASTEXITCODE -ne 0) { throw 'V2预检未通过，停止，不启动训练' }
```

确认上述返回exit0后，手动执行一次：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B -u scripts/run_positive_quality_transport_v2.py --train
```

父入口顺序启动三臂，保存claim/exit/连续1–30 CSV/args/实现/commit/权重SHA，
三臂成功后自动统一评估、精确绑定best权重/逐图预测/评估协议、保留完整指标和决定，终态封印。
独立只读核验（不重复val推理；日志如保存应在封印目录之外）：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B scripts/run_positive_quality_transport_v2.py --audit
```

新输出`outputs/positive_quality_transport_v2_three30_s0/`；
run为`runs/positive_quality_transport_v2_{reference,candidate,control}30_s0/`。
任一已存在均拒绝启动，无resume/重试/单臂绕过；失败记录不覆盖，父退出缺封印时只报未确认。
父/worker活跃时禁止改冻结代码/协议。终态须先确认父与所有worker停止。
audit工程通过与11项性能/控制通过分开。完整保留负结果，不自动修复或再训练。

## 成本与停止

总预算90学生epoch+三臂各一次统一548图评估；旧同设置三臂约8小时仅供参考，
新正目标排序吞吐尚未实测。初始化/下载/AMP bootstrap与逐epoch原默认验证另有耗时。
任一进程失败/缺连续30轮停止后续，不自动恢复、并行或扩100轮。
本轮仅零训练修复/验证，没有执行--train。实际手动启动后再建临时观察，终态清理。
