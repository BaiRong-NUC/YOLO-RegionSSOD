# 竞争负项V2：AMP修复与B2／P1复用

状态（2026-11-03）：修复、缓存重放、单批CUDA AMP及315项测试通过；
**未启动训练，尚无V2性能结论。** [验证记录](validation.md)／[独立冻结协议](protocol.md)。

原实验保留在`outputs/p1_competitive_negative_main30_s0/`，原candidate首批失败记录不改。
复用其中已完整30轮且exit0的B2与原P1，只重新初始化并训练改进组30轮。
V2将真实类别／TAL来源检查与FP16目标数值分开；下溢或零质量前景保持原P1权重1。
正常目标仍按固定r>=.25、top2负项减半；无新增网络模块、无val参数拟合。

在仓库根目录、现有yolo环境运行：

```powershell
Set-Location D:\Python\YOLO-RegionSSOD
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe scripts/run_p1_main_comparison_repair.py --check-only
```

预检核对原39,828项冻结输入、原证据／权重、新实现及验证回执；零训练／零推理。
预检可重复，回执分开保存，不占用正式新输出目录。

准备好的实际执行命令（**本次未执行**）：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -u scripts/run_p1_main_comparison_repair.py --train
```

此命令只调度一个新candidate worker，使用同COCO初始化／split0／教师／学生seed0、
30epoch、train640／batch8／SGD／AMP；不会重训B2或P1，也不恢复失败组。
学生估计约2–3小时，另加三个best权重的统一548图val960评估及审计。
实际训练启动后再建立临时监控，正常推进静默，终态处理后删除。

- 新改进run：`runs/p1_maincmp_improved_v2_30_s0/`
- 新主对照输出：`outputs/p1_competitive_negative_main30_v2_s0/`
- 过程：`status.json`、`candidate_stdout.log`、`candidate_exit.json`
- 来源：`preflight.json`、`dispatch.json`、`candidate_config.yaml`、`training_audit.json`
- 结果：`main_comparison.{json,md}`、`decision.json`、`evaluation/`、`evaluation_audit.json`

统一评估保存AP50／AP50–95／AP_small、conf.25小目标召回、双IoU完整PR、
P70／P90／严格IoU.75/P90、FP、全部类别正负结果与原训练／新worker／评估成本。
主表明确列P1−B2、改进−P1、改进−B2，不叠加旧100轮或历史30轮增益。
七项性能护栏原值不变；无strength，机制仍NOT_TESTED，全过也不自动100轮。

成功终态只读审计：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe scripts/run_p1_main_comparison_repair.py --audit
```

新输出／run已有时`--train`拒绝覆盖，无自动重试／恢复；异常保留日志，不用未满30轮判断性能。
若训练完成但评估失败，也先保留证据诊断，不能再次执行`--train`覆盖。
本修订对照复用了同一次新主对照的B2／P1，源码commit不同会记录，不能称三臂同一源码版本。

工程验证入口`python scripts/validate_competitive_negative_v2.py`已执行，默认验证目录不可覆盖；
其单批工程验证不能充当学生性能验收。
