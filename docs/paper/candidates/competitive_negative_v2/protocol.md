# 竞争负项AMP修订：只训练改进组，复用本次B2／P1

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent plan/validate
- Origin Date: 2026-11-03
- Verification Status: PLANNED（先固定协议，再工程验证）；学生性能未验证
- Version Label: competitive_negative_amp_safe_v2_reuse30

## 修复合同

原P1、旧candidate源码／配置／协议／失败目录不改。新实现为
`training/competitive_negative_v2.py`，只分开TAL来源类别检查和BCE数值检查：
原质量／真实分配类别检查继续拒绝软目标、错类及来源损坏；原P1目标转换／损失分母／
正项／回归不变。实际BCE目标质量为0（原始零质量或AMP下溢）时，新动作权重为1；
不将原正类当成普通负类。非零目标保持原r>=.25、top2负项减半和原强度控制定义。
零质量处理为确定数值边界，不是按val选择的额外参数。

## 验证范围

先重放已捕获1988前景／649伪前景缓存，再做合成边界／梯度／真实损失集成测试，
最后最多一个真实训练batch的CUDA AMP前向，零训练循环／反向／优化步骤。
固定同一预测，对比原P1与V2的分配和box／DFL，检查正目标不变及诊断计数。
不读隐藏GT、不拟合参数、不迭代val评估；setup允许读取既有val元数据及框架AMP自检。
所有样本都是已有开发数据，结果仅工程验证，不是性能确认。

## 复用与执行

从`outputs/p1_competitive_negative_main30_s0/`复用已连续1–30轮、exit0的
`p1_maincmp_b2_30_s0`和`p1_maincmp_p1_30_s0`，绑定其原preflight、退出码、
args／experiment／CSV及best／last哈希，复核原39828冻结输入与教师／初始化。
B2 best固定`76b678af53eb340860273d434a7404bd6edc4545d6d156adea92341c1ee30bcb`；
P1 best固定`6cc444decbcdcd67b4be7d9ed373e070d82aaefb0e522c6bd09debf1bb6e70a4`。

独立新输出`outputs/p1_competitive_negative_main30_v2_s0/`，新改进run
`runs/p1_maincmp_improved_v2_30_s0`，从同COCO初始化完整30轮。
不恢复失败candidate、不重新训练B2／P1；等epoch不等计算量，原B2为11095tile、P1为11310tile。
只允许candidate worker；拒绝已有新输出／run，无覆盖、自动重试或自动扩100轮。

改进组成功后，在新目录对三个冻结best权重统一进行548图val960、AP_small和双IoU完整PR评估，
保存预测绑定、逐类／全曲线／成本，输出绝对值及P1−B2、改进−P1、改进−B2。
这称为“复用同批B2／P1的修订主对照”，不声称三组同一源码commit或同时启动。
训练来源差异逐项披露；不拼接旧100轮或历史30轮增益。COCO-style非VisDrone官方AP。

七项性能护栏沿用原固定值：改进−P1的AP50–95>=−.10pp、AP_small>=0、
IoU.50/P90 small>=+.30pp、P70>=0、conf.25 small>=−.20pp、IoU.75/P90>=0、
conf.25 FP<=原P1的102%。无strength组，机制仍NOT_TESTED；全过也不自动扩训。
失败保留全部证据，不用短工程测试判断性能，不在当前val调r／衰减。

入口为`python scripts/run_p1_main_comparison_repair.py --check-only`和用户手动`--train`；
终态只读`--audit`。本轮只允许预检和工程验证，不执行`--train`。
预估仅剩一个学生约2–3小时，另加三权重评估／审计；实际成本待记录。
训练启动后再建立临时监控；目前不创建空监控。
