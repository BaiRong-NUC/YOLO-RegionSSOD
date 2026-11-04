# 人工正目标几何重排：V2独立Trainer集成修复

- Origin Skill / Mode: academic-research-suite / experiment-agent validate
- Origin Date: 2026-11-04，Asia/Shanghai
- Version Label: positive_quality_transport_trainerfix_v2
- Verification Status: 1088tests、独立工程验证与40267项绑定预检通过；学生性能NOT_TESTED；机制NOT_ESTABLISHED

V1用户手动执行在reference验证数据集初始化时报`super(type, obj)`错误，
0完整epoch，candidate/control未启动。它是集成失败，不是方法性能失败。
旧源码、合同和`outputs/positive_quality_transport_three30_s0`封印保留。

V2只修Trainer数据集派发，继承原模型/损失与mode检查，训练SourceDataset构造体逐行等价，
验证显式派发P1Trainer，避免从不在继承链中的类移植含零参数super的方法。
这不是一个新的科学候选，也不改变几何信号、控制定义或11门槛。
文献/诊断/反证继续见[V1研究材料](../positive_quality_transport/README.md)，不重做文献导入。

本次先保存[工程协议](engineering_protocol.md)，后用合成图像走实际安装包的
train/val get_dataloader和collate；两份既有缓存重放完整loss/局部head autograd。
不新增真实batch、不做检测模型forward/optimizer/学生训练/真实val推理。
这覆盖原失败路径，不代表完整Trainer启动、GPU反传或学生训练已完成验证。

新源码`src/regionssod/training/positive_quality_transport_v2.py`，
新入口`scripts/run_positive_quality_transport_v2.py`。
按[新执行协议](student_protocol.md)手动启动；拒绝覆盖和恢复，无自动重试或扩训。
原V1入口已终态，勿重跑，也不要删除旧失败目录。
