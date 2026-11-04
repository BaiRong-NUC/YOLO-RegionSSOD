# 正目标重排V2：Trainer集成修复先行合同

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate
- Origin Date: 2026-11-04，Asia/Shanghai
- Version Label: positive_quality_transport_trainerfix_v2
- Verification Status: PLANNED（修复与验证前保存）

V1 reference在验证数据集构造阶段失败：借用ReliableCompetitorTrainer.build_dataset，
其中零参数super的词法类与PositiveQualityTrainer实例继承关系不兼容。
0完整epoch，候选/控制未启动；原封印目录、源码和合同保持原样，不能把它当性能失败。
旧935tests/缓存loss/preflight没有覆盖真实Trainer train→val数据集路径，这是验证缺口。

## 修复边界

独立新增positive_quality_transport_v2模块/runner/config/报告；复用原V1 loss与模型，
只修Trainer数据集派发，不改任何科学变量、11门槛、优化/增强、三臂各30轮。
V2在本类中显式构造训练SourceDataset；val显式调用P1Trainer.build_dataset，
不移植其他类中带零参数super的方法。新三臂使用全新目录与COCO初始化，
不恢复旧reference或覆盖失败目录，不自动启动训练/创建监控。

## 零训练验证

1. 必须先有回归反例：原V1真实val方法在无数据访问前即抛TypeError；
   V2三模式都能派发train/val，来源数据集只用于train。
2. 合成磁盘图像/标签放ignored新目录。用V2真实get_dataloader与真实安装包构造train/val，
   并各取一个**合成**batch检查collate/标准验证元数据，不使用新的真实训练/val图像。
   使用stride占位模块，不执行真实检测模型、不创建optimizer、不进学生训练循环。
3. 复用已有初始化/训练后完整head缓存与V1工程回执核对loss/梯度；
   新Trainer.get_model必须创建PositiveQualityModel并绑定三mode，禁止借模型前向验证。
   必要模型构造可用合成类替身检查派发；模型构造检查不是训练性能。
4. 限定V2只改集成层，原loss/model/SourceDataset对象相同；完整pytest。
   新预检必须要求Trainer路径工程证据，不能仅凭旧accepted/AMP回执称就绪。
5. 来源绑定包括V1失败manifest/日志/exit/status和终态只读audit回执，
   新检查/实验输出拒绝覆盖；旧39k/40k来源及产物不改。

本轮真实batch/检测模型forward/optimizer步骤/训练循环/真实val推理均0。
合成batch仅用于发现继承/派发/初始化问题，不提供任何学生AP或机制证据。
