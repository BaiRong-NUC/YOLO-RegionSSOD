# V2零训练集成修复：工程记录

- Origin Skill / Mode: academic-research-suite / experiment-agent validate
- Origin Date: 2026-11-04，Asia/Shanghai
- Version Label: positive_quality_transport_trainerfix_v2
- Verification Status: 1088tests与独立工程验证通过；学生NOT_TESTED，机制NOT_ESTABLISHED

## 已确认

基点d6b95d1，工作区干净、无正目标三臂训练进程后开始。
原V1 reference验证数据集构造的TypeError已在合成fixture稳定复现。
V2只新增Trainer.build_dataset显式派发，get_model/构造器/模型/loss仍继承V1。
训练SourceDataset构造体与冻结旧实现的AST一致性检查通过。
独立runner/config/输出保留同11门槛、推理/完整评估和终态audit，worker导入V2。
所有原P1、V1、旧候选源码/合同、失败封印均未修改。

三模式各通过一个合成train及val batch的实际DataLoader派发检查，共六批；
模型工厂仅验证替身下的mode绑定。训练为SourceDataset，验证为YOLODataset，
shape分别[4,3,64,64]与[1,3,96,96]（真实默认rect pad .5），验证不依赖训练sidecar。
人工与confidence=1伪对象都在合成fixture中，来源不由权重判定。

两份既有完整head缓存经原check_cache重放CUDA AMP损失/局部head autograd：
TAL、原分母、box/DFL/其梯度、参考项、伪来源、负项、未触发项继续一致；
候选保目标值集合，控制的局部搬移L1浮点容差检查通过。
这是冻结loss在保存张量上的工程证据，不是学生性能或机制成立。
没有新增真实batch/检测模型forward/optimizer/学生循环/真实val推理。

`python -B -m pytest -q`完整通过；单独collect-only确认1088项，非只跑新测试。
新pipeline10项与独立runner143项覆盖原错误反例、模式/派发、科学配置/门槛等价、
工程receipt缺失/伪造、覆盖拒绝、workerV2导入、权重/预测绑定、只读终态、V1失败保护。
独立只读复核未发现明确新启动blocker；工程边界如下。

## 回执和复核命令

- `outputs/positive_quality_transport_v2_engineering/validation.json`：accepted=true，
  trainer_pipeline_checks=true，完整源码/两缓存/V1工程与失败封印SHA绑定。
- 同目录`trainer_pipeline.json`与`synthetic_fixture/`：实际六批的来源与shape，原反例字符串。
- 外部`outputs/positive_quality_transport_v2_validation_20261104.log`：生产命令exit0。
- 外部`outputs/positive_quality_transport_v2_collection_20261104.log`：1088项收集记录。
- 新预检`outputs/positive_quality_transport_v2_preflight_20261104.log`：exit0、accepted=true，
  40267项绑定、11310训练图；training/forward/created_files=false。
  只读核对旧失败完整清单、原输入、exit1和外部audit，不重新运行旧audit。
  本记录的结果/状态随后写入，用户启动时入口会对最终文档/commit重新生成完整preflight。

工程脚本已执行一次，独占目录存在即拒绝重跑；不要求用户再次执行它。
下一步手动命令是[student_protocol](student_protocol.md)里的check-only→train。
新check-only不创建实验目录或模型/数据集，不重复缓存重放。
旧V1入口/失败目录不要重跑、删除或覆盖。

## 不支持的结论

Trainer检查使用object.__new__、stride占位模块与模型构造替身，未覆盖完整Trainer构造/
setup、真实权重加载、AMP bootstrap、优化器或训练循环。
receipt的new_dataloader_batches=0是新增真实数据batch为0，合成batch另记6；
parameter_hash_unchanged沿用旧capture哈希证据，本次未构造并核验真实检测模型。
未测新方法吞吐/完整30轮收敛/AP、AP_small或保覆盖收益，不保证后续训练无运行错误。
V1阶段框外目标质量增加、控制熵/改变锚点数混杂等反证没有因修复消失。
