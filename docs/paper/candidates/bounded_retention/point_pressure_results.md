# 真实3U与第9轮EMA压力审查

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent。
- Origin Mode: validate；Origin Date: 2026-11-06；Version Label: pressure_results_v1。
- Verification Status: STOP_INITIAL_3U_LOSS_DEADLINE；压力总体未完成，不是学生性能证据。
- 冻结合同：point_pressure_protocol.md，943c9e9，先于本轮实现与数值检查。
- 范围：初始权重/第9轮EMA的真实3U压力批与第9轮EMA的固定32批，零优化。

## 依据及边界

用户本轮接受时间成本，并授权通过后准备独立训练入口/三臂配置/命令，明确不启动训练。
上轮完整loss旧/新0.9465、2倍速度门未通过的事实不变；本轮没有复测以挑选速度结果。
原第10轮失败在线学生现场未保存，保存的第9轮last.pt是half EMA；
两者不是同一权重。本轮不恢复旧实验，也不将EMA作为未来三臂初始化。
所有源码、原P1、冻结入口及旧产物保护；新输出仅在独立ignored目录。

## 固定检查

真实loader重放原两条完整1414批增强流，逐批核对active_dimensions；
固定closed1316为首个最大维数3U批，核对旧AMP回执tensor SHA、source/occurrence身份和63正框。
为保持历史构造与RNG顺序，构造548图验证loader但不迭代、不推理；不使用验证指标筛选。
两份权重各检查该压力批；第9轮EMA再检查real_03保存并绑定SHA的全部32批。
102次三臂loss/反向的分母不变，失败即停，不拼接后续重试。
统一诊断scale1024、每次loss/反向120秒，无数值/采样容量放宽。

存在3U观测、3U点请求、实际3U候选监督分别记录，不能互相替代。
候选/控制必须同几何/TAL/CIoU/classification；压力批梯度须可区分。
未通过前不生成训练许可或新训练命令。

## 执行与独立复核记录

独立静态复核（同模型子代理，只读；不是跨模型认证）确认：验证loader构造影响历史
Python RNG顺序，须保留构造；EMA本身全部requires_grad=False，故只将其state_dict
严格载入原可训练架构，不能直接用EMA对象替换学生。检查脚本已采用这两种接线。
初始化可能含原生stride probe及verbose THOP代理前向；显式batch计数不包含它们。

复核另指出后验计算计数不等于正权重消费、需要绑定旧初始化；这些按
point_pressure_acceptance_addendum.md（65426a5）独立只读补核，运行内PASSED不单独放行。
该补充在audit_01的source binding阶段冻结，未看到本轮压力loss结果，不改正在运行源码。

`suite_01`：完整`python -m pytest`，2094 passed /168.47秒（外部170.842秒），
包含7项新固定总体/epoch9 EMA/监督计算计数守卫；测试通过不替代真实压力验证。

已执行零训练命令：

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' -u scripts/check_bounded_retention_point_pressure.py --output outputs/bounded_retention_pressure_20261106/audit_01
```

父进程39076、worker492；2026-11-06 20:20:00北京时间启动。
输出保存独立manifest/source_snapshot/不可变heartbeats/process及运行日志。
本命令不是训练命令。

`audit_01`：219.860秒后mosaic_open第0批transport replay mismatch，worker exit2、
父终态WORKER_FAILED；显式forward/loss/backward/optimizer均0。缺少actual维数快照是
首版诊断不足，不能事后补造。进程自然结束，完整失败目录保留，不是容量故障。

`point_pressure_replay_addendum.md`（257052a）先于audit_02冻结修订：
恢复原生回调注册及历史val=True，仅构造不推理；新增原初始参数摘要硬校验、
批次不匹配时保存actual/expected/身份/原始batch、全参数非unused守卫。
独立无模型/无数据探针`registration_rng_01`证实：注册前events未导入，注册恰好调用
一次random.random，调用栈来自events.py:84的session_id；末随机状态与一次draw完全一致。
没有手动丢随机数或搜索seed；真实流是否恢复一致由同一历史逐批/张量哈希决定。

`suite_02`：完整python -m pytest，2094 passed /111.37秒（外部113.192秒）。
`audit_02`父54428、worker2320，于2026-11-06 20:30:21北京时间启动，仍在独立ignored输出。
命令同上，仅输出改为audit_02，不是训练启动。

## 终态：首次真实3U候选loss超时

`audit_02`两条1414批完整重放通过；fixed closed1316、active dimension3、63条正框、
身份和张量SHA全部与历史一致。恢复原生初始化已解决audit_01的重放差异。
初始模型参数摘要也严格匹配历史AMP；第9轮EMA已按SHA载入并保存状态，但**未执行其压力case**。

初始权重/真实压力批：1次显式模型前向完成，原P1完整loss=0.2904秒、
全参数反向=4.7054秒，梯度有限且无unused参数。候选loss阶段于20:43:03.830进入，
到冻结120秒期限仍未进入save_loss；父进程仅终止其拥有的诊断worker2320。
父终态TIMEOUT/exit1，父54428及worker2320均已退出；没有训练进程或计划任务。

| 固定总体 | 本次实际完成 |
|---|---|
| 34个case/34显式前向 | 0个完整case；1次显式前向 |
| 102次完整loss及全模型反向 | 仅原P1的1对；候选loss阶段未完成 |
| 初始及第9轮EMA真实3U | 初始候选超时；第9轮未执行 |
| 第9轮EMA固定32批 | 0/32，按首故障停止 |
| 真实3U正TAL权重消费、候选/控制同几何及不同梯度 | 本次未获证 |
| optimizer/训练/验证推理 | 均0 |

没有保存到候选point_failure、loss结果或内部调用栈，**不能把timeout写成再次point resource cap**，
该阶段还包含criterion调用前后CUDA同步，严格说也不能独立证明criterion本身未返回；
不能判断耗时在点求解、归一化、后验采样还是CUDA等待。保存的request.pt包含完整前向输入、
预测和来源绑定；initial_state.pt、epoch9_state.pt、pressure_batch.pt及P1 loss回执均保留。
强制终止意味着当前case的finally/state恢复未执行，不能声称本次全部state restored；
所有权重文件未写入，进程内状态已随worker退出。

`qualification_01`独立只读检查输出STOP/accepted=false/training_permit=false，因terminal未通过
在加载任何tensor前拒绝。它不替代尚未完成的成功路径数值认证。
`terminal_review_01`另核对374项manifest/source reads的SHA，0差异，并封存已有产物摘要。
两次失败、所有快照、不可变心跳和诊断输入均保留，未拼接为成功。

`suite_03`：新增独立资格checker的43项身份/正TAL权重/EMA/梯度/拒绝错误证据测试后，
完整`python -m pytest` **2137 passed /115.81秒**（外部117.875秒），exit0。
检查源码未在audit_02运行中修改；新的checker仅在该运行结束后执行只读拒绝路径。
unit tests通过不等于实际34case通过，也不认证尚未执行的成功qualification路径。

120秒是预先冻结的工程诊断期限，超时不证明数学机制无效，也不证明换显卡可解决。
用户接受训练时间成本，不等于已经完成可靠性认证；本轮既不放宽期限、资源上限或尺度，
也不重跑数值检查。结论为**训练准备门未通过**，不是性能下降或机制否定。
没有建立新训练Model/Trainer/launcher、三臂配置或训练许可；不给可执行训练命令。
若未来继续，须先单独审查同一保存输入的内部耗时证据与工程约束，不能直接重开三臂。
