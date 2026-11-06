# 120秒loss超时的独立定位合同

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；Mode: validate。
- Date: 2026-11-06；Version: timeout_diagnostic_v1；Status: UNVERIFIED。
- 起点0c747c3；仅定位与合同审查，不修改机制、不训练、不提高任何限额。

## 问题和已有证据

audit_02在initial/pressure/new_candidate的loss阶段120秒未到save_loss。
阶段含criterion前后CUDA同步，旧证据不能区分同步、点求解、积分和采样。
120秒来自c24603d容量审查合同：单个合成求解请求及整批loss均采用120秒，
943c9e9压力合同沿用。它是额外诊断保护，不是数学机制、原P1或训练协议推导出的上界。
本轮不直接改该期限；不把诊断timeout解释为数学/性能否定。

## 固定输入和执行

只读audit_02保存的01_initial_pressure/request.pt、initial_state.pt、trainer/args.yaml、
压力batch和P1 loss回执以及绑定manifest。先检查历史terminal/产物摘要/源码SHA。
不扫描原图或隐藏GT，不跑loader，不重新选择批次，不读取第9轮数据进行新数值检查。
保存的预测已经包含原模型前向结果。本次用其原值及原归一化batch重放完整criterion，
仅在boxes/scores建立叶子梯度；不执行新检测模型前向、全模型反向、optimizer、验证或训练。
criterion所需元数据视图取原args、nc=10/reg_max=16/stride=8,16,32和P1设置，
核对保存的预测形状。先原P1完整loss/assignment与旧产物逐位比较；不一致则全停。
此检查不是Trainer集成或全模型反向认证。

原P1校准通过后，**候选、强控制各一次，顺序执行**，均为同一保存预测、观测、
point_search_v2、连续C/名义E/TAL/CIoU及原固定数值预算；候选仍是原后验目标，
控制仍是同界最强连续点DFL。每臂新进程、冷数值缓存、同随机状态/调用序号1、相同AMP。
每臂loss阶段总计120秒，细粒度日志不得续期。初始化另设120秒保护，不包含数值loss。
仅允许本轮父进程终止自己创建的诊断子进程；发出明确预告、保留pid/命令/终态。
只在候选timeout时继续预声明的控制对照以定位共享/独有阶段；这不是重试或跳过失败求均值。
其他异常或输入/校准不一致停止；不得改参再跑。控制timeout也直接收尾。

## 观测和公平性

新增独立脚本，通过临时包装原函数记录enter/exit/error、wall/process CPU时间：
CUDA调用前同步、criterion、调用后同步；continuous_support与每次点求解；
normalizer/连续区间积分；sample_posterior及其_partition。
使用原函数返回值，不改分支、随机数、proposal、cell、bits、误差或采样数。
不得为计时在每个内部函数额外插入CUDA同步。内部GPU时间只能作为含隐式同步的墙钟描述。
CPU既有exact_logits/几何信息用于记录请求指纹，避免观测新增GPU拷贝。
每20秒faulthandler只写调用栈，不重置120秒deadline；父进程观察进程存活。
包装和日志存在开销，禁止据此宣称算法提速或精确重现原运行耗时。
累计调用、最后未返回阶段、多次栈和CPU时间共同用于定位；只见一次栈不能声称耗时全在该行。
候选/控制完整几何若未产生则只比较已完成共享前缀，不宣称全部强控制公平性已认证。

## 终态交付

保留旧失败及本轮全部完成/失败/超时，不拼成压力门通过，不生成训练入口或命令。
报告120秒属于外加工程保护，以及是否有事实支持它在此处过严；未完成请求的完成时间保持未知。
区分单请求超时、累计多请求、CPU计算与CUDA等待。若需进一步诊断，先交付实际阻塞，不自动扩预算。
完整pytest；输出仅outputs/bounded_retention_timeout_20261106/新目录。
更新论文计划第10节/必要入口，中文commit并push。无需重新询问已授权的诊断。
