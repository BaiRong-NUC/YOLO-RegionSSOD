# 连续点修复：真实3U与第9轮EMA零训练压力合同

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent。
- Origin Mode: validate；Origin Date: 2026-11-06；Version Label: pressure_v1。
- Verification Status: UNVERIFIED；先于实现与新数值运行冻结。
- 起点26d0f8d。用户授权：压力验证通过后准备独立入口和三臂命令，**不启动训练**。

## 不变项和解释

沿用原连续C、名义E/原DFL、面积耦合、后验目标、同界最强连续点控制和原P1。
两臂使用point_search_v2同一求解器。20000cells、1e-5gap、K256、16384proposals、
256bits及每次loss/反向120秒不变。用户已接受时间成本；之前2倍速度门失败结论保留，
本次不是重测速度以推翻旧结论，而是新授权的可靠性门。
不读隐藏训练GT/计数/质量缓存，不丢对象，不降级E，不按loss/性能挑批次。

原失败现场无法精确重现。第9轮candidate/weights/last.pt只含保存时half EMA，
SHA256=8b27ba426c82315c62d1120accdb05921a2d8af0427012a84ef277a963a8b56e；
只读取epoch、EMA、类别和必要架构信息，转换float用于相同AMP前向，
不恢复优化器、scheduler、scaler、训练游标或历史训练。检查epoch=8（零基第9轮）。
EMA与在线学生及后续权重不同，结果不能保证未来所有请求都可解。

## 固定总体与执行顺序

1. 合法原loader重放seed0、batch8、640、workers0的完整open/closed流；
   依据已冻结native_e_integration_01的每批active_dimensions逐批核验，
   固定首个最大维数批closed1316（3U）。对照native_amp_audit_01的batch tensor SHA、
   source/occurrence身份和63条正框；不重新选批。重放无loss或模型前向。
2. 保存该完整批次，依次用原初始权重、第9轮EMA运行P1/新候选/新控制三臂完整loss及全模型反向。
3. 第9轮EMA继续检查上一轮real_03已保存的固定open00..15、closed00..15。
   读取前核对每个batch.pt与其已完成result.json的SHA；不是选取快批或只重放失败点。
   共34次显式前向、102次三臂loss/反向；原构造器stride探测另披露。
4. 所有三臂共享同一前向、同一固定梯度scale1024（沿用已验证压力批诊断尺度），
   比较时除回scale；不依据本次结果搜索scale。无GradScaler状态恢复、无optimizer.step，
   不把诊断1024写进未来训练超参数。非有限loss/梯度停止，不通过调scale重试。
5. 冻结前向前模型状态，完成全部反向后恢复BN/其他buffer，逐批及最终SHA认证。
   每臂数值缓存冷清理，源代码前后hash；完整输入/预测/身份/参数状态在loss前保存。
   额外记录3U实际求解请求和后验监督暴露，不能以存在3U观测替代实际消费。
6. 每次loss/反向120秒硬期限；只终止本次拥有的诊断子进程。初始化及loader每批真实进展心跳，
   心跳不可变编号文件，不在loss内部续心跳。失败停止后续全部检查，无自动重试。

## 门槛和交付

必须全部完成且无容量/空集/非有限异常；两个压力case都有实际3U点请求及候选3U监督消费；
候选/控制同geometry/TAL/CIoU/classification，压力case梯度可区分；原P1无机制替换。
全部参数梯度有限、buffer/参数恢复、合法来源/输入/代码hash一致。全库python -m pytest通过。
任何失败保留现场，停止，不接入训练入口，不给出可执行训练许可。

通过后另建训练Model/Trainer/launcher，保护原V6入口与旧输出。冻结原P1/候选/控制各完整30轮，
顺序执行，均从原COCO权重开始；第9轮EMA只属诊断，不作为新实验初始化或resume。
继承原548图full-image val960、AP/AP_small、召回、同K/同FP预算、重复框与定位评估，
继承旧失败策略；固定配置/代码/证据hash、独立ignored输出，只给命令并做只读--check，**不运行--train**。
若入口实现后新增源码，必须重新认证相关接线/初始化和完整pytest，不声称检查过未运行的新路径。
旧64.63小时仅描述性成本，不能作为可靠期限。无性能/机制成立/原创性结论。

本轮输出root仅outputs/bounded_retention_pressure_20261106/；每次子目录必须全新。
完整回执/失败证据保留，最终更新论文计划第10节和入口，中文commit并push。
