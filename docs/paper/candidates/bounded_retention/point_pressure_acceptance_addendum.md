# 压力门独立只读资格审核补充

2026-11-06，在audit_01仍处于loader初始化、未进入压力loss时记录。
943c9e9原合同、固定34次前向和102次loss/反向、全部限额不改。

独立静态复核未发现必现的初始化或scale接线错误，但现有运行内计数
supervision_by_dimension只表示后验项计算过，不能单独证明其TAL权重为正。
因此即使运行终态PASSED，也只是数值执行完成，不自动获得训练入口资格。

增加独立只读审核：利用保存的loss.pt中的geometry.mask/indices、certificates.identity、
assignment.foreground/gt_index/scores与posterior_event_details.identity/anchor_row，
核验两份权重的固定压力case确实存在实际选中、权重有限且严格为正的3U后验项。
不重新计算模型预测/loss或修改权重，不把全批梯度不同当作3U有效消费证明。
若现有证据不足以对应，就不通过，不能用推测补证。

同时核验初始state与旧初始化证据、全部可训练参数非unused且梯度有限、
34case状态恢复、源码及历史证据根SHA。读第9轮EMA不等于恢复原在线学生。
运行内记录的显式batch前向数不含构造器stride probe或verbose的THOP代理前向。
这些构造器行为须披露，不能声称总共只执行34次forward。

现有audit_01源码保持不动、运行不重启；独立审核结果写入新的ignored资格目录。
只有运行与独立审核均通过，才允许准备独立训练入口。单次loss失败/超时仍立即终止全窗口。
本轮不启动训练，不追加样本，不重测选择有利结果。
