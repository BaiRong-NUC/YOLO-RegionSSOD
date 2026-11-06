# 历史增强流重放的初始化修订

2026-11-06，audit_01终止后、audit_02之前冻结。原固定总体、loss/反向限额、机制不改。

audit_01父39076/worker492于20:20启动，219.860秒后在mosaic_open第0批active_dimensions
与历史不一致而exit2；显式batch forward=0、loss=0、optimizer=0。
该失败不能解释为点求解容量故障。未记录actual dimensions是首版诊断不足，原文件保留。

已确认新包装层禁用add_integration_callbacks，而历史入口采用原生注册。
原生注册在Trainer.init_seeds之后导入ultralytics.utils.events；其模块级Events()创建
metadata.session_id时执行random.random()。新包装层少了该调用，会改变后续增强RNG。
这证明存在初始化路径差异，但是否足以解释全部重放差异仍须原固定哈希验证。

修订仅恢复原生回调注册及历史val=True配置（仍不迭代val、不调用get_validator或训练）；
禁止train/build_optimizer的守卫保留。不手动丢随机数、不搜索seed、不选择新压力批。
增加历史初始化parameter_digest断言，新增不匹配时的actual/expected维数、身份和批次保存。
所有参数必须非unused且梯度有限。原来的独立正权重3U审核仍必须通过。

源码修改后重新全库pytest；通过后在全新audit_02执行相同固定重放和34case。
不接续audit_01、不拼接回执。若仍不能匹配且无证据支持进一步修复，停止并报告来源/重放阻塞。
任何实际loss/反向容量或非有限故障仍立即停止，不调整数值预算或尺度。
本轮不启动训练，所有失败保留。
