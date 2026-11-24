# 项目测试

| 目录 | 测试内容 |
|---|---|
| [reproduce](reproduce/) | 外部论文专用测试，按论文简称分目录 |
| [data](data/) | 伪标签一致性、候选保留、忽略区域恢复、定位质量 |
| [training](training/) | P1 损失与梯度、NWD、配对种子、消融判定 |
| [evaluation](evaluation/) | 检测案例、误检诊断、精确率与召回率、操作点 |
| [paper](paper/) | 正式实验、恢复、证据保存和冻结结果重放 |
| [project](project/) | 仓库配置与忽略规则 |

在项目根目录执行 `python -m pytest` 可运行全套，`python -m pytest tests/evaluation` 只运行评测组。顶层的八个测试文件由历史实验回执按原路径和内容哈希引用。
