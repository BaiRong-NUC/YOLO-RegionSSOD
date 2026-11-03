# 主对照入口验证（2026-11-03）

状态：主对照准备/工程预检完成，未启动训练，未新增真实图像推理。
用户明确希望直观看到基线、P1和现有改进的总成果；当前基线定义为B2全局阈值自训练。

## 已完成

- 起始Git干净；实际未发现竞争负项旧三臂训练进程或运行目录。
- 新增独立B2的30轮配置及`run_p1_main_comparison.py`入口，旧机制入口/合同保留。
- 默认B2/原P1/P1＋竞争负项三个全新学生；可选第四组强度控制不混入主表。
- 预检核对新B2与P1相同教师SHA/训练参数、冻结B2的27667框/4624伪图/11095 tile，
  输入图像和标签逐项绑定；只改变B2配置的名称/学生epochs100→30。
- 输出绝对指标和P1−B2、改进−P1、改进−B2直接差值，显式区分百分数与pp；
  不复用旧学生权重或跨训练长度叠加增益。无控制时机制标记NOT_TESTED。
- 测试覆盖总/新增增量与控制分列、主三臂无机制结论、四组机制失败、不可达点、
  check-only不启动训练、默认/可选四组与B2独立30轮配置。

实际运行：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -m pytest
```

**283 passed in 10.02s**。真实全量CUDA训练/新学生性能仍未验证，测试不能替代它。

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe scripts/run_p1_main_comparison.py --check-only
```

退出0、零优化步骤；一份通过回执为
`outputs/p1_main_comparison_preflights/20261103_100120_958694.json`。
协议/训练入口见[主对照合同](main_comparison_protocol.md)。
README、baseline协议、论文计划第3.67/10节与分轨索引已同步。
本轮中文提交并push，实际提交号与远端状态见最终回复/Git。
