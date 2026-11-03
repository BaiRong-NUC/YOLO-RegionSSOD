# 验证与交接（2026-11-03）

完成：零训练缓存诊断、独立重放、隔离候选/控制实现、CPU工程测试及三臂预检。
未完成：新候选学生训练、实际全量CUDA训练链路验收、学生效果/跨split确认。
本轮未创建新训练监控；旧类别分数实验已结束，本机未发现其临时监控配置。

## 实际执行

| 项目 | 命令/结果 | 范围 |
|---|---|---|
| 起始Git | `git status --short --branch`，main与origin/main一致且干净 | 后续改动均为本轮 |
| 诊断初次计算 | `python scripts/diagnose_competitive_negative.py`，exit0 | 30,565绑定、teacher events重放、六学生各548图/双IoU/三操作点 |
| 诊断独立重放 | 同脚本`--audit`，exit0/JSON精确一致 | 原输入/历史结果未改 |
| 首批候选测试 | `python -m pytest tests/test_competitive_negative.py -q`，6通过 | 负梯度强度匹配、分配/box/正类/人工/背景梯度保护等 |
| 增加lazy criterion/验证路径及三臂决策边界测试后 | `python -m pytest`，**277 passed in 8.78s** | 全仓库测试；不代表真实全量训练验收 |
| 三臂预检 | `python scripts/run_p1_competitive_negative.py --check-only`，exit0 | 一份通过回执`outputs/p1_competitive_negative_preflights/20261103_094620_769748.json`，零优化步骤/零新推理 |
| 文献 | paper-search+官方源+Zotero6条目/附件/新增局部阅读笔记复查 | 无重复导入；PDF结构预检6份均UNAVAILABLE，原因见literature_review.md |
| 格式 | `git diff --cached --check` | 新冻结诊断协议保留原始末尾空行警告；其他空白检查通过 |

新训练入口仅在用户显式执行`--train`时运行；拒绝已有目录/自动恢复。
三臂全部完成后自动548图val960、COCO-style AP_small、PR、来源及数值重放和固定继续线。
共享数值评估器复用原文件；只在新输出目录给新三臂图替换正确标题并重新绑定哈希，
不修改旧数值评估器或原图表。COCO-style仍不冒充VisDrone官方AP。

## 工程验证的准确限制

- 控制臂匹配的是同一logits/分配下移除的负类logit梯度L1，不是BCE或参数梯度/轨迹。
- 同输入的分配/回归/正类梯度等价性不意味着训练分叉后预测和分配完全相同。
- 原class-score sidecar/对象token运输复用既有实现；新损失已测试空/无触发以外的关键梯度边界，
  实际所有增强/CUDA组合仍待真实运行，不以CPU fixture冒充端到端训练通过。
- 不执行短训练作性能判断；至少三臂各完整30轮，任何工程异常不读作算法性能失败。
- 大文件仅保存在ignored目录；原P1、旧候选源码/合同/权重/负结果不覆盖。

已更新论文计划第3.66/10节、README与baseline复现路径；本次使用中文提交并推送。
具体提交及远端结果以最终回复/Git实际状态为准。
