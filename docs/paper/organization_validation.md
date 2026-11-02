# 2026-11-02 成果保存与论文整理验收

状态：本轮整理、保存和工程验证完成。候选分类改进仍未实现或训练。

## 已通过

- `python -m pytest --basetemp outputs/pytest_paper1_preservation_final_20261102`：
  **208 passed**。新增测试覆盖归档被破坏、原件后续变化、拒绝覆盖、自身目录排除，
  以及新审计回执隔离、拒绝非预期写入、异常后恢复写入函数。
- `build_paper_tables.py --check`：原137来源及8组recall通过；
  `build_small_pr_results.py --check`、两个诊断`--check-doc`通过。
- 新正文`main_results.md`的12行主表/汇总数字对照冻结JSON通过；
  新整理文档33个本地链接及显式代码路径检查通过。
  回执：`outputs/paper1_writeup_20261102/organization_audit.json`。
- 本地58,707文件成果快照创建后，归档内哈希及原件哈希逐项一致；
  Git保存小型[校验回执](preservation_receipt.json)，不上传实验大文件。
- AP/PR完整缓存重放通过1119来源；高精度诊断完整数值/表格重放通过；
  尺度分析在原回执精确恢复后重放通过。新审计入口按阶段写独立回执，
  原评估程序及P1训练源码/配置未改动。

## 保留的工程问题与验收边界

首次直接审计和新入口第一次整体重放，在尺度阶段遇到上游回执时间戳哈希不一致；
不是性能退化或预测改变。第一轮新入口的small-pr和high-precision阶段已成功，
整体`run.json`仍保留`accepted=false`，不伪改为全程成功。
恢复后仅重跑依赖该回执的scale阶段，成功回执单独保存：

- `outputs/paper1_reaudits/20261102_preservation_check/{small-pr,high-precision}/independent_audit.json`
- `outputs/paper1_reaudits/20261102_scale_after_receipt_recovery/run.json`
- 原/新回执及精确恢复说明：
  `outputs/paper1_writeup_20261102/audit_receipt_recovery/`

恢复依据是原完成记录、原序列化与下游既有SHA256逐字节吻合，
没有修改下游预期哈希，也没有改变科学规则，详见[保存说明](preservation.md)。
尺度审计的独立重放仅覆盖摘要，不含完整fold记录列表；
其记录文件由产物哈希保护。该限制不能写成独立实现全面验证。

## 文稿检查与待完成项

六章正文和现有参考文献拼接保存于
`outputs/paper1_writeup_20261102/phase4_existing_results/draft.md`，
仅作本轮质量检查副本；可持续编辑入口仍为`docs/paper/README.md`指向的各章节。

ARS缩写检查已执行，完整报告：
`outputs/paper1_writeup_20261102/acronym_report.md`。
其覆盖为`body (partial)`，未检查英文/中文摘要；STAC、EMA、GT等仍有定义提示，
并存在S³OD/YOLO模型名、数字指标和参考文献书写被当缩写的误识别。
已修正新增正文中的IoU/PR首次定义问题，不把剩余报告解释为“排版已通过”。

目前没有摘要、最终引用编号/缩写统一、目标期刊合规或独立审稿结论。
本轮MCP文献核对不是穷尽性新颖性认证；候选方向也没有新性能结果。
下一步可继续写作，并独立准备原P1之上的教师类别歧义诊断。
