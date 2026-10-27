# RegionSSOD 论文方法与实验初稿

本目录保存基于已验收 B1、B2、P1 结果整理的中文论文初稿。核心结果为：在 10% 标注面积、同一 split/seed、30 epoch 学生对照中，P1 相对 B2 的 mAP50-95 提高 0.335 个百分点，完整原图小目标 recall 提高 1.878 个百分点。

| 文件 | 内容 |
|---|---|
| [methods.md](methods.md) | 第 3 章方法初稿：区域协议、固定教师、尺度筛选、分类软权重、ignore 掩码与损失公式 |
| [experiments.md](experiments.md) | 第 4 章实验分析初稿：实验设置、整体/分尺寸/逐类别结果、伪标签质量和开销 |
| [results_tables.md](results_tables.md) | 表 4-1 至 4-6 和补充表 4-S1，含单位、评测口径和差值附表 |
| [evidence.md](evidence.md) | 表格来源、权重 SHA256、补评估记录和数字核对方式 |
| [related_work.md](related_work.md) | 第 2 章相关工作初稿：SSOD、缺失标注、航拍小目标与最接近工作的关系 |
| [references.md](references.md) | 16 篇核心文献、阅读优先级、引用边界；[references.bib](references.bib) 供文献管理工具导入 |

表格采用三个独立 30 epoch 学生进行主对照。B1 100 epoch 单列为教师和较长训练参考；20 图工程烟测不进入这些结果表。正文中的指标为单 seed 的 Ultralytics 开发结果，小目标诊断 recall 与官方 AP_small/AR500 分别表述。

论文正文围绕方法和实验发现组织，项目 Gate 的验收记录与结果后的工作决策保留在 [paper_completion_plan.md 第 3.8 节](../paper_completion_plan.md)。当前原 Gate 3 数值门槛未达到，用户选择保留正向初步成果并优先写作；正文没有据此改写实验数字或声称通过原门槛。

核对或重新生成表格，在仓库根目录的 `yolo` 环境运行：

```powershell
python scripts/build_paper_tables.py --check
python scripts/build_paper_tables.py
```

`--check` 只核对表格与冻结结果是否一致；第二条命令重新生成表格文本。完整精度数值、输入哈希和生成核对记录位于忽略目录 `outputs/paper_tables_20261027/audit.json`。参考文献已单独整理；初稿尚未覆盖引言与最终排版。S³OD 是最接近的工作，正式写作前应重点核对其全文和方法差异。
