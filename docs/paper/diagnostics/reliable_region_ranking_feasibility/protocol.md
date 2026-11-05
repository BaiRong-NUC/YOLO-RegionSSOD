# 完整人工区域内跨对象排序：来源可行性审查协议

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate（来源与可识别性）
- Origin Date: 2026-11-05，Asia/Shanghai
- Verification Status: PROTOCOL；新增文件盘点/数量统计/合成反例前保存。
- Scope: 延续56e242f的下一项来源审查，非新方法性能实验；不宣称系统综述。

## 唯一问题与先行停止门

现有合法人工标注及已保存来源，能否证明某个非空空间区域内所有前景均已标注，从而把该区域内未匹配位置用于跨对象/跨位置的可靠背景负例？
对象身份/类别可信、某个anchor被TAL分配为正，与空间背景完整性是不同命题。
本轮不证明新的排序损失有效，不将现有BCE的对象级非真类负监督改名为背景证明。

必须同时满足：
1. **G1 原始人工区域可识别：** 实际入训人工图像与人工标签可独立定位，预算/裁片坐标和标签计数一致，不混伪标签。
2. **G2 裁片输出完整性：** 存在合法、可追溯的完整性证据；被裁片过滤的前景足迹已标注或按几何位置完整ignore。单纯未匹配、置信度低或无框都不能证明背景。
3. **G3 增强后证明运输：** 完整/未知区域及被过滤对象足迹能随resize/mosaic/裁剪等运输；越界、丢框与混合来源不能变成可靠负例。

G2/G3任一未证明即STOP。未证明不等于已观测实际漏标，亦不否定原P1已有结果。
停止后不跑排序错误统计/梯度诊断、模型forward、loss实现或训练，不自动切换新方向/新控制。

## 允许与禁止的数据访问

允许：仓库源码/冻结配置/既有合同；`data/visdrone_area10`根目录文件名；
`dataset_p1.yaml`中训练清单定位；`splits/labeled_train.txt`；**仅**`labeled/labels/train/*.txt`人工导出标签；
`summary.json`只投影`area_ratio/seed/min_box_visibility/jitter_fraction`构造参数，不提取任何隐藏计数；
`manifest.jsonl`投影字段`source_image`（仅身份字符串）、`source_size`、`requested_area_ratio`、`actual_area_ratio`、`labeled_tile.{image,tile,labels}`。
manifest容器可能还带既有隐藏计数字段，解析后直接丢弃，不输出/统计/使用；只对完整容器做SHA绑定。
文件系统盘点只列与合法人工来源有关的目录/文件名，不打开未标注标签。
允许读取现有yolo环境的Ultralytics增强源码，但不实例化模型/数据集，不加载图片或head缓存。

禁止：原始全图训练标签（含标注区域外GT）、`unlabeled/labels/`、任何hidden-GT诊断/计数/框、val GT或预测、
已有真实batch/head缓存和训练后权重；不重新生成标注、筛选图片/类别、读取全图GT后再按区域过滤。
不调用旧preflight/train/terminal-audit，不改原P1、冻结实现/合同/封印输出。

## 事先固定的审查方法

- 逐项检查人工清单/manifest投影/人工标签的身份、数目、路径白名单和格式，绑定只读输入SHA。
- 只统计实际合法人工tile与标签数量、裁片面积比范围、可见阈值配置、已保存完整性证明字段是否存在。
- 审读裁片过滤和保存字段，以及P1/SourceDataset/安装增强对裁剪丢框与来源的处理；字段缺失不能用函数名猜测补齐。
- 只用合成几何，检验“相同保留标签可否兼容额外但被过滤的前景”；保留一个全框被裁丢却覆盖tile内空间的反例即可否定普遍背景保证。合成框不是观测GT，不估计实际漏标率。
- 不能靠边缘侵蚀宽度、最大目标尺寸先验、人工对象可信或伪框ignore来填补未保存的空间证据；不得在本次结果后换筛选规则找可过线子集。
- 如已到STOP门，后续只作独立反方核对和报告/文档/Git收尾，不进行新统计方向。

## 产物与验证

新脚本/报告分别放独立路径；机器回执仅写新ignored `outputs/reliable_region_ranking_feasibility_20261105/`，已存在即拒绝覆盖。
回执记录允许输入的哈希、访问类别、合成反例、G1/G2/G3结论与停止决定；`--audit`仅重放本审查，不触碰旧终态审计。
新增源码后用既有yolo Python跑全套pytest。报告包含ARS 11类解释检查、读到的事实与推论边界、未执行项目。
最后更新论文计划第10节，中文commit并push；不创建训练监控。
