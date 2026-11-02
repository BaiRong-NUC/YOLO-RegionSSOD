# 第一篇文献地图与Zotero核对

日期：2026-11-02。本轮使用 paper-search MCP 做有界题名检索，
使用 Zotero MCP 核对 `Research / YOLO-RegionSSOD`（`BMQTT9Y7`）项目集合。
该集合返回33项；本轮采用的下列条目均已在集合中，复用原条目，没有重复导入。
集合内的DINOv3、剪枝、LoRA等远期材料不作为第一篇核心依据。

## 1. 文献服务于哪一部分

| 论文 | Zotero key | 本篇用途 | 阅读/核对范围 |
|---|---|---|---|
| S³OD: Size-unbiased semi-supervised object detection in aerial images (2025) | `MDFG3JPK` | 最近邻；约束尺度阈值/负学习的新颖性表述 | 本轮元数据/集合核对；既有[版本及代码对照](s3od_comparison.md)，不追加全文已读声明 |
| Humble Teachers Teach Better Students for Semi-Supervised Object Detection (CVPR2021) | `BKDRLGZS` | 类别分布/软伪标签先例；约束新候选创新说法 | 本轮arXiv v1摘要与Zotero元数据；已有PDF不代表本轮全文阅读 |
| End-to-End Semi-Supervised Object Detection with Soft Teacher (ICCV2021) | `5C8LJIIE` | 教师信息在不同监督项的使用位置 | 本轮集合核对；既有[相关工作](related_work.md) |
| Unbiased Teacher v2 (CVPR2022) | `R6AMZBIU` | 分类分数与回归质量不能互换 | 本轮集合核对；原局部阅读范围见[补评估文献报告](small_pr_literature.md) |
| Calibrated Teacher for Sparsely Annotated Object Detection (2023预印本) | `KV74E49Y` | 区分置信度校准、分类loss权重与类别歧义 | 本轮arXiv v1摘要；既有局部阅读范围仍有效 |
| Multivariate Confidence Calibration for Object Detection (2020) | `RQIKGRP9` | 尺度/位置条件校准已有先例 | 本轮集合核对；原局部阅读范围见补评估报告 |
| On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines (2024) | `K3VYDTZE` | 校准评测与留出划分；防止把开发集拟合当部署证据 | 本轮集合核对；原局部阅读范围见补评估报告 |
| STOD: toward semi-supervised tiny object detection (2024) | `HBL9PADC` | SSOD小目标质量监督近邻 | 仍仅摘要/元数据，无PDF；不据此证明新颖性 |
| Microsoft COCO: Common Objects in Context (2014) | `B22YC6DY` | COCO-style面积分组指标的背景 | 本轮集合核对；实际评测仍由项目协议定义 |
| VisDrone-DET2019 challenge results | `CMNDTT9F` | 数据与官方评测语境 | 本轮集合核对；不能把当前YOLO标签评测冒充官方指标 |

以上不构成系统综述或穷尽性新颖性检索。`human-read`状态不由助手代填。
现有正文编号及BibTeX保留；排版阶段再统一首次引用顺序。

## 2. 本轮检索回执摘要

- MCP：`search_arxiv`，查询
  `ti:"Humble Teachers Teach Better Students" OR ti:"S3OD" OR ti:"Calibrated Teacher"`，
  `max_results=8`，返回4项。
- 纳入已有文献定位：`2106.10456v1`（Humble Teacher）、`2303.07582v1`（Calibrated Teacher）。
  前者摘要明确提到EMA教师、丰富region proposals与软伪标签；
  后者摘要明确提到校准教师及Focal IoU Weight。只据摘要作这些机制的存在性描述。
- 排除：`2609.21619v1`是语言模型on-policy distillation，与当前检测问题无直接对应；
  `2510.21605v3`的S3OD是salient object detection，与航拍SSOD的S³OD不是同一工作。
- 精确补检 `ti:"Size-unbiased" AND ti:"object detection"` 返回0项；
  零结果不是论文不存在。Zotero `MDFG3JPK`元数据记录正式期刊 DOI
  `10.1016/j.isprsjprs.2025.01.037`、卷221、页179–192。
  早期预印本另题为 *Rethinking Scale Imbalance in Semi-supervised Object Detection for Aerial Images*
  (`2310.14718v1`)，不同版本不得互相代替引用公式。
- `zotero_get_collection_items(BMQTT9Y7, limit=100)`返回33项；
  `zotero_get_item_metadata`对S³OD与Humble Teacher进一步确认集合字段均含`BMQTT9Y7`。
- 本轮网页工具尝试补取官方页/预印本，没有返回可读取内容；不记为网页全文核验通过。
  本轮证据为MCP返回摘要/元数据与已有版本限定报告。

原始来源入口：
[Humble Teacher CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Humble_Teachers_Teach_Better_Students_for_Semi-Supervised_Object_Detection_CVPR_2021_paper.html)、
[Humble Teacher v1](https://arxiv.org/abs/2106.10456v1)、
[Calibrated Teacher v1](https://arxiv.org/abs/2303.07582v1)、
[S³OD期刊 DOI](https://doi.org/10.1016/j.isprsjprs.2025.01.037)。

## 3. 对叠加候选的实际约束

先保留原P1的尺度筛选、c权重和ignore，再检查完整类别分数是否提供超出top1分数的
信息，并确定它改变哪一项监督。只有“换成soft label”不够作为独立创新；
只看高精度wrong-class比例也不能证明该改动会奏效。
即使候选最终无效，上表和既有负向诊断仍能用于第一篇的相关工作、讨论与限制。
