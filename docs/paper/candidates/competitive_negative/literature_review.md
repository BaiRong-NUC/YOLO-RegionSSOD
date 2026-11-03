# 竞争负项候选：近邻差异与阅读记录

日期2026-11-03。使用项目academic-research-suite、paper-search MCP、Zotero MCP和官方来源。
本轮是定向、有界检索，不是系统综述或穷尽的新颖性认证。

## 1. 已建立的先例与尚未建立的差异

| 工作/版本 | 实际核对到的机制 | 与本候选的区别及限制 |
|---|---|---|
| [S³OD早期预印本2310.14718v1](https://arxiv.org/abs/2310.14718v1)，pp5–6 | 尺度阈值、尺度分配、教师引导负学习TNL | 本候选处理已接受伪正anchor内部的竞争类别负BCE，不复现旋转RoI/TNL；不能宣称尺度阈值或教师负监督首创。2025期刊版与公开代码区别继续见[s3od_comparison](../../s3od_comparison.md)，本轮未读期刊全文 |
| [Humble Teacher 2106.10456v1](https://arxiv.org/abs/2106.10456v1)，pp3–5，p4图像复核 | 大量教师proposals、类别分布软一致性、回归一致性、EMA与翻转ensemble | 不能把soft label本身作为创新；本候选保留硬正目标、仅负项乘数变动，不把sigmoid归一化当校准后验；不是该Faster R-CNN KL框架的等价复现 |
| [Soft Teacher 2106.09018v3](https://arxiv.org/abs/2106.09018v3)，pp3–5 | 教师背景可靠性加权学生负proposal；另用box jittering估计回归可靠性 | 与前景伪正anchor内“非top1类负项”位置不同；本YOLO无显式背景softmax，不可直接以1−max(s)替代背景概率 |
| [Credible Teacher 2401.00695](https://arxiv.org/abs/2401.00695v2)，附件pp1–5文本 | 学生proposals回询教师；按类别将可信输出硬化为0/1、不确定输出保留教师分数；另有DBN | 最直接的不确定类别监督先例。当前候选不改正目标、无proposal回询/在线EMA/DBN、以强度控制检验负项方向。但这些实现差别**不足以证明算法新颖性**，也不是Credible Teacher复现 |
| [LabelMatch 2206.06608v1](https://arxiv.org/abs/2206.06608v1)，pp3–5 | 可靠伪标签用硬监督，不确定标签用proposal self-assignment/软分类；ACT分布阈值及可靠性挖掘 | 已有可靠/不确定监督分治。其类别分布匹配依赖标注/未标注同分布，目标中心区域采样不应未经检查就满足此假定；当前候选不引入ACT/RPLM |
| [Soft Sampling 1806.06986](https://arxiv.org/abs/1806.06986)，附件pp2–4 | 漏标使前景被当背景；全忽略会损失有用真负，教师分数抑制也有错误忽略背景风险 | 为部分标注负监督的风险提供依据；未标注目标空间负anchor与已接受伪正框内的竞争类别是不同问题。不能称候选解决全部漏标 |

精确式子只在清晰来源中采用。Humble Teacher p4已用渲染图像检查式(3)–(4)及相关文字；
其余PDF文本中有公式错位，因此本轮表格只采用可读方法文字，不新增精确公式声称。
候选的强度控制公式来自本项目BCE导数及自动微分测试，不冒充文献原式。

原P1覆盖小目标、此候选限制分类负监督的位置，机制上可以互补；是否实际互补需要
新三臂30轮。最近邻未报告此精确组合，并不意味着已经证明首创。

## 2. Zotero、元数据和阅读范围

集合`BMQTT9Y7`本轮返回35项。上述6篇均已在集合并有PDF，复用原条目，
没有重复导入或把“已有附件”写成“此前全文已读”。

| Zotero key | 附件/实际范围 | 本轮新增机器阅读笔记 |
|---|---|---|
| MDFG3JPK | D6KACD7H；明确为2023早期预印本，pp5–6文本 | GGAHDYKT |
| BKDRLGZS | 4PQM3BES；v1 pp3–5文本+p4图像 | KFKNWAFJ |
| 5C8LJIIE | 3BQA6MIF；v3 pp3–5文本 | HPDF4DZ7 |
| RZDQGRYI | 2JB7JL38；pp1–5文本，未逐字节绑定在线v2 | XRUNYH7D |
| 3D34MMIT | DX4EP5V4；pp3–5文本，arXiv v1元数据 | 6X22IIIA |
| VLH8HGFG | 既有PDF pp2–4文本；元数据年份沿用库记录，未额外确认最终年 | PIJDKETE |

均记录为机器局部阅读，不代替作者/用户的human-read声明。
ARS PDF结构预检另有6份sidecar保存在
`outputs/p1_competitive_negative_literature_20261103/`：当前yolo环境无pypdf，
均为`UNAVAILABLE`，不是PASS，也不是文件损坏证据。Windows的`--output`还因
POSIX dirfd能力缺失被拒绝，因此改用脚本stdout原样保存JSON，未修改技能/安装依赖。
页码范围来自Zotero MCP返回的页序，本轮未独立确认PDF页树结构；
Humble Teacher所引用方法页另实际看过页面图像，但不将图像阅读提升为结构预检通过。
Credible Teacher、LabelMatch已从“只有检索入库”推进到以上具体方法段阅读，
但不声称已完成表格/补充/公式和所有实验的全文核验。

paper-search arXiv题名检索核对Humble Teacher、LabelMatch、S³OD早期版本和
Credible Teacher的题名、作者、时间/版本；Zotero元数据及附件/集合逐项核对。
官方arXiv[Credible Teacher v2](https://arxiv.org/abs/2401.00695v2)也确认作者与2024-01-03版本；
LabelMatch的[CVF出版页](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Label_Matching_Semi-Supervised_Object_Detection_CVPR_2022_paper.html)
搜索返回确认CVPR2022、14381–14390页，直接open为403，保留这一访问限制；
本轮方法阅读仍引用已读arXiv附件，不冒充已读正式版PDF。

## 3. 检索限制

- 主题检索`semi supervised object detection ambiguous negative class supervision partial labels`
  以及`negative class uncertainty semi-supervised object detection multi-label ambiguous`，
  sources=arxiv,semantic，max_results_per_source分别5/4。
  Semantic Scholar返回429；后一个arXiv返回0，不解释为无相关研究。
- `ti:"Credible Teacher" OR ti:"Partial Label" AND all:"object detection"`返回Credible
  Teacher和两篇其他部分标签/多任务记录；后两篇仅摘要筛查，未采用作本候选依据。
- `ti:"Negative" AND all:"semi-supervised object detection"`无返回，不能据此声明新颖。
- 联合题名查询Humble Teachers、Label Matching、Rethinking Scale Imbalance返回三项，
  与现有库去重；相关论文原PDF均存在，无需再上传重复附件。
- 网页广义检索的多标签识别/综述结果未作为技术依据；只采用上表原始论文。

尚缺专门覆盖“逐类互补标签/候选标签集/负项掩码”的完整引用链检索，以及同协议近邻对照。
即使新候选成功，贡献也应先表述为在特定区域局部标注协议下经控制验证的监督设计，
不使用“首次类别负监督”“首次忽略不确定类别”或“优于S³OD”等表述。
