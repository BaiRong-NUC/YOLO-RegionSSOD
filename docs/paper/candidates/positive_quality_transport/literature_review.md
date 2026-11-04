# 正目标质量排序：原文、作者代码及差异核验

## Material Passport

- Origin Skill / Mode: academic-research-suite / deep-research source verification + synthesis
- Origin Date: 2026-11-04，Asia/Shanghai
- Verification Status: 下列指定版本/片段已机器部分阅读；非用户全文已读

沿用并重新读取项目[related_work](../../related_work.md)、[references](../../references.md)及
[S³OD差异材料](../../s3od_comparison.md)。不把S³OD期刊版与早期预印本当同一实现：
正式S³OD为Zhang等，ISPRS Journal 221(2025):179–192，
[DOI 10.1016/j.isprsjprs.2025.01.037](https://doi.org/10.1016/j.isprsjprs.2025.01.037)；
具体SAT/SLA/TNL对照来自已有2310.14718v1及公开代码记录，本轮未取得期刊全文。
尺度阈值、教师负学习、困难负样本和软标签均不是本项目首创。

本轮原始机制核验由ARS source_verification原生子代理完成，主会话结合实际P1/TAL源码
整合；另有devil's advocate两阶段反方审阅。角色分工不是跨模型独立验证。
以下只声明实际读取范围，不将摘要/抽取全文文件存在称为全文读完。

## 1. TOOD：P1本来已经质量对齐

Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R. Scott, Weilin Huang.
*TOOD: Task-aligned One-stage Object Detection*. ICCV 2021。
[会议DOI](https://doi.org/10.1109/ICCV48922.2021.00349)，
实际读[arXiv:2108.07755v3](https://arxiv.org/abs/2108.07755v3)（2021-08-28修订）
PDF pp.5–6 §3.2/式9–12及p.11补充实现；未读全部实验/全文。
元数据经arXiv/DOI核对，CVF访问403时未用其他版本冒充会议正文。

原文用t=s^alpha u^beta，按实例max归一化，使同对象最大软目标的数值等于最大IoU值；
最大目标所在锚点不必就是最大IoU所在锚点。
分类用task-aligned focal loss，回归也按质量加权。
作者代码commit `1bda27c6f290c5b828c566eb95ea50fed75a53cb`：
[assigner](https://github.com/fcjian/TOOD/blob/1bda27c6f290c5b828c566eb95ea50fed75a53cb/mmdet/core/bbox/assigners/task_aligned_assigner.py#L79)、
[归一化L678–689](https://github.com/fcjian/TOOD/blob/1bda27c6f290c5b828c566eb95ea50fed75a53cb/mmdet/models/dense_heads/tood_head.py#L678)、
[分类损失L34–50](https://github.com/fcjian/TOOD/blob/1bda27c6f290c5b828c566eb95ea50fed75a53cb/mmdet/models/losses/task_aligned_focal_loss.py#L34)。
补充原文冲突写最小面积，所读代码用最大IoU，版本差异保留。

**代码事实**：本机Ultralytics 8.4.172 TAL用alpha=.5/beta=6、非负CIoU与归一化；
P1直接用target_scores作BCE目标，再乘教师置信度/ignore。它已具质量对齐，
不能把新方案写成首次利用定位质量。**假设差异**：新方案不重新计算目标值、分配或回归权重，
仅把人工同对象原目标值集合按普通IoU排列；它不是完整TOOD/T-head复现。
普通IoU与CIoU差异及去除分类分数对次序的影响同时存在，不能单独归因。

Zotero原条目7D3IXLK2已在BMQTT9Y7，无重复导入；本轮机器部分阅读笔记D3WYFQC4。

## 2. VarifocalNet：直接IoU目标也会压低正分数

Haoyang Zhang, Ying Wang, Feras Dayoub, Niko Sünderhauf.
*VarifocalNet: An IoU-aware Dense Object Detector*. CVPR 2021。
[DOI](https://doi.org/10.1109/CVPR46437.2021.00841)；
实际读[arXiv:2008.13367v2](https://arxiv.org/abs/2008.13367v2)（2021-03-04修订，初稿2020）
PDF p.4 §4.1–4.2式2，pp.6/8组件消融和迁移段；未全文读完。

式2正项为q×BCE(p,q)，q为当前预测框对GT的IoU；负项采用focal权重。
作者代码commit `f459696240b4442b75bcebbbe4f0910ee4e0f66b`：
[VFL L40–55](https://github.com/hyz-xmaster/VarifocalNet/blob/f459696240b4442b75bcebbbe4f0910ee4e0f66b/mmdet/models/losses/varifocal_loss.py#L40)、
[head L400–452](https://github.com/hyz-xmaster/VarifocalNet/blob/f459696240b4442b75bcebbbe4f0910ee4e0f66b/mmdet/models/dense_heads/vfnet_head.py#L400)，
refined box IoU detach后填真类目标，按正样本数归一化。

**推论**：正项梯度q(p−q)可以压低高于目标的分数，正监督不保证保召回；
student/pseudo IoU仅教师框一致性，不能当真实GT质量。
**差异**：本方案保存P1原目标值与分母，不用IoU直接替换所有目标，不额外乘q或改负项，
也没有VFNet星形特征/框细化分支。原论文消融不能移植为本候选预期收益。

全库按Varifocal检索无精确条目（语义fallback不是命中），新增唯一AM4MTCB9并附PDF到
BMQTT9Y7；元数据回读确认作者/2008.13367v2，机器笔记ZY9XLCK2。
Zotero日期保留预印本初稿2020-08-31，会议2021/v2日期在此分列，不混写。

## 3. Rank & Sort：正样本相对质量排序已有

Kemal Oksuz, Baris Can Cam, Emre Akbas, Sinan Kalkan.
*Rank & Sort Loss for Object Detection and Instance Segmentation*. ICCV 2021。
[DOI](https://doi.org/10.1109/ICCV48922.2021.00300)；
实际读[arXiv:2107.11669v2](https://arxiv.org/abs/2107.11669v2)（2021-08-30修订），
PDF pp.3–5 §3–5式6–12，p.8 §6.3；非全文。

RS把正负排名与正样本内部定位质量排序联合，Identity Update给出更新。
质量较好但低排名者提升、较差高排名者也可被压低，不等于保覆盖。
作者代码commit `b41f64df70b1f8e3260e5c04015e0c7b59b339d9`：
[ranking_losses.py L3–100](https://github.com/kemaloksuz/RankSortLoss/blob/b41f64df70b1f8e3260e5c04015e0c7b59b339d9/mmdet/models/losses/ranking_losses.py#L3)、
[ATSS L204–260](https://github.com/kemaloksuz/RankSortLoss/blob/b41f64df70b1f8e3260e5c04015e0c7b59b339d9/mmdet/models/dense_heads/rank_based_atss_head.py#L204)。
实现跨层/类别展平，用detached IoU，同时改变回归加权/任务平衡。
官方backward使用预存排名+排序联合梯度并仅乘out_grad1、忽略out_grad2，
不能只取第二个sorting_loss就获得独立排序梯度。

**差异**：本方案保留原BCE/回归/TAL，仅人工同对象重分原正目标值，
没有跨类别/背景排名、手写Identity Update或未知区域负例。
若后续成功，仍只能说这一受限适配的实验证据，不是首创quality sorting或复现完整RS。

Zotero已存在RI2S2NJV（原6HT9SNK7），只增添BMQTT9Y7集合成员，无重复导入；笔记FX7RJTPU。

## 4. FSS（2026）：更近的对象内质量排序重叠

Lijuan Huang, Zhixian Liu, Xinyu Zhou, Jinping Liu, Kunyi Zheng, Yimei Yang.
*FSS: Focusing on Suboptimal Samples for Detector-Agnostic Label Assignment in Object Detection*.
Computers, Materials & Continua, 2026, 88(1), 61；2026-05-08刊出，
[DOI 10.32604/cmc.2026.077655](https://doi.org/10.32604/cmc.2026.077655)。
实际机器阅读[出版社正式全文HTML](https://www.techscience.com/cmc/v88n1/67273/html)
§3.1–3.2、式3–9及相关消融/讨论/数据代码声明，核对正式题名、完整作者、年份和版本。
ScienceDirect403、PDF未核验，未找到可确认作者代码，因此实现一致性UNVERIFIED，非全文已读。

FSS以分类/IoU联合指标、Gaussian-prior dynamic-k重新选正样本，并按对象内均值归一化
幂权重调整分类/回归，真类直接使用预测IoU软目标。它已明确研究对象内质量排序。
**差异有限但具体**：本候选固定TAL分配/回归，仅在人标对象内置换原目标值集合，
没有dynamic-k、Gaussian prior、重新生成权重或直接IoU替换目标。
未发现相同受约束置换，不等于新颖性充分；宽泛“对象内排序/质量对齐首次提出”不可用。
其联合分类/几何指标也提醒：移除次序中的分类信号可能损害可学习性。
全库FSS检索无条目后只导入一次正式DOI到BMQTT9Y7，条目TZD6SP4I/PDF已附，
机器部分阅读笔记X6BT8PB4。回读发现Crossref导入卷期0/0、页1–10，
已按出版社修正88(1)、文章61及2026-05-08，不以导入成功替代元数据核验。
附件存在不代表已阅读或核验其PDF内容。

## 5. 搜索边界

另检索2024–2026质量排序/半监督检测近邻以检查较新重叠；搜索结果不是已读文献。
论文事实、当前代码、探索诊断和研究假设已分开。所有上述实验收益均来自原作者各自
COCO等协议，本文不移用其增幅预测VisDrone/P1收益。没有阅读全文或人类已读标记，
没有穷尽文献/首创声明、自动外部模型核验或训练。
