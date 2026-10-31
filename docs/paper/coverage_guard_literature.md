# 覆盖保护分配的原文核对与贡献边界

2026-10-31，唯一覆盖保护30轮训练等待期间完成；不修改冻结源码、配置或判定线。
通过Paper Search MCP核对FCOS arXiv:1904.01355v5和TOOD arXiv:2108.07755v3，
从Zotero已有PDF读取正文。文献集合为Research / YOLO-RegionSSOD；
保留已有条目，不为同一论文重复建项。

## FCOS：框内回归与覆盖问题已有先例

阅读版本PDF第4页、式(1)将位置到框四边的距离定义为ltrb目标，
并说明回归目标为正；第6页讨论best possible recall，
将至少获得一个训练样本的目标计为可覆盖。
因此“框内点更符合非负距离回归”与“需要检查分配覆盖”都是已有问题。
FCOS的最小面积冲突消解、FPN尺度范围、centerness与本项目不相同。
本候选不声称首次框内分配，也不是FCOS实现的替代名称。

本地证据为原版本PDF，父条目R3GT2QZQ，附件NRI2CTCA。
同一PDF同时包含更新说明和附录，页码指PDF物理页，不混用论文印刷页码。

## TOOD：保留任务对齐评分，不把改池说成新TAL

阅读版本PDF第5页、式(9)以分类得分和定位质量的幂积定义对齐度，
按排名选正样本，并将归一化对齐质量用于分类和回归。
第7页报告论文使用alpha=1、beta=6和m=13。
本机Ultralytics 8.4.172实际实现使用topk10、alpha0.5、beta6；
候选保持本机配置，不能将其写成TOOD原论文的完整复现。

父条目7D3IXLK2、PDF附件6VJYR7IG已补入同一RegionSSOD集合。
原有其他集合成员关系保留。该论文提供TAL来源，不证明本候选有效。

## 这次真正验证什么

本机固定版本为小框扩充候选池；缩小候选池后，top-k和多目标冲突消解
可能导致原本有分配的目标失去监督。前一版诊断已实际观察到这个问题。
当前规则比较原TAL与框内优先结果，任一原有效目标消失时整图退回原TAL。
这比单纯加框内条件多一个显式覆盖保护，但仍不据此直接宣布新颖性。

原128tile诊断100%有效目标保留和48.0992%框外质量减少是几何代理。
它不等于小目标检测召回、AP或统计显著性；也不能排除归一化目标分数、
分类监督以及计算量同时改变的影响。
完整30轮结果未知，后续先看配对学生结果，再决定是否值得公平100轮、
重复随机性及覆盖保护单因素消融。

S³OD已有尺度重平衡分配；本候选不将“航拍尺度分配”称为首次。
差异与版本边界继续依照s3od_comparison.md，本轮未执行S³OD同协议复现。

## 可复核来源

- FCOS：https://arxiv.org/abs/1904.01355v5
- TOOD：https://arxiv.org/abs/2108.07755v3
- 本地固定实现：src/regionssod/training/assignment.py、
  src/regionssod/training/coverage_guard.py。
- 本轮学生协议：assignment_coverage30_protocol.md。

提取文字缓存仅放在忽略目录outputs/coverage_guard_literature_20261031；
Git不保存PDF、全文或生成缓存。
