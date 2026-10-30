# 参考文献

以下文献按本文初稿涉及的检测基础、半监督检测、漏标/稀疏标注和航拍小目标检测整理。在线链接在 2026-10-27 核对；预印本与正式会议版本分别注明。
2026-10-30补入独立NWD候选所依据的[17]，该项版本和正文另行核对，不表示其已成为主方法。

[1] Jocher G., Qiu J. *Ultralytics YOLO11*[CP/OL]. 2024. https://docs.ultralytics.com/models/yolo11/.

[2] Lin T.-Y., Maire M., Belongie S., et al. Microsoft COCO: Common Objects in Context[C]//European Conference on Computer Vision (ECCV). 2014: 740-755.

[3] Du D., Zhu P., Wen L., et al. VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results[C]//IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). 2019. https://openaccess.thecvf.com/content_ICCVW_2019/html/VISDrone/Du_VisDrone-DET2019_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ICCVW_2019_paper.html.

[4] Sohn K., Zhang Z., Li C.-L., et al. A Simple Semi-Supervised Learning Framework for Object Detection[EB/OL]. arXiv:2005.04757, 2020. https://arxiv.org/abs/2005.04757.

[5] Liu Y.-C., Ma C.-Y., He Z., et al. Unbiased Teacher for Semi-Supervised Object Detection[C]//International Conference on Learning Representations (ICLR). 2021. https://openreview.net/forum?id=MJIve1zgR_.

[6] Xu M., Zhang Z., Hu H., et al. End-to-End Semi-Supervised Object Detection with Soft Teacher[C]//IEEE/CVF International Conference on Computer Vision (ICCV). 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html.

[7] Tang Y., Chen W., Luo Y., Zhang Y. Humble Teachers Teach Better Students for Semi-Supervised Object Detection[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2021: 3132-3141.

[8] Zhou H., Ge Z., Liu S., et al. Dense Teacher: Dense Pseudo-Labels for Semi-Supervised Object Detection[C]//European Conference on Computer Vision (ECCV). 2022. https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/961_ECCV_2022_paper.php.

[9] Wang X., Yang X., Zhang S., et al. Consistent-Teacher: Towards Reducing Inconsistent Pseudo-Targets in Semi-Supervised Object Detection[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2023: 3240-3249. DOI: 10.1109/CVPR52729.2023.00316.

[10] Wu Z., Bodla N., Singh B., et al. Soft Sampling for Robust Object Detection[C]//British Machine Vision Conference (BMVC). 2019: Article 225. https://arxiv.org/abs/1806.06986.

[11] Suri S., Rambhatla S., Chellappa R., Shrivastava A. SparseDet: Improving Sparsely Annotated Object Detection with Pseudo-Positive Mining[C]//IEEE/CVF International Conference on Computer Vision (ICCV). 2023. DOI: 10.1109/ICCV51070.2023.00623.

[12] Zhang R., Xu C., Xu F., et al. S³OD: Size-unbiased semi-supervised object detection in aerial images[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2025, 221: 179-192. https://doi.org/10.1016/j.isprsjprs.2025.01.037. **与本方法最接近的文献；已核对正式发表信息。** 早期预印本题名为 *Rethinking Scale Imbalance in Semi-supervised Object Detection for Aerial Images*，arXiv:2310.14718v1，2023，https://arxiv.org/abs/2310.14718v1。两版与公开代码的证据范围见 [方法差异对照](s3od_comparison.md)。

[13] Meethal A., Granger E., Pedersoli M. Density Crop-guided Semi-supervised Object Detection in Aerial Images[EB/OL]. arXiv:2308.05032, 2023. https://arxiv.org/abs/2308.05032.

[14] Yang F., Fan H., Chu P., Blasch E., Ling H. Clustered Object Detection in Aerial Images[C]//IEEE/CVF International Conference on Computer Vision (ICCV). 2019. DOI: 10.1109/ICCV.2019.00840.

[15] Yang C., Huang Z., Wang N. QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022: 13668-13677.

[16] Redmon J., Divvala S., Girshick R., Farhadi A. You Only Look Once: Unified, Real-Time Object Detection[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016: 779-788.

[17] Wang J., Xu C., Yang W., Yu L. A Normalized Gaussian Wasserstein Distance for Tiny Object Detection[EB/OL]. arXiv:2110.13389, 2021. 阅读版本为v2，修订于2022-06-14，https://arxiv.org/abs/2110.13389v2。原文包含分配、NMS和回归应用及VisDrone验证；本项目只借用其中度量作小框混合回归。arXiv注明另有期刊扩展工作，本条按实际阅读的预印本引用，不冒充已核验扩展全文。

## 阅读与取舍

| 优先级 | 文献 | 在论文中的作用 | 阅读重点 |
|---|---|---|---|
| 必读 | [12] S³OD | 最接近的航拍尺度 SSOD | 区分 2023 预印本、2025 期刊版与公开代码；逐项比较 SAT/SLA/TNL |
| 必读 | [6] Soft Teacher | 教师可靠性加权的相关工作 | 背景分数对负候选分类项的权重及归一化；与本文正伪框权重区分 |
| 必读 | [9] Consistent-Teacher | 伪目标一致性 | anchor 分配和动态阈值解决的问题 |
| 必读 | [11] SparseDet | 漏标与背景误监督 | 已标注/未标注区域划分及伪正样本挖掘 |
| 必读 | [13] Density Crop SSOD | 航拍 VisDrone 小目标 SSOD | 基线、评价指标与密度裁片策略 |
| 建议 | [10] Soft Sampling | 缺失标注导致错误负样本 | 与 uncertain-ignore 的差别 |
| 建议 | [4] STAC、[5] Unbiased Teacher、[7] Humble Teacher、[8] Dense Teacher | SSOD 发展脉络 | 离线伪框、在线教师和稠密伪标签的差异 |
| 建议 | [14] ClusDet、[15] QueryDet | 航拍小目标网络与分辨率处理 | 为何本文不修改网络结构，以及计算开销来源 |
| 当前候选必读 | [17] NWD | 已有几何回归度量与短程收敛解释 | 式7–9、表3的12/24epoch差异；与S³OD分配机制区分，不据此保证本项目收益 |
| 背景 | [1]-[3]、[16] | YOLO、COCO、VisDrone 基础设置 | 数据和检测器引用，具体版本与实验协议 |

引用时应注意：[12] 已更新为 2025 年期刊文献，早期预印本和公开代码用于定位机制细节，不能混用其版本参数；[13] 仍按预印本列出，正式发表信息应在投稿前再次核对。[6] 第 3.2 节的软权重具体用于负候选的背景分类项，与本文前景置信度对正 anchor 分类项的权重不同。[12] 已处理尺度相关的阈值与负学习偏差，因此不能把这些概念写成本文首次提出。本文的实验尚未直接复现 [12] 或 [13]，不能跨数据集比较数值性能。

BibTeX 条目见 [references.bib](references.bib)。当前引言和其余章节沿用本表的稳定编号；最终整合排版时再按全文首次引用顺序统一重排。
