# 参考文献

以下文献按本文初稿涉及的检测基础、半监督检测、漏标/稀疏标注和航拍小目标检测整理。在线链接在 2026-10-27 核对；预印本与正式会议版本分别注明。

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

[12] Zhang R., Xu C., Xu F., et al. Rethinking Scale Imbalance in Semi-Supervised Object Detection for Aerial Images[EB/OL]. arXiv:2310.14718, 2023. https://arxiv.org/abs/2310.14718. **预印本，也是与本方法最接近的文献。**

[13] Meethal A., Granger E., Pedersoli M. Density Crop-guided Semi-supervised Object Detection in Aerial Images[EB/OL]. arXiv:2308.05032, 2023. https://arxiv.org/abs/2308.05032.

[14] Yang F., Fan H., Chu P., Blasch E., Ling H. Clustered Object Detection in Aerial Images[C]//IEEE/CVF International Conference on Computer Vision (ICCV). 2019. DOI: 10.1109/ICCV.2019.00840.

[15] Yang C., Huang Z., Wang N. QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022: 13668-13677.

[16] Redmon J., Divvala S., Girshick R., Farhadi A. You Only Look Once: Unified, Real-Time Object Detection[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016: 779-788.

## 阅读与取舍

| 优先级 | 文献 | 在论文中的作用 | 阅读重点 |
|---|---|---|---|
| 必读 | [12] S³OD | 最接近的航拍尺度 SSOD | 尺度阈值、负学习和 DOTA 实验设置；逐项比较本文机制 |
| 必读 | [6] Soft Teacher | 置信度分类软权重来源 | 教师分数如何进入分类损失，回归与可靠性如何处理 |
| 必读 | [9] Consistent-Teacher | 伪目标一致性 | anchor 分配和动态阈值解决的问题 |
| 必读 | [11] SparseDet | 漏标与背景误监督 | 已标注/未标注区域划分及伪正样本挖掘 |
| 必读 | [13] Density Crop SSOD | 航拍 VisDrone 小目标 SSOD | 基线、评价指标与密度裁片策略 |
| 建议 | [10] Soft Sampling | 缺失标注导致错误负样本 | 与 uncertain-ignore 的差别 |
| 建议 | [4] STAC、[5] Unbiased Teacher、[7] Humble Teacher、[8] Dense Teacher | SSOD 发展脉络 | 离线伪框、在线教师和稠密伪标签的差异 |
| 建议 | [14] ClusDet、[15] QueryDet | 航拍小目标网络与分辨率处理 | 为何本文不修改网络结构，以及计算开销来源 |
| 背景 | [1]-[3]、[16] | YOLO、COCO、VisDrone 基础设置 | 数据和检测器引用，具体版本与实验协议 |

引用时应注意：[12] 和 [13] 当前按预印本列出；正式发表信息应在投稿前再次核对。[6] 的软权重作用于分类损失，[12] 的教师引导负学习较早处理尺度相关的负样本偏差，因此本文的介绍不能把这些概念写成已有文献中不存在的新思想。本文的实验尚未直接复现 [12] 或 [13]，不能跨数据集比较数值性能。

BibTeX 条目见 [references.bib](references.bib)。本文件是待整合稿，编号需在引言完成后按全文首次引用顺序统一重排。
