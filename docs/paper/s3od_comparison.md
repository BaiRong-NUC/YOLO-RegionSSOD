# S³OD 与 RegionSSOD 的方法差异对照

S³OD 已系统研究航拍半监督检测中的尺度阈值、正样本分配和教师引导负学习。RegionSSOD 当前 P1 与其在问题动机上有明显重叠；可以明确的差异是标注预算与区域构造、固定离线训练流程，以及伪正样本和不确定背景的具体损失处理。现有实验只比较 B1、B2 与 P1，不能据此判断 P1 优于 S³OD。

## 1 对照来源

S³OD 的正式发表版本为 Zhang 等的 *S³OD: Size-unbiased semi-supervised object detection in aerial images*，发表于 *ISPRS Journal of Photogrammetry and Remote Sensing*，2025 年，第 221 卷，179–192 页，DOI：[10.1016/j.isprsjprs.2025.01.037](https://doi.org/10.1016/j.isprsjprs.2025.01.037)。[出版社页面](https://www.sciencedirect.com/science/article/pii/S0924271625000425)和 Crossref 元数据确认了题名、作者和卷页；期刊摘要确认 SAT/SLA/TNL 三项机制，并将实验扩展至 DOTA-v1.5 与 SODA-A。

可完整阅读的早期版本是 *Rethinking Scale Imbalance in Semi-supervised Object Detection for Aerial Images*，arXiv:2310.14718v1，2023-10-23：[版本记录](https://arxiv.org/abs/2310.14718v1)、[全文 PDF](https://arxiv.org/pdf/2310.14718v1)、[全文 HTML](https://arxiv.org/html/2310.14718v1)。下表的段落、公式和页码均对应这份预印本，不将其百分位阈值或参数当作期刊版已逐项核验的设置。期刊全文在本次访问中不可用；公开代码提供了额外实现依据，见第 3.4 节。

P1 对照依据为本仓库 `9dd79f7` 的实现、冻结配置及已验收的 10% 面积预算、seed0、30 epoch 学生结果。实现索引见[证据文件](evidence.md)，指标与统计范围见[结果表格](results_tables.md)。论文文献编号沿用[参考文献](references.md)中的 S³OD[12] 和 Soft Teacher[6]。

## 2 逐项差异

| 维度 | S³OD 2023 预印本 | 当前 P1 实现 | 差异的含义 |
|---|---|---|---|
| 标注预算 | 先将 DOTA-v1.5 原图裁片，再抽取 1%/5%/10% 裁片作为有标注样本，其余作为无标注样本；IV-A，p.7 | 每张 VisDrone 训练原图选择一个目标中心标注 tile，以其面积占原图的比例定义预算；当前为 10% | 裁片数量比例与每图标注面积比例不是同一种预算，不能直接比较“10%”设置 |
| 未标注区域 | 在其部分标注实验中，有标注和无标注样本按裁片划分；IV-A，p.7 | 每图另采样一个尽量与标注 tile 低重叠的未标注 tile；不是使用全部区域外像素，也不保证两个 tile 完全互斥 | 这是可控的区域标注协议差异，不能表述为 S³OD 完全没有区域裁剪 |
| 检测器与框类型 | Rotated Faster R-CNN，ResNet-50/FPN、MMRotate，预测旋转框；III-A、IV-B，pp.5、7 | YOLO11s，水平框；网络结构不变 | 检测器与任务形式不同，不能仅靠更换检测器确立新机制或更高性能 |
| 教师与伪标签更新 | 在线师生框架，教师由学生 EMA 更新，弱增强教师与强增强学生配对；III-A，pp.4–5 | B1 教师训练 100 epoch 后冻结，一次离线生成伪标签；学生独立从 COCO 权重初始化 | P1 是固定教师的单轮自训练，不复现 S³OD 的在线更新流程 |
| 尺度阈值 | SAT 对批次内置信度大于 0.5 的教师预测按 small/large 分组，分别取第 P 百分位作为阈值；III-B，p.5 | 按原始 tile 框面积分为 small/medium/large，固定阈值为 0.30/0.50/0.50；候选下限 0.05 | P1 属于固定尺度分层筛选，不能称为分布自适应阈值；中、大尺度当前阈值相同 |
| 正样本分配 | SLA 将旋转框建模为高斯分布，按 Wasserstein 距离相似度选择 top-K anchors；III-C，pp.5–6 | 只让人工框和正伪框进入原 YOLO 任务对齐分配；ignore 框不进入分配 | P1 没有实现 SLA，也没有独立解决尺度造成的正样本分配数量不平衡 |
| 正样本权重 | SLA 按批次 small/large 正样本数重新加权正样本损失；式(6)，p.6 | 匹配伪框的正 anchor 的整个类别 BCE 向量乘教师前景置信度；人工框乘 1，TAL 分母不变 | 尺度组数量平衡与逐伪框可靠性衰减的依据、作用位置不同 |
| 负样本处理 | TNL 先筛选教师背景分数大于 0.7 的负 proposals，再将置信度低于 0.5 的预测作为加权困难负样本；III-D、式(7)，p.6 | 对 0.05≤conf<尺度正阈值的候选，仅屏蔽其框内负 anchor 的所有类别 BCE；已匹配正 anchor 优先 | TNL 包含负样本筛选与困难负样本学习；P1 使用空间二值 ignore，二者不能等同 |
| 回归与候选限制 | 采用旋转检测框架的监督及 SLA/TNL 处理；III-A–D | 原 box/DFL 计算规则不加教师置信度乘数；仅含正伪框的未标注 tile 入训 | P1 的置信度权重作用于分类；回归仍承受被接受伪框的定位噪声 |
| 实验与指标 | DOTA-v1.5 验证集、IoU=0.5 的 mAP；部分标注主结果报告五次划分统计；IV-A、表 I，p.7 | VisDrone 548 张完整原图、960 验证；Ultralytics mAP50/mAP50-95 与固定操作点分尺寸 recall，当前单 split/seed | 数据、框类型、预算、训练流程及评测口径均不同，不可把论文数字并列后宣布优劣 |

S³OD 的 small/large 划分以 32² 像素面积区分小目标与其余目标，见 p.2 脚注；P1 采用原始 tile 坐标中的预测框面积，额外保留 96² 的 medium/large 边界。两者都关注小目标，但坐标及预处理条件仍须各自说明。

P1 的实现位置为：[区域构造](../../src/regionssod/data/partial.py)、[尺度筛选](../../src/regionssod/pseudo/scale.py)、[阈值配置](../../configs/pseudo_p1_10.yaml)、[分类权重和 ignore](../../src/regionssod/training/p1.py)、[学生配置](../../configs/train_p1_10.yaml)。S³OD 对应段落可直接定位至 [III-A](https://arxiv.org/html/2310.14718v1#S3.SS1)、[III-B](https://arxiv.org/html/2310.14718v1#S3.SS2)、[III-C](https://arxiv.org/html/2310.14718v1#S3.SS3)、[III-D](https://arxiv.org/html/2310.14718v1#S3.SS4) 和 [IV-A](https://arxiv.org/html/2310.14718v1#S4.SS1)。

## 3 三个容易混淆的机制

### 3.1 固定尺度阈值与自适应阈值

SAT 的阈值随当前批次预测分布变化，P1 的阈值在配置中固定。两者都按尺度区分伪标签接受条件，但 P1 没有在线统计分布或更新阈值。更直接的重叠来自 S³OD 表 IV：该文已经比较固定的“大目标 0.9、小目标 0.8”方案与尺度自适应方案。因此，“给小目标更低固定阈值”本身也不能作为本文首次提出的思想。表 IV 的具体结果属于其 DOTA 协议，不能迁移为当前 P1 的效果证据。[S³OD IV-D2](https://arxiv.org/html/2310.14718v1#S4.SS4.SSS2)

### 3.2 正伪框置信度权重与尺度重平衡

SLA 的权重取决于一个批次中不同尺度正样本的数量，旨在平衡尺度组的监督贡献。P1 则依据每个被接受伪框的教师置信度衰减正 anchor 分类项，不进行尺度组数量归一化，不修改任务对齐分配或回归损失的计算规则。因此，P1 的软权重不能描述为 SLA 的等价实现，也不能仅凭组合实验断言它独立改善了尺度平衡。

Soft Teacher 的关联也需要准确区分：其第 3.2 节式(4)–(5)用教师背景分数估计负候选可靠性，并归一化加权背景分类损失；P1 用接受伪框的前景置信度衰减正 anchor 分类项，保留原 TAL 分母。两者共享“利用教师信息调节监督可靠性”的思想，具体的加权对象、分数含义和归一化方式不同。[Soft Teacher 3.2](https://arxiv.org/html/2106.09018v3#S3.SS2)

### 3.3 困难负样本学习与不确定区域忽略

S³OD 不只是丢弃低置信度预测。其 TNL 在筛出可靠背景后，还引入困难负样本，并按式(7)的 2(1−s²) 系数加权。P1 则不让不确定框承担额外正、负类别监督，而是将中心落入其中的负 anchor 分类乘数设为 0。P1 没有显式的背景类别概率，也不能把多类前景最大分数的补数直接当作经过校准的背景概率。

这一空间 ignore 只保护教师下限以上、NMS 后且随正伪框 tile 入训的候选覆盖区域；完全没有候选的目标仍可能受到错误背景监督。扩大 ignore 范围也可能减少有用的负样本，因此其净收益必须通过去掉 ignore 的单因素对照判断，不能由方法定义直接推出。

### 3.4 正式版本公开代码的补充核对

出版社给出的代码仓库为 [ZhangRuixiang-WHU/S3OD 的 master 分支](https://github.com/ZhangRuixiang-WHU/S3OD/tree/master)。本次读取固定提交 `d7806b78947cdc403d525e67a28193a88a4f8598`，其 README 使用期刊版题名并列出 DOTA-v1.5 和 SODA-A。仓库默认 main 分支只有简短 README，不能据此判断实现尚未公开。

该提交的 [10% DOTA 配置](https://github.com/ZhangRuixiang-WHU/S3OD/blob/d7806b78947cdc403d525e67a28193a88a4f8598/configs/dota1.5/s3od_dota1.5_10percent.py#L262)设置 `cls_thr_percent='gmm'`、阈值裁剪范围 0.6–0.92、`small_size=1024`。[阈值函数](https://github.com/ZhangRuixiang-WHU/S3OD/blob/d7806b78947cdc403d525e67a28193a88a4f8598/s3od/models/utils/bbox_utils.py#L286)包含双分量 GaussianMixture 分支，以两个分量均值的中点取阈值；分尺度统计在同文件 `get_adaptive_size2` 中完成。因此，公开实现的这个配置不能直接描述为预印本的“第 35 百分位”。P1 与其共有按尺度区分筛选条件的动机，但 P1 没有此 GMM 自适应计算。

[学生基础配置](https://github.com/ZhangRuixiang-WHU/S3OD/blob/d7806b78947cdc403d525e67a28193a88a4f8598/thirdparty/mmrotate/configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_for_s3od_student_dota15.py#L61)在 RPN 中使用 GWD 的 `TopKAssigner`、`topk=2`，在 RoI 分配中设置 `true_neg_thr=0.7`，并使用负样本采样器。[师生模型](https://github.com/ZhangRuixiang-WHU/S3OD/blob/d7806b78947cdc403d525e67a28193a88a4f8598/s3od/models/s3od.py#L255)还会以教师 RoI head 复核学生 proposals，构造负样本分类权重；它不是 P1 的不确定区域空间掩码。该处权重计算也不宜不加核对地写成预印本式(7)。本次是静态源码对照，未执行 S³OD 训练，也未据这些代码确认期刊全文的所有公式与默认参数。

## 4 本文的贡献定位

当前可支持的贡献是：建立以每图标注面积为预算的航拍区域局部监督实验协议；在 YOLO11s 单轮离线自训练中实现固定尺度阈值、伪正框置信度分类权重和不确定区域负项屏蔽；通过可追溯的 B1/B2/P1 对照，分析教师伪标签精度与覆盖的取舍及学生完整原图检测表现。

该区域协议使用完整 GT 选择目标中心，再隐藏区域外标签，是合成的局部标注条件。10% 面积不能等同于 10% 目标数或 90% 人工工时节省；目标中心采样、裁片缩放和区域重叠也会影响数据分布。协议差异及 YOLO 适配提供研究范围与实现贡献，但尚不足以证明独立算法新颖性，仍需结合单因素消融和同协议相关方法对照判断。

可用于正文的关系表述为：

> S³OD 已从尺度阈值、正样本分配与负学习三个环节分析航拍半监督检测的尺度偏差。本文在此基础上关注每张训练图像仅开放局部标注区域的情形，采用固定离线教师和 YOLO11s 学生，考察固定尺度分层筛选、伪正框置信度分类加权以及不确定区域负项屏蔽的组合效果。本文保留基础检测器的目标分配规则，并在完整原图上分别分析整体检测、分尺寸召回与伪标签精度—覆盖取舍。

现有结果支持“本次配对 pilot 中获得正向收益”。尚不支持“首次结合尺度阈值和教师负学习”“优于 S³OD”“达到最先进水平”“各模块均有独立贡献”或“相同计算预算下更高效”。P1 学生耗时高于 B2，二者仅学生 epoch 配对，教师及伪标签生成成本另计。

## 5 下一步需要回答的实验问题

优先复用冻结教师、split 和完整原图 val960，对完整 P1 分别去掉 ignore、去掉软权重、将尺度正阈值统一为 0.50。前两项可复用正框与元信息；阈值对照需另建 policy 和输出目录，保留其他开关与训练条件，并记录入训 tile 数和批次数的变化。这些对照用于区分组合中各机制的作用，不等同于复现 S³OD。

重复 seed 用于检查学生训练随机性；固定 split 与更换区域 split 回答不同的稳定性问题。若后续主张相对 S³OD 的优势，还需把原方法或明确注明改动的适配版本放到同一标注协议下，统一评测并交代框类型、检测器、在线教师与计算成本的差别。当前不从两套论文数字推断这种优势。
