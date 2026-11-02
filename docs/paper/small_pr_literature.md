# 文献侧短报告：冻结预测能检验的机制假设
日期：2026-11-02；状态：文献检索、局部原文阅读及 Zotero 归档完成；诊断与假设均未执行/验证。

## 范围与证据边界

按项目 ARS 的 `deep-research/WORKFLOW.md` 采用 **three-way-scan**，内联执行 bibliography、source-verification，并使用 synthesis 的有界综合约束；角色共享当前上下文，不是独立多模型评审。仅用通用检索词、公开题名/DOI和公开论文访问外部服务。不是系统综述，也不是完整的新颖性排查。

本轮完整读取 `docs/paper/{s3od_comparison,related_work,references,p1_b2_seed1_full100_results}.md`；核对计划§3.60及最新日志，浏览实验协议、baseline与README的范围/执行说明。最新100轮报告仅用来确认“已有固定操作点覆盖与误检取舍、完整低分PR待补”的问题背景；未重算其数字，未读取或解释主会话新评估结果。旧30轮与正式100轮不混合；历史Gate、目标和阈值不变。

下面五篇均补足当前 `references.md` 的空缺；其中四篇此前已在 Zotero，故“补缺”不表示此前从未检索过。S³OD仍是最近的尺度SSOD边界：更换YOLO、降低小框阈值、加入教师负学习不能单独推出新颖性；本次不重复把它列入五篇。

## 五篇实际采用文献

### 1. Guo等（2024），STOD: toward semi-supervised tiny object detection

[出版社原文页面 / DOI 10.1007/s00521-024-09936-z](https://link.springer.com/article/10.1007/s00521-024-09936-z)。Neural Computing and Applications，36，17107–17123。

- **WHY/HOW/WHAT**：针对微小目标伪框定位质量；摘要明确列出 dense IoU-aware head、结合分类分数与预测overlap的GMM多阈值强/弱正样本挖掘、考虑正负样本定位质量的加权损失；摘要报告VisDrone-Partial实验。<!--ref:guo2024stod--><!--anchor:section:Abstract-->
- **与项目重叠**：小目标、弱置信候选、定位质量及负项处理均已有直接先例。“分类/定位解耦+强弱伪标签”也不能直接作为新概念。VisDrone-Partial这个名字不证明其预算与本项目每图标注面积协议相同。
- **确切阅读范围**：出版社题名/作者/卷页/日期与Abstract，网页抽取行36–40；**未取得正文**，未读公式、实验划分和附录。方法弱点：`not assessed (read scope: abstract_only)`。本条支持机制存在，不支持复现参数或跨论文性能比较。

### 2. Wang等（2023），Calibrated Teacher for Sparsely Annotated Object Detection

[arXiv:2303.07582v1全文](https://arxiv.org/html/2303.07582v1)；Crossref核验正式出版为AAAI 2023，[DOI 10.1609/aaai.v37i2.25349](https://doi.org/10.1609/aaai.v37i2.25349)，出版社页面本次拒绝访问。

- **WHY/HOW/WHAT**：伪标签阈值随检测器和训练阶段失去统一含义；用在线logistic校准及预测队列使分数对应匹配精度，另用FIoU降低潜在漏标负样本的分类权重。队列来自与现有标注有重叠的预测，负项权重依据与已有标注/伪标签的最大IoU。<!--ref:wang2023calibrated--><!--anchor:section:Method-->
- **与项目重叠**：已经把“置信度可靠性”和“漏标背景抑制”结合。P1是冻结教师、固定尺度阈值及空间二值ignore；不能把校准+背景抑制重新包装成首创。
- **确切阅读范围**：v1 Abstract、Introduction、Method五个小节（含Algorithm 1、式2–4）、Experimental Results/Datasets、Failed Cases、Conclusion and Limitations；未核验全部实验表、代码或正式版全文。
- **方法边界**：Online Calibration假设选中/未选中预测统计相似；若标注区域由目标中心选择并造成分布偏移，此假设可能失配。这是**reader-inferred、未验证的迁移风险**，不是作者在本项目上失败的证据。作者在Failed Cases明确讨论密集邻居被IoU过滤及极小物体漏检。<!--ref:wang2023calibrated--><!--anchor:section:Online Calibration-->

### 3. Küppers等（2020），Multivariate Confidence Calibration for Object Detection

[CVPR Workshops官方索引](https://openaccess.thecvf.com/content_CVPRW_2020/html/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.html)；[实际阅读arXiv:2004.13546v1](https://arxiv.org/html/2004.13546v1)。

- **WHY/HOW/WHAT**：检测校准依赖框的位置和尺度；以类别、分数及预测框属性定义条件匹配精度，建立多变量校准与D-ECE。它校准已有检测的可靠性，不能为完全无候选的GT生成分数。<!--ref:kuppers2020multivariate--><!--anchor:section:3.1–3.2-->
- **与项目重叠**：`score+scale`校准及位置条件校准已有先例。潜在问题应落在区域选择偏差、有限标注下的可迁移性和受精度约束的覆盖，而不是“首次按尺度校准”。
- **确切阅读范围**：v1 Abstract、§1、§3.1–3.2（式1–3）、§3.4末尾Histogram Binning段、§4（式11–14）、§5.1–5.2文字与图4说明；表1浏览；未完整核验§3.3–3.4分布推导。
- **方法边界**：作者指出高维分箱样本稀疏、不同分箱/维度的D-ECE不能直接比；不能把总体ECE下降当作小目标可靠性提高。<!--ref:kuppers2020multivariate--><!--anchor:section:4-->

### 4. Kuzucu等（2024），On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines

[ECCV官方索引](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3148_ECCV_2024_paper.php)；[本次阅读的正式版PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03148.pdf)。

- **WHY/HOW/WHAT**：固定分数阈值、不同检测集合及不合适的数据划分会扭曲校准比较；提出LaECE0/LaACE0与LRP联合评价，以及留出集上的类别特定Platt/Isotonic基线。其目标含定位质量，不能将LaECE0等同于某个IoU门槛下的precision。<!--ref:kuzucu2024calibration--><!--anchor:section:3–4-->
- **与项目重叠**：直接支撑“相同conf不等于相同风险”和需区分排序、校准、定位。该文采用LRP选操作点；本报告建议的等精度召回是本轮诊断提议，**不是该论文已证明的精度约束保证**。
- **确切阅读范围**：官方Abstract；PDF§1/表1；§2开头匹配/AP段；§3中阈值、二值目标、数据划分/训练设置片段；§4.1–4.4（式5–9）；表5浏览，不引用数值；未完整读§5/附录。以可见章节作定位，未在本机执行PDF页完整性预检，不宣称逐页全文审读。
- **方法边界**：论文采用留出校准/评测划分；若在同一548图拟合并评价校准或挑最佳阈值，会产生本项目的选择偏差（**reader-inferred**），不能继承原文的留出结论。

### 5. Liu、Ma和Kira（2022），Unbiased Teacher v2

完整题名：*Unbiased Teacher v2: Semi-supervised Object Detection for Anchor-free and Anchor-based Detectors*。[CVPR官方索引](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Unbiased_Teacher_v2_Semi-Supervised_Object_Detection_for_Anchor-Free_and_Anchor-Based_CVPR_2022_paper.html)；[实际阅读arXiv:2206.09500v1](https://arxiv.org/html/2206.09500v1)。

- **WHY/HOW/WHAT**：分类分数不足以代表回归质量；Listen2Student预测逐边界不确定性，仅在教师相对学生更可靠时施加相应边界回归监督；§3.2还指出有限监督下FCOS centerness可能失真。<!--ref:liu2022utv2--><!--anchor:section:3.2–3.3-->
- **与项目重叠**：单个教师分数直接控制回归已有明确质疑；P1分类软权重不等于回归可靠性约束。但相对教师—学生不确定性和逐边界门控也已有先例。
- **确切阅读范围**：v1 Abstract、Introduction、§3.2、§3.3.1–3.3.2（含式1）；表4/5浏览，不引用性能数值；未完整读§4/附录。CVF PDF本次403，使用作者预印本定位机制。
- **方法边界**：需要训练定位不确定性分支。现有最终框/分类分数不能恢复该分支输出；把它们当定位不确定性代理是否成立未验证（**reader-inferred**）。

**有界方法检查记录**：按ARS经验研究六项，以上2–5仅检查“变量定义”（`checked: none found`，指所读范围未发现明确含混，非全篇认证）；2、4、5的迁移/内部效度风险及3的分箱测量风险为`checked: found`，来源如各条标记。研究假设清晰度、统计报告完整性、结论保守性均`not checked`；外部效度仅作上述迁移风险提示，未完成泛化审查。STOD六项均未评估。此记录不是正式审稿。

## 综合与三个可证伪假设（AI提出，均未确认）

共同结论是“候选存在、分类可靠、定位准确、避免错误背景监督”是不同环节。校准既不能创造不存在的框，也不自动改善定位。Küppers的二值匹配校准与Kuzucu的IoU目标是**条件差异**而非互相否定；STOD的定位加权与UTv2对FCOS质量分数的警告也不能跨模型判谁正确。仅检查这两组关联，未穷举五篇的全部配对；这些解释待作者评判。

### H1：小目标覆盖存在“超出分数整体平移”的收益，但受尺度条件风险约束

**可探索机制**：未来将区域内可见标签提供的可靠性、类别和预测尺度共同用于风险约束的候选分配；候选方向是“有限可见区域下校准如何迁移到隐藏区域”，不是尺度校准本身。

**本轮能检验的必要条件**：用每臂同设置的低分冻结预测，比较全PR和共同precision范围内的small recall；同时画按预测尺度/类别的分数—经验正确率曲线。若P1在相同precision下覆盖仍较高，才有超出单一conf操作点的证据；若优势消失或仅出现在少数精度区间，就收缩该假设。纯粹严格单调的类别内分数变换，在固定候选/NMS/max_det和无新增并列的条件下不改变类别内排序及AP；尺度条件映射可能重排，需另作拟合/留出验证，**本轮不拟合**。

**仍不可推出**：等精度曲线优势不是校准机制已有效，也不能证明收益来自ignore；全尺寸precision约束不等于small组也具有相同precision。

### H2：部分弱置信小框有“可辨认存在、边界不可靠”的双重属性

**可探索机制**：以后把“可用于分类/保护背景的存在证据”和“可用于回归的几何证据”分开决策，并研究像素误差相对框宽高的放大效应。STOD、Soft Teacher、UTv2已覆盖重要组成思想；新增价值只能来自区域协议下可重复的诊断及不同于既有机制的具体设计。

**本轮诊断**：分别在匹配IoU 0.50和0.75重放small PR/等精度召回；对稳定关联到同一GT的候选描述中心偏移、各边误差/GT宽高、类别错误和重复框。按已有尺度组与类别分层，避免把IoU几何敏感性误认为单一回归模块故障。若低分新增主要是近邻重复/类别错误，或精细定位质量与高分相当，双重属性解释被削弱；若差异在控制尺度后消失，应优先保留几何解释。

**本轮不能验证**：最终学生预测不等于训练时教师伪框；没有逐边界不确定性输出便不能复现Listen2Student，亦不能因IoU0.75下降断言需要某种损失。

### H3：ignore应受“新增独立目标覆盖/屏蔽面积”约束，而不只由分数区间决定

**可探索机制**：在未来训练中将潜在漏标保护视为空间预算分配：优先覆盖未被高可信正框解释的候选区域，限制重叠重复候选和无目标区域消耗的负样本屏蔽面积。不能把“负项降权/去重/空间ignore”单独视为创新；关键待证问题是区域选样偏差下，其边际收益是否比普通阈值更可迁移。

**本轮低分全图预测可做的筛查**：按固定规则逐级纳入低分候选，记录新增唯一GT匹配、重复框、未匹配框及候选并集面积，分密度/尺度看边际覆盖；GT仅用于诊断。若更多候选只增加重复和面积，不增加独立目标覆盖，则该方向缺少必要依据。

**更直接但有条件的诊断**：只有既有冻结教师tile候选、原入训tile清单、坐标变换、ignore元数据和隐藏GT审计缓存齐全时，才可重放“未被正伪框覆盖的隐藏GT保护率”与背景屏蔽代价；hidden GT必须保持审计oracle身份。仅主会话完整val学生预测不能重建训练时负anchor梯度、实际mask或证明因果。无候选目标、无正伪框而未入训的tile必须单列，不能从可见候选样本外推。

## 交给主会话的最小诊断合同（建议，不改现有协议）

1. **冻结输入**：各seed分别配对；记录权重/预测哈希、置信下限、原图坐标、NMS、max_det、类别、ignore及匹配策略。先核验候选缓存是否在低分端/密集图被截断。旧conf≥0.25缓存无法补回其下方PR。
2. **同时给两种比较**：保留原conf0.25；新增经验函数
   `R_small@P≥π = max_t R_small(t), subject to P(t)≥π`。
   π沿用主会话预先固定的共同网格，不在看到P1收益后选择；并列分数一起处理，实际可达到点与插值包络分开，无非空可行点报NA。注明P为全尺寸还是small专用定义；small precision若涉及未匹配预测，需要清楚的预测面积/忽略规则，不能用GT尺度给FP分组。
3. **指标不得冒名**：原诊断“全尺寸一对一匹配后按GT面积分组”与COCO式area-range AP/AR的ignore逻辑并不等价。AP_small、AR及maxDets必须随评测器定义写清，不自动称VisDrone官方指标；经验等精度前沿亦不提供部署风险保证。
4. **配对与不确定性**：同图像配对，两个学生seed分别呈现；图像bootstrap只反映固定权重、固定验证样本条件的不确定性。若重采样后重新选经验阈值，应明确它是前沿的不确定性；若阈值来自外部固定校准集，则在评测bootstrap中保持固定。重复图像不能增加训练seed样本量。
5. **无泄漏的解释**：本轮只做描述性诊断，不在同一验证集拟合校准器、选择政策再报告“验证提升”。未来如探索后处理，需独立校准/评测划分或图像级交叉拟合；这仍只是开发性证据，不能把已多次使用的val/test-dev改称盲测。

**确需新训练且本轮不做**：改伪标签接受/ignore政策、增设IoU或不确定性分支、分类/回归解耦训练及其成对消融、校准策略回灌教师/学生。仅重算PR、匹配与可靠性图不需新训练。冻结结果最多支持或削弱下一步假设，不证明新训练机制有效，也不追认Gate通过。

## Zotero实际归档与附件状态

已用工具重新核验：父目录Research=`BW83SN4A`，目标Research / YOLO-RegionSSOD=`BMQTT9Y7`。先按题名查重，STOD另按短题名/DOI查重；4篇复用、1篇新增。随后用目标collection+本轮标签查回，恰好5篇，逐条子附件/笔记已核对。

| 文献 | 操作 / item key | DOI或公开URL | PDF及笔记状态 |
|---|---|---|---|
| STOD | **新增 HBL9PADC** | [10.1007/s00521-024-09936-z](https://doi.org/10.1007/s00521-024-09936-z) | OA自动下载未成功，出版社订阅；**无PDF**。机器摘要阅读笔记 `TKFPXEPL` |
| Calibrated Teacher | 复用 `KV74E49Y`，原已在目标目录 | [2303.07582v1](https://arxiv.org/abs/2303.07582v1)；正式DOI见第2条 | 已有PDF `9N87GK65`；笔记 `Z64STA8I` |
| Multivariate Calibration | 复用 `RQIKGRP9`，加入目标目录，保留原目录 | [2004.13546v1](https://arxiv.org/abs/2004.13546v1) | 已有PDF `4HFERNLI`；笔记 `AFP9RSUS` |
| On Calibration | 复用 `K3VYDTZE`，加入目标目录，保留原目录 | [2405.20459v1](https://arxiv.org/abs/2405.20459v1)；[ECCV正式PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03148.pdf) | 原预印本PDF `S69F7FDK`保留；本次新增正式版PDF **`2HGW5IHR`**；笔记 `6HCAUXHG` |
| Unbiased Teacher v2 | 复用 `R6AMZBIU`，原已在目标目录 | [2206.09500v1](https://arxiv.org/abs/2206.09500v1) | 已有PDF `AAVY3YSS`；笔记 `N5N68YE8` |

共同标签：`literature-sidecar-20261102`、`YOLO-RegionSSOD`、`human-read:not-attested`；另分`machine-read:partial`与`machine-read:abstract-only`。五份笔记只含公开机制、版本与实际阅读范围，不含本项目私有结果/假设；没有标成人已读。已有四条保留预印本身份，笔记区分正式出版与实际阅读版本。未运行可能调用外部embedding模型的语义索引更新；不影响条目、附件及collection归档。

## 检索回执与限制

[检索/归档回执](../../outputs/paper1_small_pr_20261102/literature_search_receipts.json)保存MCP实际请求与响应、公开来源、范围和Zotero写入/核验回执。MCP首轮Semantic Scholar返回HTTP429，arXiv宽查询0条；改用OpenAlex/DBLP与arXiv/Crossref题名检索。Calibrated Teacher的arXiv/Crossref命中成功；STOD的Crossref通用题名查询没有精确命中，最后以Springer官方页面与DOI元数据核验，**零命中/限流不当作不存在**。

纳入规则为四个主题的机制直接相关、可核查原始公开来源、优先补当前参考表空缺；不按“支持P1”筛选。未入选五篇的PseCo/LabelMatch及新近遥感候选仅作检索线索；不为其推导机制缺失或性能判断。排除视频/跟踪、RGB-IR、纯网络模块及仅二手来源；新近结果未全面覆盖。本次五篇均为计算机视觉计算实验研究（5/5），年代2020–2024；这是机制导向的覆盖偏斜，不是最新文献全集，更不是“无人做过”的证明。

本文保留独立文献侧任务的原始完成记录；该子任务本地只写本报告和回执；未调用训练/GPU/推理，未修改源码、正式论文或计划，未提交主会话变更。主会话产生的源码/测试/协议文件保持独立。本轮实际补评估结果与本报告分开，见[补评估结果](small_pr_results.md)；本报告中的“未执行”仅指文献侧未运行诊断，不表示主会话未执行评估。文献侧任务完成；所有候选假设、诊断建议及后续训练均未确认有效。

