# 论文完成计划与项目交接

> 最后更新：2026-11-06<br>
> 最新定位：固定保存预测中，候选/最强控制2590点请求及已记录证书字段完全一致；同一3U对象2233锚点的CPU精确几何求解约101–104秒。控制完整loss108.391秒，候选120秒截断于刚开始的第7次后验采样；积分约0.42秒，不支持点容量或CUDA等待主导的解释。120秒源自额外工程合同，整批口径很紧但未放宽，候选完成时间未知；2147tests通过。没有训练资格/开训，详见3.94及[定位报告](paper/candidates/bounded_retention/loss_timeout_diagnostic_results.md)。<br>
> 最新压力门：真实3U/第9轮EMA零训练审查停止。恢复原生回调注册后两条1414批流及固定closed1316身份/张量SHA通过；原P1同批loss/反向通过，初始权重候选loss超过冻结120秒，父仅终止自有诊断worker。第9轮及32批未执行，独立资格复核STOP；不把timeout写成已确认点容量故障/机制无效。未接入新入口/冻结新三臂/启动训练，失败保留；见3.93及[压力报告](paper/candidates/bounded_retention/point_pressure_results.md)。<br>
> 容量审查终态：独立v2通过8例压力检查、固定真实32批及2087tests；完整loss旧/新0.9465（新耗时+5.65%），未达2倍门槛。局部容量反例已修复，但真实窗口无3U/后期覆盖，不能保证原故障消除。停止本轮扩展，不接入生产、不提高上限、不恢复或新开训练。见第3.92节及[容量审查](paper/candidates/bounded_retention/point_capacity_review_results.md)。<br>
> 当前终态：V6候选第10轮触发全局点求解容量故障`global point resource cap`；原P1连续30轮exit0、候选连续9轮后exit1、控制未启动。三臂未完成，无最终548图val960或性能/机制判断。只读audit exit0且terminal_evidence_verified=true，但accepted=false；24004输入/61封印产物通过。父/worker/监控均结束，临时脚本已清理，不重试/恢复/提高限额。见第3.91节及[失败与成本报告](paper/candidates/bounded_retention/three_arm30_terminal_failure.md)。旧合同、工程通过记录及全部失败保留。<br>
> 此前V3终态：E可表示性门是额外合同，V3恢复原P1编码语义后旧E反例通过；第13真实batch的U连续区间约[637.5464,640]无字面DFL原子，因此停止离散质量候选。1316tests/31绑定/28快照通过；V1/V2/V3合同、源码与失败全部保留。见第3.83节及[编码复核报告](paper/candidates/bounded_retention/p1_encoding_results.md)。<br>
> 既有B审查：ARS合法人工实例跨尺度观测B已完成，停止泛化候选（STOP_GENERIC_B_NO_DISTINCT_SIGNAL）。6471个人工tile长边131–548，载入640均放大，首次640下采样细节缺口不成立；全图960确增上下文，但同目标名义线性尺度仅为tile640的35.54%–41.10%。人工实例对应不同于旧伪框稳定性诊断，亦可能新增训练约束；对照SoCo/PseCo/MixTeacher/密度裁片/MSAD，尚无独立新信号/动作，不实现loss或训练，也不宣称跨尺度学习普遍无效。见第3.80节及[独立审查](paper/diagnostics/human_multiscale_observation_review/results.md)。<br>
> 上一轮来源决定：已完成ARS“完整人工区域内跨对象排序A”来源审查：6471个人工tile/105058个导出框身份与当前P1清单一致，但裁片过滤丢失足迹/空间完整性证书，增强后证明也未建立；G1通过、G2/G3 NOT_ESTABLISHED，停止A，不做排序统计/loss/训练。实际面积比.09931134–.10027778低于可见门槛.35，纯合成反例排除仅靠固定边缘剔除认证背景；不代表实际可靠负例不存在。6484项绑定/新审查精确重放及1093tests通过，不读隐藏标签或用隐藏计数补齐。原P1/失败封印/冻结合同保护，无新监控；见第3.79节、第10节及[来源报告](paper/diagnostics/reliable_region_ranking_feasibility/results.md)。以下既有记录不替代上方当前方法决定。<br>
> 已有三臂终态：正目标几何重排V2三新学生各完整30轮exit0，548图val960及终态audit工程通过，但11线仅4过，停止候选；候选−P1 AP50–95−.3145pp/AP_small−.3453pp/conf.25 small−.9444pp，IoU.50/P90+.5418pp却对控制−.0301pp。同FP少160个small/同K少103，去掉任一类别后AP_small差值仍负。该轮提出的A仅获来源审查资格，已按上一条完成并停止；原三臂正负结果不改。见第3.78节与[复盘](paper/diagnostics/positive_quality_negative_review/results.md)。<br>
> 最新修复准备：新增独立V2 Trainer集成修复，仅改变train/val数据集派发，原科学loss/模型与11门槛不变；原V1失败封印/源码/合同全部保护。三mode真实dataloader合成batch回归、两份旧缓存完整loss重放、完整1088tests及40267项绑定check-only exit0通过；新执行入口`run_positive_quality_transport_v2.py`，详见paper/candidates/positive_quality_transport_v2/。本次未启动训练/新真实batch/检测模型forward/optimizer，未验证完整Trainer setup，学生性能NOT_TESTED、机制NOT_ESTABLISHED；真实启动后再建临时观察。<br>
> 最新工程终态（10:47:57）：用户手动启动正目标几何重排三臂入口，reference在验证数据集初始化时因Trainer方法复用的零参数super继承不兼容而exit1，0完整epoch、无results.csv/best；candidate/control未启动。父/worker均已停止，独立只读audit exit0且terminal_evidence_verified=true，但accepted=false/performance=NOT_ACCEPTED；不作方法性能失败判定。原封印目录/源码/合同保留，未修复或重试；监控终态清理，详见第10节。<br>
> 本轮新研究：在b1db01e干净工作区上完成正目标几何排序文献/零训练诊断与独立最小实现，未启动学生训练。实际CUDA AMP人工small多锚点126个中100个逆序、32个大差逆序，small目标搬移8.7506%；框外质量35.3208→38.6970的反证保留。新原P1/候选/匹配扰动控制各30轮及11条保覆盖门槛已冻结；935tests、完整loss/来源/audit及零训练预检通过，学生性能/机制仍未验证。见第3.77节及paper/candidates/positive_quality_transport/，不覆盖下列旧终态。<br>
> 当前终态：人工GT可信竞争负监督新参考／候选／控制各完整30轮exit0，统一548图val960及独立只读audit通过。候选−P1 AP50–95−0.3502pp／AP_small−0.1088pp／conf.25 small−3.1682pp；P90 small+1.1890pp／FP−22.5377%／wrong-class FP−25.5983%。10项门槛5过，停止候选；对控制AP−0.2034pp，机制未建立。工程通过与性能投入条件分开；保留原P1及旧V2失败，不重试／恢复／扩训；详见第3.76节和新结果／独立验证报告，临时rcn-30终态收尾删除。<br>
> 最新诊断：V2固定conf.25 small净+99，但同7889 FP预算为−137、同27237预测数为−84；新增623 FP含错类218／背景166／重复121／近定位78／近异类40。原P1 P90的small FN中60.03%有同类低分可匹配框，高分FP中69.91%错类；严格IoU另有定位上限，不能认定唯一训练根因。完成零训练／零推理缓存精确重放及318tests；当时的新方向计划与诊断见第3.74节，后续工程准备见第3.75节，本次仍不训练。<br>
> 最新终态：AMP修复V2改进组完整30轮／exit0、三方法统一548图val960及独立audit通过。原P1−B2的AP50–95+0.3349pp／conf.25 small+2.2012pp；V2−P1的AP−0.1628pp／AP_small−0.2279pp／P90 small+0.3650pp／FP+7.8971%，七护栏仅2项通过，停止本候选、保留原P1，不扩训。完整表与限制见第3.73节及paper/candidates/competitive_negative_v2/results30.md；终态收尾删除p1-v2。<br>
> 最新排障：真实单批CUDA AMP复现表明，1个伪前景TAL质量8.7088e-9转FP16成为0，触发竞争负项过强断言；649个伪前景无教师／分配类别不一致，原P1损失正常。完成零优化步骤定位和dtype控制重放，286tests通过；未修订原训练实现或重训，见第3.70节。<br>
> 历史工程失败：原新主对照B2／P1完成30轮，但旧candidate首批断言exit1、0完整轮／无权重；原目录未统一评估，不给旧candidate性能结论。39,828输入和原证据保留，旧p1已删除，见第3.69节；该失败不覆盖顶部V2完整结果。<br>
> 当前对照：B2、原P1、P1＋竞争负项V2的绝对值及P1−B2／V2−P1／V2−B2均已完成。复用本次B2／P1，源码commit差异披露；无strength，机制NOT_TESTED。两个已有输出目录均勿重跑训练，只读复核用run_p1_main_comparison_repair.py --audit。<br>
> 历史研究准备：竞争负项曾完成六学生缓存／教师分数零训练诊断与独立工程准备，支持信号和误报反证见第3.66节；实际主对照未含强度控制，V2终态按原护栏停止。<br>
> 成果保留：现有P1源码基点c1ecf6c已制作独立代码/产物副本，58,707文件约4.20 GiB；正文入口paper/README.md、100轮主结果及讨论/结论已整理。新方向“原P1＋类别分数分类监督”已完成两臂各30轮及独立验收，IoU.50/P90 small +0.8203pp但AP/AP_small/P70退化，未通过扩大条件；详见第3.65节，原成果保留。<br>
> 当前执行范围：只推进A第一篇方法稿；B/C、蒸馏剪枝和树莓派部署留在远期规划，不并行启动。<br>
> 最新补证据：四个正式100轮冻结权重的高精度退化原因和尺度条件操作点已完成零训练诊断。90%precision下P1少找回的small GT分别695/711、730/752个已在完整缓存匹配但低于分数阈值；共同GT几何未变差。尺度条件阈值提高两臂约4.4–5.4pp small recall，但P1独有优势未通过固定继续线，见第3.62节。<br>
> 最新安排：P1/B2学生seed0/1正式100轮均已独立验收。seed1 P1−B2的mAP50-95+0.3351pp、small recall+2.4194pp（净+643），未匹配预测+1435、F1+0.4031pp。两个seed小目标覆盖收益重复出现，建议本轮主线训练收束、优先写作与已有机制分析；不自动扩训或恢复持续目标。详见第3.60节。<br>
> 当前阶段：正式100轮两seed平均配对增益为mAP50-95 +0.1939±0.1996pp、small recall +2.2726±0.2075pp（样本SD），与原30轮两seed的+0.2893/+1.7854pp分开报告。原Gate3不追认为通过，原Gate4效应量亦未证实达到。正式结果见paper/p1_b2_seed1_full100_results.md。<br>
> 运行更新：P1 seed1在55轮后由用户恢复，2026-11-02 17:20:27包装真实exit0、两进程停止；原55轮CSV前缀及21文件归档不变，原退出码仍未知。恢复独立验收、同seed1 B2重验及两seed配对完成；收尾删除临时监控p1-1，不重复训练。<br>
> 论文暂定题目：**面向高分辨率航拍图像的区域局部标注小目标半监督检测方法**  
> 仓库：`D:\Python\YOLO-RegionSSOD`  
> 计划起点：2026-10-26，可按实际启动日期整体平移

## 0. 一句话交接

本论文研究“高分辨率训练图只有局部区域有标签”时的航拍小目标半监督检测：
未标注区域中的目标仍然存在，不能当作纯背景。主实验固定 VisDrone、YOLO11s、
640 训练、完整原图 960 验证，先建立 B1 部分监督和 B2 朴素自训练基线，再验证
P1 尺度感知伪标签与 uncertain-ignore 是否带来稳定收益。

## 1. 论文定位

### 1.1 研究问题

每张高分辨率训练图只标注一个目标中心 tile，tile 外仍有大量未标注目标。
训练时如果把所有未标注像素当作背景，会产生错误负样本；如果直接丢弃 tile 外
区域，又会损失上下文和潜在目标信息。

### 1.2 核心假设

- H1：在不使用未标注区域时，B1 会明显低于完整监督参考 B0。
- H2：只用全局阈值伪标签的 B2 能获得一定收益，但对小目标存在系统性漏检。
- H3：按目标尺度设置伪标签阈值、按教师置信度加权，并把低置信度候选视为
  unknown 而不是背景，能在相同训练预算下优于 B2。

### 1.3 论文贡献

1. 建立区域局部标注下的高分辨率航拍小目标检测协议，记录固定 split 和未标注
   区域中的隐藏 GT，用于分析伪标签漏检。
2. 提出 tile-aware 自训练流程，将尺度感知阈值和教师置信度权重用于伪标签筛选。
3. 将低置信度教师预测作为 unknown/ignore 区域，减少小目标漏检造成的背景负样本
   污染，并按尺寸、密度和类别报告机制证据。

不要在论文中使用“首次”或“全面解决”表述。贡献必须由 B0/B1/B2/P1 对照、
消融和机制分析支撑。

## 2. 当前范围与边界

- 当前论文只使用图像检测，不切换到视频。
- 当前主数据集为 VisDrone train/val；外部确认集优先 UAVDT，但必须完成类别映射
  和版本审计后才纳入。
- 当前方法实现基于 YOLO11s，不把 backbone、neck、head、P2 或注意力改造作为
  第一轮变量。
- `docs/RGB_IR_Reliability_Experiment_Design.md` 是另一项 RGB-IR 可靠性融合
  设计，不属于本论文。不得混用数据、模型、指标或结论。
- 现有 VisDrone test-dev 已被旧项目多次查看，不再作为新论文的盲测集。

## 3. 当前状态

以下状态更新至2026-11-02。后续会话必须先运行 `git status` 并核对实际产物，
不能只依赖本文档的旧描述。

| 项目 | 状态 | 证据或位置 | 下一步 |
|---|---|---|---|
| Git 仓库 | B2 seed1、案例与论文更新已提交 `80e9f2b`；新增三篇文章/学位论文/树莓派路线为规划文档修改 | 第 3.20–3.21 节；`git status --short --branch` | 规划不等于训练/投稿或专利申请已启动 |
| 数据构造 | 已实现；10% seed0 已生成 | `src/regionssod/data/partial.py` | 生成正式 5%/20% split |
| B0 完整监督 | 配置已加入，未正式训练 | `configs/train_full.yaml` | 复用旧结果或重训 seed 0 |
| B1 部分监督 | 正式 100 epoch 和独立 30 epoch pilot 均完成；val960 mAP50-95 分别为 0.248835、0.229114 | `runs/partial10_sup_s0/`、`runs/partial10_sup_pilot30_s0/` | 冻结为正式参考与 pilot 对照 |
| B2 朴素自训练 | 两seed独立30轮及正式100轮均已验收；100轮seed1 AP50-95=25.6683%、small recall=39.3611% | 第3.20、3.54–3.60节；paper/b2_seed1_full100_results.md | seed0恢复事实保留，seed1未中断；已完成同seed1 P1配对，不重训 |
| P1 尺度感知方法 | 两seed正式100轮均已验收；seed1 AP50-95=26.0034%、small recall=41.7805%，相对B2 +0.3351/+2.4194pp、净+643小目标 | 第3.60节；paper/p1_b2_seed1_full100_results.md；实验4.21 | 保留两seed覆盖收益与误检/类别/成本代价，优先收束稿件；不宣称总体AP稳定领先或原Gate3/4通过 |
| 第①组 no-ignore 消融 | 30 epoch、val960、固定操作点 recall 及独立验收完成；相对完整 P1 的 mAP50-95 +0.0217 pp、small recall −0.1392 pp | `outputs/p1_no_ignore_pilot30_s0/{comparison,acceptance_audit}.json`；第 3.13 节 | 不能声称 ignore 独立提高整体 mAP；继续拆分软权重与尺度阈值，补配对 seed |
| 软权重低成本筛选 | 1024 源图、两组各 5 epoch 和 val960 完成；mAP50-95 +0.0884 pp、small recall +0.0790 pp，低于 0.3 pp 候选线 | `outputs/screens/p1_soft_weights_1024_e5/{comparison,acceptance_audit}.json`；第 3.15 节 | 保留当时中性停止结论；最新A补证据授权下单独启动正式30epoch消融，见第3.26节，不追认短筛过线 |
| 软权重全量单因素消融 | 30epoch/val960/recall/独立验收完成；去软权重mAP50-95−0.1623pp、small recall+0.9143pp | `outputs/p1_no_soft_weights_pilot30_s0/independent_audit.json`；第3.26节；表4-S5 | 未达扩大训练条件；保留完整P1主配置及取舍结果，不自动扩训 |
| 尺度阈值低成本筛选 | 统一 0.50 对比尺度分层：mAP50-95 +0.2237 pp、small recall −3.1230 pp | `outputs/screens/p1_global_t050_1024_v2_e5/{comparison,acceptance_audit}.json`；第 3.16 节 | 自动候选线未触发，但小目标信号值得全量 30 epoch 确认；数据已准备，用户手动启动 |
| 尺度阈值全量确认 | 30 epoch、val960、固定操作点 recall 及独立验收完成；统一 0.50 相对完整 P1 的 mAP50-95 −0.6177 pp、small recall −0.3424 pp，逐类 1 升/9 降 | `outputs/p1_global_t050_pilot30_s0/{comparison,acceptance_audit}.json`；表 4-S2；第 3.18 节 | 全量结果支持尺度分层阈值更有利于总体检测和小目标覆盖；仍为单 split/seed，不扩写为跨 seed 稳定性 |
| 尺度阈值配对学生 seed | seed1 两组各 30 epoch、完整 val960 和预测重放完成；尺度分层相对统一 0.50 的 mAP50-95 +0.2420 pp、small recall +0.6434 pp，与 seed0 两项同向 | `outputs/p1_scale_threshold_pair_s1/{comparison,acceptance_audit,independent_audit}.json`；表 4-S3；第 3.19 节 | 保留尺度分层；单 split/固定教师下两学生 seed 的平均配对增益为 +0.4299/+0.4929 pp，不作显著性或跨 split 主张 |
| 正式 5/10/20% 数据 | 10% seed0 已生成；20 图同 seed 检查通过；5%/20% 未生成 | `data/visdrone_area10/summary.json`、`audit.json` | 生成 5%/20% split |
| 正式 100 epoch 主矩阵 | B1 seed0及B2/P1学生seed0/1完成；P1 seed1的55轮后恢复已完整验收 | 第3.60节；两seed正式100轮比较及P1恢复证据 | 当前配对已收束；B0、其余预算/外部确认/S³OD同协议比较仍有缺口，按投稿主张排序，不自动铺满历史矩阵 |
| 论文正文 | 第1–4章、两seed统计/案例，以及表4-S5/实验4.14软权重全量消融已同步；原seed0主表保留 | `docs/paper/{experiments,results_tables,evidence,case_studies}.md`；第3.20、3.26节 | 正式矩阵与最终排版仍待补，后续先选一个低成本问题，不自动扩训 |
| 文章与树莓派部署规划 | 用户新增三篇文章目标及最终树莓派部署要求；论文按录用、专利按“申请通过”，硬件与学校节点待确认 | `docs/publication_and_edge_deployment_plan.md`；第 3.21 节 | 先 A 方法稿、C 轻量化/部署稿，再扩展 B 可靠补全稿；树莓派先做导出与板端基线，均未启动 |
| 第一篇实验缺口 | 已完成零训练漏检归因；6,471 tile/38,245 hidden GT 精确重放，按低成本列出 L0→L1→L2 | `docs/paper1_experiment_gaps.md`；`scripts/audit_paper1_gaps.py`；第 3.22 节 | 先做 ignore 可恢复增量与 NMS/max-det 候选覆盖小试；未过线不启动训练 |

### 3.1 已完成的烟测

- 20 张 VisDrone 图像，10% 标注面积：20 个标注 tile、321 个监督框。
- 20 个未标注 tile 中有 141 个隐藏真实框，说明“未标注不等于纯背景”。
- B2 使用 1 epoch、320 分辨率教师：20 个未标注 tile 中接受 13 个，生成 38 个框；
  合并后训练集为 33 张 tile。
- B1/B2 的伪标签生成、列表合并、训练和完整 val 验证链路均成功。
- 最后一次记录：`pytest` 9/9 通过。

**证据边界：**烟测只有 20 张图、1 epoch，B2 烟测 `mAP50-95 ≈ 0.0252`。
这只证明工程链路，不是方法效果，不得写入论文结果表，也不得据此判断研究方向。

### 3.2 正式 10% seed0 数据

- 2026-10-26 在 Git Bash 中加载 `D:/MiniConda/etc/profile.d/conda.sh`，激活已有
  `yolo` 环境后运行 `make-partial`；命令见 README 和 baseline 协议。
- 6471 张训练图全部使用，无跳过；6471 个标注 tile、6471 个未标注 tile。
- 标注 tile 共保留 105058 个监督框，未标注 tile 中共有 38245 个可见隐藏 GT 框。
- 实际平均标注面积比例为 9.993%；完整原图验证集仍为 548 张。
- `data/visdrone_area10/audit.json` 保存了 12 项基础检查、当前 commit 和关键文件
  SHA256；检查包括 split/manifest 一致、标签合法、隐藏 GT 未导出为训练标签、
  验证路径指向完整原图。

**证据边界：**该次数据构造只完成基础审计。后续的 20 图工程验收和正式 B1
训练分别见第 3.3、3.4 节。

### 3.3 Gate 1 工程验收通过（2026-10-26）

- 用户已完成 `partial10_sup_smoke5`：20 张标注 tile、5 epoch、640、batch 8、
  SGD、seed 0，`results.csv` 包含完整 5 行；所有数值有限，best/last 权重齐全。
- box loss 从 2.04970 降至 1.77198；该趋势仅用于排除明显工程异常。
- 本次补跑 `best.pt` 的完整原图 960 验证：548 张图、38759 个实例；Ultralytics
  开发指标 mAP50=0.117764、mAP50-95=0.080882。
- 同 seed 重生成 20 图数据，manifest 在归一化输出路径后完全一致；40 张裁剪
  图像与 20 个标签文件的 SHA256 均一致，且与正式数据的前 20 图对应。
- 正式数据基础审计所记录的 8 个文件哈希均未变化；`python -m pytest` 9/9 通过。
- 证据：`runs/smoke/partial10_sup_smoke5/gate1_audit.json`、
  `runs/val/partial10_sup_smoke5_val960/metrics.json`、
  `data/gate1_checks/area10_seed0_20261026`。

**结论：**第 9 节定义的 Gate 1 已通过。上述数字仍是 20 图工程烟测，不能进入
论文结果表，也不能证明 B1/B2/P1 的正式效果。当时建议的 B1 seed0、100 epoch
现已完成，见第 3.4 节；仍需完成 B1/B2 的 30 epoch pilot，才能决定是否实现 P1。

### 3.4 正式 B1 完成，B2 数据准备完成（2026-10-26）

- 用户已完成 `runs/partial10_sup_s0`：6471 张标注 tile、100 epoch、YOLO11s、
  640、batch 8、SGD、seed 0；从 COCO 预训练初始化。日志共 100 行、全部数值
  有限，CSV 累计训练耗时 11460.1 秒（约 3.18 小时）。训练 commit 为 `14eaf2c`。
- 本次补跑 `best.pt` 的完整原图 960 验证：548 张图、38759 个实例；Ultralytics
  mAP50=0.417503、mAP50-95=0.248835、precision=0.559604、recall=0.408240。
  训练日志的 640 验证最佳 mAP50-95 为 0.16704（epoch 90），须与 val960 区分。
- 8 项正式 run 审计通过；原 split/manifest 等关键哈希未变。证据和 SHA256 保存在
  `runs/partial10_sup_s0/audit.json`、`SHA256SUMS.txt`、`validation/metrics.json`。
- 固定教师 `runs/partial10_sup_s0/weights/best.pt`，SHA256 为
  `58f40e1e7b397c3f0b873bbc9cbd9e50660b41c48c9cadb1e7ef3d0f0b0ebf54`。
- 正式伪标签首轮暴露了批处理缺陷：Python 路径列表使 Ultralytics 将全部 6471
  图当作一个 batch，触发 CUDA OOM。已改为持久化文件清单输入，按 batch 8
  流式读取；回归测试在旧实现上复现问题，修复后全部 11 项测试通过。
  失败目录归档为 `data/visdrone_area10/pseudo/naive_t050_failed_all_images_batch_20261026`。
- 修复后以 threshold=0.50、conf-floor=0.05、imgsz=960、batch=8 完成 6471 图
  推理；接受 4624 个 tile（71.46%）、27667 个框，10 类均有伪标签。合并后
  11095 张训练 tile、132725 个框，完整验证集保持不变，未将空预测 tile 当背景。
- 离线隐藏 GT 诊断：按类别、IoU>=0.50、降序 IoU 一对一匹配，伪标签 precision
  为 81.47%、recall 为 58.94%，漏检隐藏 GT 15704 个；小目标 recall 为 46.99%。
  尺寸按原始 tile 中裁剪后框面积划分（small<32²、medium<96²），这不是 AP_small
  或官方评测，不能证明 B2 学生已有收益。隐藏 GT 只用于离线分析。
- B2 数据通过 10 项审计；其标签、清单、配置、源代码及教师共 11106 个文件的
  SHA256 已保存。证据位于 `data/visdrone_area10/pseudo/naive_t050/` 下的
  `summary.json`、`audit.json`、`quality_audit.json`、`SHA256SUMS.txt`；合并配置为
  `data/visdrone_area10/dataset_naive_t050.yaml`。诊断脚本在忽略目录
  `outputs/audit_naive_t050_20261026.py`。

**当时的下一步：**按 baseline 协议第 4.1 节运行 B1/B2 各 30 epoch 的独立学生并
完成完整 val960，现已完成，见第 3.5 节。B2 和后续 P1 pilot 固定使用上述 B1
100 epoch 教师；学生训练轮数配对，实际训练数据量、批次数和耗时分别记录，
教师成本另记。不得将 B2 30 epoch 与 B1 100 epoch 或其第 30 行 CSV 作 gate
比较。官方 AP、AP_small 和多 seed 仍待补。

### 3.5 Gate 2 配对 pilot 通过（2026-10-27）

- 用户已完成 `partial10_sup_pilot30_s0` 和 `partial10_naive_t050_pilot30_s0`，
  训练 commit 均为 `f7b711c`。两者均从 COCO 预训练独立开始，完整训练 30 epoch；
  640、batch 8、SGD、seed 0、增强及其他实际训练参数一致，仅训练数据和 run
  名称不同。训练 CSV 数值全部有限，best/last 权重齐全。
- 本次分别补跑两者的 `best.pt` 完整原图 960 验证，均为 548 张图、38759 个实例。

| 方法 | 训练 tile | 学生训练耗时 | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| B1 30 epoch | 6471 | 3523.76 秒 | 0.382766 | 0.229114 | 0.512319 | 0.385004 |
| B2 30 epoch | 11095 | 5073.12 秒 | 0.389202 | 0.234534 | 0.520438 | 0.386509 |

- B2−B1 的 mAP50-95 为 +0.005421，即 **+0.5421 个百分点**；mAP50 为
  +0.6436 个百分点。Gate 2 要求差值不低于 -0.002，因此 **Gate 2 通过**。
- 每类 AP 的方向为 9 类上升、1 类下降；awning-tricycle 为 -0.1817 个百分点。
  这是单 split/seed 的开发 pilot，不做显著性或跨 seed 稳定性结论。
- 核对了原数据 8 个关键文件和伪标签包 11106 个文件的哈希，全部与冻结记录一致；
  固定教师未变。两个 run 下均保存了 `audit.json`、`SHA256SUMS.txt` 和
  `validation/metrics.json`。汇总证据为 `outputs/gate2_20261027/comparison.json`，
  输入审计为同目录 `input_integrity.json`。
- 本对照固定的是学生 epoch；B2 的数据量、每轮批次数和训练耗时更多，并额外使用
  B1 100 epoch 教师。不能据此宣称同等计算量下的收益。指标仍为 Ultralytics
  开发指标，未补官方 AP、AP_small、AR500 或多 seed。

**当时的下一步：**开始实现 P1 的尺度分层阈值、教师置信度软权重和 uncertain-ignore。
先完成候选/置信度保留和损失侧的权重、ignore 路径及测试，再使用同一固定教师、
同一 split、COCO 初始化和 30 epoch 学生协议对照 B2。Gate 3 要求 P1 相对 B2
至少 +0.5 个百分点 mAP50-95（即当前基线下至少 0.239534），并且 AP_small 或
小目标 recall 同向改善。Gate 3 通过前不添加网络模块或扩展 P1 正式主矩阵。
本次仅完成训练验收、验证与记录；P1 代码和训练均尚未开始。

### 3.6 P1 已实现，全量数据就绪，30 epoch pilot 已启动（2026-10-27）

- P1 第一轮只加入尺度阈值、教师置信度分类软权重、uncertain-ignore，没有修改
  YOLO11s 结构或 box/DFL gains。绑定 Ultralytics 8.4.172；方法、阈值边界、
  增强传递方式和损失公式语义见 `docs/baseline_protocol.md` 第 4.3 节。
- 固定原 tile 面积 small<32²、medium<96²、large>=96²；正阈值分别为
  0.30/0.50/0.50，下限 0.05。仅含正框的 tile 入训，其余候选仍保存在离线记录。
  教师、split、960 推理、batch8、NMS 和 max_det 均沿用 B2。
- `python -m pytest --basetemp outputs/pytest_p1_20261027` **28/28 通过**。覆盖
  尺度边界、阈值合法性、单位权重时与原 loss 完全一致、空正框、分类梯度、
  ignore/正样本优先、回归不变、跨图隔离、mosaic/翻转/close_mosaic 对齐、
  元信息与标签错配拒绝、实际流式推理 batch，以及固定 recall 匹配。
- 真实 GPU 工程烟测：同一教师对前 20 个未标注 tile 生成 17 个伪标签 tile、
  139 个正框和 325 个 ignore 框；与 20 个标注 tile 合并为 37 张，640/batch8
  完成 1 epoch 和 548 图训练内完整验证。best/last 齐全，CSV 全部有限，独立
  进程重新加载 `P1DetectionModel` 推理正常。证据为
  `runs/smoke/p1_integration_20261027`、`outputs/p1_smoke_*_20261027.log`。
  **这些烟测结果不进入论文结果表，也不用于判断 Gate 3。**
- 全量 P1 生成完成：6471 个未标注 tile 中接受 4839 个，35067 个正框、52117
  个 ignore 框；正框按尺度为 small 20666、medium 12637、large 1764。合并学生
  训练集 11310 张，原始正标签行共 140125。标准 Ultralytics loader 会删除人工
  标签中的 1 个重复框，实际加载 140124 个正框；没有修改原标签文件，去重逻辑
  与 B1/B2 相同。数据在 `data/visdrone_area10/pseudo/p1_s030_m050_l050/`。
- 隐藏 GT 仅用于**离线**诊断：P1 precision=73.20%、recall=67.12%、小目标
  recall=60.36%；B2 分别为 81.47%、58.94%、46.99%。增加覆盖的同时精度下降，
  不能据此声称学生性能改善。相同 IoU/匹配规则和边界可见性规则的证据为 P1
  目录下 `quality_audit.json`，脚本为 `outputs/audit_p1_20261027.py`。
- B2 学生已补固定操作点完整原图分尺寸诊断：548 张、38759 GT，conf=0.25、
  IoU>=0.50、max_det=500；小目标 recall=8987/26577=0.33814952778718443。
  证据为 `runs/size_recall/b2_pilot30_s0/metrics.json`，其权重 SHA256 与 Gate 2
  一致。此指标不是官方 AP_small/AR500，也不是上述伪标签 recall。
- 全量预检通过：8 个原 split/manifest 关键文件和 B2 的 11103 个数据/权重文件
  哈希未变；全部 P1 正框与置信度元信息一致，完整 val548 路径保持原图。配置、
  源码/测试和全部训练/验证输入已冻结 SHA256，脏源码快照存于
  `outputs/gate3_20261027/source_snapshot/`；预检为同目录 `preflight.json`，
  生成审计脚本为 `outputs/prepare_p1_20261027.py`。没有把新源码哈希与 B2 的
  历史源码哈希混作“未变化”证据。
- 北京时间 **2026-10-27 10:33** 启动后台 `scripts/run_p1_pilot.py`，run 为
  `runs/partial10_p1_pilot30_s0`：同 COCO 初始化、640、30 epoch、SGD、batch8、
  seed0。P1/B2 每 epoch 批次数分别为 1414/1387；学生耗时与教师成本另记。
  脚本会依次完成训练参数/CSV 审核、完整 val960、同操作点小目标 recall、
  权重哈希和 `comparison.json`；失败写入 `status.json` 并停止，不会扩训。
  日志为 `outputs/p1_pilot30_20261027.{stdout,stderr}.log`，当前状态必须读取
  `outputs/gate3_20261027/status.json`，不得仅凭本启动记录认定训练已完成。

**当时尚未完成：**30 epoch pilot 和 Gate 3 的最终验收，现已完成，见第 3.8 节。
原通过条件为 P1 mAP50-95
至少 0.239534294688062，且小目标 recall 严格高于 0.33814952778718443。
未测得两个条件前不扩展正式 P1 主矩阵，不宣称 P1 有效。

### 3.7 P1 训练进度核查：尚未完成（2026-10-27 11:04，北京时间）

- 后台 Python 进程 PID 9500 仍在运行，命令为 `scripts/run_p1_pilot.py`；
  `status.json` 仍处于 `training`。该文件仅记录阶段切换，实时 epoch 进度应以
  stdout 日志和 `results.csv` 为准，不把阶段时间戳较旧误判为训练停滞。
- CSV 已记录完整、连续的 epoch 1–6，全部数值有限；最近一行累计训练时间
  1618.76 秒。11:04:53 日志已到第 7/30 epoch 的 1344/1414 batch（约 95%）。
  best.pt、last.pt 均存在；stderr 为 0 字节，核查时未见训练错误。
- 实际 `args.yaml` 与 B2 对比，除 data/name/save_dir 外无参数差异。训练内的
  640 验证值属于中间曲线，不能当作最终完整原图 val960 结果或提前判定 Gate 3。
- P1 的完整 val960 `metrics.json`、分尺寸 recall 和 Gate 3 `comparison.json`
  均尚未生成。因此本次仅完成**训练进度核查**，不能验收最终效果；Gate 3 是
  **尚未测量**，不是通过或失败。原后台训练继续运行，完成后由已启动脚本自动
  执行完整验证、固定操作点小目标 recall 和比较，不另行启动重复训练。
- 核查证据：`outputs/gate3_20261027/progress_check_20261027_110453.json`。

### 3.8 P1 结果验收：取得初步收益，保留阶段成果（2026-10-27）

P1 的 30 epoch 训练、完整原图 val960 和固定操作点小目标 recall 已于北京时间
**12:36:10** 全部完成，12:44 完成独立验收。**结果显示正向收益，可作为方法的
阶段成果整理。**原 Gate 3 的 +0.5 个百分点 mAP50-95 门槛未达到，小目标 recall
同向改善条件已达到，故原 Gate 3 判定为未通过。

| 方法（独立 30 epoch，seed0） | 训练 tile | 学生耗时（秒） | mAP50 | mAP50-95 | 小目标 recall |
|---|---:|---:|---:|---:|---:|
| B2 全局阈值自训练 | 11095 | 5073.12 | 38.9202% | 23.4534% | 33.8150% |
| P1 尺度阈值 + 软权重 + ignore | 11310 | 7286.61 | 39.4990% | 23.7883% | 35.6925% |
| P1−B2（百分点） | — | — | **+0.5788** | **+0.3349** | **+1.8776** |

- 两者使用同一完整原图 val：548 张、38759 个 GT。小目标 recall 固定
  conf=0.25、匹配 IoU>=0.50、max_det=500、原图 GT 框面积<32²，匹配规则一致。
  P1 匹配小目标 9486/26577，B2 为 8987/26577，即多匹配 499 个小目标。
  这是学生完整验证集的诊断 recall，与隐藏训练 GT 的伪标签 recall 分开报告。
- 原门槛要求 P1 mAP50-95>=0.239534294688062，实际为
  0.23788331849556607，差 0.1651 个百分点；因此自动判定与人工复核均为
  `gate3_passed=false`。该检查项未通过不等于没有成果：mAP50、mAP50-95
  和小目标 recall 均为正向变化。
- **验收检查 19/19 通过**：30 行 epoch 连续、CSV 数值全部有限、best/last 齐全、
  训练参数与 B2 除 data/name/save_dir 外一致；共 23773 个冻结输入、19 个 run
  记录的源码/输入和 8 个输出文件哈希均匹配。教师与 B2 所用教师一致，P1
  评估权重对应本轮 best.pt，均可追溯；
  两次分尺寸评估的 548 图列表/顺序和 GT 数一致，stderr 为空。
- best.pt SHA256：
  `8d20c9645328d9f1a18429c7223d40e3a123f211ec04e86b2f697a844a51e115`。
  last.pt SHA256：
  `03cf3b6c4f9bb238001179a44667610c5a9344935fd3ed9cc820add79c7253ca`。
- 证据：`runs/partial10_p1_pilot30_s0/{audit.json,SHA256SUMS.txt,results.csv}`、
  `runs/val/partial10_p1_pilot30_s0_val960/metrics.json`、
  `runs/size_recall/{b2_pilot30_s0,p1_pilot30_s0}/metrics.json`、
  `outputs/gate3_20261027/{comparison,acceptance_audit}.json`。
  独立验收脚本为 `outputs/audit_gate3_acceptance_20261027.py`。
- 当前证据是单 split/seed 的 Ultralytics 开发结果。可以表述为“在本次配对
  pilot 中，P1 提高整体检测指标和小目标召回”；跨 seed 稳定性和统计显著性
  尚待补充，官方 AP/AP_small/AR500 也需与这些开发指标分别报告。
- **结果后的工作决策：**用户明确要求“有提升，有成果就好”。因此将本轮作为
  积极的初步成果保留，优先整理方法、配对结果表、伪标签覆盖/精度取舍和案例。
  原 Gate 3 数值和判定保留用于复盘，后续工作不以本次未达 0.5 个百分点门槛
  为由否定已有成果。若需要增强论文证据，可按具体需要选择单因素消融或重复
  seed 验证；不把扩大完整矩阵作为整理现有成果的前提。

### 3.9 论文结果表格与方法和实验初稿（2026-10-27）

- 写作入口为 `docs/paper/README.md`。`methods.md` 已写出区域协议、固定离线
  教师、原始 tile 尺度阈值、置信度分类软权重、ignore 掩码及总体损失公式；
  描述与现有 P1 实现对应。`experiments.md` 已写出设置、整体结果、分尺寸
  recall、伪标签精度与覆盖取舍、逐类别表现和运行开销分析。
- `results_tables.md` 包含表 4-1 至 4-6、表 4-S1 和 4-S2，共八组：主结果/差值、
  分尺寸 recall、伪标签覆盖、伪标签质量、逐类别 AP50-95、学生规模/耗时，
  以及单列的 B1 30/100 epoch 参考。主对照只使用三个独立 30 epoch 学生，
  未混入工程烟测；百分数保留三位小数，差值用未舍入值计算。
- 为补齐三组的相同操作点评估，使用冻结 B1 30 epoch best.pt 补跑完整 val960
  `size-recall`，没有重新训练。结果 small/medium/large recall 分别为
  34.379%/76.302%/89.316%，small 匹配 9137/26577；权重哈希仍为
  `5c9e531a5e04929261fbeb57854f78703c011db00b226a8250683cbe232f86f4`。
  证据为 `runs/size_recall/b1_pilot30_s0/metrics.json`，命令见 baseline 协议。
- 分尺寸分析明确记录：B2 整体 mAP 提高，但固定 conf=0.25 下 small recall
  低于 B1；P1 相对 B1/B2 均提高该 recall。逐类别 AP 中 P1 相对 B2 为
  **7 类升、3 类降**；论文草稿据此描述收益分布，没有表述为全部类别改善。
- 新增 `scripts/build_paper_tables.py`，从冻结 JSON/CSV 生成表格并用
  `--check` 逐字核对；同时校验 best.pt 哈希、30 行日志、配对参数、548 图
  和 GT 分组/操作点。27 个来源的哈希、完整精度数值和核对记录保存为
  `outputs/paper_tables_20261027/audit.json`；写作来源与权重索引见
  `docs/paper/evidence.md`。
- 当前交付为**方法与实验章节初稿**。引言、相关工作、案例图和最终排版可在
  此基础上继续整合。原 Gate 3 与用户保留初步成果的决定按第 3.8 节记录。
- `python scripts/build_paper_tables.py --check` 核对通过；
  `python -m pytest --basetemp outputs/pytest_paper_writing_20261027` 28/28 通过。

### 3.10 相关工作与参考论文整理（2026-10-27）

- 新增 `docs/paper/related_work.md`，按半监督目标检测、漏标/稀疏标注、航拍
  小目标检测及与本文方法的关系组织，并直接服务于后续第 2 章写作。
- 新增 `docs/paper/references.md` 和 `references.bib`，共 16 项文献。核心必读
  为 S³OD、Soft Teacher、Consistent-Teacher、SparseDet 和 Density Crop-guided
  SSOD；另整理阅读重点及引用限制，后续不应只按题名堆积参考文献。
- **最近的参考工作是 S³OD**（arXiv:2310.14718，当前按预印本引用）：其尺寸感知
  自适应阈值、尺度重平衡标签分配和教师引导负学习与本文尺度阈值及不确定区域
  处理直接相关。相关工作已明确，本文不能声称首次结合尺度阈值和教师负学习；
  当前可支持的差异主要是区域局部标注协议、不修改 YOLO11s 网络的实现，以及
  完整原图上的伪标签覆盖/精度和大小目标收益分析。正式投稿前应核对其预印本
  后续版本，并尽可能做同协议对照或单因素消融。
- `methods.md` 和 `experiments.md` 已加入关键引用。文献编号当前按基础检测器/
  数据、SSOD、缺失标注、航拍尺度工作的顺序编排；完成引言后需按全文首次引用
  顺序统一重排。
- 文献线索经论文搜索和官方出版页/论文页核对；S³OD 与 Density Crop-guided SSOD
  均标注为预印本。未把不同数据集上的论文数值直接与本文 VisDrone 开发结果比较。

### 3.11 S³OD 方法差异对照与引言初稿（2026-10-27）

- 新增 `docs/paper/s3od_comparison.md`：按标注预算、区域构造、师生更新、尺度
  阈值、正样本分配、分类权重、负学习、框类型和评测口径逐项对照，并连接
  S³OD 原文段落、公式与 P1 实现；不从不同协议的数值推断方法优劣。
- 更新第 3.10 节的历史文献信息：S³OD 已以 *S³OD: Size-unbiased semi-supervised
  object detection in aerial images* 正式发表于 ISPRS Journal of Photogrammetry
  and Remote Sensing，2025，221：179–192，DOI `10.1016/j.isprsjprs.2025.01.037`。
  `references.md` 与 BibTeX 已更新；详细公式仍明确归于已读的 2023 预印本。
- 期刊全文本次未取得。核对作者公开 master 提交
  `d7806b78947cdc403d525e67a28193a88a4f8598`：其 DOTA 配置含 GMM 自适应阈值，
  与预印本百分位方案不同；另核对 GWD top-K 分配和教师 RoI 负候选处理，
  没有声称复现或运行 S³OD。期刊摘要的实验集为 DOTA-v1.5 与 SODA-A。
- 新增 `docs/paper/introduction.md`，包含研究背景、区域面积预算、错误负监督
  问题、已有工作关系、方法概述、三项工作和单 seed 初步结果。明确目标中心
  采样为合成协议，面积预算不能等同于实测工时节省，不声称首次或领先 S³OD。
- 按 Soft Teacher 第 3.2 节修正方法/相关工作/文献阅读说明：其具体公式对负候选
  背景分类项按教师背景分数加权，与 P1 的正 anchor 前景置信度权重不同。
- README 与论文入口、证据索引已同步；现有引用编号暂保持一致，最终排版时
  再按全文首次引用顺序重排。本次为文稿与文献核对，不新增实验或改变 Gate 3。
- `python scripts/build_paper_tables.py --check` 通过，27 个结果来源核对一致。
  忽略目录中的 `outputs/verify_writing_20261027.py` 核对 9 份论文 Markdown、
  36 个本地链接、16 项文献/BibTeX 及引言 6 个结果数值，全部通过；审计记录为
  `outputs/literature/s3od/writing_audit_20261027.json`。`git diff --check` 通过。
  未改训练源码，未重跑训练测试或启动训练。

### 3.12 第①组去掉 uncertain-ignore 的训练入口就绪（2026-10-27）

- 按用户要求新增 `configs/train_p1_no_ignore_10.yaml`，run 为
  `partial10_p1_no_ignore_pilot30_s0`。与完整 P1 的冻结配置相比仅更改名称和
  `p1.uncertain_ignore=false`；保留软权重、尺度阈值、11310 个训练 tile、元信息、
  增强链路、同一 COCO 初始化、640/30 epoch/batch8/SGD/seed0。
- 新增 `scripts/run_p1_no_ignore.py`，执行后依次训练、验证连续 30 epoch 和实际
  参数、完整原图 val960、相同操作点分尺寸 recall，再与完整 P1 自动生成 JSON
  及 Markdown 对照。差值为“去掉 ignore − 完整 P1”，不预设结果方向，不改变
  原 Gate 3；失败写入阶段记录并停止，已有 run/派发记录时拒绝重复启动。
- 已在原 `yolo` 环境执行 `python scripts/run_p1_no_ignore.py --check-only`：
  **23781 个冻结输入与完整 P1 参考产物哈希核对通过**，确认仅变更目标开关。
  预检在 `outputs/p1_no_ignore_pilot30_s0/preflight.json`，不表示训练已完成。
- `python -m pytest --basetemp outputs/pytest_p1_no_ignore_20261027` 28/28 通过；
  README 和 baseline 协议第 4.4 节已加入自动入口与独立训练/评估命令。
- **当时状态仅为配置、脚本与预检完成，尚未启动训练**，按用户“给我训练指令”
  提供可直接运行的 `python -u scripts/run_p1_no_ignore.py`。启动后的阶段文件为
  `outputs/p1_no_ignore_pilot30_s0/status.json`；全部完成后查看同目录
  `comparison.md`、`comparison.json`；本组现已完成并验收，见第 3.13 节。

### 3.13 第①组 no-ignore 消融完成验收（2026-10-27）

- 用户已完成 `runs/partial10_p1_no_ignore_pilot30_s0`，自动流程于北京时间
  **17:12:53** 完成。CSV 为连续 1–30 epoch，所有数值有限；best/last 权重齐全，
  实际训练参数与完整 P1 除 name/save_dir 外一致。唯一方法变量为
  `p1.uncertain_ignore=true→false`，软权重、数据、教师和训练源码保持一致。
- 完整原图 val960 与固定操作点分尺寸 recall 均已完成。差值定义为“去掉
  ignore − 完整 P1”，单位为百分点：

| 指标 | 完整 P1 / % | 去掉 ignore / % | 差值 / pp |
|---|---:|---:|---:|
| mAP50 | 39.4990 | 39.5405 | +0.0415 |
| mAP50-95 | 23.7883 | 23.8100 | +0.0217 |
| Small recall | 35.6925 | 35.5533 | −0.1392 |
| Medium recall | 77.5709 | 77.1390 | −0.4318 |
| Large recall | 89.4096 | 88.0037 | −1.4058 |

- recall 操作点保持 conf=0.25、匹配 IoU=0.50、NMS IoU=0.70、max_det=500，
  两组为相同 548 张完整原图和 38759 个 GT。small 匹配数 9486→9449（−37），
  medium 为 8622→8574（−48），large 为 954→939（−15）；此指标不是 AP_small。
- 逐类别 AP50-95 为 4 类升、6 类降。去掉 ignore 后，bus +1.2027 pp、
  tricycle +0.8362 pp；truck −0.7854 pp、van −0.7845 pp。总体近似持平包含
  不同类别的增减抵消，不能描述为所有类别一致改善。
- 日志累计学生耗时为 7169.38 秒，完整 P1 为 7286.61 秒；单次运行的差值不能
  解释为 ignore 模块的精确耗时。no-ignore best.pt SHA256 为
  `9c98ea42a1d656135ca562e84c86adf8ab53894fbaed4f0e6487643611f8aef0`。
- 独立审计 `outputs/audit_p1_no_ignore_acceptance_20261027.py` 已通过：23795 个
  冻结输入/源码/参考及输出文件哈希匹配，run 记录的 19 个输入哈希匹配；
  用两组保存预测分别重算全部 548 图的 GT 匹配，分尺寸计数与报告一致。
  审计为 `outputs/p1_no_ignore_pilot30_s0/acceptance_audit.json`。
- **结论：**本轮去掉 ignore 后整体 mAP 几乎不变，固定操作点召回下降。
  不支持将整体 mAP 收益主要归因于 ignore，也不能从单 seed 断言该机制稳定
  有效或无效。保留两组结果，下一步优先做去掉软权重的同条件对照，随后补
  尺度阈值消融及重复 seed；本次没有启动新训练或 GPU 评估，原 Gate 3 不变。

### 3.14 用户约束更新与软权重低成本筛选启动（2026-10-27）

- 用户要求按步骤继续实验，每个完成实验提交并推送 Gitee，提交信息用中文；
  随后补充“每次验证实验成本不要太大，当找到了值得后续训练的东西后给出
  训练指令停止”。后续以这项低成本及停止条件为当前执行范围，不自动扩训
  30/100 epoch 或完整主矩阵。
- 第①组验收内容已在 `4fe664d`，本次用 `git ls-remote origin refs/heads/main`
  核实 Gitee main 为同一提交，初始工作区干净，不重复提交已有结果。
- 准备了全量去软权重配置/入口 `configs/train_p1_no_soft_weights_10.yaml`、
  `scripts/run_p1_no_soft_weights.py`，仅关闭软权重、保留 ignore。23781 个冻结
  文件预检通过，但按用户更新要求**没有启动这组全量 30 epoch 训练**。
- 已于北京时间 **17:44:44** 启动 `scripts/screen_p1_soft_weights.py`，PID **5784**。
  固定抽取 1024 张源训练图，样本选择 seed=20261027，不读取隐藏 GT 作筛选。
  对应 1024 个标注 tile、741 个已接受伪标签 tile，共 1765 个训练 tile，
  两组共用同一列表与相同元信息；分别保留/去掉软权重，均保留 ignore。
- 两组从同一 COCO 权重独立训练 5 epoch，640/batch8/SGD/seed0；短程对照
  统一 `close_mosaic=0`，默认 warmup=3 epoch。每组完成后各做完整原图 val960
  和固定操作点 recall。提前固定的候选信号为：mAP50-95 差绝对值至少 0.3 pp，
  small recall 同向。它是探索性筛选标准，不是论文效应门槛或显著性检验。
- 31/31 tests 通过，其中新增独立梯度检查，验证仅关闭某一开关时另一个机制
  仍起作用。原论文表格的 27 个来源核对通过，原 Gate 3 判定不变。
- 当时仅已启动筛选，尚无结果；阶段和日志在
  `outputs/screens/p1_soft_weights_1024_e5/status.json`、同层 stdout/stderr 日志。
  此轮结果不得进入正式论文表或与既有 30 epoch 结果直接作方法增益比较。

### 3.15 软权重低成本配对筛选完成（2026-10-27）

- 配对筛选于北京时间 **17:53:05** 完成，PID 5784 已结束。两组各完成连续
  5 epoch，CSV 数值有限；共享固定 1024 张源图、1765 个训练 tile 和相同
  val960/recall 操作点，唯一方法变量为 `soft_weights`。
- 保留软权重与去掉软权重结果如下；15% 左右的 mAP 和 28% 左右的 recall 是
  1024 图子集、5 epoch 模型的探索值，不能与已有全量 30 epoch 数字并列：

| 指标 | 保留软权重 / % | 去掉软权重 / % | 差值 / pp |
|---|---:|---:|---:|
| mAP50 | 26.8181 | 26.5932 | −0.2249 |
| mAP50-95 | 15.3751 | 15.4635 | +0.0884 |
| Small recall | 28.0129 | 28.0920 | +0.0790 |

- 去掉软权重后 mAP50-95 与 small recall 同向为正，但绝对值均小于预先固定的
  0.3 个百分点筛选线，mAP50 下降，因此 `promising_direction=null`。**结论为
  中性，不启动全量去软权重 30 epoch 训练**；不能据此说软权重无效。
- 独立审计 27 项检查全部通过：1024 图/1765 tile、两开关、5 epoch、权重哈希、
  完整 val960、548 图 recall 操作点和差值复算均一致；本轮总日志学生耗时
  417.319 秒。记录在 `outputs/screens/p1_soft_weights_1024_e5/`。
- 全量确认配置和入口保留，但只作为之后需要时的人工入口，不自动派发。下一步
  继续下一项低成本候选筛选，仍须先固定筛选信号，完成一个实验后提交并推送。

### 3.16 尺度阈值低成本筛选完成（2026-10-27）

- 沿用相同的 1024 张训练原图。尺度分层组复用软权重筛选中的完整 P1 5 epoch
  臂；新组仅把 small/medium/large 阈值统一为 0.50，软权重和 ignore 均保留。
  `scripts/screen_p1_scale_threshold.py` 从冻结教师候选重划正/ignore，并新写
  对应标签文件，避免复用尺度分层标签；新组为 713 个伪标签 tile、1737 个训练 tile。
- 新组于北京时间 **18:00:37** 完成，学生训练 238.414 秒。独立审计验证 713 条
  候选的状态和标签逐项可从冻结候选复现，5 epoch、权重哈希、完整 val960、
  548 图 recall 与基线均一致，全部检查通过。

| 指标 | 尺度分层 / % | 统一 0.50 / % | 统一−分层 / pp |
|---|---:|---:|---:|
| mAP50 | 26.8181 | 26.7279 | −0.0903 |
| mAP50-95 | 15.3751 | 15.5988 | +0.2237 |
| Small recall | 28.0129 | 24.8899 | −3.1230 |
| Medium recall | 67.1615 | 68.2681 | +1.1066 |
| Large recall | 83.9738 | 85.7545 | +1.7807 |

- 自动候选规则要求 mAP 变化至少 0.3 个百分点且 small recall 同向，本轮因总体
  增量 0.2237 且方向相反，结果为 `promising_direction=null`。不过本研究的主要
  目标是小目标覆盖，统一阈值导致 small recall 下降 3.1230 个百分点、约少匹配
  830 个目标，而中、大目标更好。该现象支持尺度分层对小目标覆盖的预期方向，
  值得进行一次全量 30 epoch 确认，而非直接宣称机制成立。
- 证据：`outputs/screens/p1_global_t050_1024_v2_e5/{comparison,acceptance_audit}.json`。
  结果仍属于 1024 图、5 epoch 探索筛选，不进入论文正式表。

### 3.17 全量尺度阈值确认数据已准备（2026-10-27）

- 新增 `configs/{pseudo_p1_global_t050,train_p1_global_t050_10}.yaml` 和
  `scripts/run_p1_global_t050.py`。全量数据从冻结教师候选重划，不重跑教师：
  4624 个伪标签 tile、27667 个正框、57219 个 ignore 框，合并 11095 个训练 tile。
- `python scripts/run_p1_global_t050.py --prepare-only` 已通过：23781 个冻结输入、
  4624 条 metadata/标签对应关系、训练列表 11095、数据集 YAML 指向和教师摘要均
  通过。生成数据与审计在 `data/visdrone_area10/pseudo/`、
  `outputs/p1_global_t050_pilot30_s0/dataset_audit.json`（忽略目录）。
- 确认配置保留 `soft_weights=true`、`uncertain_ignore=true`，只替换数据且将三个
  尺度阈值统一为 0.50；学生仍为 COCO 初始化、640、30 epoch、batch8、SGD、
  seed0、`close_mosaic=10`。脚本将自动完成训练、val960、分尺寸 recall 和与完整
  P1 对比。
- **当时仅完成数据准备与预检，全量训练尚未启动。**按用户最新约束，给出
  `python -u scripts/run_p1_global_t050.py` 后停止，不自动占用 GPU 跑 30 epoch。

### 3.18 全量尺度阈值确认完成验收（2026-10-27）

- 用户执行全量确认，自动流程于北京时间 **20:00:53** 完成。新 run 为
  `runs/partial10_p1_global_t050_pilot30_s0`；CSV 为连续 1–30 epoch、全部数值有限，
  累计训练日志耗时 6915.15 秒。实际学生超参数与完整 P1 相同，配置差异仅指向
  阈值策略生成的另一数据集；两组共用冻结教师、软权重和 ignore 处理。
- 阈值策略只把 small/medium/large 正阈值从 0.30/0.50/0.50 改为
  0.50/0.50/0.50，从同一教师候选重新生成数据：新组为 4624 个伪标签 tile、
  27667 个正框、57219 个 ignore 框，合并 11095 个训练 tile。完整 P1 为 4839 个
  伪标签 tile、35067 个正框、52117 个 ignore 框、11310 个训练 tile。阈值造成的
  正/ignore 划分差异会改变入训 tile 数和每轮 batch 数，因此不能写成等计算量对照。
- 差值为“统一 0.50 − 完整 P1”，单位为百分点：

| 指标 | 完整 P1 / % | 统一 0.50 / % | 差值 / pp |
|---|---:|---:|---:|
| mAP50 | 39.4990 | 38.6992 | −0.7998 |
| mAP50-95 | 23.7883 | 23.1706 | −0.6177 |
| Small recall | 35.6925 | 35.3501 | −0.3424 |
| Medium recall | 77.5709 | 77.9847 | +0.4139 |
| Large recall | 89.4096 | 89.5033 | +0.0937 |

- recall 操作点仍为 conf=0.25、匹配 IoU=0.50、NMS IoU=0.70、max_det=500，
  两组使用相同 548 张完整原图。small 匹配数由 9486 降至 9395，少 91 个；medium
  和 large 分别由 8622→8668、954→955。该指标不是 AP_small 或官方 AR500。
- 逐类别 AP50-95 为 **1 类升、9 类降**。统一 0.50 仅使 tricycle 提高 0.6425 pp；
  people、motor、bus、car、van、pedestrian 分别下降 1.2516、1.1630、1.0498、
  0.8112、0.8046、0.8024 pp。总体和小目标差异不是由单一类别或中等/大目标支配。
- 独立验收入口 `outputs/audit_p1_global_t050_acceptance_20261027.py` 已通过：24 个
  dataset/run 输入及输出哈希匹配，30 epoch、实际参数、教师和损失开关配对，完整
  val960 与 recall 协议一致；对两组各自 548 图保存预测重新进行 GT 匹配，分尺寸
  计数与报告一致。验收记录为 `outputs/p1_global_t050_pilot30_s0/acceptance_audit.json`，
  run 审计为 `runs/partial10_p1_global_t050_pilot30_s0/audit.json`。
- best.pt / last.pt SHA256 分别为
  `94571aae799119becc033fa6a7296eb0326dbf7fb83e44f650ef7ab488728a4a` /
  `cd96237d62d121ec8132cdfd30dbce94aefe39afd887bfde6a39759ebae3b5ff`。
  结果已接入 `scripts/build_paper_tables.py` 并生成表 4-S2。
- **结论：**全量 30 epoch 与此前 5 epoch 筛选的 small recall 方向一致，但总体
  mAP50-95 方向相反（筛选中统一阈值 +0.2237 pp，全量中 −0.6177 pp）。全量
  结果支持尺度分层相对统一 0.50 同时改善总体检测和小目标覆盖。此节证据为单 split/seed 的
  Ultralytics 开发对照，不改变原 Gate 3，不据此声称尺度阈值首次提出或跨 seed 稳定。

### 3.19 尺度阈值配对学生 seed 复验完成（2026-10-27）

- 用户授权优先补尺度分层阈值的配对 seed，并在完成后更新论文、记录和提交 Git。
  本轮固定 split seed0、同一 B1 教师、两套既有全量伪标签与损失，仅学生 seed
  改为 1；不生成新 split，不扩展正式 100 epoch 主矩阵或其他消融。
- `scripts/run_p1_scale_seed_pair.py` 与两份 seed1 配置已准备。33,047 个冻结文件
  预检通过，统一阈值 metadata/标签对应关系核对通过；测试 36/36。
- 北京时间 20:31 启动顺序流水线，主进程 PID 29248；先运行
  `partial10_p1_pilot30_s1`，随后 `partial10_p1_global_t050_pilot30_s1`，各 30 epoch。
  日志、状态及后续验收位于 `outputs/p1_scale_threshold_pair_s1/`。
- 北京时间 **23:47:09** 两组训练、548 图完整 val960、固定操作点 recall 及自动
  验收完成；两 run 均为连续 1–30 epoch、训练值有限。尺度分层和统一阈值学生
  日志累计耗时分别为 6567.53 和 5080.26 秒，训练 tile 分别为 11310 和 11095。
  实际参数在同 seed 内配对，相对各自 seed0 仅学生 seed 和 run 名变化。
- seed1 差值为“尺度分层 − 统一 0.50”：

| 指标 | 尺度分层 / % | 统一 0.50 / % | 尺度−统一 / pp |
|---|---:|---:|---:|
| mAP50 | 39.5056 | 38.9834 | +0.5222 |
| mAP50-95 | 23.6126 | 23.3706 | +0.2420 |
| Small recall | 36.1403 | 35.4969 | +0.6434 |
| Medium recall | 77.6428 | 77.3639 | +0.2789 |
| Large recall | 88.9410 | 89.5033 | −0.5623 |

- small 匹配数为 9605 对 9434，多匹配 171 个小目标。seed1 逐类 AP50-95
  在统一阈值下为 2 升/8 降，仅 car、truck 上升；medium recall 的相对方向与
  seed0 不同，large recall 在两个 seed 中均为统一阈值更高，不作全面改善表述。
- seed0/1 的 mAP50-95 和 small recall 均支持尺度分层。种子内“尺度−统一”
  的平均配对增益±样本 SD（ddof=1）分别为 **+0.4299±0.2657 pp** 和
  **+0.4929±0.2128 pp**；两方案各自的均值与 SD 及全部单次值见表 4-S3。
- `scripts/audit_p1_scale_seed_pair.py` 独立验收 **36 项检查**通过，重新核对
  33,047 个冻结文件、教师/伪标签/损失、best/last 和完整 val 协议，并重放两组
  各 548 图保存预测进行 GT 匹配。验收与两组权重 SHA256 见
  `outputs/p1_scale_threshold_pair_s1/{acceptance_audit,independent_audit,comparison}.json`
  和论文证据索引；执行脚本与方法源码在训练期间未改变。
- 论文表生成器已接入表 4-S3，实验第 4.11 节及证据/复现路径同步更新。
  最终 `build_paper_tables.py --check` 核对 59 个来源，6 组 recall 共用 GT 计数；
  `python -m pytest --basetemp outputs/pytest_scale_seed_pair_final_recheck` 36/36 通过。
  这不是烟测；结论限于单 split、固定教师下两个学生 seed，仍是 Ultralytics
  开发评测，不等同于 AP_small、统计显著或跨 split 泛化，不改变原 Gate 3。

### 3.20 B2 学生 seed1 与主对照案例图（2026-10-30，验收完成）

- 用户授权补 B2 学生 seed1 并整理案例，完成后验收、更新论文及本地 Git 提交。
  固定 split seed0、原 B1 教师、B2 的全局 0.50 伪标签和标准损失；30 epoch、
  640、batch8、SGD、COCO 初始化及 `close_mosaic=10` 沿用已有 B2 pilot。
  P1 seed1 已完成，直接复用；不训练 B1 seed1，不扩大正式 100 epoch 矩阵。
- 10 月 28 日 09:27 的原进程 PID 42232 已退出，日志停在第 3 epoch，CSV 仅有
  连续 1–2 两行，无异常栈，原因尚不能由日志确定。10 月 30 日核实无对应训练
  进程后，将原 run 与流水线目录的 10 个文件、来源脚本/配置和哈希归档至
  `outputs/b2_seed1_interrupted_20261030_091251/`。中断权重不作为论文结果。
- 10 月 30 日 **09:13:39** 以未修改的执行脚本和原初始化重新启动，PID 31944；
  37,695 个冻结输入再次通过预检。`outputs/b2_pilot30_s1/recovery.json` 链接归档；
  新运行不拼接旧 epoch，不覆盖已完成 run。
- 10 月 30 日 **10:10** 心跳发现 PID 31944 再次消失：日志最后更新于 10:07:29，
  停在第 16 epoch，CSV 完成 1–15，stderr 为空；操作系统未重启，也没有能确认
  训练退出原因的应用异常记录。将该次 12 个文件、原脚本和配置/哈希另行归档至
  `outputs/b2_seed1_interrupted_20261030_101330/`，原档案链均保留。
- **10:15:05** 改由本机 `Win32_Process.Create` 启动 yolo 环境的 `pythonw.exe`，
  PID 14304，父进程为 WMI 服务进程而非 Codex 子进程。仅启动和日志包装不同，
  仍执行原 `scripts/run_b2_seed1.py`、相同配置/源码、同一初始化，37,695 文件
  预检再次通过；从 epoch1 重跑连续 30 epoch。`launch_receipt.json` 保存启动包装
  文件及哈希，独立验收递归复核两次归档链。该次运行已按用户要求停止并归档，转交终端手动训练；当时尚未验收，后续完整结果见下文。
- 流水线自动完成 548 图完整 val960、固定操作点 recall，以及 B2/P1 seed0/1
  四组各 548 图保存预测的 GT 匹配重放。随后运行 `scripts/audit_b2_seed1.py`，
  核对连续 1–30 epoch/有限值、实际参数配对、教师/伪标签/损失、best/last 和评估
  哈希，再生成表 4-S4 的单次值、各方法均值±样本 SD 及种子内配对差均值±SD。
- 已在新增 B2 seed1 结果出现前，从冻结 seed0 验证预测选定四类案例：P1 新检出、
  P1 丢失 B2 已检出、共同漏检、P1 未匹配预测。选图按原验证顺序及固定事件顺序，
  取首个合格目标，并排除重复文件名前缀和相同图像内容；验证集本身不变。
  总览为 1776×1910、300 dpi PNG，输入/输出哈希及裁剪坐标均记录在 manifest；
  详见 `docs/paper/case_studies.md`。图像、日志和权重均留在忽略目录。
- seed0 的 26,577 个 small GT 中，两者均检出 8,151、仅 P1 1,335、仅 B2 836、
  均漏检 16,255，净多匹配 499。案例解释覆盖成功与失败，不据此推断因果或
  官方错误分类；主结果继续以完整验证集指标为准。原 Gate 3 不变。

- **10:27** 按用户要求停止后台 PID14304，原 run 的 13 个文件及来源链保存在
  `outputs/b2_seed1_interrupted_20261030_102720/`；前两次归档也完整保留，心跳暂停。
  **10:33:25** 用户终端直接执行原入口，从同一 COCO 初始化开始，最终 CSV
  连续 1–30 epoch 且全部数值有限，累计训练日志耗时 5845.51 秒。
- **12:11:53** 完整训练、548 图 val960、recall 和自动验收完成。新增 seed1：

| 指标 | B2 seed1 / % | P1 seed1 / % | P1−B2 / pp |
|---|---:|---:|---:|
| mAP50 | 38.9843 | 39.5056 | +0.5214 |
| mAP50-95 | 23.3688 | 23.6126 | +0.2438 |
| Small recall | 34.4471 | 36.1403 | +1.6932 |
| Medium recall | 76.3383 | 77.6428 | +1.3045 |
| Large recall | 89.5970 | 88.9410 | -0.6560 |

- P1 seed1 多匹配 450 个 small（9605 对 9155）、145 个 medium（8630 对 8485），
  少匹配 7 个 large（949 对 956）。seed0/1 的 mAP50、mAP50-95、small/medium
  recall 均为 P1 更高，large recall 则一升一降；seed1 逐类 AP50-95 为 6 升/4 降。
- 两 seed 种子内 P1−B2 的 mAP50-95 平均差±样本 SD 为
  **+0.2893±0.0644 pp**，small recall 为 **+1.7854±0.1304 pp**（ddof=1）。
  B2/P1 各自 mAP50-95 均值为 23.4111±0.0598% / 23.7005±0.1243%；small recall
  为 34.1310±0.4470% / 35.9164±0.3166%。全部单次值/指标见表 4-S4，不选报。
- 独立验收 **116 项检查**通过：37,695 个冻结文件、四 run 连续 30 epoch/有限值、
  实际参数及原初始化配对、同教师/既有伪标签/损失、best/last 与评估哈希、完整
  548 图/38,759 GT 和固定 recall 操作点、四组保存预测 GT 重放、独立配对统计；
  三层归档的 35 个文件和执行脚本/配置快照均核对。中断日志不拼接、不入论文。
- `build_detection_cases.py --check` 再核对四类案例、1,106 输入哈希及图像输出。
  论文表 4-S4、第 4.12–4.13 节、证据和复现入口均已同步；表格核对 123 个来源、
  7 组不同 recall 目录；最终 `python -m pytest --basetemp outputs/pytest_b2_seed1_final_20261030_1308`
  **39/39 通过**，`git diff --check` 通过。B1 未重复，原主表仍为 seed0；仅是单 split/固定教师
  下两个学生 seed 的开发复验，不等同于独立教师、多 split、显著性或官方 AP_small，
  不完成正式 100 epoch 主矩阵，也不改变原 Gate 3；不自动启动新 seed/消融。

### 3.21 三篇文章、学位论文与树莓派部署目标（2026-10-30，仅规划）

- 用户希望围绕相关成果规划三篇文章，最高按北核层级，最终模型需要轻量化并部署树莓派。
  用户补充论文按录用计算，专利按“申请通过”；该词暂不擅自解释成受理或授权。
  学校/专业、考核截止日期、作者/单位及专利类型/认可节点待核实。
- 新增 `docs/publication_and_edge_deployment_plan.md`：A 局部标注半监督检测、
  B 漏检感知可靠补全、C 蒸馏压缩与树莓派部署；三者须有独立问题与新增证据，
  不将已有阈值/权重/ignore 三模块拆成三篇。实际顺序 A→C→B，先保障毕业必需
  成果和硬件落地；暂按两篇录用论文+学位论文，或一篇录用论文+两件认可专利+
  学位论文理解，最终仍以学院要求为准。
- 硬件未回复，排期暂按树莓派5/8GB/纯CPU/图像检测，明确是待确认假设。
  先验证未剪枝模型/原生小模型的导出与上板，再选择结构化剪枝和蒸馏；FP32、
  后端支持的量化、CPU/NPU结果分开报告。完整原图精度、端到端时延、P95、内存
  和热稳定性均纳入验收，不能用GPU速度或单tile速度代替树莓派整图结果。
- 另给三篇文章边界、学位论文章节和约20周相对研发排期，审稿/录用等待不作固定
  时长承诺；专利先核对公开内容与申请顺序。LoRA/DINOv3/CLIP均为候选工具，
  不直接假定有效或同时接入，不覆盖原Gate3判定，不自动启动训练或购买硬件。

### 3.22 第一篇论文实验缺口与低成本排序（2026-10-30）

- 用户决定先推进第一篇方法稿，要求先梳理缺口、低成本优先，只有出现成果后再投入；
  需要训练时由用户运行。已新增 `docs/paper1_experiment_gaps.md`。
- 首项零训练漏检诊断完成。工具 `scripts/audit_paper1_gaps.py` 对 6,471 个未标注 tile 的
  38,245 个 hidden GT 精确重放既有 P1 正伪标签及 25,669 个真匹配；核对 11,319 个输入哈希。
  small 未匹配 hidden GT 8,984 个，其中同类别 ignore IoU≥0.50 3,652、异类 IoU≥0.50 1,644、
  同类别 IoU 0.10–0.50 1,243、无保存候选 IoU≥0.10 1,597。类别为描述性证据，不能证明
  ignore 可独立转正或原因归于 NMS；候选文件已是 NMS 后结果。
- 计划依次做：零训练“可恢复增量/候选覆盖”检查；仅在信号充分时运行一次 5 epoch 配对筛选；
  筛选过线后才补完整 30 epoch、seed0/1 确认。B0 和新 split 均是更昂贵、按投稿需要决定的
  后续验证，不是当前应立即启动的任务。软权重全量状态先审计，不能把短程筛选写成正式消融。
- 本轮未启动训练、推理或伪标签生成；原 Gate3 和已有正负结果不变。

### 3.23 Ignore 可恢复增量分析完成（2026-10-30，零训练）

- 用户明确先做 ignore 可恢复增量，不训练；只有明确改进方向后再交训练命令。
  新增 `scripts/audit_ignore_recovery.py` 与 `docs/paper/ignore_recovery_analysis.md`，
  用类别一致/IoU≥0.50 的旧 positive 匹配锁定基线，ignore 仅与剩余 GT 联合一对一匹配。
- 当前 4,839 入训 tile 的 52,117 个 ignore 框覆盖 5,340 个剩余 GT；普通贪心新增
  匹配 5,303，锁定旧匹配后的最大基数容量上界 5,315；贪心新增 small 为 3,627。
  3,490 个 ignore 框只覆盖旧匹配 GT；新增独立匹配/新增框仅 10.18%。
- 仅按预测 small 且 conf≥0.25 保留原 ignore，额外 2,548 框新增独立匹配 691，
  其中 small 为 672、medium 为19，新增匹配比例 27.12%。0.20/0.15/0.10/0.05
  对应比例 22.85/18.76/14.60/9.55%。不能将其当作总体 precision 或学生 mAP。
- 排除 tile 的 4,152 ignore 框单列，普通新增187、条件上界188，不能计入当前训练。
  最大匹配按总量求解，各尺寸计数不代表单独最大化各尺寸的上界；所有 oracle 仅诊断，
  隐藏GT不用于训练标签。旧正框可重分配的 union IoU 贪心结果另存，不混用。
- 存在可恢复空间，但尚无不依赖 GT 的可靠转正规则；当前不启动扩训。下一步可以
  考虑固定小样本的跨视图支持核验，仍先测伪标签精度—覆盖，不直接训练学生。
  软权重全量30epoch训练/val/recall/派发均核实不存在，缺口保留，不重复中性短程筛查。
- 工具复核 11,322 输入哈希，完整6,471 tile计数；新匹配函数经过全部2×3二部图
  的穷举容量对比和重复/空输入等测试。Pyright初始化修正前的脚本/输出和来源已归档，
  计数算法不变，两份重生成 analysis.json 已与旧产物逐字节核对一致。全套 pytest 46/46通过、
  两诊断脚本 Pyright 0错误/0警告，既有论文表123来源/7组recall仍通过。当前训练方法和已验收指标未改。

### 3.24 跨视图与候选保留筛查后停止扩训（2026-10-30）

- 已完成 256 tile 翻转探索及另取 128 tile 的 NMS/max-det 检查；后者排除前者
  全部下标，原入训 89/排除 39 tile 分列，冻结教师和原配置在全部128 tile重现。
- NMS IoU 0.70→0.90：原入训全候选1647→3438、GT匹配617→628；正框650→860、
  匹配500→508，precision76.92%→59.07%。新增210正框在锁定基线匹配后仅新增9个
  small GT，新增匹配比例4.29%；按原正框同类IoU≥0.70过滤后仅余2框/1新增匹配。
  max-det500→1000无变化，原配置单tile最多213候选。
- 原初分析误将与任意原候选重叠者全部排除，口径过于保守；保留初始源码/预测/manifest，
  v2复用推理，区分新增预测身份、未过滤恢复容量和正框重叠过滤，不重新推理。
  原始和修订来源均保存于忽略目录，详见 `docs/paper/candidate_retention_analysis.md`。
- 两方案均未达到探索性继续条件，故不训练、不输出新训练标签。只读重放入口为
  `python scripts/audit_candidate_retention.py --check`，成本和结果在
  `outputs/candidate_retention_128_20261030_v2/`；不改变原Gate3。
- 全套pytest52/52通过，359个输入哈希及全部128 tile重放通过；既有论文表
  123个来源与7组recall计数核对通过，原训练配置和正式结果未改。
- 用户最新授权替代先前“只由用户手动启动”安排：可靠信号出现后可自主进行至多
  30 epoch有界训练，启动时建临时进度心跳，完成/失败/需处理时通知、结束清理；
  更长训练仍由结果决定。每轮修改中文提交。本轮无训练，无需创建进度定时器。

### 3.25 跨尺度支持确认与总安排更新（2026-10-30）

- 用户指定`docs/publication_and_edge_deployment_plan.md`为总安排依据；新增第10节
  当前工作队列、职责和完成定义，替换旧的全部训练由用户手动启动规则；
  按用户最新收束要求，当前目标只做A，C/B为远期规划，不并行执行。
  Codex负责可靠信号后的至多30epoch有界小试与临时监控，用户负责有证据后的更大规模训练。
- 新增GT-free候选支持函数，保持原正框、原ignore坐标与置信度，按同类/同尺度候选
  的640视图支持与正框去重决定是否可转正；隐藏GT只用于离线计数，不输出训练标签。
- 原256 tile探索中，floor0.25/support0.30新增22框、11个small匹配，达到探索继续线。
  固定该唯一候选后，在排除既有384 tile的另256 tile确认；192个原入训tile新增23框、
  9个small匹配，新增匹配比例39.13%，未达继续线，不改选其他规则。
- 本轮未启动学生训练；保留探索和确认的预测/manifest/成本，入口为
  `scripts/audit_scale_support.py`，报告见`docs/paper/scale_support_analysis.md`。
  当前仍未找到可交给用户扩大训练的新方向；下一项先核对A的必要消融、基线和
  正式训练预算缺口，不继续横向扩展新教师或B类补全机制。
- 全套pytest55/55通过，探索722/确认719个输入哈希与各256tile只读重放通过。

### 3.26 A的软权重单因素补证据（2026-10-30，已验收）

- 收束到A后，优先补现有稿件软权重开关的完整30epoch消融；旧1024图5epoch中性
  筛查不追认为过线。用户最新至多30epoch自主小试授权用于这一组必要机制检验，
  不是启动新教师、补全论文或大规模矩阵。
- 原入口/配置预检通过23781冻结文件、11310 tile；只关闭soft_weights，保留
  原教师、数据、增强、初始化与ignore。预算一组30epoch，复用已完成完整P1参考。
- 新增后台包装和独立验收：保存父/子PID与退出码，检查连续epoch、实际参数、
  评估/权重哈希，并重放两组548图预测。全套pytest56/56通过。
- 预算/预先判据见`docs/paper1_experiment_gaps.md`第7节。北京时间2026-10-30
  14:30:20实际启动，包装PID37236、训练PID34612，命令与创建时间见
  `outputs/p1_no_soft_weights_pilot30_s0/launch_receipt.json`。
  临时心跳`a-30`每10分钟核查进程身份、CSV和阶段，只在完成、失败或需处理时通知，
  完成/失败处置后删除；不自动重启或扩训。本组于16:28:18正常退出，exit_code=0。
- 连续30epoch、完整val960/固定操作点recall、实际参数配对、23783个冻结输入及
  两组各548图预测重放通过16项独立验收。CSV累计训练7016.44秒；
  入口`python scripts/audit_p1_no_soft_weights.py`，证据同目录`independent_audit.json`。
- 去软权重−完整P1：mAP50−0.0585pp、mAP50-95−0.1623pp，
  small recall+0.9143pp（+243），medium+0.6658pp（+74），large−0.3749pp（−4）；
  逐类别AP50-95为3升7降。整体mAP与召回有取舍，不满足预先记录的后续线，
  不启动100epoch或追加训练；保留完整P1主配置，不改写旧5epoch中性筛查或原Gate3。
- 表4-S5、实验4.14和证据/复现记录已更新；表格核对137个来源/8组recall，
  全套pytest56/56通过。单split/固定教师/学生seed0，不作显著性或官方AP_small主张。
- 本轮训练和验收均结束，临时心跳`a-30`已删除；未启动新训练或长期监控。

### 3.27 A的关键词文献调研（2026-10-30）

- 使用paper-search MCP围绕漏标补全、定位不确定性、损失和基础模型查询原论文。
  7份PDF及SHA256清单位于忽略目录；另外5篇目前仅核对元数据/摘要，阅读范围
  和固定版本逐项见`docs/paper/a_keyword_research.md`，不虚构全文阅读。
- 优先候选是“固定现有正伪框，仅检验跨视图定位稳定性能否预测定位质量”，
  先做离线诊断，尚未实现或训练。明确与Soft Teacher、RUPL、UTv2重叠，
  NWD只列为独立备选，不先验宣称创新或收益。
- 第3.26节软权重消融已完成，未达到扩大训练条件；LoRA/DINOv3/CLIP未下载模型或运行，
  蒸馏后结构剪枝留在C。下一步可检验既有正伪框定位稳定性，当前仅文献假设，
  尚未执行该诊断或启动第二项训练。

### 3.28 已有正伪框定位可靠性（2026-10-30，诊断完成）

- 下一项只复用两个不重叠256tile缓存，检验跨尺度定位稳定性在控制置信度、
  类别组成后是否区分已有正框的定位质量，不增加标签、推理或训练。
- GT-free匹配、固定分层及继续线见`configs/positive_localization_diagnostic.yaml`和
  `docs/paper/positive_localization_analysis.md`；协议由`2a42fa5`先提交，再执行诊断。
- 已完成2710个正框、1442来源哈希的逐框重放。探索/确认small正框906/641，
  稳定822/570、不稳定各6、无匹配50/51；没有足够分层可比样本，未通过继续线。
  稳定组同类GT IoU≥0.75仅52.7981%/49.4737%，不能由同教师多尺度稳定推出正确。
- 无匹配组仍有20.0000%/23.5294%原一对一匹配，不直接剔除；
  不合并无匹配组重试、不改组阈值，不实现或训练新回归加权。
  全套pytest61/61通过；只读复核`python scripts/audit_positive_localization.py --check`。

### 3.29 原图上下文类别纠正（2026-10-30，探索失败后停止）

- 跨尺度定位诊断停止后，下一项只检查新的观察范围：同一B1教师的原图960预测
  映射回既有未标注tile，能否纠正低置信度small正伪框的类别。
  不增加正框、不改几何/置信度/ignore，不引入新教师、模块或训练。
- 协议`docs/paper/context_rectification_analysis.md`及
  `configs/context_rectification_diagnostic.yaml`固定128原图探索、过线后另128图确认；
  排除历史640尺度与NMS样本。类别冲突/匹配选择无隐藏GT输入；
  隐藏GT只离线计数，未导出训练标签。与CrossRectify/密集裁片SSOD重叠已说明。
- 入口`python scripts/audit_context_rectification.py`；`--check`只读验收；
  确认组`--confirm`必须先重放并通过探索继续线。协议和代码先提交再推理。
  协议/代码由`ef8f565`先提交，实际推理commit为`f1c997e`。
- 唯一128图探索完成：98个已接受tile、799正框，只触发1次类别纠正，
  将正确van改为car；总体匹配562→561、small302→301，继续线失败。
  推理含warmup4.3874秒、新增训练0epoch；不执行确认或更改阈值追试。
  372来源/全部决策与GT统计只读重放通过，确认入口拒绝失败探索已实际验证。
  结果与限制见`docs/paper/context_rectification_results.md`。
- 预检修复历史`data/`忽略规则屏蔽四个数据源码的问题；原文件内容不改，
  Git归档核对源码齐全且不含生成数据，全套pytest74/74通过。
- Zotero已按用户要求调整为“我的文库”下并列的`YOLO-RegionSSOD`和
  `YOLO-VisDrone Research`；前者28篇均有PDF，S³OD附件为注明的2023早期预印本。
  文献清单见`docs/paper/zotero_literature_library.md`；扩展关键词归档不代表启动B/C。

### 3.30 开放区域拟合的置信度校准（2026-10-30，探索未过线）

- 基于已读Calibrated Teacher方法页，单独检验分类置信度可靠性；不复现其在线EMA、
  IoU队列或FIoU，不宣称校准概念新颖。执行前协议
  `docs/paper/confidence_calibration_protocol.md`和固定配置先提交，再推理。
- 512个开放标注tile拟合全局分数段/类别×尺寸单调收缩映射；拟合仅用已有
  `labeled/labels`，与探索/确认源图不重叠，不用隐藏GT拟合或调整分数段。
  探索/确认各256个未标注tile，仅复用已有正伪框；类别/几何/原分数/ignore不改。
- 八项继续线同时要求分组Brier优于原分数和全局简化对照、small同向、
  足够样本/覆盖且错误权重占比不明显恶化。探索失败不执行确认，不调规则追试。
  入口`python scripts/audit_confidence_calibration.py`，`--check`只读重放拟合和诊断。
  执行前由`f3d77cd`提交；全套84/84tests通过。
- 已完成512开放tile推理（8.1703秒）和7548正框拟合；不重叠256tile探索中
  194接受tile/1421正框（small816），分组映射覆盖96.4110%。
  分组Brier整体0.160310，高于原分数0.158864；虽small由0.183221降至0.176334，
  仍不满足预先整体继续线，未运行确认或学生训练，也不改成small-only追认通过。
- 拟合1045来源/7548正框、探索474来源/1421正框只读重放一致，
  失败探索的确认阻断已实测；结果见`docs/paper/confidence_calibration_results.md`。
  当前两项新规则均不支持扩训，后续须有新的可验证信息，不在同批隐藏GT上调参追试。

### 3.31 小框混合NWD回归筛查（2026-10-30，完成后停止）

- 在先前两项规则失败后，单独检验几何损失；原NWD论文式7–9及Sec.3.3已核对，
  其已包含VisDrone及回归/NMS/分配应用，不把接入NWD称为原创。
  固定协议`docs/paper/small_nwd_protocol.md`先提交，再估计C和执行。
- 仅增强后640像素空间面积<32²的目标框回归项改为0.5CIoU+0.5NWD；
  C来自同一1024子集开放GT resize后的平均small边长，不读隐藏GT或val调参。
  新增独立模块，不修改原P1/runner；TAL、DFL、分类、ignore及标签集合保持固定。
- 复用已验收1024源图/1765tile的完整P1独立5epoch参考，只新增一组5epoch，
  配置`configs/screen_p1_small_nwd_5ep.yaml`。继续线为同预算val960
  mAP50-95≥+0.30pp且small recall≥0；未过线停止，不自动30/100epoch。
- 当前全套pytest94/94；stride/像素换算、零/非重叠/包含梯度、AMP类型、
  空前景、DFL与分类不变、关闭恢复P1及继续线测试通过。
  已通过23794个冻结输入预检；1024开放tile的7519个resize后small框得到
  C=20.22804306训练像素。真实YOLO11s、2×3×640×640、AMP前反传通过，
  255个梯度张量均有限，优化器更新0步；证据
  `outputs/screens/p1_small_nwd_1024_e5/{preflight,geometry_gpu_check}.json`。
  2026-10-30 18:16:07实际启动唯一5epoch，包装PID32304、训练PID28860，
  记录`outputs/screens/p1_small_nwd_1024_e5/launch_receipt.json`；
  命令/创建时间经Get-CimInstance核实，CSV实际增长。18:19核查时已完成2epoch，
  stderr为空，仍在原训练，不将中途640指标当作最终val960结果。
  包装于18:22:23正常退出，exit0；连续5epoch累计299.989秒、完整val960/recall完成。
  相对同子集P1：mAP50-95−0.1746pp、small recall+1.5464pp、
  medium+0.0630pp、large−0.7498pp；未达继续线，不扩训或调参追试。
  独立核验23794来源、源码/参数/产物哈希与两组各548图预测匹配通过；
  详细结果与限制见`docs/paper/small_nwd_results.md`。
  临时心跳`a-nwd`收尾后删除，不将本组结果写入正式主表。

### 3.32 至少30epoch判断性能与NWD全量确认（2026-10-30）

- 用户指出短程可能不能体现最终性能，并明确至少30epoch后再判断改进。
  该要求覆盖此前“5epoch筛查未过即停”的后续安排，历史结果与门槛不改写。
  当前自主单候选固定30epoch；更短仅工程检查，更长仍交用户执行。
  用户另授权中文Git提交后push，替代此前不push偏好。
- 缓存548图离线重放：固定conf0.25/IoU0.50，NWD小目标+411、总FP+888；
  等FP后small recall仍+0.6321/+0.7713pp（P1 conf0.25/0.50两档预算）。
  IoU0.75为+0.2521/+0.3951pp，IoU0.90为−0.0489/−0.0564pp。
  10类AP为5升5降；仅事后描述，缓存conf≥0.25，不生成部署阈值或重建完整AP。
  详见`paper/small_nwd_results.md`，重放入口
  `python scripts/audit_small_nwd_tradeoff.py --check`。
- 独立登记`paper/small_nwd30_protocol.md`，只新增完整11310tile的NWD学生30epoch，
  与已验收完整P1 seed0/30epoch配对；从原COCO权重初始化，
  close_mosaic10，不续接五轮。原NWD源码及C=20.22804306/mix0.5/面积边界均冻结。
  新入口`python scripts/run_p1_small_nwd.py --check-only`、后台包装与独立验收已实现；
  工程预检通过后于2026-10-30 18:54:03启动，不提前声称性能改善。
- 后续线在30epoch未知时登记：mAP50-95≥+0.30pp且small recall非负，
  或small recall≥+1.00pp且mAP50-95≥−0.10pp。采用与既有30epoch
  软权重消融相同的两路线；过线才准备更大训练交接，不自动扩训。
  工程检查108/108 tests；新增五个Python模块Pyright零错误/警告。
  缓存诊断1111来源SHA256及两组预测精确重放通过，论文表137来源/8组recall核对通过。
  `0d75739`已提交并push协议/源码；23793来源预检通过，完整11310tile。
  包装PID30248、训练PID46428命令行/创建时间通过Get-CimInstance核实；
  18:55为第1/30轮、batch221/1414，stdout实际增长、stderr为空，
  首轮尚未结束，results.csv尚无首行，不能把此时缺少CSV判为中断。
  临时心跳`a-nwd`已创建，每10分钟检查本次全量30epoch，
  仅终态/需处理通知；结束或失败处置后删除。日志与回执位于
  `outputs/p1_small_nwd_pilot30_s0/`，不重复启动，不自动更长训练。
  19:19核实已完成5轮、正在第6/30轮，原包装/训练身份匹配、stderr为空。
  文献定位核对已由`70b6884`提交push，按用户允许的等待安排选择暂停持续目标，
  保留原训练及10分钟临时监控；监控已补充终态验收与后续接续职责。
  如果工具不能把paused状态恢复为active，完成时需明确提醒用户点击继续，
  不虚报自动恢复；训练正常时不再由持续目标反复输出同类进度。

### 3.33 NWD全量30epoch中断处置（2026-10-30）

- 20:11心跳发现异常，20:14再次用Get-CimInstance核实：原包装PID30248、
  训练PID46428均不存在，也没有同入口Python进程；日志/CSV未继续增长。
  `launch_receipt.json`无exit_code，`status.json`仍停留在启动时的training，
  未覆盖原始记录或虚构退出码。
- CSV连续13轮，累计训练3975.89秒；最后修改20:00:42，stdout最后修改20:01:20，
  停于第14/30轮181/1414batch。stderr及包装日志为空；548图val960和分尺寸
  recall均未生成。现有best/last保留为中断权重，不能作为已完成30轮结果。
- Windows应用事件1002（记录8771）于20:01:24报告Python停止响应并关闭；
  PID十六进制0x7628=30248，创建时间与原包装回执一致，可执行路径为yolo环境。
  因此确认包装发生AppHang；子进程退出机制和挂起根因仍未知，不归因于NWD性能。
  系统最后启动为10月24日，本次未见对应时段重启证据。
- 原输出/run目录及Windows事件已复制到忽略目录
  `outputs/p1_small_nwd_interrupted_20261030_201452/`；
  `interruption_report.json`保存逐文件SHA256、进程核实和缺失退出码/阶段证据。
  13个原文件/副本哈希一致；108项pytest、论文表137来源/8组recall核对通过。
  详见`paper/small_nwd30_interruption.md`。损失源码、配置和执行前协议保持冻结。
- **实验未完成，性能未评定**；不把13轮拼成30轮，不运行成功验收，不自动重启、
  扩训或并行候选，不因中断判定继续线失败。既有5轮探索与Gate3历史结论不变。
  中断处置后删除临时监控`a-nwd`。持续目标工具返回paused，现有控制不能恢复active；
  需要用户点击继续后再安排后续工作，未声称恢复或新建替代目标。

### 3.34 用户终端重新训练交接（2026-10-31）

- 用户明确上次由本人中断，要求训练命令，自行执行后通知。
  该确认补充第3.33节当时未知的中断背景，不将Windows挂起事件解释为NWD缺陷。
- 重新核实无原包装/训练或同入口进程，原13个文件SHA256与中断报告一致。
  使用原生PowerShell、校验绝对路径位于仓库后，将旧输出/run移入
  `outputs/p1_small_nwd_manual_handoff_20261031/{output_original,run_original}`，
  保留中断快照及`handoff.json`路径映射，没有删除原证据。
- 保持原执行源码、损失、配置、C/mix、参考P1和固定后续线；
  由用户执行`python -u scripts/run_small_nwd30_background.py`，
  从COCO初始化重新训练完整30轮，不使用13轮权重续接或拼接。
  包装入口等待全部阶段并记录日志/身份/退出码；依次执行完整val960、recall和比较。
  Git Bash命令与日志查看方式已同步README/baseline。
  `--check-only`核对23793项哈希、11310tile及C=20.22804306通过；
  108项pytest与论文表137来源/8组recall核对通过。
- 本次只做归档、预检和交接，未启动新训练；按本次用户要求由其完成后通知，
  不创建新定时监控或恢复持续目标。用户报告结束后核实完整30轮、exit0/阶段，
  再执行独立验收并判断是否值得更大训练。

### 3.35 全量NWD30轮完成与独立验收（2026-10-31）

- 用户报告完成后核实回执：09:02:42启动，11:18:53包装exit0；
  包装16400/训练44996均已停止，全部阶段completed，CSV连续1–30轮。
  训练11310tile，原COCO独立初始化，未接续13轮中断。
- `python scripts/audit_p1_small_nwd.py`独立验收通过：
  23793项冻结输入/来源哈希、实际参数、权重及产物核对，
  两组各548图预测匹配重放；accepted=true，worth_larger_training=false。
- 完整val960：P1/NWD mAP50-95为23.7883%/23.5761%（−0.2123pp），
  mAP50为39.4990%/39.1453%（−0.3537pp）。
  small recall为35.6925%/35.0228%（−0.6698pp，匹配9486→9308，−178），
  medium/large分别下降0.0900/0.1874pp。逐类AP为4升6降，truck−0.9732pp。
- P1/NWD学生CSV累计7286.61/8092.09秒，本次多805.48秒（约11.05%），
  不是独立速度基准。原5轮子集的小目标收益未在本轮保留，
  但数据规模和训练预算同时不同，不能只归因于epoch或否定全部NWD用法。
- 固定两条继续线均未达到；满足至少完整30轮后判断的要求，
  不准备此变体的100轮扩训，不调C/mix/面积边界追试。
  保留原P1、负面结果与单split/固定教师/学生seed限制，不宣称完全收敛，
  不改写原Gate3，也不自动启动其他候选。
- 结果单独整理为`paper/small_nwd30_results.md`，同步实验4.15、证据索引及命令入口。
  既有论文主表/补充表仍由原生成器核对，新NWD表由独立NWD验收入口核对；
  未修改冻结训练源码、配置或执行前协议。旧临时监控已删除，本次用户手动训练
  未创建新监控；不把本次验收等同于已找到值得大规模训练的方向。
  本次108项pytest及既有论文表137来源/8组recall核对通过。

### 3.36 小框分配支持域诊断准备（2026-10-31）

- 用户恢复持续目标后，上一轮NWD验收属于有效进展但未找到可扩大训练方向。
  当前仍仅A，先寻找尚未验证的单因素，不重复翻转/跨尺度阈值扫描或增加基础模型。
- 源码核实当前Ultralytics TAL已将小框短边扩至stride[1]=16后筛选候选点，
  DFL仍对原框计算非负ltrb；不能提出“原实现从不扩大tiny候选池”的错误动机。
  框外点可能存在回归支持域不匹配，也可能帮助覆盖，须先测实际有效分配占比。
- 新协议`paper/assignment_geometry_protocol.md`在推理前固定256基础tile、
  128/128探索/确认、原P1 checkpoint、真实训练增强、FP32原始头预测与CPU分配重放。
  候选只在原框有点时使用原框内点，否则保留原扩框池；其他分配与损失不变。
  不读取隐藏GT/val，不训练；mosaic来源可能重叠，不宣称独立来源确认。
- `scripts/audit_assignment_geometry.py`及几何/覆盖保护测试已实现；
  111项pytest通过，两个新增Python文件Pyright零错误/警告。
  FCOS原文第2.1节/式1核对框内点的四边距离定义，
  RFLA只取得MCP题名/DOI，不虚报精读；当前假设不称首次或S³OD SLA复现。
- FCOS父条目R3GT2QZQ、原PDF附件NRI2CTCA经MCP核实，
  加入顶层YOLO-RegionSSOD并保留原YOLO-VisDrone Research归属。
  首次推理前中文提交协议/代码并push；诊断结果及是否值得30轮确认尚未知。

### 3.37 初版小框分配诊断未通过覆盖条件（2026-10-31）

- `cb365b4`提交协议后实际完成256个固定增强输入的冻结P1前向，
  零优化器步；缓存32批原始预测，在CPU计算现有与候选TAL分配。
  探索/确认1930/2222个正框，其中small1035/1170个。
- 基线small框外目标分数质量占比7.9472%/4.6675%，确有作用空间；
  候选将框外质量245.4662/180.8721降至29.0962/11.8627。
  但small目标有效分配保留率98.1151%/98.8676%，均低于预先固定99%，
  分别丢失19/13个已有有效分配的small目标；初版不进入训练，不改阈值追试。
- `paper/assignment_geometry_results.md`保留完整范围与失败结论。
  这不是新的学生性能结果，至少30轮要求未被诊断替代。
  后续可另登记带整图覆盖保护的新规则；旧确认缓存已参与设计，不能重新声称独立确认。

### 3.38 覆盖保护的新规则与新样本确认准备（2026-10-31）

- 初版失败结论保留并重放一致，不降低99%保留门槛。
  新规则仍框内优先，但若任一原有效目标在最终分配后消失，
  整图使用原TAL结果；保护训练标签目标，不使用隐藏GT或val做选择。
- 新模块`src/regionssod/training/assignment.py`、工程覆盖保护测试及
  `scripts/audit_assignment_coverage.py`完成，114项pytest通过，
  三个新增Python文件Pyright零错误/警告。新增独立模块，不改原P1/NWD源码。
- `paper/assignment_coverage_protocol.md`在首次运行新规则前固定：
  旧256基础tile参与过设计，不当新确认；排除后固定抽新128基础tile，
  仅新增128增强输入的冻结P1前向，保持原诊断阈值及CPU重放，不训练。
  mosaic来自同一完整训练池，不能声称独立来源或显著性检验。
- 提交协议和代码后才执行；若确认过线，也只支持另登记唯一完整30轮学生，
  不能把几何/覆盖性质当作mAP或recall改善。

### 3.39 整图覆盖保护的新样本确认未通过（2026-10-31）

- `833cd83`提交后实际新增128基础tile的增强前向，2350正框、1457small，
  基线1396个有效small框，757个small框存在原框外有效分配。
- 新规则保留全部基线有效目标，16/128图退回原TAL；
  small框外目标分数质量352.9970→183.2082，仅降低48.0992%，
  未达到预先固定至少50%条件。small总质量保留95.6121%，其他条件通过。
- 不追认48.10%达到50%，不调阈值或换样本继续该规则；两种分配候选均不训练。
  `paper/assignment_coverage_results.md`保留范围和限制，缓存可CPU重放。
  这不是新的学生性能结果，至少30轮的性能判断要求仍保留。
  后续转向其他单因素；未发现值得大规模训练的方向，持续目标保持未完成。

### 3.40 原教师置信度用于回归的缓存诊断准备（2026-10-31）

- 两项分配路线均停止，另考察已接受伪框回归监督分配。
  原P1分类项乘教师c，box/DFL不额外乘c；新假设只将这两个回归项的伪框监督
  乘原c，人工框保持1，原TAL分母不变，不改标签或拟合新评分。
- `paper/regression_confidence_protocol.md`在统计前固定两组已有缓存、
  small样本量、同类IoU/高质量比例/一对一匹配率、类别控制与有效样本量门槛。
  只重用2710个已缓存的伪框质量，不增加推理或训练，不将隐藏GT用于拟合/选择框。
  分类分数不等于定位可靠性，不冒充RUPL/UTv2/Soft Teacher定位筛选。
- `scripts/audit_regression_confidence.py`和权重/类别组成保护测试完成，
  117项pytest通过；两个新增Python文件Pyright零错误/警告。
  通过也只支持另登记30轮；即使学生正向，后续还需常数衰减对照区分
  整体伪监督强度与逐框置信度排序，不能直接推断独立算法创新。

### 3.41 回归置信度单因素30轮准备（2026-10-31）

- `6c8bf7e`统计前冻结后，两组缓存small同类GT IoU均值按c加权提高
  0.0472/0.0526，高质量比例提高7.3397/6.7066pp，
  类别组成控制后IoU提高0.0411/0.0470；预登记诊断条件全部通过且重放一致。
  详见`paper/regression_confidence_diagnostic.md`，不是学生检测指标。
- 隔离模块`src/regionssod/training/regression_confidence.py`只使box/DFL
  对伪框乘原c、人工框乘1，原未加权TAL分母不变；分类/ignore/分配/标签固定。
  原P1/NWD源码不改，不同时启用NWD或分配候选。
- 唯一候选固定完整11310tile、同split/教师/伪标签、COCO独立初始化、
  seed0、640/batch8/SGD/close_mosaic10、完整30轮；参考复用完整P1。
  配置/入口/包装/验收及`paper/regression_confidence30_protocol.md`已实现。
  预登记性能继续线沿用AP≥+0.30pp且small非负，或small≥+1.00pp且AP≥−0.10pp。
- 122项pytest通过，新增五个Python文件Pyright零错误/警告，
  既有论文表137来源/8组recall核对通过。
  中文提交协议/代码并push后先预检与真实AMP零优化步工程检查，
  通过才启动唯一30轮并创建临时监控；目前无本候选学生性能结果。
  即使通过，也只准备用户更大训练交接，不自动100轮，尚不认定目标完成。

### 3.42 唯一回归置信度30轮实际启动（2026-10-31）

- `5f47671`已中文提交并push协议/配置/源码/验收。
  工程预检23806来源通过，真实YOLO11s AMP前反传255个梯度张量有限，
  优化器步数0；工程产物记录源码哈希。实际训练前重新预检23807来源，
  包含该工程检查文件，完整11310tile不变。
- 12:31:07通过WMI独立启动一次，包装PID21156、训练PID10196，
  Get-CimInstance核实命令行/创建时间匹配回执；不存在PID复用。
  12:32为第1/30轮179/1414batch，stdout实际增长、stderr为空，
  CSV尚待首轮结束，不能因首次缺CSV判断中断。
- 实际args与完整P1除name/save_dir外一致，experiment.json已写入，
  status=training。证据目录`outputs/p1_regression_confidence_pilot30_s0/`。
  临时监控ID为`a`，每10分钟核实本进程，正常静默，仅终态/需处理通知；
  完成/失败处置后删除。先前a-nwd已删除，不复用其提示。
- 当前仅实际启动，不声称性能改善、收敛或已值得大规模训练。
  不重启/拼接，不并行新候选，不改冻结源码/配置/协议；
  原进程停止且exit0/全部阶段结束后执行独立验收，
  过线也只准备用户大训练交接，不自动扩训。
  持续目标实际保持active，未虚报暂停或完成。

### 3.43 回归置信度实验首轮验证工程失败（2026-10-31）

- 12:37:02包装记录真实exit_code=1，status=failed；再次Get-CimInstance
  核对包装21156和训练10196均不存在，全进程命令搜索也无该入口运行。
  原始launch_receipt/status/stdout/stderr均原位保留，未覆盖回执或伪造成功阶段。
- stdout记录第1轮训练1414/1414 batch结束，但验证0/35时报错：
  `ValueError: confidence regression requires augmented per-object provenance`。
  验证集由标准全标注数据加载器提供，没有训练专用的`pseudo_weights`；
  新损失函数未区分验证输入，在缺字段时无条件抛出异常。
- 这是新增实现的工程错误，真实AMP前反传检查未覆盖验证调用路径。
  无results.csv、无best/last checkpoint，完整训练加验证epoch数为0；
  未完成30轮，不能判定该候选有效或无效，也不运行成功终态验收。
- 失败证据及10个原始文件SHA256保存在
  `outputs/p1_regression_confidence_pilot30_s0/failure_report.json`；
  详见`paper/regression_confidence30_failure.md`。
  不自动重启、拼接或另启候选，监控a按失败终态清理。
- 后续工程修复需保留训练缺元数据时的保护，同时兼容正常全标注验证，
  补充实际验证调用路径检查；另行冻结独立运行后再交用户训练。
  本次未修改被冻结的训练源码、配置或协议，持续目标未标为完成。

### 3.44 回归置信度验证兼容修复与独立入口（2026-10-31）

- 原失败运行六个源码/配置/协议文件已按旧preflight哈希核对，
  保存至原输出的`frozen_source_snapshot/`；旧回执、日志和状态未覆盖。
- 新模型在eval时用原P1损失，不依赖EMA中缓存的训练criterion；
  train模式仍保留原c加权及缺元数据异常，不关闭逐轮验证或跳过验证loss。
  单元测试覆盖有框/空框验证、缓存训练criterion、切回train及必要字段保护。
- 增加`--validation-fix`显式入口及独立valfix命名，原无标志训练入口拒绝执行；
  新配置只有run名不同，仍是同参考、同11310tile、同30epoch和固定继续线。
- 工程检查新增实际验证集前2张原图及训练期DetectionValidator的完整调用，
  包含EMA/eval/标准数据预处理/验证loss/NMS和统计，与原P1验证loss精确核对；
  工程期间0优化器步骤，小样本指标不用于性能判断或参数选择。
- 先中文提交并push修复/新协议，随后预检及工程检查；检查后再记录真实结果。
  本次不自动重试完整训练。用户命令见README及baseline第4.16节，
  协议为`paper/regression_confidence30_valfix_protocol.md`。
  当前未获得本候选完整30轮结果，持续目标未完成。
- `84817b2`已中文提交并push。124项pytest全部通过，
  五个相关Python文件Pyright零错误/零警告。实际工程检查成功完成：
  255个梯度张量有限、2张标准验证图/1batch的训练期DetectionValidator正常，
  eval损失与原P1精确一致，train缺元数据仍异常，0优化器步骤。
  首次预检23808来源，加入工程产物后23809来源再次核对通过；
  主表137来源/8组recall核对仍通过。
  工程证据为`outputs/p1_regression_confidence_valfix_pilot30_s0/engineering_check.json`。
  未创建修正版run/status/dispatch/launch_receipt，不存在新训练或监控。

### 3.45 回归置信度的框级监督份额预登记（2026-10-31）

- 修正版尚无launch_receipt/status/run或对应进程；不自动重试训练。
  新增零推理检查，仅统计固定训练清单中已接受正伪框的原c权重，
  读取原metadata/summary，不读隐藏GT、验证GT或新预测。
- 原开发缓存small平均c为0.5817/0.5928，medium为0.7953/0.7792，
  表明加权也可能改变尺度组成。本项预登记全量/各尺度/各类的框数、
  sum(c)、mean(c)、Kish样本量及框数份额与置信度质量份额差。
- mean(c)只是单位框权重的代理保留比例，不能写成实际TAL加权
  box/DFL损失、梯度或召回的降幅。不选择常数、幂指数或补偿规则，
  不改变已交用户的科学变量和30轮继续线。
- 协议`paper/regression_confidence_mass_protocol.md`、
  脚本`audit_regression_confidence_mass.py`在首次统计前中文提交并push。
  131项pytest通过，新增两Python文件Pyright零错误/警告。
  本次为预登记，尚未取得全量统计，不代表学生性能结果。
- `245501b`中文提交并push后首次统计并精确重放：
  4839个正伪框tile、35067框，全量平均c=0.670235；
  small/medium/large平均c=0.585343/0.787948/0.821497。
  small框数份额58.9329%，sum(c)份额51.4685%，差−7.4644pp；
  car份额+4.9497pp，说明需区分监督强度、排序与尺度/类别组成。
  这是单位框权重代理，不能写成实际损失/梯度减少或recall变化。
- RUPL Sec.3.4/式7–8及UTv2 Sec.3.3已重读固定版本相关段落，
  Zotero MCP核实两父条目归属和PDF附件，不新增重复条目。
  细节见`paper/regression_confidence_mass_results.md`，
  不称独立定位不确定性、Listen2Student复现或新颖性证明。
  训练变量和继续线不变，修正版仍待用户运行完整30轮。

### 3.46 小训练启动责任更新与首轮中断归档（2026-10-31）

- 用户最新明确“小训练由你开启，需要大训练找我”，目标已重新active；
  替代3.44–3.45中等待用户启动本次30轮的安排。不以旧责任划分继续阻塞目标。
  30轮确认由助手执行，更大训练仍需完整结果支持并交用户。
- 启动前发现13:28已有一次valfix启动，包装20616/训练1540；
  两次Get-CimInstance核实进程均不存在，stdout停在首轮265/1414batch，
  stderr为KeyboardInterrupt。用户随后确认自己主动终止该次，
  是为交给助手开始训练，不能推断为方法失败。
  回执缺exit_code、原status仍training，均原样保留，不写成exit0或完成。
- 9个原始文件逐项SHA256核对后原生PowerShell归档至
  `outputs/p1_regression_confidence_valfix_interrupted_20261031_133202/`，
  含original_output/original_run/interruption_manifest.json；
  六份冻结源码/配置/协议与原preflight核对后另存frozen_source_snapshot。
- 本次按新授权从原COCO独立开始完整30轮，保留11310tile及原所有配对参数，
  不拼接中断epoch、不修改c或继续线、不新增候选、不自动100轮。
  原工程检查哈希匹配后复用；更换的只是启动责任说明，不改训练源码。
- 先中文提交并push授权/协议记录，再预检和单次独立后台启动，
  核实进程命令/创建时间及日志增长后创建临时监控。
  正常静默，终态验收/失败处置后删除；仅A，目标未达成。
- `1d6b682`/`12b6886`已中文提交并push责任更新及用户中断确认。
  新运行13:36:03启动，Win32_Process.Create的ShowWindow=0，
  原包装22020创建UTC05:36:02.298455，训练22512创建UTC05:36:03.904606；
  Get-CimInstance命令/创建时间与回执匹配，排除PID复用。
  首次加CREATE_NO_WINDOW的WMI调用返回21，没有创建运行；
  去掉该不兼容启动标志后返回0，只创建上述唯一实际进程组。
- 启动前和入口预检均通过23809来源，实际args仅name/save_dir有别，
  model/data/p1/train_args/pseudo_summary与完整P1参考一致，
  工程检查的配置/入口/训练源码哈希未变化。13:37为首轮346/1414batch，
  stdout增长、stderr空，尚无首轮CSV不视为中断。
- 临时监控ID为`a`，名称“A回归置信度修正版三十轮临时监控”，
  每10分钟核实上述原进程和真实日志增长，正常静默，完成/异常/需处理才通知。
  原进程停止且exit0/全部阶段completed后执行
  `python scripts/audit_p1_regression_confidence.py --validation-fix`。
  目标保持active，不把已启动或工程检查通过当作性能过线或目标完成。
- 13:41核实首轮训练及35批验证完整结束，results.csv已写入epoch1，
  三项验证loss及该行所有数值均有限，stderr仍0字节，随后已进入epoch2。
  之前首轮验证缺pseudo_weights的实际崩溃点已越过，并且验证后可继续训练；
  只证明真实训练/验证调用兼容，不根据首轮精度判断方法改进。
  原始首行快照与检查范围保存为输出目录中的`first_epoch_validation_check.json`；
  仍等完整30轮/val960/recall及独立验收，不改训练源码或继续线。

### 3.47 回归置信度修正版30轮完成与取舍结果（2026-10-31）

- 用户在训练期间暂停持续目标，当前保持paused；临时监控只跟进既有训练，
  不恢复目标、不新建替代目标、不并行研究/实验。
- 13:36:03独立开始，15:41:20包装真实exit0，status=completed；
  原包装22020/训练22512均已停止，CSV连续1–30轮、11310tile。
  完整548原图val960、固定操作点recall及对比全部结束，不混入旧中断进度。
- 15:48执行`python scripts/audit_p1_regression_confidence.py --validation-fix`成功，
  23809冻结来源、实际args/元数据/权重/产物哈希及两组548图保存预测重放一致。
  输出目录`outputs/p1_regression_confidence_valfix_pilot30_s0/`的
  independent_audit记录accepted=true、worth_larger_training=false。
- 相对同预算完整P1，mAP50-95−0.1755pp、mAP50−0.3134pp、
  small recall+0.9444pp（+251匹配）、medium/large−0.1350/−0.1874pp；
  类别AP为1升9降。学生CSV累计7450.65秒，参考7286.61秒，
  多164.04秒（约2.25%），负载未控制，不作独立速度结论。
- AP≥+0.30pp路线未达；small≥+1.00pp且AP≥−0.10pp路线两项均未达。
  保留召回/精度取舍，不降低继续线、不按val改权重或扩大本变体训练。
  完整结果及类别/尺寸/证据在`paper/regression_confidence30_results.md`，
  实验4.16、证据索引及复现入口同步；训练源码/配置/协议保持冻结。
- 本轮满足至少完整30轮后判断，但单split/固定教师/学生seed0不证明
  显著性、泛化、充分收敛或100轮必定无效；常数衰减/排序/组成作用仍未分离。
  固定操作点recall非官方AP_small/AR500，原Gate3和既有正向P1结果保持。
- 本次只做终态验收、文稿记录、适当检查、中文提交push及临时监控a清理。
  131项pytest通过，原主表137来源/8组recall检查通过，新结果页15行指标
  与原始JSON及best/last哈希核对通过，23809冻结来源再次匹配；
  检查记录位于本次输出的`documentation_check.json`，只修改Markdown文稿。
  持续目标保持paused；后续研究或更大训练交接需用户点击继续，
  当前尚未找到通过固定继续线的新候选，不宣称目标已完成。

### 3.48 用户更新探索授权与覆盖保护30轮准备（2026-10-31）

- 用户重新发起持续目标，实际状态active；小训练由助手启动，大训练由用户执行。
  最新要求“门槛不要设计的太高”用于后续探索，不追认历史实验通过，
  不改原Gate3或已完成NWD/回归置信度的判定。
- 原覆盖保护诊断保留100%有效目标、小框框外质量降低48.0992%，
  未达旧50%代理线。明确更新原不训练安排，投入一次完整30轮直接检验学生，
  不改规则、重采样或把旧缓存包装为独立确认。
- 新协议`paper/assignment_coverage30_protocol.md`固定唯一分配改动，
  同11310tile/教师/split/seed0/COCO独立初始化，原P1其余因素不变。
  新探索交接条件为AP≥+0.10pp且small不降，或small≥+0.50pp且AP不降。
  这是用户授权的低门槛探索，不是论文显著性或Gate3成功。
- 新隔离适配器和完整预检/工程检查/训练/验证/独立验收入口已实现；
  先中文提交并push，再核对真实AMP前反传和标准验证路径，零优化器步骤，
  通过后才由助手单次启动30轮及临时监控，不并行新候选。
- Zotero MCP确认29篇已有论文及PDF；将原YOLO-RegionSSOD集合BMQTT9Y7
  移到新Research集合BW83SN4A下，保留条目、附件和其他集合归属。
  Paper Search MCP再次核实FCOS版本1904.01355v5，不重复导入。
- `dbc8e76`已中文提交并push；141项pytest通过，原主表137来源和8组recall核对通过。
  真实工程检查255个梯度张量有限，标准两图训练期验证正常，eval损失与P1
  精确一致，train缺元数据拒绝，0优化器步骤。23809冻结来源预检通过。
- 16:07:31由助手WMI隐藏独立启动，包装15432/训练30956创建时间和命令核实，
  `os_dispatch`返回0，实际args与完整P1配对；首轮CSV待验证结束，
  stdout已增长到518/1414batch，stderr为空，不将其作为性能结果。
  回执/工程/身份记录位于`outputs/p1_coverage_guard_pilot30_s0/`。
- 临时heartbeat监控30已创建，每15分钟检查，正常静默，终态独立验收、
  记录及中文提交push后删除。仅跟进原进程，不重启/续训/并行候选。
- 等待期间核对FCOS PDF第4/6页及TOOD第5/7页，TOOD既有父条目7D3IXLK2
  和PDF补入同一集合，现共30篇。依据及贡献边界见`paper/coverage_guard_literature.md`；
  没有更改冻结训练源码、参数或探索继续线。

### 3.49 训练等待期准备条件式正式配对入口（2026-10-31）

- 已核实原包装15432/训练30956存活；覆盖保护首轮真实验证和第二轮验证结束，
  CSV连续1–2轮且有限，stderr为空，已进入第3轮。不是30轮性能证据，
  不按中间指标改变量、停止或判定；本轮保持active，监控30继续。
- 等待期新增独立`scripts/run_coverage_guard_full100.py`及
  `paper/coverage_guard_full100_protocol.md`，只准备用户未来运行的公平100轮入口，
  不修改正在运行的小训练、冻结源码/配置或输入，不启动大训练。
- 100轮入口由原P1确定性生成两臂配置，仅epochs从30到100；
  候选额外分配规则，其他因素配对。两臂各自从COCO初始化，依次运行，
  自动完整val960/recall、100行连续性、参数/来源/产物和预测重放验收。
  不从30轮权重续训，不自动扩成其他seed或预算。
- `--plan-only`无输出/训练地显示配置；实际`--check-only`/`--train`
  必须先有本次30轮完整exit0、独立验收和正向继续条件，再重放原始证据。
  当前在真实未完成状态调用`--check-only`已按预期拒绝，
  未创建正式输出或训练进程。准备交接代码不等于已找到值得扩训方向。

### 3.50 按用户明确授权暂停等待覆盖保护训练（2026-10-31 16:24）

- 用户再次明确训练期间可由助手选择暂停；本次选择暂停持续目标等待，
  不继续重复轮询或进度汇报。仅停止额外的只读连续观察会话49603，
  原包装15432/训练30956仍存活，训练进程及监控30保留。
- 监控按已有职责在终态后验收、记录、中文提交push并删除自身；
  等待期间不改冻结源码、不重启或并行训练，不自动启动100轮。
- 现有目标控制只提供complete/blocked/paused，不能将paused自动恢复为active。
  不声称自动恢复；终态收尾后如需接续研究或正式交接，明确通知用户点击继续。
  此限制不影响已运行训练或已授权的终态验收。

### 3.51 覆盖保护完整30轮终态验收（2026-10-31）

- 本次只按监控职责完成既有训练的终态收尾；get_goal实际状态仍paused，
  不恢复目标、不新建替代目标、不启动新研究或训练。
- 原包装15432/训练30956均已停止，真实exit0、status=completed；
  16:07:31启动、18:29:34结束，CSV连续1–30轮且有限，11310tile。
  完整548原图val960/recall和比较均结束，stderr为空。
- 18:42执行python scripts/audit_p1_coverage_guard.py通过：
  23809冻结来源、实际参数、源码/配置/权重/产物哈希及两组548图保存预测
  匹配重放一致；independent_audit中accepted=true、worth_larger_training=false。
- 相对同预算完整P1，mAP50-95−0.0652pp、mAP50+0.0719pp、
  small recall+0.1016pp（+27匹配）、medium/large−0.1350/−0.3749pp；
  类别5升5降。CSV累计8431.75秒，参考7286.61秒，多15.72%，
  负载未控制，不作独立模块速度结论。
- 本轮AP≥+0.10pp且small不降路线未达，small≥+0.50pp且AP不降路线
  两项未达。不按结果改线/改规则追试，不启动条件式100轮。
  完整结果见paper/coverage_guard30_results.md，实验4.17、证据和命令入口同步。
- 用户提醒应认可小目标召回的正向成果：本次固定验证净增27匹配、
  small recall+0.1016pp，作为阶段性结果和AP/召回取舍保留。
  “未达预设扩训条件”不等于无成果，不据此宣布机制无效；
  单seed尚不能判断小幅收益是否稳定。历史预登记判断不追改。
- 旧几何诊断的原有效目标保留100%并不等于所有GT有监督或学生recall100%；
  48.0992%框外质量减少未转化为本次扩大训练条件。
  单split/教师/seed0不证明显著性、泛化或100轮必定无效；原Gate3不变。
- 中文记录并push后删除临时监控30。目标继续paused；
  现有控制不能自动设回active，需用户点击继续后再接续后续研究。
- 18:48收尾核对本页15行指标与原始JSON、结果产物SHA256；
  原主表137来源/8组recall检查通过。当前未达条件时100轮入口确实拒绝预检，
  未创建正式输出，记录见outputs/p1_coverage_guard_pilot30_s0/documentation_check.json。
  本次只修改Markdown文稿，没有训练源码或参数变更。

### 3.52 综合盘点训练价值，优先交接原主线（2026-10-31）

- 用户明确“成果衡量不一定只有两个标准，需要自己盘点是否继续训练”。
  按本次直接请求综合评估，不要求用户重复点击继续才能做盘点和交接；
  持续目标实际状态仍paused，不虚报自动恢复。
- 新脚本`audit_training_investment.py`重放12组完整30轮和548原图保存预测，
  纳入整体/类别AP、尺寸/逐目标增失、受益图数、固定点未匹配预测/F1、
  IoU0.75匹配、重复seed、成本与论文缺口。未新增训练或GPU推理。
- 覆盖保护small新增973/丢失946，净+27；191图净改善、175图净下降。
  全尺寸净+8，未匹配预测+66，固定点F1−0.0414pp，IoU0.75 small−14。
  因而保留局部成果，但暂不优先追加其100轮/seed。
  不是把旧条件失败等同于没有成果，也不证明长程一定无效。
- 原P1−B2两个学生seed的AP/小目标召回/F1均同向，
  small净+499/+450，IoU0.75 small+251/+167；同时未匹配预测+642/+877，
  固定点precision下降，完整报告代价。两seed平均AP+0.2893pp、
  small+1.7854pp，尺度消融另有重复支持。
- 独立研究决定：继续原B2/P1正式100轮主对照，优先补第一篇最直接缺口；
  不再以寻找新模块为前提，不改原Gate3或覆盖保护预登记判定。
  依据见`paper/training_investment_review_20261031.md`，机器记录绑定原配对结果SHA。
- 新入口`scripts/run_paper1_full100.py`保持原数据/教师/方法与初始化，
  生成两臂各100轮，顺序运行，自动val960/recall/真实exit/预测重放验收，
  拒绝已有目录/并发正式臂/来源变化，P1实际训练须先验收B2。
  协议`paper/paper1_full100_protocol.md`，助手仅预检、不启动大训练。
- 历史时间外推学生总计10.78–12.16小时，加评估/I/O可预留12–15小时；
  本轮只交接seed0，不自动扩seed/split/教师/面积预算，后续按完整结果综合判断。
- 155项pytest通过，原主表137来源/8组recall检查通过；
  新增盘点表7行全部逐项对应原始重放数据。
  在源码提交a65cc05实际运行两臂`--check-only`均exit0，各37772冻结来源，
  stderr为空，未创建正式输出、未训练或新增推理。
  记录`outputs/training_investment_20261031/handoff_preflight.json`。
  交接已完成，用户可按协议启动；这不表示100轮结果或第一篇论文已经完成。

### 3.53 用户误中断的B2正式运行与显式恢复（2026-10-31）

- 用户报告误中断并要求检查恢复。原B2 19:13:36启动，23:01:05
  在第64轮初期记录KeyboardInterrupt；PID8248已停止，无其他正式进程。
  原包装未写入真实exit_code，保留未知，不仅凭status推断退出码。
- CSV连续完整63轮，累计13621.5秒；last.pt可读取，
  epoch=62、原总100轮，优化器255状态、EMA updates11683、AMP scaler存在且有限。
  SHA256 d44cfb626e7b82bbec5c1f8081d0cde6416d46d30ffb709475347a6f1cdf0f8e。
- 保存并逐个核对原13文件到outputs/paper1_b2_formal100_recovery_20261031，
  包含best/last/CSV/args/experiment、原始日志/启动/状态/来源及原入口源码。
- 根据本次新请求增加显式恢复，不自动重试：默认--train仍拒绝已有目录；
  --resume-check只检查该捕获中断，--resume由用户从64轮继续原计划到100轮。
  不改变训练方法/数据/超参，原experiment保留初始COCO记录并单列恢复事实。
- 原入口哈希变更通过原源码归档和新恢复来源清单共同校验，只放行此处恢复代码；
  原训练源码/数据哈希仍须一致。原CSV字节前缀保留，新段正常追加，
  训练时间累计采用13621.5秒加恢复段累计时间。
- 真实yolo恢复初始化在独立副本目录通过：start_epoch63/总100、
  scheduler.last_epoch62、255项优化器逐张量一致、EMA和scaler恢复；
  在首个训练batch前停止，0优化器步骤，原中断产物哈希未变。
- 文档paper/paper1_full100_recovery.md明确这不是逐位连续训练，
  需在论文记录checkpoint恢复；前63轮不丢弃，仅重算未保存的64轮部分。
  剩余37轮按近期178.68秒/轮估计约1.84小时，另加评估/I/O。
  P1待B2最终验收后由用户单独运行，不自动启动。
- 源码提交49d114a后实际`--arm b2 --resume-check`退出0：
  completed_epochs63、resume_epoch64、total_epochs100，归档/来源/检查点均通过。
  全库159tests及原主表137来源/8组recall检查通过。
  原中断文件哈希未变，未创建recovery/dispatch、未启动剩余训练；
  记录为归档handoff_check.json，恢复代码及文档已中文提交并push。

### 3.54 B2正式100轮完成验收与P1交接（2026-11-01）

- 用户报告训练完毕，核实仅B2正式100轮完成，P1正式输出均不存在。
  恢复工作PID29152身份与原回执对应，已停止；2026-11-01 01:24:17
  真实exit_code0，status=completed，resume_stderr为空。
- 重新执行`--arm b2 --audit`返回accepted=true：
  CSV连续1–100轮且全部有限，原1–63轮字节前缀保持，
  源码/恢复归档/原冻结输入/实际args/配置/权重与产物哈希一致，
  548张完整原图固定操作点预测重放一致。
- val960 mAP50=42.7885%、mAP50-95=25.8264%，
  small/medium/large recall=39.2858%/79.5052%/89.5033%；
  GT与匹配分别为26577/10441、11115/8837、1067/955。
  固定点recall不是AP_small；训练640指标不混用。
- 与既有B1同100轮参考比较，AP50/50-95+1.0381/+0.9429pp，类别9升1降。
  两者训练数据量/计算量及教师成本不同，B2包含恢复；
  保留正向阶段成果，不据此称P1已提高或跨split稳定。
- 训练计时为原13621.5秒+恢复7728.22秒=21349.72秒（5.9305小时），
  避免只计恢复段；原退出码未知及中断证据仍保留。
  结果见paper/b2_full100_results.md，实验4.18和证据页同步。
- 实际执行`--arm p1 --check-only`，37773冻结来源通过，无新训练/推理。
  下一步仍值得完成原P1同100轮主对照；由用户执行`--arm p1 --train`，
  不自动启动、不追加seed或新模块，也不拿B2 100轮与P1 30轮判优劣。
  论文A与正式配对未完成，本次只完成B2验收和P1交接。
- 本次仅修改Markdown：新增结果页13行总体/固定点/参考比较全部与原JSON及
  保存预测计数一致；原主表137来源/8组recall检查通过。
  记录outputs/paper1_b2_formal100_s0/documentation_check_20261101.json，
  未修改冻结训练/恢复源码、协议或参数，未新增训练。

### 3.55 P1/B2正式100轮配对验收与综合投入判断（2026-11-01）

- 用户报告P1完成，重新执行`python scripts/run_paper1_full100.py --arm p1 --audit`，
  双方accepted=true。P1工作PID15920与启动回执的创建时间/命令一致且已停止，
  09:33:02启动、16:17:55真实exit0，stderr为空；CSV连续完整100轮且数值有限。
  两臂冻结来源37772/37773项、实际args、权重/产物及548原图val960预测重放通过。
  B2原63轮后恢复、原退出码未知、前缀和归档证据继续保留，不称逐位连续等价。
- P1的mAP50/50-95=42.9769%/25.8791%，相对B2 +0.1884/+0.0528pp；
  small/medium/large recall=41.4117%/79.9730%/89.6907%，
  分别+2.1259/+0.4678/+0.1874pp。small匹配10441→11006（净+565），
  medium/large分别净+52/+2。完整细表见`paper/p1_b2_full100_results.md`。
- 新增只读入口`scripts/audit_paper1_full100_effects.py`重放已有预测，不训练/新推理。
  small逐目标新增1350、丢失785；图像266改善/113下降/169持平。
  同conf0.25/IoU0.50操作点未匹配预测7819→9040（+1221），
  precision−2.3690pp、全尺寸recall+1.5970pp、F1+0.1800pp。
  类别AP7升3降，bicycle−1.1677pp最多；收益和代价同时成立。
- 固定conf0.35/0.50重放small净+535/+334、F1+0.6567/+0.8396pp；
  匹配IoU0.75下small净+229，但全尺寸F1−0.3767pp。
  不据此选择新阈值或修改训练。548图配对bootstrap的small差95%区间
  [+1.7157,+2.5438]pp，仅为固定权重条件下的图像抽样，不是seed显著性证据。
- B2/P1学生21349.72/24229.10秒，P1多约48分钟、13.49%；
  数据量和负载不同且B2恢复，不把差值全部归因于损失机制。
  原30轮两seed与本次100轮分别报告，不合并统计或追改原Gate3。
- 综合决定：小目标收益值得一次正式预算的学生seed1重复，检查随机性和误检取舍；
  不直接延长本次seed0到200/300轮，不再横向叠加模块，也不增设任意AP门槛。
  当前已具备阶段成果。seed1仍为同split/教师，不称跨split或独立教师确认。
  若基本重复优先收束稿件；若变化明显先报告波动及缓存诊断，不自动追加seed2。
- 新建隔离入口`scripts/run_paper1_seed1_full100.py`和
  `paper/paper1_seed1_full100_protocol.md`，只改学生seed与输出名；
  B2先于同seed1 P1，完整验证、真实退出码、来源/args/预测重放自动验收。
  167项测试通过。两臂实际只读预检结果在第10节记录；未启动任何新训练。
  预计学生合计12.66小时，含完整评估等预留13–15小时，由用户执行大训练。
- 新结果页24行总体/尺寸计数/误检F1/类别表已逐项核对原始JSON，
  记录`outputs/paper1_full100_effects_s0/documentation_check_20261101.json`；
  原表格生成器137来源/8组recall检查及`git diff --check`通过。
  原seed0训练/恢复入口及冻结协议未修改。
- 交接源码/协议在`e62e456`中文提交并push后，16:46–16:47实际执行学生seed1
  B2/P1两臂`--check-only`，各37803冻结来源、真实exit0且stderr为空，
  8个新训练/评估输出目录均不存在。回执
  `outputs/paper1_full100_effects_s0/seed1_handoff_preflight_20261101.json`。
  另核对两seed旧pilot实际args，除seed/输出名外无差异。
- 只读分析改为追踪稳定metrics，而非会随复核更新时间戳的independent_audit；
  同时要求两者测量完全一致。重放结果未变，557项分析来源哈希及24行表格再次通过，
  167项测试再次通过。此调整不改变冻结训练入口或新seed1协议。

### 3.56 B2学生seed1启动核实与临时监控（2026-11-01）

- 用户已手动启动`scripts/run_paper1_seed1_full100.py --arm b2 --train`。
  核对包装PID10476（17:44:48.599021）与训练PID31640（17:45:25.622818），
  创建时间/命令与启动回执一致、两者存活；stdout推进到第1/100轮666/1387批，
  stderr为空，首轮尚未完成、CSV暂未生成，不据此判定失败或性能。
  初始核实记录于`outputs/paper1_b2_formal100_s1/monitor_initial_check_20261101.json`。
- 按用户请求创建本会话临时心跳`b2-1`，每15分钟检查。
  正常推进保持安静；只在完成、失败、显著异常或需要用户操作时通知。
  结合进程身份/CSV/日志/真实退出码判断，禁止重启、续训、改冻结来源或并行训练。
- 100轮之后仍须等待完整val960/recall和包装真实退出，再用yolo环境执行
  `python scripts/run_paper1_seed1_full100.py --arm b2 --audit`，
  核对连续完整100轮、实际args/冻结来源/产物及548图预测重放。
  验收结果入论文/证据及第10节，中文提交并push，随后提醒用户手动启动同seed1 P1。
  失败/中断保留证据、不自动重试；终态收尾后删除`b2-1`。
  本次仅观察既有训练及建立监控，不改变持久研究目标状态。

### 3.57 B2学生seed1正式100轮验收与P1交接（2026-11-01）

- 原唯一B2训练于23:04:51真实exit0，状态completed。
  包装PID10476、训练PID31640按创建时间和精确命令持续跟踪，
  23:18终态核实均已停止，stderr为空；CSV连续完整1–100轮且有限，
  未中断或恢复，学生累计19111.9秒（5.3089小时）。
- 使用yolo环境独立执行`python scripts/run_paper1_seed1_full100.py --arm b2 --audit`，
  accepted=true；37803冻结来源、实际学生seed1/split0与args、
  权重/产物哈希、548原图val960和固定操作点预测重放通过。
  原始日志/回执保留于`outputs/paper1_b2_formal100_s1/`。
- AP50/50-95=42.5503%/25.6683%，small/medium/large recall=
  39.3611%/79.1273%/89.1284%；small匹配10461/26577。
  论文细表见`paper/b2_seed1_full100_results.md`及实验4.20；
  固定点recall不是官方AP_small，不用B2单臂或跨seed比较声称P1收益。
- 23:21实际执行P1 `--check-only`，exit0、37803来源通过、stderr为空；
  P1的output/train/val/recall目录前后均不存在，无新训练或推理。
  回执`outputs/paper1_b2_formal100_s1/p1_handoff_preflight_20261101.json`。
  当前只提醒用户手动执行`python -u scripts/run_paper1_seed1_full100.py --arm p1 --train`；
  按既有P1学生6.73小时估算，含评估预留7–8小时，实际随负载变化。
- 本次仅更新论文/状态/复现文档，冻结源码/协议不改，新增表格单独核对，
  8行新增表格及best/last哈希复算通过，原30轮表格137来源/8组recall检查通过，
  `git diff --check`通过；回执为本次output下`documentation_check_20261101.json`。
  中文提交并push后删除临时监控`b2-1`；
  不代启动P1、不重试B2、不自动恢复持久研究目标。
  P1完成后才更新同seed配对与两seed正式100轮统计，原Gate3及30轮证据保持。

### 3.58 P1学生seed1启动核实与临时监控（2026-11-02）

- 用户手动执行`python -u scripts/run_paper1_seed1_full100.py --arm p1 --train`。
  包装PID4692，创建时间09:53:19.249228；训练PID45248，
  创建时间09:54:10.710917。09:54:59核对两者创建时间/命令、父子关系及存活状态，
  工作命令与launch_receipt一致；status=training，stdout已推进至首轮197/1414批，
  stderr为空。首轮尚无CSV，不据此判失败或效果。
- 初始核实保存于`outputs/paper1_p1_formal100_s1/monitor_initial_check_20261102.json`，
  原日志/回执及训练目录保持不变；未启动新训练、重启或修改冻结来源。
  按用户请求创建临时心跳`p1-1`，每15分钟核实进程身份、CSV/日志和真实退出码，
  正常推进保持安静；只在完成、失败、显著异常或需要用户操作时通知。
- 完整100轮之后仍须等待548原图val960/recall及包装真实退出。
  正常exit0后用yolo环境执行
  `python scripts/run_paper1_seed1_full100.py --arm p1 --audit`，
  核对来源/实际args/产物/预测重放及同seed1 B2；
  核对paired_comparison和仅含正式100轮的two_seed_comparison。
  不能跨seed拼配或与30轮混合统计，固定点召回不称官方AP_small。
- 终态后将AP、尺寸/匹配、类别退化、缓存可核实的误检/F1、成本和后续投入判断
  写入论文/证据及第10节，中文提交并push，再删除`p1-1`。
  失败/中断保留证据、未知退出码如实记录、不自动重试；
  不启动后续训练、不自动恢复或替换已暂停/清除的持续研究目标。
  本次仅建立观察和交接记录，不更改训练源码/协议或判断方法效果。

### 3.59 P1学生seed1的55轮检查点恢复交接（2026-11-02）

- 用户报告VS Code窗口崩溃并要求继续训练。截图为窗口oom，
  训练status/stderr在13:43:11记录KeyboardInterrupt，原包装4692和工作45248
  按身份核实均停止；launch_receipt未记录真实exit_code，保留未知。
  13:43:44实测物理内存约63.85GiB、可用34.27GiB、使用46.3%，
  这是崩溃后的测量，不能据此反推瞬时占用；不把窗口oom等同于整机RAM或CUDA耗尽。
- CSV连续有限1–55轮，累计13602.5秒；last.pt零起始epoch54、
  原seed1/总100轮参数一致，优化器255项、EMA10496次更新及AMP scaler完整。
  权重及优化器有限，last.pt SHA256为
  `19c2aa987aa21465190c86e9f48a6b005006679501d7769265d19741e5fe6160`。
  21个原始文件复制并逐项核对，保存在
  `outputs/paper1_p1_formal100_s1_recovery_20261102/`，不覆盖原日志/状态/回执。
- 新增独立入口`scripts/resume_paper1_p1_seed1.py`及
  `paper/paper1_p1_seed1_recovery.md`，原训练源码、入口和冻结协议均不改。
  显式传P1Trainer及原soft_weights/uncertain_ignore，禁止退回通用训练器。
  实际续训只允许model/resume记账变化，单次入口默认拒绝重复恢复。
- 已用真实检查点副本完成初始化后立即停止的验证，训练批次及优化器步骤均0：
  start_epoch55/总100、scheduler54、255项优化器逐张量及参数组一致、
  EMA权重/更新次数与AMP恢复一致，P1类型和选项正确，21个原始文件未变。
  回执为归档`engineering_verified/check.json`。
  首次副本验证因隔离目录派生name不同而拒绝，证据保留；
  仅副本核对name/save_dir为隔离路径，不放宽真实续训参数。
- 用户在独立Git Bash执行`python scripts/resume_paper1_p1_seed1.py --check-only`，
  通过后执行`python -u scripts/resume_paper1_p1_seed1.py --resume`。
  从第56轮重算，仅补45轮至原100轮，不由助手启动；按原速度约3.1小时，
  含验证预留3.5–4小时。续训写resume前缀日志/回执，原退出码不补写为0。
- 完整结束后用独立恢复入口`--audit`，验证原55轮CSV字节前缀及连续100轮、
  原来源/实际args/548原图val960与recall预测重放、真实恢复exit0和工作停止；
  计时为13602.5秒加恢复段，重验同seed1 B2及两seed正式100轮配对。
  披露seed0 B2和seed1 P1分别恢复，不承诺逐位连续等价。
  临时监控p1-1在准备期间暂停，交接后改为等待resume回执再观察，
  无回执不自动启动；终态处理后删除，不自动重试/扩训或恢复持久研究目标。
- 验证：完整`python -m pytest -q`的179项通过（含12项新恢复检查）；
  原论文表137来源/8组recall检查及`git diff --check`通过。
  交接源码/说明在`e54eb61`中文提交并push后，14:01实际执行新入口`--check-only`，
  exit0、stderr空、全部37803冻结来源和同seed1 B2通过，
  未创建resume/评估输出，不执行训练；归档回执`handoff_check.json`。
  临时监控p1-1已更新为恢复入口/回执并启用，等待用户启动后静默核实。
  VS Code本地main.log确认13:42:45.872为renderer process gone/oom，
  补充证据为归档`window_crash_evidence.json`，不据此推定整机内存峰值。

### 3.60 P1学生seed1恢复完成与两seed正式100轮收束（2026-11-02）

- 用户14:05启动既有恢复，包装28560/工作41744的PID、创建时间和精确命令
  与resume回执一致；17:20:27包装真实exit0、stderr为空，终态核实时均停止。
  CSV连续有限1–100轮，原55轮字节前缀、21文件中断归档及原未知退出码保留。
- 17:35独立`python scripts/resume_paper1_p1_seed1.py --audit`通过，
  accepted=true；37803冻结来源、学生seed1/split0实际args、P1Trainer与
  soft_weights/uncertain_ignore、255项优化器/EMA/AMP/调度恢复、权重/产物、
  548原图val960/固定点recall及预测重放核对一致，同时重验B2与seed0配对。
- seed1 B2/P1的AP50-95=25.6683%/26.0034%（+0.3351pp），
  AP50+0.5534pp；small recall=39.3611%/41.7805%（+2.4194pp），
  匹配10461→11104，净+643；medium/large净+176/+3。
  small逐目标新增1406、丢失763，286图改善/105下降/157持平。
- 同conf0.25/匹配IoU0.50下，未匹配预测7826→9261（+1435）、
  precision−2.6573pp、F1+0.4031pp；类别AP5升5降。
  两seed的bicycle/awning-tricycle AP均下降；更严IoU0.75时small仍净+251，
  但全尺寸F1−0.4233pp，不称为AP75或官方误检。新增只读入口
  `scripts/audit_paper1_seed1_effects.py`沿用seed0的缓存分析规则，独立输出s1。
- 两学生seed正式100轮平均配对增益±样本SD：AP50-95
  +0.1939±0.1996pp、small +2.2726±0.2075pp，small各净+565/+643。
  固定split/教师，披露seed0 B2及seed1 P1分别恢复，不承诺逐位连续等价。
  不与旧30轮混合，不作显著性/跨split泛化/官方AP_small主张；
  原Gate3不追认为通过，原Gate4预注册效应量也未证实达到。
- seed1 P1学生13602.5+11590.7=25193.2秒（6.9981小时），
  B2为19111.9秒（5.3089小时），多31.82%；数据量、负载与恢复不同，
  非纯模块开销，教师/伪标签/最终评估另计。
  综合两seed覆盖收益、整体AP幅度、类别/误检和投入，建议本轮主线训练收束，
  优先写作与已有机制分析；不自动补seed2、延长轮数或新增模块。
  B0、其他预算/外部确认/S³OD同协议对照仍缺，按投稿主张检查必要性，
  不宣称论文证据已齐全或无条件追随历史完整矩阵。
- 细表与来源见`paper/p1_b2_seed1_full100_results.md`、实验4.21及evidence。
  缓存结果`outputs/paper1_full100_effects_s1/analysis.json`，
  ARS validate报告保存在同目录，11/11谬误检查，状态ANALYZED，
  仅重放既有证据而未重训验证模型可复现性。179项pytest已通过；
  新表数字逐行对JSON核验，原表137来源/8组recall和`git diff --check`通过。
  中文提交push后删除临时监控p1-1；不自动恢复或新建持续研究目标。

### 3.61 冻结权重AP_small、完整PR与等精度覆盖补评估（2026-11-02）

用户要求“先补AP_small、完整PR曲线和相同精度下的小目标召回”，明确暂不启动新训练，
并允许skill/MCP找创新线索；另要求论文放入Zotero项目目录、中文提交并push。

- 固定四个已验收100轮best.pt、split0/教师及548完整原图，保存conf≥0.001、
  960/batch8/FP32/NMS IoU0.70/max_det500的validator原图预测；原Ultralytics
  mAP50/50-95四臂均精确重放，权重不变，无优化器步骤。
- COCO-style AP_small的P1−B2为seed0 −0.2460pp、seed1 +0.6129pp，均值
  +0.1834±0.6073pp（样本SD）；方向不同，不能声称小目标AP稳定领先。
  该评估用当前YOLO标签及COCO面积忽略，不是VisDrone官方AP，不追认Gate3/4。
- 探索性AP_medium平均+0.2242pp、AP_large两seed+3.0256/+2.2382pp，均值
  +2.6319±0.5568pp；large仅1067GT，car占778、部分其他类仅3–7个，需交代
  类别支持。保留此正向线索，不事后用它替换小目标主问题或作稳定泛化宣称。
- 全尺寸micro precision约束下，IoU0.50的small recall在50/60/70%三个档位两seed
  均增加，平均+0.7036/+0.7375/+0.7036pp；80%方向不同，90%均下降、平均−0.2032pp。
  对齐本轮B2 conf0.25约71.8%precision的锚点，small净增171/201（+0.6434/+0.7563pp）。
  全部IoU0.75/逐类结果、实际阈值和精度均报告；经验前沿不是独立部署保证。
- 新validator与旧predict有分批缩放/填充口径差异，固定点计数分开保留；大量图像
  达到500框，完整PR仅覆盖保存的候选范围。没有回选检查点或校准拟合。
- 首次数据配置预检在GPU前失败，修正目录读取；第一版分析之后的独立代码复核
  发现来源绑定和乱序类别名称两个问题，新增回归测试后在bound_v2独立重建绑定。
  条件不变、四臂预测与第一版字节哈希相同，失败/初版日志保留，不择优或覆盖。
- 190tests通过；正式缓存审计核对1119输入哈希、四臂预测绑定与全部重算数字精确一致。
  数据/NPZ/CSV/PNG/PDF/SVG及回执仅在`outputs/paper1_small_pr_20261102/`。
  `scripts/build_small_pr_results.py --check`单独核对新增表，不冒充原137来源/8组recall检查。
- 文献MCP和原始来源核对五篇实际采用论文，已在Research / YOLO-RegionSSOD
  （BMQTT9Y7）查重归档：新增STOD HBL9PADC、复用四篇，四篇有PDF、STOD仅摘要。
  阅读版本/范围和三个AI待证方向见`paper/small_pr_literature.md`；未标成人类已读，
  未证明新机制有效或完成全面新颖性排查。

本轮完成补评估及文献线索收集，下一步优先整合第一篇的条件性覆盖收益和代价；
精度区间风险约束、存在/几何证据分离与ignore空间预算均仅为候选假设，不启动训练。
现有预测已完成；只读复核用`python scripts/evaluate_paper1_small_pr.py --audit`及
`python scripts/build_small_pr_results.py --check`，不要再次`--predict`。

### 3.62 高精度退化原因与尺度条件操作点（2026-11-02）

用户要求从H1–H3选择一个方向，先用现有缓存做零训练诊断，找到高精度退化原因，并在需要时
给出训练指令。选择H2存在/几何证据分离；结果削弱H2后，按已有缓存补做H1必要条件的
图像跨拟合。全程新增推理0、训练步数0。

- H2诊断：四个冻结权重的IoU0.50、90%precision下，P1−B2 small净增为−65/−43；
  B2命中而P1未命中的711/752个small GT中，695/730个在P1完整缓存中实际匹配，
  但低于P1全局阈值。粗定位、类别错误、近邻和竞争只占少数。
- P1高精度阈值为0.568531/0.573130，B2为0.553496/0.559138；共同small GT的
  P1−B2平均IoU差仅+0.000737/+0.000791，中心/边界误差略低。没有证据表明P1丢回的
  目标系统性定位更差，H2不能支持新的回归/NWD训练。
- 错误组成中wrong-class/GT未解释背景占多数但两臂共有；GT-oracle删除单类FP只显示条件
  容量，wrong-class删除使P1−B2 gap变化+0.44/+0.09pp，背景为+0.29/+0.58pp，
  不可部署、不可相加，也不能作为训练因果。
- H1跨拟合：固定5个图像split seed，各274/274两折交换校准/评估，201点对数阈值网格，
  每臂独立选pred-small/other阈值。目标0.90时，B2/P1 scale−global small召回分别约
  +4.44/+4.51pp（seed0）和+4.69/+5.38pp（seed1），评估precision约0.900；但P1−B2 gap
  改善seed0 +0.0662pp（2/5 split正），seed1 +0.6856pp（5/5正），预先两seed≥+0.20且
  ≥4/5同向的继续线未通过；0.95也未通过。尺度条件阈值不是P1独有优势。
- 正式输出：`outputs/paper1_high_precision_diagnostic_20261102/`与
  `outputs/paper1_scale_operating_point_20261102/`；缓存审计分别通过，逐GT事件、四臂数组、
  逐fold阈值/实际precision/召回、计算源码快照、PNG/PDF/SVG和失败回执均保留。
- 201 tests通过；两份文档表与JSON精确重放。`scripts/evaluate_paper1_small_pr.py`等旧入口不变。

结论：第一篇应把高精度退化写成“全尺寸precision约束下全局阈值/候选排序限制较低分小目标覆盖”
的适用边界，而不是已证明的回归质量恶化。尺度条件操作点可作为通用分析，但不进入P1方法主表。
**当前不提供训练命令、不启动模型**：无GT可实现的新质量信号、具体损失动作及区别于旧c/NWD的
配对协议尚未定义；若未来补足，再进行至少30完整配对epoch。原Gate3/4、原30/100轮主作用和
旧失败探索不作追认。

### 3.63 现有P1成果保存、正文整理与叠加候选隔离（2026-11-02）

用户要求尽量保留当前成果，并在原P1上加入新改进。
据此保留原尺度阈值、c分类权重及ignore主线；类别歧义只作为独立叠加候选，
不把先前NWD、回归权重或分配保护默认叠加，不因候选失败替换原结果。

- `paper/preservation.md`及`preservation_receipt.json`：原源码基点c1ecf6c、
  Git标签`paper1-p1-preserved-20261102`、本地两个ZIP及逐文件manifest。
  已复制58,707个文件、4,507,658,156字节，归档内及原件SHA256逐项核对通过。
  保存范围data/runs/weights/outputs；外部原始数据集/环境不包含，副本仍在同盘。
- `paper/README.md`提供正文1–6章入口；引言及方法改用100轮主线，
  `main_results.md`整合正式两seed、完整精度网格和独立30轮消融；
  新增`discussion.md`、`conclusion.md`。旧正文/表格保留历史链接，原件已存快照。
- `paper/paper1_writeup_plan.md`映射结构、主张、证据及缺口；
  `paper/research_tracks.md`区分已有主线、旧改进、评估工具和候选。
- MCP复核Zotero项目集合33条，已有S³OD/Humble Teacher等均已归档，无重复导入；
  题名检索排除同名salient-object S3OD与语言模型蒸馏。文献版本/阅读边界见
  `paper/literature_map.md`。本轮不宣称穷尽检索或人类全文已读。
- `paper/candidates/class_ambiguity/README.md`规定教师全类别信号诊断、
  不使用隐藏GT拟合、分类单变量、监督强度控制和至少30完整配对epoch；
  仍为PLANNED，没有提取器/损失/runner，没有训练命令或新性能数字。
- 尺度操作点报告补充“继承完整缓存匹配、筛选后未重匹配”的代理统计限制；
  数值和原停止决定保持不变。

另发现旧`--audit`会改写历史回执时间戳，导致下游哈希链失配；保存本轮开始时及
新复核回执，按首次退出记录与原序列化恢复出下游既有SHA256完全一致的首次回执。
新增`scripts/audit_paper1_frozen.py`将后续复核回执写到独立目录，不改原计算源码。
详细恢复和验证范围见`paper/preservation.md`，不修改预期哈希或实验数字。

本轮新代码仅为保存/审计工具。当前可继续完成初稿，投稿证据仍按目标主张补充。
原Gate3/4不追认通过，不自动训练。
收尾208项测试通过，原表137来源/8组recall及全部文档表检查通过，
新增主表12行数字核对通过；AP/PR、高精度及尺度摘要分别完成重放。
旧审计时间戳失败和精确恢复记录保留；ARS缩写检查仍有提示，
详见`paper/organization_validation.md`，不据此声称排版或投稿已验收。

### 3.64 类别歧义诊断通过与P1单变量叠加实现（2026-11-02）

按用户“在已有成果上扩大，假设之后要继续做”的要求，执行ARS实验run/validate，
把上节候选从设计推进到零训练诊断、全量侧文件、独立损失和可执行确认入口。

- 冻结512原图开放标注tile，256设计/256确认；另采原16完整batch的128未标注tile。
  不读隐藏GT，不用val拟合。教师已见开放区域，因此不是教师未见域的泛化证据。
- 确认3337可定位无多类重叠正框/191错类，r≥.25组149框/69错类，
  r<.1组2991框/71错类。控制top1/面积/类别后，AUROC 0.7916→0.8214，
  log-loss约降低6.8%；原图bootstrap改善区间[.0089309,.0159789]。
  七项预先门槛通过，只支持准备学生确认。详见候选`diagnostic_results.md`。
- 原6471tile重新提取完整类别分数，91336候选全部身份对齐，框/最高分差均0；
  4839伪图/35067正框全部覆盖，侧文件独立保存，全量额外推理97.953秒。
- 新`training/class_ambiguity.py`只把伪正anchor分类目标从`m e_y`改为
  `m*s/sum(s)`，保留原c、尺度筛选、正框/标签、TAL、ignore、box/DFL和人工标签。
  新旧目标质量守恒不等于损失/梯度守恒；每epoch首batch保存机制快照。
- `scripts/run_p1_class_ambiguity.py --train`由用户执行，顺序训练**全新原P1对照30轮
  ＋候选30轮**，再自动548图val960、AP_small/完整PR/P70/P90与配对及数值审计。
  两臂原配置相同，仅候选分类目标变化；不复用历史100轮或覆盖旧权重。
- 固定继续线：P90 small≥+.30pp，且AP50–95≥−.10pp、AP_small≥0、
  P70 small≥0、conf.25 small≥−.20pp。不可达/非有限/未满30轮不能通过。
  只作为是否值得100轮投入的决定，不自动扩训，不改原Gate3/4。
- 诊断首次排序/序列化工程失败记录保留。集成预检发现OMP线程数变化引起
  最大约1.4e−16浮点尾差；冻结诊断源码/数字不改，预检对float允许1e−12，
  事件、计数和门槛仍精确一致。原始独立`--audit`在原环境下精确重放通过。
- 原P1源码/配置未修改；工程和全套验证见候选`validation.md`。
  学生训练尚未启动，不能声称候选已提高检测精度；相关文献已在Zotero项目集合。

### 3.65 类别分数叠加完整30轮验收（2026-11-03）

用户启动的reference/candidate均连续完成1–30轮并exit0；2026-11-03 01:05:20
自动评估与内部重放完成。本轮另行`run_p1_class_ambiguity.py --audit`退出0，
实际args、22898冻结来源、30条首batch机制快照、548图AP/PR数组与权重绑定通过。
原包装47852、候选15496已停止，没有启动额外训练或推理。

候选−同轮数原P1：Ultralytics AP50 −0.3422pp、AP50–95 −0.2631pp、
COCO-style AP_small −0.4162pp；IoU.50/P90 small +0.8203pp（净+218），
P70 −0.5418pp（净−144），conf.25 +0.1994pp（净+53，但FP+489）。
IoU.75/P90 −0.2446pp（净−65），五档精度约束均下降；AP_small类别3升7降，
bus −4.2528pp。学生时间7340.13/8139.00秒，候选多10.8836%；教师全量导出97.953秒另报。

五项固定条件仅P90与conf.25护栏通过，其他三项未通过，**不扩大当前变体到100轮**。
保留P90局部收益作为机制取舍证据，不认定全面升级/显著性/首创；原P1正式100轮
成果与原Gate3/4不变。分类质量守恒但BCE/梯度改变，信息与正则化作用尚不能分离。
完整表格、双IoU全部曲线、逐类结果、成本、来源及限制见
`paper/candidates/class_ambiguity/results30.md`；当前入口改为只读`--audit`。
本轮仅文档及已有缓存图表整理，原训练/评估源码与输入不变，收尾后删除临时心跳p1。

### 3.66 原P1互补方向：竞争负项诊断与三臂30轮准备（2026-11-03）

先核实Git干净、旧类别分数两臂结束、无训练进程和本机临时p1监控配置。
使用项目ARS experiment-agent validate/plan、paper-search与Zotero，先保存诊断协议，
再只读重放512开放tile、全35067伪正框分数及六学生548图缓存；不读隐藏GT。

- 30,565项路径哈希运行前后校验；新30轮/正式100轮六缓存的双IoU PR与固定操作点
  精确一致，独立`--audit`再次重算通过，未改写原始证据。
- 固定r>=.25，歧义错类中真实top2分别71/75、62/69；信息条件通过。
  但全部可分析动作中错误top2为81/152、87/149，另有不可定位/重叠框，误报风险明确。
  全量只触发1748/35067框（1422 small），977伪图；不把teacher信息当学生收益。
- 新30轮IoU.50/P90错类FP 969→1014；conf.25 FP+489，其中错类+214。
  small FN以低于阈值的同类可匹配框为主；.75仍有大批只有.50框的目标。
  六模型触顶尾分数均低于P70/P90/conf.25阈值，不支持将这些点直接归因于500框上限；
  不排除NMS/.001或更早过滤。全部负向分解保留。
- 隔离`training/competitive_negative.py`，保留原硬正目标，只将歧义top2负BCE减半；
  控制臂匹配相同logits下移除的负梯度L1，不声称匹配参数梯度/训练轨迹。
  三臂全新原P1/semantic/strength各30轮，约6–9小时学生粗估；本次只预检/CPU测试，
  实际全量CUDA训练链路未执行。原P1、旧候选、配置和冻结合同不改。
- 固定继续线保留AP/AP_small/P70/P90/conf.25，并增加严格IoU不退化、FP<=102%、
  semantic相对强度控制P90>=+.15pp与AP_small不退化。失败停止该动作，不在val追阈值；
  全过只建议新seed三臂30轮，不自动100轮。
- 最近邻S³OD/Humble/Soft/Credible/LabelMatch及Soft Sampling均在BMQTT9Y7且有PDF，
  复用6原条目，保存机器局部阅读笔记；不代填human-read，不认定首创。
  Semantic Scholar429和正式版/公式访问范围明确披露。
- 最多三方向及论文补证据成本已评审；若暂不追性能，优先新10%区域split配对，
  但不称独立图像/场景盲测。B0正式参考与同协议近邻/5%20%仍缺。

交付入口`paper/candidates/competitive_negative/README.md`；当前命令为
`python scripts/diagnose_competitive_negative.py --audit`与
`python scripts/run_p1_competitive_negative.py --check-only`；真实训练命令仅在独立学生合同中交用户。
验证与Git收尾见该目录validation.md；旧类别分数不扩训、原Gate3/4和正式数字不变。

### 3.67 用户明确“基线、P1、现有改进”成果主对照（2026-11-03）

用户要求直观看到相对基线的真正成果大小。实际核实Git干净、旧三臂尚无运行/进程后，
保留3.66的机制合同和源码，在独立新协议/入口中加入全新B2的30轮配对。

- 当前默认主三臂：B2全局阈值基线→原P1→P1＋竞争负项，各完整30epoch。
  不把强度控制当基线；可选`--with-control`增加第四组，另列机制判断。
- 主表/JSON同时报告绝对AP50/AP50–95/AP_small、固定/等精度召回、严格IoU、FP/成本，
  以及P1−B2、改进−P1、改进−B2三项直接差值；不拼旧30/100轮数字。
- 新B2沿用冻结27667伪框/4624伪图、11095训练tile，P1/改进11310；
  与P1相同教师SHA、split0/初始化/seed0及train args，实际args/数据绑定另审计。
- 主三臂无机制控制时标记NOT_TESTED，不能声称语义独立因果或扩训通过；
  原性能护栏保留，失败不按val调参。全部性能线通过只建议另行准备控制。
- 新入口`python scripts/run_p1_main_comparison.py --check-only`和手动`--train`，
  输出`outputs/p1_competitive_negative_main30_s0/`，独立run前缀p1_maincmp_。
  主三臂学生粗估5–8小时，带控制约7–11小时，实际待计时。

本轮仅准备/预检/工程测试，不启动训练或真实图像推理。新合同及验证记录见
`paper/candidates/competitive_negative/main_comparison_protocol.md`和`main_comparison_validation.md`。
原P1、旧类别分数负结果、原竞争负项机制合同和历史证据均保留。

### 3.68 主三臂已启动与临时完成监控（2026-11-03 10:06）

用户已执行新入口并要求核对、建立监控和完成后比较。包装进程8452（10:04:16）、
baseline子进程46736（10:04:40，父8452）的CIM完整命令、创建时间和进程链均已核对，
确认为`run_p1_main_comparison.py --train`及`--worker baseline`，没有with-control。

- 冻结commit为b75e58c；dispatch/preflight绑定和39,828项来源SHA256当前全部通过。
  三组名称为p1_maincmp_b2_30_s0 / p1_maincmp_p1_30_s0 / p1_maincmp_improved30_s0。
- 当前B2实际args逐项符合新配置：30epoch、640、batch8、SGD、seed0、原B2数据、
  resume=false等；P1/改进尚未轮到，不能声称三臂实际训练均已验证。
- 验证期间日志46460→54713字节，首轮批次持续增长，无提前性能结论。
  回执`outputs/p1_competitive_negative_main30_s0/monitor_initial_check_20261103.json`。
  回执保存后控制台输出Unicode进度条遇GBK编码错误，后续UTF-8显示复核通过，
  是核对脚本显示问题，不是训练/哈希失败；原训练完全未改。
- 已创建并读回ACTIVE临时心跳p1，每15分钟检查本对话；正常推进静默，
  只在完成、失败或需用户操作时通知。源码/合同/输入不修改，不停止、重跑、恢复或扩训。
- 三臂各满30轮、exit0及自动评估结束后，只读`run_p1_main_comparison.py --audit`，
  比较同次新30轮的绝对AP/AP_small/PR/P70/P90/严格IoU/FP/类别/成本，
  列P1−B2、改进−P1、改进−B2及固定护栏；无strength时机制仍NOT_TESTED。
  更新独立报告和第10节，中文提交push，终态处理后删除p1；失败保留证据，不自动重试。

本轮仅核对/监控与文档记录，没有新训练或推理启动；主计划文件已确认不在该运行
冻结输入映射中，本次文档修改不会使后续worker的输入校验失效。

### 3.69 主三臂candidate首批工程失败终态（2026-11-03 14:08）

心跳发现包装stage=failed；原B2／reference均连续1–30轮且exit0，CSV累计时间
6697.56／7078.94秒。candidate在第1轮0/1414批的损失前向抛出
`ValueError: original hard pseudo-positive TAL target required`，exit1，无CSV／权重。
断言同时检查教师top1一致性、正目标质量及单类非零性，原日志不能确定具体触发分支。
这是工程失败，不是完整30轮的性能负结果；此前工程测试不能替代真实CUDA验证。

终态只读复核39,828输入，缺失／哈希不一致均0；无本入口Python进程。
17份原日志／状态／参数／CSV独立快照哈希一致，四份已完成权重哈希另存于
`outputs/p1_competitive_negative_main30_s0/terminal_failure_20261103_140708_a233c4/failure_receipt.json`。
自动统一评估、训练审计和main_comparison／decision均未生成，未运行成功终态audit，
未追加推理；七项性能护栏NOT_EVALUATED，无strength机制NOT_TESTED。
报告见`paper/candidates/competitive_negative/main_comparison_failure_20261103.md`。
源码／冻结合同／输入完全不改，不自动修复、重训、恢复或扩100轮；
后续若排障须独立验证和新冻结输出，已完成B2／P1保留。中文提交push后删除临时p1。

### 3.70 用户要求定位：竞争负项AMP下溢断言（2026-11-03）

先写独立debug_protocol，再在新ignored目录重建真实增强首批，禁止训练循环／backward／
optimizer.step，只做一次CUDA AMP前向及同预测原P1损失检查。
1988前景中649伪前景，教师top1与TAL分配原类别全部一致；其中1个FP32质量
8.708838983295664e-9转FP16成为0，原始TAL零质量为0个。
该对象两端类别均9、top2/top1=.01424，未达动作触发线；代码先检查严格正质量而误拒绝。
同批原P1损失有限；新候选报同一断言。缓存CPU控制重放固定其余输入，只改检查目标dtype，
FP16在semantic／strength均失败、FP32均通过。历史首批无张量，不宣称逐位同一锚点。

入口`scripts/diagnose_competitive_negative_failure.py`；结果
`outputs/p1_competitive_negative_debug_20261103_v1/{diagnosis,controlled_replay}.json`及
`foreground_tensors.pt`。参数哈希未变、监测原源码／缓存／失败文件未变，无隐藏GT／val指标。
报告`paper/candidates/competitive_negative/debug_results_20261103.md`解释修复边界：
分开来源类别检查和实际BCE目标，保护下溢／零质量锚点及strength正类，不能简单删断言。
本轮只定位／验证，不修改冻结训练源码／合同，不重启／恢复／补评估。
新增定位代码及三项边界测试，全套pytest286 passed；中文commit并push。

### 3.71 AMP独立修复验证与复用B2／P1的单组执行准备（2026-11-03）

按用户明确授权修复／验证／准备命令，不启动训练。先写V2独立协议，新增
`training/competitive_negative_v2.py`及独立semantic30配置，原P1／旧candidate源码、
合同及失败输出完全保留。真实分配类别／原TAL目标负责来源校验；FP16下溢或原零质量
前景的额外动作权重为1，原正类始终受保护；其余仍用原r>=.25、top2负项减半。

- 重放1988前景／649伪前景；semantic／strength各19个active，非零目标与旧权重精确一致。
- 真实单批8图CUDA AMP：旧实现仍报错、V2正常，1下溢目标保留；同预测分配／box／DFL
  完全一致、正目标及分配正类权重变化0，训练循环／反向／优化均0、参数哈希不变。
- 新入口`run_p1_main_comparison_repair.py`仅允许candidate worker，预检核对原39828输入、
  失败证据、B2／P1连续1–30轮exit0及四权重；新输出／run存在就拒绝，无自动重试／恢复。
- 复用同次`p1_maincmp_b2_30_s0`与`p1_maincmp_p1_30_s0`；新run为
  `p1_maincmp_improved_v2_30_s0`，主输出`outputs/p1_competitive_negative_main30_v2_s0/`。
  只剩改进学生约2–3小时粗估，另加548图val960／三权重AP_small与双IoU PR及审计。
- 主表包含三绝对结果、P1−B2／改进−P1／改进−B2、全部类别正负差值／FP／成本；
  原七项护栏不变，源码commit差异／复用如实披露，无strength机制NOT_TESTED，不自动100轮。
- 全套pytest315 passed in 8.78s；调度／报告合成集成测试不启动模型或推理。
  真实验证回执在`outputs/p1_competitive_negative_v2_validation_20261103/`，
  只读预检回执独立保存到`outputs/p1_main_comparison_v2_preflights/`。

精确命令、来源和限制见`paper/candidates/competitive_negative_v2/{README,protocol,validation}.md`。
本轮没有正式训练／评估，也无V2性能结论；训练启动后才建立临时监控。
README／baseline协议已更新；中文commit并push，保留原历史负结果及Gate3／4边界。

### 3.72 用户启动V2改进组：真实核对与临时监控（2026-11-03 14:38）

用户执行`python -u scripts/run_p1_main_comparison_repair.py --train`并要求核对／监控／
完成后审计三方法。包装PID48360创建14:36:49.029936，worker35148创建14:37:07.993851，
父48360；CIM完整命令、创建时间及yolo解释器均符合预期。

- dispatch仅candidate，无strength。来源fe17ce69e7b584397493ac9adf29327374cde1f6，
  preflight SHA256 `c6a6b3b2a345dc0ef5db649b1132b9f3e70d72582e3859f4308f0b7859d86f3d`；
  39,864项绑定只读复核通过，B2／P1的best SHA与原冻结值一致，未重新训练这两组。
- 新组实际args与原P1除name／save_dir外无差异，30轮／640／batch8／SGD／AMP／seed0、
  resume=false等均通过；experiment的实现标记、配置、来源及输入哈希核对通过。
- 已成功完成首批并写机制快照：19 active、1个FP16下溢前景保持原权重，
  changed_positive_target_weights及changed_assigned_class_weights均0，目标不变。
  随后日志22628→71338字节、第1轮432/1414批，持续推进，未作提前性能判断。
- 独立回执`outputs/p1_competitive_negative_main30_v2_s0/monitor_initial_check_20261103_143811_980833.json`；
  主计划文件已确认不在本次冻结映射中，本轮仅改本文档，不影响worker／终态输入校验。
- ACTIVE临时心跳p1-v2每15分钟检查；正常静默，完成／失败／确需操作才通知。
  不修改源码／冻结输入，不停止／恢复／重训／追加手动推理／自动扩100轮。
- 完整30轮／exit0及自动评估结束后，新唯一日志保存
  `python scripts/run_p1_main_comparison_repair.py --audit`输出，比较同次统一评估的
  B2／P1／改进绝对值与三项差值、全部类别正负结果／成本、七项护栏。
  复用和源码commit差异明确披露，无strength机制NOT_TESTED；更新独立报告／第10节，
  中文commit并push，再删除p1-v2。终态失败保留证据、不自动重试，也收尾删除监控。

### 3.73 V2改进组30轮完成：三方法统一比较与停止（2026-11-03）

用户启动的AMP修复V2完成连续1–30轮、exit0，自动统一评估548图val960完成；
`python scripts/run_p1_main_comparison_repair.py --audit`独立exit0、accepted=true，
stdout／stderr保存在新唯一`terminal_audit_20261103_171004_802028.log`。
39,864冻结绑定、原失败证据、三组args／best／预测、双IoU完整PR／COCO-style AP_small
及全部曲线数组、主表／decision精确重放通过。没有追加训练／模型推理。

| 同次统一评估 | AP50 | AP50–95 | AP_small | conf.25 small | IoU.50/P70 small | IoU.50/P90 small | IoU.75/P90 small | FP@.25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2（复用） | 38.9202 | 23.4534 | 13.8172 | 34.1423 | 35.6963 | 16.7288 | 5.4671 | 7071 |
| 原P1（复用） | 39.4990 | 23.7883 | 13.6792 | 36.3435 | 36.9304 | 16.6610 | 5.6778 | 7889 |
| V2改进组 | 39.2772 | 23.6255 | 13.4513 | 36.7160 | 36.4601 | 17.0260 | 5.5763 | 8512 |

单位百分数；总体AP用Ultralytics，AP_small用同预测COCO-style，非VisDrone官方AP。
P70／P90为全尺寸micro precision约束，不是small precision或部署保证。

- **P1−B2**：AP50／AP50–95／AP_small为+0.5788／+0.3349／−0.1380pp；
  conf.25 small／P70／P90／严格P90为+2.2012／+1.2341／−0.0677／+0.2107pp，FP+818。
- **V2−P1**：对应−0.2218／−0.1628／−0.2279pp；
  +0.3725／−0.4703／+0.3650／−0.1016pp，FP+623（+7.8971%）。
- **V2−B2**：对应+0.3570／+0.1721／−0.3659pp；
  +2.5737／+0.7638／+0.2972／+0.1091pp，FP+1441。
- 七护栏只有P90≥+.30pp与conf.25 small≥−.20pp通过，其余五项失败；
  固定合同停止本候选，不调r／衰减、不追加strength／第二seed／100轮，保留原P1。
- 全10类AP_small保留，V2−P1为5升5降，bus−1.7331pp最大；
  conf.25 small匹配+99，但全尺寸TP仅+84／FP+623，不包装为纯提升。
- 原B2／P1累计训练6697.56／7078.94秒为复用历史成本；V2学生8430.70秒，
  新worker墙钟8472.594秒；三组本轮评估／缓存分析共184.046秒，独立审计另计。
- 原preflight来源b75e58c，实际experiment记录B2为b75e58c、P1为9ab6189，V2为fe17ce6；
  冻结哈希与实际args通过，不能声称三臂同commit。无strength机制NOT_TESTED，
  30个首批工程快照不是整轮语义因果证据。同split／teacher／seed0、反复开发val，
  不外推显著性／跨划分泛化，不叠加旧100轮或历史30轮增益。

详表／11项解释检查／审计边界见`paper/candidates/competitive_negative_v2/{results30,terminal_validation}.md`。
Ultralytics总体AP是冻结验证器输出及主表重放，缓存重新计算的是COCO-style与micro PR，
没有独立重算Ultralytics内部AP或重跑随机训练。原失败目录／旧合同未改。
当前无新训练任务；下一项最值得准备的是原P1／B2独立区域split配对证据，B0仍缺，
均需另行冻结和执行，不能承诺性能提升。本轮仅报告／状态文档，中文commit并push后删除p1-v2。

### 3.74 V2召回／AP分歧与原P1瓶颈缓存诊断（2026-11-03）

用户要求解释本次改进的召回上升／AP下降／FP增加并推荐一个新方向，不启动训练。
先保存`paper/diagnostics/competitive_negative_v2_bottleneck/protocol.md`，再只读当前统一
三模型548图预测与既有教师开放区域事件；val、旧design／confirmation均已看过，明确探索性。
隐藏GT不读，不拟合／选参数。59项绑定、双IoU固定点／micro PR、新增逐预测／逐GT数组
及JSON精确重放；诊断44.766秒、独立audit exit0；全套318tests通过。

- conf.25 V2−P1 small+99，预测+707、全尺寸TP+84／FP+623；同FP7889时small
  9659→9522（−137，−.5155pp），同K27237时9659→9575（−84，−.3161pp）。
  P90局部+97仍保留，但P70−125、严格P90−27，不是全前沿改善。
- IoU.50新增623 FP中错类218、背景166、重复121、同类近定位78、近异类40；
  严格IoU新增743中近定位337最大。V2减少top2抑制却保留top1目标，符合放松正确负监督
  的风险，但无strength，不能区分语义／强度因果；分类梯度也可改变共享特征和后续分配。
- 原P1 IoU.50/P90的22149个small FN中13297（60.03%）有同类低分可匹配框；
  1386 FP中969（69.91%）错类、重复仅21。GT-only删除错类使P90 small
  16.6610%→29.5970%，仅不可部署／不可加和的容量探针，不是未来方法收益。
- 严格IoU下9939个small GT仅有同类.50候选、FP近定位422/857，定位限制真实存在；
  但P1／B2共同16109个small GT的P1平均IoU+.005229、中心／边界误差更低，
  不支持原P1首先因回归恶化的概括。V2共同16471个small IoU−.001381，不据此重启旧NWD。
- 三方法双IoU全部固定点均无cap末分数≥点阈值，500数量上限不能解释这些点直接删框；
  仍不能排除NMS／.001／pre-NMS及低分AP截断。
- 教师r≥.25可定位单类框top1错类率49.34%／46.31%，非歧义5.26%／3.83%；
  但动作中top2真实正确仅46.71%／41.61%。歧义含信息不推出负项放松安全，
  开放区域教师见过，亦不能外推未标注伪框真实错误率。

最多三个方向：优先只对已匹配人工GT且非零质量的正锚点，将学生当前最难非真类负BCE
乘2，保留伪标签／正目标／ignore／TAL／回归，对照为同锚点相同新增负logit梯度L1均匀控制。
另两个方向是伪框类别监督弃权（继续减弱语义监督有风险）和定位质量排序
（尚无新的可实现GT-free信号），本轮不实现。最小下一步仅零优化步骤的来源／AMP／
梯度验证，通过后才准备原P1／候选／控制各完整30轮，粗估6–9小时学生+评估。
新方案门槛在独立计划中明列，不改原V2失败，不在val调权重／类别名单；当前没有新训练入口。

重读S³OD预印本III-D／Soft Teacher3.2–3.3；Humble Teacher PDF3–5页且视觉核验第4页、
Credible Teacher PDF2–5页／方法HTML部分阅读。原集合BMQTT9Y7条目／PDF保留，新增机器
部分阅读笔记S9WF6SIN／BPGJUBME，无重复导入、无用户全文已读或首创新颖性主张；
LabelMatch仍未读全文。完整结果／11项解释检查见
`paper/diagnostics/competitive_negative_v2_bottleneck/results.md`，候选只计划见
`paper/candidates/reliable_competitor_negative/README.md`。产物`outputs/p1_v2_bottleneck_20261103/`，
复核`python scripts/diagnose_p1_v2_bottleneck.py --audit`。不训练、不新推理、不建监控，
原P1和全部失败保留。若暂不试方法，独立区域split的原P1／B2配对及B0仍是补证据优先项。
中文commit并push。

### 3.75 人工GT正锚点竞争负监督：工程实现与新三臂准备（2026-11-03）

用户授权完成实现／工程验证／实验准备，明确本次不训练。起点核对main干净、HEAD bd1435d。
使用项目ARS experiment-agent validate；先写工程合同，再保存新源码、配置、测试和报告。
原P1继续为主方法，V2完整30轮2／7失败和全部历史证据保留，不追改第3.73／3.74结论。

新候选仅增强人工GT有效正锚点最难非真类负BCE为2倍；detach学生概率，固定ID解并列。
控制同锚点所有非真类乘1+p_k/sum(p_negative)，仅局部负logit梯度L1匹配。
沿原P1标量token路径显式运输来源／真实类别／对象ID，通过mosaic／裁剪后仍绑定。
置信度为1的伪框仍为伪来源；损坏来源先拒绝，零有效质量／FP16目标下溢不动作。
不改P1 TAL／分母／ignore／正类权重／伪框监督／box／DFL，不加模块／教师推理／重采样。

合法旧前景缓存1988锚点重放通过；为检查新运输和完整loss，只额外取一个真实8图batch、
640／CUDA FP16 AMP前向一次，未反传／未创建优化器／训练循环0／val推理0，参数SHA相同。
新参考完整loss和三臂TAL／box／DFL逐位一致；1341个人工正锚点动作，497个伪正锚点保护，
26个人工零有效质量不触发；候选／控制逐锚点新增负logit L1最大绝对误差8.67683e-8。
来源负向审查加强ignore解码前及TAL索引整数转换前拒绝，之后仅重放同一缓存，无第二真实batch。
mosaic开启／close路径与原P1同种子图像／框／类别／ignore逐位一致；合成实际autograd覆盖AMP。
训练模式临时模型BN buffer可更新，参数不变不等于所有buffer不变；无学生checkpoint保存。

默认新配对原P1／候选／等强度控制各完整30轮，同split0／教师／COCO初始化／seed0，
640／batch8／SGD／AMP／548图val960，不重训B2。预计学生6–9小时，实际新增开销未测。
训练前冻结新8项性能／2项机制投入门槛，任一失败即停，不追val调系数／名单／筛选，
不自动重试／恢复／扩大100轮；已有输出拒绝覆盖，完成后统一评估／权重预测绑定／只读audit。
本次只做预检和工程，未启动三臂或创建空监控；实际启动时再建立临时监控。
**工程通过，学生性能NOT_TESTED，机制NOT_TESTED**；同已看开发val不称独立确认／首创。
新增文献0，不重复导入Zotero。源码、工程证据／未解决风险／命令见
`paper/candidates/reliable_competitor_negative/{README,engineering_protocol,validation,student_protocol}.md`。
最终完整pytest、预检和中文提交push状态见第10节与validation.md验收记录。

### 3.76 人工GT可信竞争负监督新三臂完成：10项门槛5过，停止候选（2026-11-04）

用户11月3日18:18:53手动启动父进程54292，reference49560、candidate23324、control8744
顺序独立训练，三臂均完整连续1–30轮、exit0、未中断／恢复。启动及切换已核对PID／创建时间／
父子／claim及完整实际args和experiment；源码均91cc57d，39906冻结来源绑定，新参考不复用旧学生。
原P1／旧V2源码合同与失败证据保留。本轮不是新增方法或调整2倍系数，仅终态验收。

自动统一548图val960／COCO-style AP_small／双IoU完整micro PR及wrong-class固定FP分解完成，
11月4日02:28:15封印终态outcome=stop。原进程停止后02:35启动独立只读audit，exit0／accepted=true，
39906输入／57产物／args／best/last／预测／GT／数组／全部正负结果与10项决定精确核对。
日志在封印目录之外`outputs/rcn30_terminal_audit_20261104_0235.log`，原目录未添加文件或修改回执；
新外部`outputs/rcn30_terminal_receipt_20261104.json`绑定日志及manifest/decision哈希。
Ultralytics总体AP仅绑定原validator输出，缓存独立重算范围是COCO/PR，未重跑训练／推理。

| 方法 | AP50–95 | AP_small | conf.25 small | IoU.50/P70 small | P90 small | IoU.75/P90 small | FP | 错类FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 新P1参考 | 23.7883 | 13.6792 | 36.3435 | 36.9304 | 16.6610 | 5.6778 | 7889 | 3719 |
| 候选 | 23.4381 | 13.5705 | 33.1753 | 36.3472 | 17.8500 | 5.7719 | 6111 | 2767 |
| 等强度控制 | 23.6415 | 13.3642 | 34.1574 | 36.6859 | 17.5227 | 5.7606 | 6569 | 2842 |

绝对百分数；总体AP为Ultralytics，AP_small为COCO-style非VisDrone官方AP；P70/P90均全尺寸micro约束。
候选−P1：AP−0.3502pp／AP_small−0.1088pp／conf.25 small−3.1682pp（−842匹配）／P70−0.5832pp，
四条失败；P90+1.1890pp（+316）／严格P90+0.0941pp（+25）／FP−1778（−22.5377%）／错类FP−952
（−25.5983%）四条通过。性能4/8，误报减少同时TP少1172，不能称无代价提升。
候选−控制AP−0.2034pp失败，AP_small+0.2063pp通过；机制1/2，合计5/10，NOT_ESTABLISHED。
固定conf.25候选比控制small少261／FP少458；P90比控制small多87；全部正负结果保留。

学生累计9748.08／9227.06／9718.43秒，总28693.57秒（约7.97小时）；三权重评估/缓存分析202.234秒，
包装预检/审计另计。负载／启动顺序未控制，不将更短时间归因纯方法效率。
候选对P1逐类AP_small为2升8降、对控制4升6降；全10类／所有FP类型／两IoU全部PR点见独立报告。
工程通过；学生性能已验证但未达投入条件；语义机制未建立。按合同停止，不按当前val调系数／名单／
筛选／阈值，不自动重试／恢复／第二seed／100轮；同split／teacher／seed0开发val不代表独立确认／显著性。
新增论文0／Zotero导入0；困难负监督非首创。只新增`paper/candidates/reliable_competitor_negative/`
`results30.md`和`terminal_validation.md`，冻结已有候选文件原样；更新第10节／主README／baseline，
中文commit并push仅本轮报告，终态处置后删除临时监控rcn-30。未改源码，不重复运行已通过的806tests。

### 3.77 人工同对象正目标几何重排：研究、零训练诊断与最小实现（2026-11-04）

用户明确要求提高方法收益/可辩护贡献，本轮不启动学生训练。起点main干净、HEAD b1db01e，
保护原P1和全部正负证据。使用项目ARS deep-research/experiment-agent流程，
先保存独立诊断协议、来源补充协议和工程合同；核验S³OD差异及TOOD/VFL/RS原文/作者代码，
另读2026 FSS正式HTML方法/实验相关段，未找到可确认作者代码，不声称其实现已核验。
原生子代理做来源核验与反方检查。最多比较三个方向：同对象正目标几何排序、可靠区域
跨锚点正负排名、教师多视图可靠性；后两者缺空间来源或可靠新增信息，当前不实现。

首选只在人工来源同对象既有有效正锚点中，将原真类BCE目标值按detach普通IoU重排；
不改变TAL/正集合/分母/box/DFL/负类/伪框，没有新权重、阈值、名单、骨干或分支。
原TAL已质量对齐；特定区别是固定分配与原目标值集合的人工同对象排列，尚非创新成立。
原32/后16批训练后缓存small逆序对象73.4129%/64.4564%、大差逆序22.8162%/21.3570%，
目标搬移6.3368%/6.6057%；混合来源不能用confidence=1识别人标，旧insufficient结论保留。

随后仅复用旧8图工程缓存像素做一次训练后原P1 AMP前向，参数及buffer SHA不变；
无新DataLoader batch/optimizer/学生训练循环/val推理/隐藏GT读取。显式人工来源补齐后，
再次只读完整head张量按原P1 CUDA AMP/TAL/BCE dtype路径重放：人工small有效多锚点126，
100个逆序、32个>.05差逆序、质量搬移8.7506%。FP32代理为33个，二者不混写。
反证是框外small目标质量35.3208→38.6970，可能增加分类/回归不一致；加权IoU改善
只是排序构造性质，不能称学生收益，也不能由代理统计确定唯一训练根因。

独立新loss/src/config/runner不修改旧P1或旧失败合同，来源运输复用冻结实现。
初稿散列排列控制搬移L1过大（251.0567 vs111.5723），在学生运行前修订为非几何方向
凸插值，匹配同对象总目标质量/支持/搬移L1；旧工程回执另存unmatched_control。
最终候选/控制总L1为111.572260618/111.572259753；逐对象L1误差≤1.78814e-7，
实际FP16附加head梯度L1误差≤9.53674e-7。控制不保值集合、changed866/1167，
不称最终绝对/参数梯度完全匹配。参考完整loss/scoregrad和TAL/分母/boxDFL精确一致，
伪来源/负类/未触发目标与梯度保持；全部935tests通过，不把工程通过当机制成立。

准备全新参考/候选/控制各完整30轮、原split0/teacher/COCO/学生seed0、640/batch8/SGD/AMP，
自动统一548图val960/COCO/双IoU PR/全FP/错类、权重预测绑定及只读终态audit；
11项固定条件要求AP≥+.10pp/AP_small≥+.10pp/P90≥+.30pp，同时conf.25/P70/严格P90不退、
全尺寸FP/错类不增，并胜过匹配扰动控制。不可达/非有限失败，任一失败即停止，
不重训B2/重试/恢复/100轮；旧实际三臂约8小时，新排序开销未测，预留6–12小时非保证。
本轮仅check-only/审计，训练目录不存在、无空监控。文献部分机器阅读与Zotero处理
见literature_review：TOOD原条目、RS只增集合、VFL新增AM4MTCB9/FSS新增TZD6SP4I，
均附PDF但不据此称已读；FSS导入卷期页过时已按出版社纠正，不作人类全文已读标记。
准确手动命令及所有限制见`paper/candidates/positive_quality_transport/student_protocol.md`。
原P1正式100轮与Gate3/4保持原判定；学生性能NOT_TESTED、机制NOT_ESTABLISHED。
最终`--check-only`真实exit0/accepted=true，40,242项绑定；11310训练图和548图val人口一致，
training/forward/created_files均false，四个新训练输出/run目录不存在。
完整pytest最终935 passed in17.46s，git diff --check通过；仅本轮23个文件中文提交并push。

### 3.78 正目标几何重排V2三臂负结果与方法投入复盘（2026-11-05）

V2是5966f58的Trainer集成修复，不是新的科学变量。全新原P1参考/候选/匹配扰动控制
各连续1–30轮、exit0，同10%面积split0/固定教师/COCO初始化/学生seed0/640/batch8/SGD/AMP；
统一548图full-image val960完成。原V1仍是0epoch初始化失败，两套封印/合同/源码独立保留。
本聊天终态阶段父与全部worker停止后只读--audit一次exit0，无新推理，随后删除临时v2。
本轮不重复终态audit、预检或学生循环。

| 同次30轮，单位%（FP为个） | P1参考 | 候选 | 控制 |
|---|---:|---:|---:|
| AP50–95 | 23.7883 | 23.4738 | 23.6304 |
| COCO-style AP_small | 13.6792 | 13.3340 | 13.6527 |
| conf.25 small recall | 36.3435 | 35.3990 | 35.5872 |
| IoU.50/P70 small recall | 36.9304 | 36.1628 | 36.3209 |
| IoU.50/P90 small recall | 16.6610 | 17.2028 | 17.2329 |
| IoU.75/P90 small recall | 5.6778 | 5.9450 | 5.4483 |
| conf.25全尺寸FP/错类FP | 7889/3719 | 7573/3434 | 7758/3480 |

候选−P1 AP/AP_small/conf.25/P70/P90/严格P90为−.3145/−.3453/−.9444/−.7676/+.5418/+.2671pp；
候选−控制AP−.1566pp/AP_small−.3187pp/P90−.0301pp。11线4过、stop，机制NOT_ESTABLISHED；
工程accepted不等于性能条件通过。P70/P90约束全尺寸micro precision，非small precision；
不与正式100轮/B2历史拼接，不当VisDrone官方AP或跨split/seed显著性。

用户要求ARS深入分析且不训练。先存独立协议，再用封印后NMS缓存做双IoU匹配/同FP与K/
GT增失/FP分解/共同几何/全部类别敏感性；64项绑定及新诊断精确重放通过，完整pytest通过。

- 候选同FP7889时small9659→9499（−160），同K27237时→9556（−103）；严格IoU对应−51/−47。
- conf.25候选small新增774、丢失1025；P90新增650、丢失506，局部正结果保留。
- 错类FP−285但重复+270；不是只有分数统一降低或只因一个类别。10类全部保留，
  逐一去掉任一类，AP_small差值对P1仍[−.4266,−.0941]pp、对控制仍[−.4395,−.0645]pp。
- 同对象目标总量守恒不保证参数梯度/全局排序/覆盖守恒；控制不保熵/目标集合/改变锚点数/独立轨迹累计强度。
  即时几何与联合可学习性、分类回归支持协调、非特异优化均是替代解释，非唯一根因。
- 只保留可靠完整区域内跨对象排序的限时可行性问题；当前SourceDataset只传对象来源，
  可见比例<.35丢框/增强过滤使“未匹配=背景”不成立。来源无法证明就停止，不用隐藏GT补齐。
  多尺度观测/独立教师仅备选；MixTeacher/PseCo/密度裁片与RS/S³OD限制新颖性，不给新loss或训练命令。
- 三臂学生累计14.62小时/worker14.67小时，预检评估另计；超过旧6–12小时预估，
  负载/时段未控制，不能当纯模块开销。未来三臂至少90学生轮，须先完成可识别控制和预算验证。

详见`paper/diagnostics/positive_quality_negative_review/{protocol,results}.md`、
`outputs/positive_quality_negative_review_20261105/{results,receipt}.json`。
新入口`python scripts/analyze_positive_quality_negative.py --audit`只读核验本轮诊断，不重跑旧终态audit；
新训练/推理/真实batch/隐藏训练GT读取均0。全部11类解释谬误与局限逐项披露。
中文commit并push；原P1主线及失败证据保护，无新监控或自动实验。

### 3.79 完整人工区域内跨对象排序来源审查：未建立，停止A（2026-11-05）

承接56e242f，用户要求继续ARS可行性审查，不能证明就停止，不用隐藏GT补齐，不训练。
先保存`paper/diagnostics/reliable_region_ranking_feasibility/protocol.md`，再白名单核对当前P1
合法人工标签、清单、manifest/summary投影字段和已绑定源码；不读原始全图/未标注tile标签、
val/预测/真实batch/权重，manifest/summary既有隐藏计数不提取或用于判断。

- G1通过：6471个人工tile/标签文件、105058个保留框、实际P1 11310个唯一输入身份一致；
  area_ratio=.10/seed0/可见门槛.35/抖动.20，实际tile面积比.0993113426–.1002777778。
- G2未建立：裁片会过滤有正面积交集但可见比例<.35的框，导出只留幸存标签，
  没有保存被过滤框交集足迹/穷尽标注证书；保留人工对象可信不等于未匹配空间是背景。
- 纯合成反例：1000²来源、316² tile，全来源框可见比.099856而被过滤，却覆盖整个tile；
  两组来源框数同为2/类别数同/导出同为1，也不能确定背景位置。每个实际tile几何均容许
  全来源框被过滤的反例，固定边缘侵蚀不能补证；不是实际漏标观测或完整选区seed重放。
- 独立反方提出几何安全内部区的充分条件dx·dy/(W·H)≥阈值，但本次其最大值仅约.02507；
  保留条件性反驳，不泛化成所有未来子区都不可能。未匹配TAL锚点也可能位于已知前景内。
- G3未建立：SourceDataset只传对象token，人工ignore初始为空，增强会继续筛框；
  库已有semantic_mask接口不等于当前P1已建立完整性证明，MixUp语义掩码规则也不能直接当可靠性掩码交集。
- 决定STOP_BEFORE_RANKING_DIAGNOSTICS：不统计排序错误、不做梯度诊断/新loss/模型forward/训练，
  不切换新教师/其他候选。停止是来源可识别性判断，不是新学生性能失败；实际可靠负例不存在未被证明。

新`outputs/reliable_region_ranking_feasibility_20261105/{results,receipt}.json`保存6484项绑定和决定；
新入口`python scripts/audit_reliable_region_ranking_feasibility.py --audit`精确重放通过，
首次16.469秒/审查重放17.593秒；这不是重跑旧终态audit。5项新增访问/合成反例测试，
完整pytest为1093 passed in21.75s；11类解释检查、独立反方及全部边界见独立results.md。
原P1/全部失败封印/旧源码合同不变；README/baseline仅更新当前状态和只读入口，
第10节追加本次停止记录，中文commit并push；无新训练监控或论文/Zotero改动。

### 3.80 合法人工实例跨尺度观测B：输入事实与近邻差异审查（2026-11-05）

承接708b45c，用户要求继续ARS审查与旧失败/现有文献的实质区别，没有新增信息就停止；
本轮不训练、不用隐藏GT，保护原P1及失败证据。先保存独立协议，再做源码/原文核对、
纯几何来源盘点和独立反方审查；没有把A的背景证明失败直接推广到positive-only实例配对。

旧同教师跨尺度支持/正伪框定位稳定性/上下文纠类均检验预测可靠性或修标签，
人工GT身份配对则可固定类别、对应可见部分与变换关系，目标是训练约束；
旧停止没有反证此命题，也不是新命题的正证据。现有P1增强有resize/Mosaic/RandomPerspective，
但没有显式同实例双视图耦合；确定性视图可以改变归纳偏置，不能据此一票否决。

新事实：只投影人工manifest尺寸/坐标/身份，SHA与前轮合法来源回执一致。
6471个人工tile均为正方形，原生长边131–548，S>640为0；640/S为1.16788321–4.88549618。
因此原生ROI不能主张补回这些人工tile在首次640载入时下采样丢失的细节。
全图960相对tile640同对象名义线性比例.3554279749–.411，确可增加上下文而非更高目标分辨率。
这些是输入前名义几何，不是像素频率、增强后实际对象尺度/有效实例或性能统计；
后续增强仍可缩小，ROI放大/更多网格曝光仍可能有用，不做普遍无效推断。

直接原文核对：SoCo已有同proposal三视图对象尺度/位置一致性，PseCo已有shifted FPN与
同伪框监督两视图及额外视图控制，MixTeacher明确也扩展有标注部分到多尺度；
密度裁片已有原图/zoom裁片SSOD，MSAD已有高多分辨率教师—低分辨率学生对齐蒸馏。
人工来源/区域部分标注作用域有差别，但没有建立区别这些近邻的新增信号—动作—反证解释。
全图上下文与配对约束的真实增量保留，停止的是泛化B的新候选升级，不是声称没有任何新输入。
合法局部框不能凭身份补完整全图GT；直接沿用现P1全图loss还会对未知位置产生背景项。

决定REVIEW_COMPLETE / STOP_GENERIC_B_NO_DISTINCT_SIGNAL：不写新loss，不做尺度/系数重试，
不启新教师、forward、val或学生，不自动转向其他方法。新几何11项输入绑定，精确重放字节一致；
SHA b8bc076aa73f6a1d19a932e8a908f6bcace7d4bebbed92fe0e52d1dfc38e45cd。
标签文件打开/像素解码/隐藏计数值提取/模型forward/训练/val推理均0。
正式仓库只变更文档与入口；ignored分析脚本/JSON置于独立新目录，不冒充公开发布或完整学生验证。
SoCo去重归档为Q6Z5S2AM，复用MSAD Q5P6YRER并加项目集合；部分机器阅读范围/笔记与检索索引限制在报告披露。
原P1/历史正负结果/全部封印/冻结源码配置合同不变，第10节追加本次记录，中文commit并push。

详见`paper/diagnostics/human_multiscale_observation_review/{protocol,results}.md`与
`outputs/human_multiscale_observation_review_20261105/{geometry.py,geometry.json,geometry_receipt.json}`。

### 3.81 原P1之后的机制方向：公开保留事件约束的有界定位分布监督（2026-11-05）

承接aa1d621，按用户要求继续academic-research-suite定向研究、原文核验与独立反方。
本轮只保留一项研究假设，状态REVIEW_COMPLETE / CONDITIONAL_ZERO_TRAINING_AUDIT_ONLY；
没有实现候选loss/Trainer，没有训练放行、性能结论或已确立原创性。

新约束来自合法正框的已保留事件：源码条件A_visible/A_input+1e-12≥.35，
故输入标注框面积≤A_visible/(.35−1e-12)。它不恢复具体隐藏框、不提供背景保证；
本轮不访问隐藏GT数值，但必须披露保留事件包含模拟生成时依赖输入GT面积的侧信息。
不能默认自然人工标注具有同一保证；所有强对照均须获得同样规则及合法几何界。
初始贴边只保留多种相容解释，不能认作真实cut；后续视觉裁剪也不能主动丢弃已知精确坐标。

CABB（CVPR2021）已定义裁剪相容完整框集合及点框回归，GFL（NeurIPS2020）已有分布定位/DFL。
本问题的可测试差异是公开保留事件给出的有限、宽高耦合观测集合及其概率质量监督；
同界CABB点/众数适配、MAP及最紧单边界均是必须公平控制的近邻解释。
待检验混合公式保留精确边原DFL，只对未知边在固定精确连续坐标的集合求质量；
全已知时显式严格恢复原DFL，不将单点/两bin集合概率冒充插值交叉熵。

固定合成例：两个平均框同为6×6且可行，集合质量仍可为1与.75；
反方补充控制唯一众数也同为6×6后，集合质量仍可为1与11/16。
均值/众数坐标约束、集合内MAP与联合质量不普遍等价，且面积事件不能拆成独立边系数缩放。
但因子化分布仍不表达任意边相关性，可能只让尺寸收缩/分布变尖；Z=1不识别真实框。
这是机制区别，不是裁片造成旧负结果、提高AP、校准或新颖性证明。

先行协议后只重读6471合法人工标签，SHA与A旧结果/回执一致；105058框中4145严格文本触边，
4053仅单边、90双边、2三边；横纵均涉及边界88个。非真实截断率、最终有效数或tiny收益；
不加epsilon，不按这些数调门槛，不制造增强歧义。来源区间/事件运输、DFL有限支撑/负距离、
TAL/IoU/DFL同观测消费者尚未工程认证，未过则停止，不补隐藏GT或加模块救活。

第一轮6483项输入绑定，结果SHA d25358cbb2723f28bc7d0238e2a35eac5f32becb46ef72ee96f65b6b1f9f4d06；
反方补充单独执行前声明、4项绑定，SHA cf8a1611f024aa80158c07c0a7db86b95d3eb18900ea3ff727f0c38d65f0a5c1。
两份新结果精确字节重放；文档链接/diff/提交白名单检查。正式仓库只改文档，未重复完整pytest；
新ignored分析脚本/JSON未公开发布。不读取新manifest/隐藏标签或计数，不解码像素/加载权重，
模型forward/backward/训练/val推理/旧终态audit/新监控均0。原P1及失败证据/合同不变。
CABB去重新增NF7VIMS2，GFL复用2WWVF6XL并补齐元数据，部分机器阅读笔记/索引实际范围见报告。

详见`paper/diagnostics/next_mechanism_review/{protocol,adversarial_supplement,results}.md`与
新ignored `outputs/next_mechanism_review_20261105/`。第10节记录、中文commit并push；无新训练命令。

### 3.82 有界保留事件监督实现与精确边支撑阻塞（2026-11-05）

从干净2dcece9开始，用户授权独立实现、零训练检查及通过后自动顺序三臂各30轮。
ARS先保存工程合同及来源/舍入、消费者、运输、覆盖补充，再实现独立模块；原P1及旧失败不改。
来源绑定6471合法标签、105058原行；1条完全相同token重复按native规则合并并保留alias，105057对象。
来源为有条件SOURCE_OUTER，不声称精确反演历史完整框；GT衍生公开保留侧信息明确披露。

实现包含二进制误差/联合物理见证、原对象/增强实例身份、累计裁剪域和区间运输、
同支撑TAL/CIoU/DFL、同界MAP控制、原P1全known数值/梯度严格恢复及正确Trainer lexical继承。
不主动制造未知边；支持外/负距/空集合fail closed，不为启动学生削弱合同。

真实检查01发现native重复行并停止；独立修复后02通过真实11310train/548val loader和首批AMP forward/backward。
02只属版本1组件证明，零optimizer。扩大的03在第二batch的U-only名义负距停止，证据保持原判。
两份独立复核后先保存原生TAL事件修订，另立v2 objective/Trainer/checker，仅恢复原native池并让U原子自行认证。
真实03反例在v2仅bin0合法，bin1受实际累计clip排除；没有把理想合成视为真实证明。

v2真实`native_event_integration_01`，PID2456，UTC04:57:56至05:01:19自然结束，202.3255秒。
第二batch object142895/occurrence51的框为[341.26489,636.80811,377.25269,640]，仅right未知。
7个native候选的精确top距离均负，全部原子为空；全部8400网格最大y636仍低于固定top0.80811像素。
因此不是U舍入或候选池过窄；按先行合同停止，不再通过改E、丢对象、clamp或新head绕过。

终态只读audit核对25输入/24快照和合法来源行，重新构造原native池与联合支撑，全网格反证及JSON字节重放一致。
当前报告仅声明必要工程门FAIL；v2显式增强batch forward0/optimizer0/val inference0，模型构造stride probe另行披露。
首epoch几何流未扫完、三个学生完整轮数均0；没有AP/召回/同FP/K/重复框/定位结果，也没有训练进程或监控需保留。
合成MAP与mass损失梯度可区分不等于最强控制已冻结、机制成立或原创性通过。完整测试数量见工程报告。

详见`paper/candidates/bounded_retention/results.md`，独立输出`outputs/bounded_retention_engineering_20261105/`。
可重放入口`python scripts/audit_bounded_retention_terminal.py --audit`；仅本轮终态复核，不重新运行旧实验audit。
README/baseline入口同步；中文commit并push。后续若改变固定E或定位表示，须明确是新科学合同，不能续跑本停止版本。

### 3.83 复核额外E合同、原P1编码修订及U事件终态（2026-11-05）

从干净f87cf6f开始，用户要求ARS审查停止是否来自额外合同，合理修订独立验证通过后直接三臂各30轮。
核对安装8.4.172：原P1/BboxLoss用原连续框算CIoU，bbox2dist/DFLoss另用clip(d,0,14.99)编码。
因此此前“E必须可表示”确属附加门槛；旧V2 STOP只否定旧合同，不能据此否定整个机制。

先冻结`p1_encoding_contract.md`，新增独立v3 loss/Trainer/checker，保留raw E/物理几何/身份/原子oracle。
原生DFL只在克隆编码上饱和；不改标签或U mask，不丢对象。独立31项CPU/CUDA AMP验证通过，
全known完整loss/梯度逐位等价，旧object142895的E-top仍636.8081，合法原子变[9,9,9,9,4,4,4]。

新真实`p1_encoding_integration_01`，PID49196，UTC05:16:50至05:20:51，自然停止，241.2241秒。
前12batch通过；第13batch object164630/occurrence399，仅bottom未知。139个U实例通过，第140个失败。
物理边界下包络637.5463731至画布640，而画布内字面DFL原子最大636；32native候选全空。
全部E距离0.06868..5.15417均合法，不再是E门问题。独立连续outer见证bottom638/639/640均通过保留条件；
bin0/1的(.625,.375)也可解码bottom639，所以不得称真实框或原P1连续定位无解。

进一步审查编码支撑union：它能消除整数格点空集，却丢掉原DFL插值权重，C=[.2,.4]与[.6,.8]均变{0,1}。
mass/编码MAP不能区分，最优物理point-DFL能够区分；原强控制资格不能自动继承。
故不把新编码事件冒充原物理质量的无损修复；未来若更换概率测度，需独立定义和认证连续点强控制。
本轮明确停止当前字面离散质量方向，保留事件信息本身未被否定，不声称所有连续替代方案不可行。

完整1316tests通过；`audit_bounded_retention_p1_encoding.py --audit`核对31输入/28快照及当前合法来源行，
旧E修订成功/新U空集/连续可行/编码信息损失均字节重放一致。未读隐藏训练GT或旧质量缓存。
V3完整真实AMP比较未执行，显式batch forward尝试0/optimizer0/val inference0；构造stride probe另记。
首epoch覆盖未完成、三臂0epoch，无性能/机制判定；原P1/V1/V2/V3及历史合同/失败保护。
独立输出`outputs/bounded_retention_p1_encoding_review_20261105/`，详见`paper/candidates/bounded_retention/p1_encoding_results.md`。
入口和第10节同步，中文commit并push；未启动学生，未创建训练监控，不自动重试或扩训。

### 3.84 连续相容集合上的归一化DFL密度（2026-11-05，实施中）

起点92aa1b8，academic-research-suite。用户明确要求审查连续集合而非整数原子，保留
区间位置、DFL插值权重和面积耦合，并以最强同界连续point-DFL控制；通过独立验证后
直接顺序三臂各完整30轮。先冻结continuous_review_contract.md、continuous_density_contract.md
及continuous_numeric_addendum.md，未修改任何旧合同或失败。

选择`(原E DFL + log integral_D exp(-ell) - log integral_C exp(-ell))/4`，D为canvas内与
固定E组成正面积框的完整未知坐标域，C为原SOURCE_OUTER联合见证及声明运输外集。
强控制为同C上的全局`inf ell`。不要求U单点密度极限等于原DFL；全known仍严格恢复原P1。
此为新连续密度lift，不声称校准/均值等价/任意联合相关性；比较还含log Z归一化的作用。

当前仅定义与独立实现阶段。必须完成真实连续域、数值误差、全局点最优、完整loss和
Trainer/增强流/AMP零优化验证及全库pytest，才有资格冻结学生运行。尚无新学生或验证推理，
无性能/机制结论；公开保留事件的模拟GT衍生侧信息、CABB/GFL边界和隐藏GT禁令全部保留。

### 3.85 连续目标不变的后验采样计算（2026-11-05，先冻结）

V4完整3U例全局point gap=0通过；矩形积分19999cells/111.35秒后仍有0.4246866
残差，未满足1e−4。receipt为`outputs/bounded_retention_continuous_20261105/curved_3u_capacity_01/`，
运行只导入先保存的源码快照。体积独立参照28240.81257落在[23437.88571,33391.64082]内。
这是该积分算法的容量失败，不是连续定义失败。

ARS独立审查认可`grad log I_C=E_C q`：保持原连续目标，以完整C后验拒绝采样
提供无偏一阶梯度；1U及D仍解析积分。已在新实现前冻结
[V5合同](paper/candidates/bounded_retention/continuous_sampling_contract.md)，固定K256、
接受包络下界1/8、随机比特及误差义务、日志区间和独立RNG，不能因只留下inside部分而假通过。
全部真实工程门和全库测试尚待执行，不构成训练许可；学生0epoch，无验证推理。

### 3.86 连续候选独立实现与真实集成前验证（2026-11-05）

V5已实现：一U精确区间并集解析积分，多U完整曲面后验拒绝采样，D解析归一化；
固定K256、私有RNG、有理全局point-DFL及同集合TAL/CIoU。修正独立审查发现的
诊断区间记录、audit不覆盖原采样证据、train11310硬门。全库1715 passed/133.38秒，
receipt为`outputs/bounded_retention_continuous_20261105/suite_01/`。
四例3U容量（两种明示理想tau×flat/ramp logits）全部通过，9.80秒，
`posterior_capacity_01/`保留源码快照；这不是生产逐对象SOURCE_OUTER阈值的替代证据。
旧容量例tau文字与实际默认不符已单独勘误，旧receipt不改。

独立来源审查确认：`E_nominal=F_native(tokens,actual_augmentation)_E`不是对latent E
的新独立测量；不得新增无依据的物理E实数等值门。混合表示与保守外集边界冻结于
`continuous_observation_semantics.md`，不声称每个外集见证可重放tokens或完整物理运输。
下一步执行真实Trainer、11310/548 loader、mosaic开/关两条完整流及唯一真实最大U维数
batch AMP前后向/参数不变检查；尚未执行，不构成学生许可。全部通过才冻结/直接顺序3×30。

### 3.87 原生名义E越界的独立V6修订（2026-11-05）

V5实际完成Trainer/权重/11310 train与548 val loader，前20增强batch通过；第21batch
source181722/occurrence662因left E=-1/65536被额外canvas门拒绝，实际未知边为翻转后的right。
无显式batch forward、optimizer或val inference。原结果和快照位于`posterior_integration_01/`；
`v5_stop_audit.json`核对51输入/48快照一致，失败receipt SHA为
`3c27289dbbf0d0a9ef9e4ec488b6653a0ee2c559761db71c596421638c6da03e`。
独立见证证明C约[249.36527259407916,640]且[300,301]完全inside，所以不是物理相容集为空。

先提交66f3bfc修订合同，再新增V6域/normalizer/loss/Trainer及独立检查入口，不改V4/V5文件。
仅U与canvas相交；原E不clamp、不降级、不加容差。域成员/cell/解析及Decimal积分同步，
原DFL/CIoU、来源、面积、采样和全局点控制不变。完整1862 passed/98.78秒，
`suite_v6_01/`保留日志；真实V6检查尚未执行。贡献可为有依据的已有方法适配，
不把全新基础理论证明设为门；来源/实现/公平比较仍须通过，再直接顺序三臂各30轮。

### 3.88 完整增强流与原生AMP语义的独立审计（2026-11-05）

V6 `native_e_integration_01`于UTC08:24:58–09:10:43运行2744.809秒，PID52840自然结束。
两阶段各1414batch的几何全部通过；载入11310 train/548 val，105058原人工行对应
105057原生对象及1个完全相同文本duplicate alias。原预定首次最大3U压力batch在
mosaic_closed index1316；显式前向1次，7项实际head数值审计含3U通过。
随后在`nonfinite original P1 loss/gradient`停止，未完成全模型有限反向；optimizer=0、
val inference=0。原receipt SHA256 `39c03aa3a4243a0f3169abe174c4701d964f905df749c208148fc0bccee7d259`。

旧检查把原生AMP初值65536下的梯度溢出与未缩放loss非有限合并。安装版PyTorch明确
允许初期缩放溢出后跳步并降低scale，故初值必须有限是额外诊断条件，不能直接否定机制。
先以1648327冻结`amp_audit_revision_contract.md`：保留V6及全部失败，独立重放同压力batch，
不重复平坦数值扫描，以同一forward的预定共同缩放序列审计真正原P1/候选/控制反向。
未缩放loss非有限仍直接失败；实际训练的默认AMP不改。新审计尚待执行，三臂仍0epoch。
独立checker、AMP helper、只读诊断、顺序启动器及548图评估器已实现；完整
`python -m pytest`为2034 passed/144.49秒，输入未变，日志与hash回执在`suite_amp_01/`。

用户明确允许已有方法的有依据适配作为贡献；不要求新基础理论。最终工程和公平比较仍须
通过；准备中的三臂启动器将统一禁止OOM减batch重试、NaN回载及包装异常杀worker，
评估保留三对同FP/精确同K、重复框和定位诊断，尚无真实评估结果。

### 3.89 连续集合三臂放行与最终冻结（2026-11-05）

独立AMP审计PID22280自然结束，727.922秒；旧2828batch重放一致，仍选closed1316/3U。
三臂未缩放loss有限，共同scale依预定序列首次在1024下head/全部参数梯度均有限；
完整候选backward通过，参数不变，原P1包装等价，旧7项head审计与三臂只读预览通过。
候选−控制去缩放head梯度L1=0.23438087879912928；分类/CIoU相同。没有optimizer或val推理。
证据见`continuous_amp_results.md`；这证明旧压力batch阻塞是缩放诊断问题，不是未缩放loss失败。

修正联合验收器对旧STOP成功专属字段的误要求后，2037 tests通过。首次只读启动核验
另发现两侧Windows路径规范化不一致；保留原freeze/失败/源码，独立启动V2仅修复路径比较。
最终完整2042 passed/126.04秒，`suite_launch_03/`；没有新loss版本，也未重跑失败训练。
最终24004输入核验通过，freeze SHA256为
`a04ec4f117472fc2e11f988a61534d3de5898c9c9bff5d7f7a592928a6a57901`，
路径`outputs/bounded_retention_continuous_freeze_20261105_v2/freeze.json`。

三臂合同及启动修订已在训练前冻结：原P1真实Trainer、V6集合质量候选、同界全局连续点DFL；
原baseline超参数、顺序各30轮、548图val960/best.pt、COCO与Ultralytics分别报告、同FP/同K/
重复框/定位及收缩/变尖诊断。三臂同禁OOM减batch、NaN回载、重试与自动杀worker；
正常AMP跳步降scale允许。原P1、旧合同/失败不改；允许有依据的既有方法适配贡献。
先记录工程放行和最终冻结；实际进程身份/启动状态另记，不把此段写成训练已启动。

### 3.90 三臂顺序任务实际启动（2026-11-05）

源码/合同提交5f96181已push，最终freeze保持
`a04ec4f117472fc2e11f988a61534d3de5898c9c9bff5d7f7a592928a6a57901`。
UTC10:21:51.7107639（北京时间18:21:51）启动调度PID30104，命令为：

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' -u scripts/run_bounded_retention_continuous_v2.py --train --spec outputs/bounded_retention_continuous_freeze_20261105_v2/freeze.json
```

已观察到原P1 worker PID45600在第1/30轮执行真实batch。后续candidate/control只在
前一臂exit0、完整30轮和封印审计通过后派发；目前二者尚未启动，不声称三臂已完成。
真实输出`outputs/bounded_retention_continuous_three30_s0/`，每臂process/claim/exit/seal
独立记录；外层命令和创建时间在freeze目录的`launch_identity.json`。
临时只读监控PID18428，`outputs/monitors/bounded_retention_continuous_three30_s0/health.json`，
监控不杀进程、不重试、不启动实验，正常进展静默；终态处理后移除临时脚本。
此记录仅认证实际启动，没有性能或机制判断。

### 3.91 连续集合候选全局点容量故障终态（2026-11-06）

本轮真实训练已结束，状态`STOP_ENGINEERING_POINT_RESOURCE_LIMIT`。
参考PID45600完整1–30轮exit0，耗时1.922小时；候选PID51256完整1–9轮，
第10轮日志89/1414附近抛出`NumericalCertificationError: global point resource cap`，
exit1，耗时20.706小时；强控制未启动。约北京时间17:00退出，约30分钟复查时发现。
不恢复、不自动重试或继续控制臂；此前无已记录的中断/重启。

异常位于TAL使用的共享连续点求解器`optimize_point`，在20000单元限额内未取得
1e-5误差证书，不能说相容集为空或连续机制无效。不是OOM/未用GPU；换4090
不直接改变这个上限。日志未保存失败对象/实际cells/gap/logits，不推断具体U维数。
候选前9轮平均2.273小时/轮，对原P1同前9轮累计耗时比35.534；未做成本剖析，
不能把比例全归给一个函数。开训前工程正确性覆盖并不替代训练成本/长尾评估，
本次缺少充分预算说明；改进希望是机制假设而非已证实性能收益。

原`--audit`只读重放exit0，`accepted=false, terminal_evidence_verified=true`；
24004冻结输入及61产物封印通过，原P1独立单臂审计通过，两臂实际args与冻结一致，
已记录CSV均有限。候选9份诊断共12726batch/63008后验事件，仅证明计算发生，
不等于正权重机制效应。没有最终548图val960，不给AP/同K/同FP/定位比较，
不把候选9轮与参考30轮配对或追改历史门槛。

详见[终态报告](paper/candidates/bounded_retention/three_arm30_terminal_failure.md)。
原输出`outputs/bounded_retention_continuous_three30_s0/`保持封印；独立收尾证据
在`outputs/bounded_retention_continuous_terminal_20261106/`。父30104、参考45600、
候选51256、监控18428均自然结束；保存监控源码副本/hash后删除临时monitor.py，
保留health/checks/logs，没有创建应用自动化任务。仅报告/入口文档变更，
不重跑已通过的2042项源码测试；本次终态验证为只读文件/配置/封印审计。

### 3.92 同目标连续点容量修复审查（2026-11-06）

先冻结`point_capacity_review_protocol.md`（c24603d），使用ARS实验验证流程和独立静态反方审查。
只做新文件实现、零训练与成本评估；用户最新授权不包含恢复或启动训练。
原失败现场未保存完整请求，无法精确重现；不以新合成例冒充原第10轮现场。

独立`bounded_retention_point_search_v2.py`及`bounded_retention_point_bound.py`实现
必要源投影收缩、同轴联合DFL仿射上包、认证候选点和细分方向提示。
源集合/原E/插值/面积耦合/两臂目标均不改，20000 cells及1e-5 gap不变；
1U沿用旧解析求解，全known保留原P1。两臂共用新求解器，但新旧同分几何不保证逐位相同。

最后`synthetic_03`新8/8通过，旧7/8通过/1容量错误，证书无冲突，前后源码SHA一致。
3U ridge从旧19999cells容量失败变为新41cells、0.63070秒、gap=1/131072；
3U sharp为旧19.4934秒→新6.8677秒。普通例仍有退化，不能用kernel结果推断整个训练。
此前失败、超时以及`synthetic_02`源码运行中变回首版的未知来源事件全部保留，
该轮整体不认证；用户确认未主动修改，最终修订另用v2独立路径。
三版本共24次多U压力请求，没有增大预算。最终`suite_04`完整2087tests通过/165.56秒。

真实入口前两次分别因初始化进展监控、Windows心跳替换失败结束，全部保留，
先冻结2cc19a6/f47d205补充再修监控；不计作机制失败，不拼接结果。
`real_03`父46776/worker59204于19:07:48启动，553.690秒exit0，进程已退出。
固定mosaic开/关首16+16批、五臂160次完整loss及全模型反向通过；
参数/BN/source SHA不变，全部梯度有限。候选/控制共享几何/TAL/CIoU通过；
本窗口新旧loss/几何/分配/梯度逐位相同，候选/控制30批梯度不同（其余2批无supported objects）。
每臂20548个1U、390个2U请求，无3U；未检查第9轮权重，原失败现场仍无法精确复现。
完整loss旧102.2471秒、新108.0245秒，旧/新0.9465；含数据/前后向比0.9512。
原2倍门槛未通过，不挑较快的closed子组。初始窗口30轮外推新P1/候选/控制为
5.1212/43.8247/15.6851小时，合计64.6311小时，未含验证/保存等，非训练时长承诺。
原P1历史实测1.922小时不能被这个含首次冷开销、workers0的诊断外推替代。
局部容量修复不等于整体提速或后期可靠；**停止本轮扩展，不接入生产、不重新开训**。详见
`paper/candidates/bounded_retention/point_capacity_review_results.md`。
产物仅在`outputs/bounded_retention_point_review_20261106/`；原P1/冻结生产入口/旧训练证据不改。

### 3.93 真实3U与第9轮EMA压力门停止（2026-11-06）

用户接受时间成本，要求先补真实3U及第9轮权重零训练，**通过后**才准备独立入口/三臂命令，
本轮明确不训练。采用ARS experiment-agent validate及独立只读代码复核；先冻结
`point_pressure_protocol.md`（943c9e9）和正TAL权重/完整梯度审核补充（65426a5）。
固定两条1414批流、closed1316压力批、两份权重及第9轮固定32批，共34case/102臂；
原20000cell/1e-5/256sample/16384proposal/256bit/诊断scale1024与120秒期限均不放宽。

audit_01在open第0批重放不一致、未进入显式模型前向/loss即停止，失败保留。
独立无模型/无数据探针证明：新包装层屏蔽回调注册，少了events.py构造session_id的一次
random.random，改变增强RNG。修订合同257052a先冻结，恢复历史原生注册/val配置，
不手动丢随机数或搜索seed；audit_02完成两条流和原固定批身份/张量SHA一致检查。

audit_02父54428/worker2320于20:30:21启动，初始压力批原P1 loss0.2904秒/反向4.7054秒，
全部参数梯度有限，无unused。候选loss阶段于20:43:03.830进入，120秒未完成，父仅终止自有
诊断子进程，TIMEOUT/exit1，均已退出。1次显式前向、1对P1 loss/反向、0完整case；
第9轮EMA只载入并保存状态，未运行；固定32批也未执行。无候选内部栈/point_failure，
不能把超时归因为再次点容量故障；该阶段含criterion前后CUDA同步，也不能独立证明
criterion本身未返回。未获真实3U正权重消费/同界控制/候选梯度资格。
硬终止未走case恢复finally，不声称全部state restored；所有磁盘权重未写。

独立qualification_01在读tensor前拒绝失败terminal，STOP/accepted=false；
terminal_review_01核对374项绑定0变化并封存已有产物摘要。完整pytest 2137 passed/115.81秒。
新检查/资格checker独立文件，旧源码/P1/冻结训练合同/全部失败保留；输出仅在
`outputs/bounded_retention_pressure_20261106/`。120秒是预冻工程约束，不是数学机制否定，
也不是性能或显卡结论。**压力门未通过，不接入训练Model/Trainer/launcher，不冻结新三臂，
不给执行训练许可，不训练/重试/提高限额。** 详见`point_pressure_results.md`。

### 3.94 120秒期限来源与共享点求解成本定位（2026-11-06）

用户只授权独立零训练定位和期限审查，不直接提高上限/开训。使用ARS validate，
先冻7e9686c主合同；仅保存预测的完整loss重放，原P1逐位校准，不构造模型或Trainer/新前向/反向。
候选/控制各独立进程冷缓存、同point_search_v2、同随机状态，120秒绝对deadline不被内部日志续期。

所有诊断故障公开保留：trace_01因观测器pickle缓存方法失败，冻结8612dd1纯数据修订；
suite_02新增测试误读缓存属性失败时主会话误启动trace_02，发现后核验并停止自有worker，
ccf2134记录并排除其全部数值/耗时，不读取轨迹调参。修正测试后suite_03通过；
trace_03在首次faulthandler转储截断后原生退出0xC0000005，根因未确证，控制未启动。
b45f95e冻结普通Python栈快照修订。最终suite_04为2147 passed/156.19秒。

trace_04父45356/P1 worker46232/候选22816/控制47724顺序执行。P1逐位loss/assignment通过；
候选2590个点请求全部完成，含357个1U与同一合法实例(89155,55771)的2233个3U锚点。
3U累计104.217秒，单请求max0.328秒/max15cells；CPU时间与五次栈定位在精确几何收缩/
顶点/联合score上界的Fraction运算。normalizer47次0.341秒、区间40次0.078秒，
posterior完成6次7.486秒，第7次在119.703秒刚开始即碰120秒deadline。无单请求容量失败证据。
控制全部2590请求输入/点/gap/cells与候选相同，完整loss108.391秒；共享3U点求解100.904秒。
候选未完成，不能声称完整TAL/梯度/正权重资格通过。三臂没有新训练，所有诊断进程已结束。

120秒首次来自c24603d对单请求与整批同时采用的工程保护，非机制/原P1数学要求。
本次证据支持整批门很紧且不能用来否定机制；不证明提高到某个新值就能完成或稳定。
主要瓶颈是共享多锚点CPU精确几何，不是积分或显式CUDA同步；到期时所在后验采样不等于采样卡死。
建议后续只审查两臂共享的纯几何缓存/等价重复查询，保留全部锚点及强控制证书；本轮未实现。
不改旧TIMEOUT、不放宽期限/cell/proposal/bits/误差，不开训/给训练许可。
证据位于outputs/bounded_retention_timeout_20261106/{trace_04,analysis_01,source_archive_04,suite_04}/。

## 4. 固定实验协议

### 4.1 训练与评估

| 项目 | 固定值 |
|---|---|
| 检测器 | YOLO11s |
| 初始化 | COCO 预训练 `D:\Python\YOLO\yolo11s.pt` |
| 训练输入 | 640 |
| 主评估输入 | 完整原图，960 |
| 优化器 | SGD，`lr0=0.01`，`lrf=0.01` |
| 训练轮数 | 主实验 100 epoch |
| batch | 8；OOM 时所有对照统一降低 |
| 其他 | `close_mosaic=10`、`patience=0`、`workers=0` |
| 数据 | VisDrone train/val |

主实验阶段不修改网络结构、损失权重和增强策略。任何额外模块必须先通过
30 epoch pilot，再进入 100 epoch 主矩阵。

### 4.2 区域标注协议

- 5%/10%/20% 表示标注面积比例，不是图片数量比例。
- 每张训练图选择一个目标中心的 square tile，允许最多 tile 边长 20% 的抖动。
- 框与 tile 的可见面积比例低于 0.35 时丢弃。
- 每张图额外采样一个低重叠未标注 tile，其隐藏 GT 只用于分析，不能进入训练。
- `manifest.jsonl` 必须保留 tile 坐标、保留框数、隐藏 GT 数和随机种子。
- 标注 split seed 与训练 seed 必须成对记录。

### 4.3 主指标

- 主指标：mAP50-95、mAP50、AP_small、AR500。
- 分析指标：按尺寸的伪标签 precision/recall/覆盖率、每类 AP、隐藏 GT 漏检数、
  参数/FLOPs/FPS。
- 报告均值 ± 标准差；同一 split 做配对比较。
- 预注册主结论：10% 标注面积下，P1 相对 B2 至少 `+1.0 mAP50-95` 或
  `+1.5 AP_small`，且至少两个 seed 方向一致。
- Ultralytics mAP 只能作为开发指标；最终表必须补 VisDrone 官方 AP 或明确说明
  评测工具差异。

## 5. 方法矩阵

| 名称 | 训练数据 | 论文作用 |
|---|---|---|
| B0 Full-sup | 完整 VisDrone 标签 | 性能上界，不作为方法变量 |
| B1 Partial-sup | 仅标注 tile | 回答“直接丢弃未标注区域损失多少” |
| B2 Naive self-training | 标注 tile + 全局阈值 0.50 伪标签 tile | 判断半监督信号是否存在 |
| P1 Scale-aware | 标注 tile + 尺度分层阈值 + 置信度软权重 + uncertain-ignore | 论文主方法候选 |

P1 第一轮只做一次离线自训练。只有 P1 明确优于 B2，才考虑第二轮或在线 EMA
教师；不得用不断增加模块来掩盖 pilot 失败。

## 6. 8 周执行计划

| 周次 | 日期建议 | 核心任务 | 必须交付 | 进入下一周的条件 |
|---|---|---|---|---|
| 第 1 周 | 10/26–11/01 | 正式 5/10/20% seed0 数据；B0/B1 启动；论文骨架 | `summary.json`、split/manifest、5 epoch 曲线、B1 seed0 训练已启动、引言/相关工作骨架 | 数据审计通过；20 图与正式 split 链路均可复现 |
| 第 2 周 | 11/02–11/08 | 10% B1/B2 30 epoch pilot；伪标签尺寸诊断 | pilot 结果表、伪标签 precision/recall/覆盖率图、失败案例、方法章节初稿 | B2 相对 B1 不低于 `-0.2 mAP50-95`；否则先诊断漏检 |
| 第 3 周 | 11/09–11/15 | 实现 P1 尺度阈值、软权重和 uncertain-ignore | 单元测试、P1 配置、30 epoch 对照、方法流程图 | P1 相对 B2 至少 `+0.5 mAP50-95`，AP_small 或小目标 recall 同向改善 |
| 第 4 周 | 11/16–11/22 | 10% 标签完整 100 epoch，seed 0；启动 seed 1 | B1/B2/P1 主结果、吞吐和显存记录、实验章节表格模板 | P1 未出现数据泄漏或明显的训练不稳定 |
| 第 5 周 | 11/23–11/29 | 10% seed 1/2；5%/20% 预算实验 | 完整主矩阵初稿、均值 ± 标准差、标注预算曲线 | 主方法在两个 seed 上方向一致 |
| 第 6 周 | 11/30–12/06 | 消融、阈值敏感性、第二轮自训练 | 尺度阈值、ignore、软权重、轮数消融；阈值曲线 | 不增加新模块；只回答核心问题 |
| 第 7 周 | 12/07–12/13 | 冻结版本；官方/外部确认；补 AP_small 和效率 | 最终表、权重 SHA256、评测日志、冻结配置、案例图 | 所有数字都来自冻结版本 |
| 第 8 周 | 12/14–12/20 | 图表、论文全文、复现实验包、内部审阅 | 论文初稿、图表源文件、README、配置归档、答辩材料 | 他人能按 README 复现最小实验 |

如果 GPU 占用或训练排队导致延期，优先保持“数据→pilot→主方法→消融→写作”的
顺序，不要为了赶日期同时启动未筛选的模型。

## 7. 实验矩阵与运行顺序

主矩阵固定 YOLO11s 640/100 epoch：

| 方法 | 5% | 10% | 20% |
|---|---:|---:|---:|
| B0 Full-sup | 参考 | 参考 | 参考 |
| B1 Partial-sup | seed 0,1 | seed 0,1,2 | seed 0,1 |
| B2 Naive self-training | seed 0,1 | seed 0,1,2 | seed 0,1 |
| P1 Scale-aware | seed 0,1 | seed 0,1,2 | seed 0,1 |

建议运行顺序：

1. 第 1 周启动 B0 和 10% B1 seed0。
2. 第 2 周完成 10% B1/B2 30 epoch pilot。
3. 第 3 周只训练 P1 pilot，不扩到全矩阵。
4. 第 4 周完成 10% seed0 的 B1/B2/P1 100 epoch。
5. 第 5 周按“已有结果价值”排序，补 seed 和 5%/20% 预算。
6. 第 6 周只跑预先登记过的消融。

完整矩阵约需 150–220 GPU 小时；单张 RTX 4070 Ti 连续运行约 2–3 周。每个
100 epoch run 预计 6–10 小时，实际以本机测得时间为准。

## 8. 论文写作计划

| 论文章节 | 主要证据 | 最晚完成时间 |
|---|---|---|
| 摘要、引言 | 标注成本、未标注不等于背景、研究问题和三点贡献 | 第 4 周初稿，第 8 周定稿 |
| 相关工作 | SSOD、稀疏/部分标注、航拍小目标、伪标签 | 第 1 周骨架，第 3 周补全 |
| 方法 | tile-aware 流程、尺度阈值、软权重、uncertain-ignore | 第 3 周完整初稿 |
| 实验设置 | 数据协议、B0–P1、固定参数、评测工具 | 第 2 周模板，第 7 周冻结 |
| 主结果 | 10% 主表、5%/10%/20% 曲线、均值±标准差 | 第 5 周 |
| 消融与机制 | 四组消融、阈值曲线、伪标签精度/召回/漏检 | 第 6 周 |
| 讨论与局限 | 可见性阈值、合成区域协议、官方评测差异、外部集限制 | 第 7 周 |
| 结论与复现包 | 结果边界、失败案例、README、SHA256 | 第 8 周 |

### 8.1 必做图表

- 图 1：原图、区域标注 tile、未标注 tile 和隐藏 GT 的协议示意图。
- 图 2：B1 教师、B2/P1 伪标签筛选和学生训练的流程图。
- 图 3：5%/10%/20% 标注面积与 mAP/AP_small 的预算曲线。
- 图 4：按尺寸的伪标签 precision、recall、coverage 和漏检率。
- 图 5：成功/失败案例与 gate/uncertain-ignore 可视化。
- 表 1：数据与隐藏 GT 统计。
- 表 2：B0/B1/B2/P1 主结果。
- 表 3：尺度阈值、ignore、软权重、轮数消融。
- 表 4：参数量、FLOPs、FPS 和显存。

### 8.2 投稿前的最低完成标准

- [ ] 10% 标签下 B1/B2/P1 的 100 epoch 主结果和至少两个 seed。
- [ ] 5%/20% 预算曲线，或者明确说明未完成的原因和结论边界。
- [ ] 至少三组核心消融：尺度阈值、uncertain-ignore、置信度软权重。
- [ ] AP_small、伪标签质量和隐藏 GT 漏检分析。
- [ ] 官方 AP 或明确的 Ultralytics/官方评测差异说明。
- [ ] 所有表可由冻结配置、权重哈希和日志追溯。
- [ ] 论文、图表源文件、README 和复现实验包归档。

## 9. Go/No-Go 门槛

### Gate 1：工程可复现

20 张图的 split 生成、5 epoch 训练、完整 val 推理均成功；同 seed 生成结果一致；
9 项以上单元测试通过。

### Gate 2：半监督信号存在

10% 标签、30 epoch pilot 中，B2 相对 B1 不得低于 `-0.2 mAP50-95`。若下降，
先检查教师伪标签 recall、未标注 tile 密度和阈值，而不是继续加模块。

### Gate 3：主方法有独立收益

30 epoch pilot 中，P1 相对 B2 至少 `+0.5 mAP50-95`，且 AP_small 或小目标 recall
同向改善。未达到时停止扩训，优先改单一机制或转诊断分析。

### Gate 4：完整训练可成立

100 epoch 主实验在 10% 标签、两个 seed 上均值相对 B2 达到预注册效应量。若未
达到，不把 0.1–0.3 个点写成显著提升；改为区域标注失效分析或更换单一伪标签机制。

## 10. 进度日志与后续会话交接

每次工作结束必须在本节追加一条记录，至少包含：日期、commit、执行的命令/
实验、产物位置、关键结果、是否通过 gate、下一步。不要把“启动训练”写成
“完成训练”，也不要把烟测写成正式结果。

| 日期 | Commit | 工作或实验 | 产物/命令 | 结果 | Gate/下一步 |
|---|---|---|---|---|---|
| 2026-10-26 | `d3503c9` | 初始化仓、数据/B1/B2 工具链；20 图烟测 | `data/smoke_area10`、`runs/smoke/*` | 9/9 tests；B2 13/20 伪标签图、38 框、33 训练图；完整 val 成功 | 仅过 Gate 1；正式数据未生成 |
| 2026-10-26 | working tree | 建立论文完成计划、新会话入口和文件地图 | `docs/paper_completion_plan.md`、`AGENTS.md`、README 链接 | 8 周计划、证据边界和交接规则已落地；测试仍为 9/9 | 等待审阅后提交 |
| 2026-10-26 | `23fdcd5` + working tree | 修正 Git Bash 环境命令；生成并审计正式 10% seed0 数据；明确 20 图烟测的数据路径 | Git Bash 加载 `conda.sh` 后 `conda activate yolo`；`data/visdrone_area10/{summary,audit}.json`；README/baseline 命令 | 6471 张训练图、105058 个监督框、38245 个可见隐藏 GT；完整 val 548 张；12/12 基础审计通过；未改源码，未运行训练 | 数据构造完成；Gate 1 尚缺同 seed 一致性和 5 epoch/640、完整 val960 验收；之后启动 B1 |
| 2026-10-26 | `8501b73` + working tree | 审核用户完成的 5 epoch 烟测；补完整 val960、同 seed 重生成检查和测试 | `runs/smoke/partial10_sup_smoke5/gate1_audit.json`；`runs/val/partial10_sup_smoke5_val960/metrics.json`；`python -m pytest` | 20 图/640/batch8 训练完成；完整 val 548 图；开发 mAP50-95=0.080882；60 个数据文件哈希一致；9/9 tests；正式数据关键哈希未变 | Gate 1 通过；仅烟测，无正式方法结果；建议启动正式 10% seed0 的 B1 100 epoch，本次未启动 |
| 2026-10-26 | `14eaf2c` + working tree | 验收正式 B1 100 epoch，补 val960/权重哈希；修复全量伪标签批处理 OOM；生成/诊断/合并 B2 数据 | `runs/partial10_sup_s0/{audit.json,SHA256SUMS.txt,validation/metrics.json}`；`data/visdrone_area10/pseudo/naive_t050/`；`python -m pytest --basetemp outputs/pytest_pseudo_batch_20261026` | B1 正式开发 mAP50-95=0.248835；4624 个伪标签 tile、27667 框，合并 11095 图；11/11 tests；教师哈希固定，隐藏 GT 仅离线分析 | B1 完成，B2 数据就绪；Gate 2 未测，两个 30 epoch 学生均未启动；P1 尚未实现 |
| 2026-10-27 | `f7b711c` + working tree | 验收用户完成的 B1/B2 30 epoch pilot，分别补完整原图 val960 和权重/输入审计 | 两个 pilot run 下的 `audit.json`、`SHA256SUMS.txt`、`validation/metrics.json`；`outputs/gate2_20261027/comparison.json` | B1 mAP50-95=0.229114，B2=0.234534，差 +0.5421 个百分点；30 epoch 及实际训练参数配对，11106 个伪标签包文件哈希未变 | Gate 2 通过；仅单 seed 开发 pilot，非最终方法结论；下一步实现 P1 并测 Gate 3，本次未改源码或启动新训练 |
| 2026-10-27 | `f7b711c` + working tree | 实现 P1 三项机制、固定分尺寸 recall；单元测试/GPU 烟测、全量生成和哈希预检；10:33 启动 30 epoch pilot | `python -m pytest --basetemp outputs/pytest_p1_20261027`；`configs/{pseudo_p1_10,train_p1_10}.yaml`；`outputs/gate3_20261027/{preflight,status}.json`；`scripts/run_p1_pilot.py` | 28/28 tests；37 tile/1 epoch 烟测通过；全量 4839 伪标签 tile、35067 正框、52117 ignore 框，合并 11310 图；B2 学生固定小目标 recall=33.815%；详见 3.6 | 工程实现完成，正式 pilot 仅已启动；Gate 3 未测，待 val960 和小目标 recall 同时达标；不扩训 |
| 2026-10-27 11:04 | `f7b711c` + working tree | 核查 P1 训练进程、CSV、权重文件、实际配对参数和评估产物；更新进度记录 | `outputs/gate3_20261027/progress_check_20261027_110453.json`；stdout/stderr、`results.csv`、`args.yaml` | 已完成 6/30 epoch，第 7 epoch 到 1344/1414 batch；CSV 连续且全部有限，best/last 存在，stderr 为空；与 B2 实际参数配对一致 | 训练未完成，最终 val960/小目标 recall/比较文件尚未生成；Gate 3 尚未测量，继续原后台流程；本次未改源码或重启训练 |
| 2026-10-27 | `8f02449` + working tree | 增加 P1 本机实时进度查看器并打开浏览器，方便用户持续查看和完成后通知验收 | `python scripts/watch_p1.py`；地址见 `outputs/p1_monitor/server.json`；`python -m pytest --basetemp outputs/pytest_p1_monitor_20261027` | 页面每 3 秒读取日志/CSV/阶段；显示 epoch/批次、估计剩余训练时间、最近日志和评估完成提示；28/28 tests，HTTP 页面及完成/评估中状态检查通过；核查时完成 17 epoch、正在 18 epoch | 仅增加只读查看工具，训练继续，Gate 3 未测；全部完成后验收结果 |
| 2026-10-27 12:44 | `8f02449` + working tree | 验收 P1 30 epoch、完整 val960、小目标 recall，独立核对冻结输入/产物哈希与 Gate 3；按用户目标保留正向阶段成果 | `outputs/audit_gate3_acceptance_20261027.py`；`outputs/gate3_20261027/acceptance_audit.json`；run 下 `audit.json` | 19/19 验收检查通过；mAP50-95=23.7883%，比 B2 +0.3349 个百分点；小目标 recall=35.6925%，+1.8776 个百分点；23773 冻结输入哈希匹配 | 原 Gate 3 未达 +0.5 个百分点门槛，小目标条件通过；用户决定认可初步收益，优先整理方法、结果和分析；后续验证按论文需要安排 |
| 2026-10-27 | `8f02449` + working tree | 整理 B1/B2/P1 论文表格，起草方法与实验分析；补齐 B1 分尺寸 recall，增加表格生成核对命令 | `docs/paper/`；`python scripts/build_paper_tables.py --check`；`runs/size_recall/b1_pilot30_s0/metrics.json` | 七组结果表及两章中文初稿；B1 small recall=34.379%，P1 对比 B2 的 AP 为 7 类升/3 类降；27 个来源哈希/表格核对通过，28/28 tests | 写作已启动，主表为配对 30 epoch 单 seed 开发结果；后续审阅与整合正文 |
| 2026-10-27 | `8f02449` + working tree | 检索并核对 SSOD、稀疏标注与航拍小目标参考论文；新增相关工作及 BibTeX；调整方法/实验引用边界 | `docs/paper/{related_work,references}.md`、`docs/paper/references.bib`；第 3.10 节 | 16 项核心文献；识别 S³OD 为最近邻工作，明确其尺度阈值与教师负学习重叠；正文不宣称这些概念首次提出 | 后续精读核心文献、按首次引用顺序统一编号、写引言；比较性实验按现有结果需要安排 |

| 2026-10-27 | `9dd79f7` + working tree | 按用户“下一步怎么做”请求复核进度、协议、论文缺口及现有 Gate 3 结果，给出后续优先级建议 | `python scripts/build_paper_tables.py --check`（已有 yolo 环境）；`outputs/gate3_20261027/{comparison,acceptance_audit}.json` | 表格核对通过，27 个来源哈希及三组 recall 的 GT 计数核对通过；现有正文缺引言、案例图，核心单因素消融与重复 seed 尚未完成；本次未改源码、未启动训练 | 建议先精读 S³OD 并明确方法差异、补引言和图；随后优先做去掉 ignore、去掉软权重、全局阈值三组 30 epoch 消融，再按证据需要安排重复 seed 与正式训练；这是建议，尚未实施或改变原 Gate 判定 |

| 2026-10-27 | `9dd79f7` + working tree | 完成 S³OD 方法差异对照和第 1 章引言；核对正式发表版本、早期预印本与公开代码；修正 Soft Teacher 权重对象 | `docs/paper/{introduction,s3od_comparison}.md`；`references.md`、BibTeX 与论文入口/证据索引；第 3.11 节 | 第 1–4 章初稿齐备；S³OD 更新为 ISPRS Journal 2025 期刊文献；预印本百分位与公开代码 GMM 阈值分开描述；表格 27 来源及文稿链接/文献/引言数字核对通过；仅文稿，无新训练 | 原 Gate 3 判定不变；后续补案例图、核心消融和重复 seed，期刊全文细节与最终引用排序待补 |

| 2026-10-27 | `9dd79f7` + working tree | 准备第①组 no-ignore 配置及训练→val960→分尺寸 recall→完整 P1 对比脚本，提供用户执行命令 | `configs/train_p1_no_ignore_10.yaml`；`scripts/run_p1_no_ignore.py --check-only`；README、baseline 第 4.4 节 | 23781 个冻结文件核对通过；11310 tile 与完整 P1 条件一致，仅关闭 uncertain-ignore；28/28 tests | 仅准备和预检完成，训练尚未启动；用户执行 `python -u scripts/run_p1_no_ignore.py` 后自动评估对比；原 Gate 3 不变 |

| 2026-10-27 | `1e2ef5a` + working tree | 验收用户完成的 no-ignore 30 epoch、val960 和分尺寸 recall；独立核对输入/输出哈希、参数与保存预测匹配；记录首组消融结果 | `outputs/p1_no_ignore_pilot30_s0/{comparison,acceptance_audit}.json`；`outputs/audit_p1_no_ignore_acceptance_20261027.py`；第 3.13 节 | 23795 冻结/产物哈希和 19 个 run 输入匹配；两组各 548 图 recall 重算一致；去掉 ignore 的 mAP50-95 +0.0217 pp、small recall −0.1392 pp（−37）；逐类 4 升/6 降 | 第一组正式配对消融已完成，非烟测；整体 mAP 近似持平，不支持 ignore 独立提高整体 mAP；继续软权重/尺度消融及重复 seed，本次未启动新训练 |

| 2026-10-27 17:44 | `4fe664d` + working tree | 按用户低成本/候选出现后停止约束，启动软权重 1024 源图×两组各 5 epoch 配对筛选，准备全量确认指令但不启动 | `scripts/screen_p1_soft_weights.py`，PID 5784；`outputs/screens/p1_soft_weights_1024_e5/`；README/baseline 第 4.5 节 | 全量配置 23781 文件预检通过；31/31 tests；当前 1765 tile 配对筛选已启动、尚无最终结果；Gitee main 已核实为 4fe664d | 等待筛选完成及验收，发现值得确认的方向后用中文提交并推送，给训练指令并停止；不自动扩训 |

| 2026-10-27 17:53 | `4fe664d` + working tree | 完成软权重 1024 源图×两组各 5 epoch 配对筛选、val960、分尺寸 recall 和独立审计；按预算约束停止全量确认 | `outputs/screens/p1_soft_weights_1024_e5/{comparison,acceptance_audit}.json`；第 3.15 节 | 27 项审计通过；去掉软权重 mAP50-95 +0.0884 pp、small recall +0.0790 pp、mAP50 −0.2249 pp，低于 0.3 pp 筛选线；总学生耗时 417.319 秒 | 中性、不进论文表、不启动 30 epoch；本轮代码与记录用中文提交并推送 Gitee；继续下一项低成本机制筛选 |

| 2026-10-27 18:01 | `919d71e` + working tree | 完成尺度阈值 1024 图单臂 5 epoch 筛选和独立审计；准备全量统一阈值数据与训练入口，按成本约束不启动 30 epoch | `outputs/screens/p1_global_t050_1024_v2_e5/{comparison,acceptance_audit}.json`；`data/.../p1_global_t050_from_candidates_20261027`；`scripts/run_p1_global_t050.py --prepare-only` | 新组 1737 tile/238.414 s；mAP50-95 +0.2237 pp 但 small recall −3.1230 pp；全量数据 11095 tile、4624 伪标签 tile、27667 正框、57219 ignore，预检通过 | 自动候选线未触发但小目标信号值得确认；30 epoch 尚未启动，向用户给命令后停止；本轮代码中文提交并推送 |

| 2026-10-27 20:21 | `ad298ce` + working tree | 验收用户完成的统一 0.50 阈值全量 30 epoch、val960、分尺寸 recall，并接入论文表 4-S2 | `outputs/p1_global_t050_pilot30_s0/{comparison,acceptance_audit}.json`；`outputs/audit_p1_global_t050_acceptance_20261027.py`；`scripts/build_paper_tables.py`；第 3.18 节 | 24 项哈希/参数检查及 548 图预测重放通过；mAP50-95 −0.6177 pp、small recall −0.3424 pp（−91），逐类 1 升/9 降；best hash `94571a…28a4a`；表格 36 个来源及 31/31 tests 通过 | 全量确认支持尺度分层更有利于总体和小目标覆盖；仍为单 seed 开发结果，原 Gate 3 不变；中文提交并推送，不自动扩训 |
| 2026-10-27 20:31 | `5d0d514` + working tree | 按用户授权启动尺度阈值配对学生 seed1；固定 split seed0/教师/两套伪标签与损失，顺序运行两组各 30 epoch | `python -u scripts/run_p1_scale_seed_pair.py`；PID 29248；`outputs/p1_scale_threshold_pair_s1/`；第 3.19 节 | 33,047 冻结文件与 metadata 预检通过；36/36 tests；已有论文表 36 来源核对通过；第一组运行中，尚无 seed1 结果 | 完成两组训练/val960/recall/保存预测重放后更新论文两 seed 配对统计并提交 Git；原 Gate 3 不变 |
| 2026-10-27 23:47（后续收尾见本次提交） | `5d0d514` + working tree | 完成尺度阈值配对学生 seed1 两组 30 epoch、548 图 val960/recall 与独立验收；更新论文表 4-S3、第 4.11 节、证据及复现路径，并纠正短程筛选与全量总体 mAP 方向不同的表述 | `scripts/audit_p1_scale_seed_pair.py`；`outputs/p1_scale_threshold_pair_s1/{comparison,acceptance_audit,independent_audit}.json`；`python scripts/build_paper_tables.py --check` | seed1 尺度−统一：mAP50-95 +0.2420 pp、small recall +0.6434 pp（+171 匹配）；两 seed 平均配对增益 +0.4299±0.2657/+0.4929±0.2128 pp；36 项独立验收、33,047 冻结文件、两组预测重放、59 来源表格与 36/36 tests 通过 | 尺度阈值总体与小目标方向在两学生 seed 保持，支持保留；medium 方向不一致，large 均为统一阈值更高；单 split/固定教师，无显著性或官方 AP_small 主张，原 Gate 3 不变；按用户要求本地 Git 提交 |

| 2026-10-30 09:13（进行中） | `9295c3e` + working tree | 跟进 B2 seed1：原训练第 3 epoch 中断且进程已退出；归档后同配置从初始化重跑完整 30 epoch；整理 seed0 四类案例 | `outputs/b2_seed1_interrupted_20261030_091251/recovery.json`；`outputs/b2_pilot30_s1/`；PID 31944；`docs/paper/case_studies.md` | 37,695 冻结文件核对通过；四类案例和来源 manifest 已生成；尚无 B2 seed1 完整结果 | 等待 30 epoch/完整 val960/recall/四组预测重放，独立验收后更新论文两 seed 统计、测试并本地提交；不推送，原 Gate 3 不变 |

| 2026-10-30 10:15（进行中） | `9295c3e` + working tree | 心跳发现第二次进程中断：日志停于第16epoch且PID31944消失；归档12个文件后用WMI独立启动原入口，避免依赖Codex子进程存续 | `outputs/b2_seed1_interrupted_20261030_101330/recovery.json`；`outputs/b2_pilot30_s1/launch_receipt.json`；PID14304 | 37,695冻结文件再次核对通过；原训练脚本/配置未改；独立验收已支持两级归档来源链及包装哈希 | 当前从原初始化重跑完整30epoch；不使用两次中断指标；完成后按原标准验收、更新论文并本地提交，原Gate3不变 |

| 2026-10-30 10:27（手动交接） | `9295c3e` + working tree | 用户要求由自己终端执行训练；核实PID14304身份后停止后台进程，完整归档当前run和来源链，暂停自动跟进 | `outputs/b2_seed1_interrupted_20261030_102720/`；`python -u scripts/run_b2_seed1.py` | 37,695冻结输入复核通过；13个文件及脚本/配置来源已保留；标准输出/错误由用户终端记录 | 等待用户手动完成连续30epoch及自动评估后，继续独立验收、更新论文和本地Git提交；不得自动重启训练 |

| 2026-10-30 13:05（验收收尾） | `9295c3e` + 本次提交 | 验收用户终端完成的 B2 seed1 连续30epoch及完整评估，复用P1 seed1；完成两seed主对照统计、四类案例和论文/复现记录 | `outputs/b2_pilot30_s1/{comparison,acceptance_audit,independent_audit}.json`；表4-S4；实验4.12–4.13；`outputs/paper_cases_b2_p1_s0/` | seed1 P1−B2：mAP50-95 +0.2438 pp、small recall +1.6932 pp（+450匹配），large −0.6560 pp；两seed配对差 +0.2893±0.0644/+1.7854±0.1304 pp；116检查/37,695冻结文件/四组预测重放、案例1,106输入核验、论文表123来源及7组recall、pytest39/39均通过；中断35文件归档不入统计 | 总体/小目标两seed同向，large方向不同；单split/固定教师/B1未重复，不作显著性或AP_small主张；原Gate3不变；按授权本地Git提交、不push，清理已暂停心跳，不自动扩训 |

| 2026-10-30 13:21（研究安排） | `80e9f2b` + working tree | 按用户三篇文章、毕业成果组合和树莓派轻量化要求制定分阶段计划；论文录用/专利“申请通过”口径保留待校方核定 | `docs/publication_and_edge_deployment_plan.md`；第3.21节 | A方法、B补全、C端侧压缩三项独立问题；执行A→C→B；单独规划学位论文及专利备选；CPU/NPU、分辨率和完整图像延迟分开验收 | 仅规划，无新训练/硬件购买/投稿/专利申请；硬件、毕业期限及认可细则待补；原Gate3不变 |

| 2026-10-30 13:31（第一篇规划） | `main` + working tree | 按用户低成本优先要求审计第一篇缺口，完成冻结候选漏检归因与下一步排序 | `scripts/audit_paper1_gaps.py`；`outputs/paper1_gap_audit_20261030/`；`docs/paper1_experiment_gaps.md`；第3.22节 | 6471 tile/38245 hidden GT/35067正伪标签/25669匹配精确重放；8,984个未匹配小目标的描述性证据类别已分开记录；3/3新测试通过 | 不做因果外推；先L0候选覆盖检查，再决定是否需要用户执行5epoch筛选；尚无新训练，原Gate3不变 |

| 2026-10-30 13:43（ignore增量） | `80e9f2b` + working tree | 按用户指令只做ignore可恢复增量：锁定旧匹配、全尺度联合贪心与最大基数匹配，单列排除tile、重复与低阈值代价 | `scripts/audit_ignore_recovery.py`；`outputs/paper1_ignore_recovery_20261030/`；`docs/paper/ignore_recovery_analysis.md`；第3.23节 | 入训52117 ignore→普通新增5303/条件上界5315，其中普通新增small3627；全纳入新增匹配比例10.18%，small conf≥0.25为27.12%；11322来源哈希与完整6471tile重放 | 有几何恢复空间，尚无足够可靠的无GT转正规则，不下发扩训；隐藏GT只诊断、不是学生AP；先考虑小样本可靠性验证；原Gate3不变 |

| 2026-10-30（跨视图离线验证） | 本次提交 | 固定随机256 tile进行同教师水平翻转推理，重放少量候选的精度与覆盖 | `scripts/audit_cross_view_flip.py`；`docs/paper/cross_view_flip_analysis.md`；`outputs/cross_view_flip_20261030/` | 原入训196 tile；small conf≥0.25新增97框/30匹配，翻转conf≥0.25及IoU≥0.50新增63框/22匹配；新增匹配比例30.93%→34.92%，small匹配增量29→22；pytest 48/48 | 探索性小样本，组合精度低于原基线；不训练、不改变Gate 3，不下发训练命令 |

| 2026-10-30（候选保留检查） | 本次提交 | 128个独立抽样tile比较原NMS/max-det与分别放宽；保留初版和v2重放来源 | `scripts/audit_candidate_retention.py --check`；`outputs/candidate_retention_128_20261030_v2/`；第3.24节 | NMS放宽新增210正框/9独立small匹配，precision76.92%→59.07%；max-det无变化；未达到继续条件；52/52tests、359输入哈希/128tile重放、论文表123来源/7组recall通过 | 不训练、不新增训练标签；更新最多30epoch自主小试/临时监控/中文提交规则；原Gate3不变 |

| 2026-10-30（跨尺度确认与范围收束） | 本次提交 | 固定教师640视图支持960 uncertain候选；固定唯一规则后用不重叠256tile确认；更新总计划，当前只做A | `scripts/audit_scale_support.py --check`及`--confirm --check`；`docs/publication_and_edge_deployment_plan.md`第10节 | 探索22框/11small匹配，确认23框/9small匹配，未达继续线；55/55tests及两样本重放通过 | 不启动该规则训练；先A的必要证据和稿件，不并行B/C；Codex有界小试/临时监控/中文提交，用户执行有证据的大规模训练；A未完成，原Gate3不变 |

| 2026-10-30（A软权重补证据准备） | 本次提交 | 准备唯一一组去软权重30epoch正式消融、后台包装与独立验收，固定预算和继续条件 | `scripts/run_p1_no_soft_weights.py --check-only`；`scripts/audit_p1_no_soft_weights.py` | 23781冻结文件/11310tile预检通过；56/56tests；尚未启动 | 仅A，旧5epoch中性结论不变；启动后创建临时监控，不自动扩训，原Gate3不变 |

| 2026-10-30 14:30（A训练启动与文献调研） | `d218720` + 本次文档提交 | 实际启动唯一一组去软权重30epoch并配置临时监控；检索六类用户关键词，明确A单机制后续假设 | `outputs/p1_no_soft_weights_pilot30_s0/launch_receipt.json`；PID34612/包装37236；心跳a-30；`docs/paper/a_keyword_research.md` | 训练进程和CSV增长核实正常，尚无最终学生指标；7份PDF与哈希，5篇摘要级来源；本次仅改文档，沿用启动前56/56测试 | 等完整训练/val960/recall及独立验收后决定下一项；不并行新损失/基础模型/蒸馏剪枝，不推送，原Gate3不变 |

| 2026-10-30 16:37（A软权重验收收尾） | `dbff977` + 本次提交 | 完成去分类软权重30epoch及独立验收，新增自动表4-S5、实验4.14和来源记录 | `outputs/p1_no_soft_weights_pilot30_s0/{comparison,independent_audit,launch_receipt}.json`；第3.26节 | exit0；16项检查/23783冻结文件/两组548图预测重放通过；去软权重mAP50-95−0.1623pp、small recall+0.9143pp（+243），3类升7类降；137来源/8组recall，pytest56/56 | 体现取舍、未达后续线，保留完整P1主配置；不扩训、不push，临时监控验收后清理；下一项仅为定位可靠性待验证假设，原Gate3不变 |

| 2026-10-30（正伪框定位诊断协议） | 本次提交 | 固定已有正伪框跨尺度匹配、置信度/类别分层和两样本继续线，再开始读取本项GT统计 | `configs/positive_localization_diagnostic.yaml`；`docs/paper/positive_localization_analysis.md`；第3.28节 | 仅协议，无该项结果；两组各256tile缓存已存在且不重叠 | 下一步实现GT-free匹配与只读离线诊断；不改现有训练输入、不新增推理或训练，原Gate3不变 |

| 2026-10-30（正伪框定位诊断完成） | `2a42fa5` + 本次提交 | 实现GT-free一对一稳定性特征，复用缓存完成探索/确认，按预先继续线停止 | `scripts/audit_positive_localization.py --check`；`outputs/positive_localization_20261030/`；第3.28节 | 2710正框/1442来源重放，pytest61/61；small不稳定两组各6框，分层证据不足；稳定组严格定位质量比例52.7981%/49.4737% | 不增推理/训练，不改分组重试、不直接删除无匹配框；下一项需新信息来源，原Gate3不变；中文本地提交、不push |

| 2026-10-30（原图上下文纠错协议与Zotero归档） | 本次提交 | 固定128原图探索、过线后独立确认的单规则；实现GT-free映射/类别纠错及只读验收；按用户要求将Zotero分类调整为并列顶层 | `scripts/audit_context_rectification.py`；第3.29节；`docs/paper/zotero_literature_library.md` | pytest73/73；28论文均有PDF；尚无本项推理/纠错效果 | 先提交再推理；探索未过线则不做确认、不训练；当前仅A，不push，原Gate3不变 |

| 2026-10-30（原图上下文探索完成） | `f1c997e` + 本次提交 | 固定128原图上下文类别纠正探索、只读重放及失败确认阻断；修复历史数据源码误忽略 | `scripts/audit_context_rectification.py --check`；`docs/paper/context_rectification_results.md` | 98接受tile/799正框，仅改1框且改错；总匹配及small各−1；372来源一致，推理4.3874秒/训练0epoch；pytest74/74，Git归档源码齐全 | 继续线失败，不运行确认、不调阈值或训练；当前未找到值得扩训的新机制，原Gate3不变；中文提交、不push |

| 2026-10-30（置信度校准协议与代码） | 本次提交 | 固定开放区域拟合、来源不重叠探索/确认、全局简化对照和八项继续线；实现单调收缩校准/验收 | `scripts/audit_confidence_calibration.py`；第3.30节 | pytest84/84；尚未推理或拟合，不改既有训练输入 | 先提交再运行512开放tile推理；隐藏GT只验收，不绕过简化对照或启动训练 |

| 2026-10-30（置信度校准探索完成） | `f3d77cd` + 本次提交 | 512开放tile拟合、来源不重叠256tile探索、全局简化对照及失败确认阻断 | `scripts/audit_confidence_calibration.py --check`；第3.30节 | 拟合7548正框；探索1421正框；整体分组Brier0.160310高于原0.158864，small0.183221→0.176334；1045+474来源重放；84tests，表格137来源/8recall核验 | 整体继续线失败；不重拟合/改分组追试，不运行确认或学生训练，不发长期命令，原Gate3不变；中文提交、不push |

| 2026-10-30（小框NWD协议与工程实现） | 本次提交 | 固定开放GT像素常数、小框混合回归及单组5epoch缓存配对筛查；实现GPU检查、冻结输入、验收和后台包装 | `scripts/screen_p1_small_nwd.py --check-only`；第3.31节 | pytest94/94；尚未估计C/做GPU检查或训练 | 先提交协议再预检；预检通过才启动唯一5epoch并创建临时监控；不自动30/100epoch |

| 2026-10-30（小框NWD预检完成） | `626fa9e` + 本次提交 | 完成23794冻结输入预检和真实YOLO11s GPU AMP前反传；只用开放GT估计常数 | `python scripts/screen_p1_small_nwd.py --geometry-only` | C=20.22804306像素，1024开放tile/7519小框；255梯度张量有限，0优化器步；94tests | 工程可执行，不代表有效；下一步仅新增单组5epoch及临时监控，不自动扩训 |

| 2026-10-30 18:16（小框NWD启动） | `d1da33a` + 本次提交 | 实际启动唯一五epoch子集筛查，复用缓存P1；创建临时监控a-nwd | `outputs/screens/p1_small_nwd_1024_e5/launch_receipt.json`；包装32304/训练28860 | WMI隐藏启动返回0；命令/创建时间验证，CSV增长；18:19已完成2epoch、stderr为空；尚无最终val960/recall | 继续原进程，完成后独立验收；未判定效果，不自动扩训，当前仅A；结束清理监控，中文提交、不push |

| 2026-10-30 18:22（小框NWD验收） | `78b92ae` + 本次提交 | 完成唯一5epoch、val960/recall及独立验收，按固定继续线停止 | `python scripts/audit_small_nwd_screen.py`；`docs/paper/small_nwd_results.md` | exit0/连续5epoch/299.989秒；mAP50-95−0.1746pp、small+1.5464pp、large−0.7498pp；23794输入与两组548图重放 | 继续线失败；不改C/mix/阈值重试、不扩训；清理a-nwd，原Gate3不变；中文提交、不push |

| 2026-10-30（用户更新30epoch性能判断预算） | 本次提交 | 完成五轮缓存等FP/严格定位诊断；按至少30epoch要求准备唯一全量NWD确认，并接受中文提交后push授权 | `scripts/audit_small_nwd_tradeoff.py --check`；`scripts/run_p1_small_nwd.py --check-only`；第3.32节 | 两组548图/1111来源精确重放；等FP时IoU0.50小目标+0.6321/+0.7713pp，IoU0.90略降；108tests/Pyright零错误/论文表核验通过；全量30epoch尚未启动 | 五轮保留为历史探索，不再据此永久淘汰；完整P1缓存复用，仅新增30epoch，不自动扩训；启动后配置临时监控 |

| 2026-10-30 18:54（NWD全量30epoch启动） | `0d75739` + 本次进度提交，均push | 原C/mix/源码冻结，复用完整P1，实际启动唯一30epoch并创建临时监控a-nwd | `outputs/p1_small_nwd_pilot30_s0/{preflight,launch_receipt,status}.json`；第3.32节 | 23793来源/11310tile预检；包装30248/训练46428创建时间及命令核实；18:55第1轮batch221增长、stderr为空、CSV首行待首轮结束；尚无30轮性能结果 | 继续原进程、不重启/拼接/并行新候选；完整val960/recall和独立验收后判定，终态清理监控；中文提交并push |

| 2026-10-30（训练等待期间工作选择） | 本次安排提交并push | 用户最终允许选择暂停等待或核对论文，当前选择完成NWD文献定位，保留训练与a-nwd | `outputs/p1_small_nwd_pilot30_s0/launch_receipt.json`；临时监控a-nwd | 原训练46428存活，已核实进入第4/30轮；未新增候选或改冻结损失 | 不高频发送重复进度；终态后验收接续目标；若后续选择暂停且无法恢复状态，需如实提醒，不虚报恢复 |

| 2026-10-30（NWD文献定位补充） | 本次文档提交并push | 核对NWD原文式7–9/表3/表5及S³OD早期III-C，补入第17条参考及机制归因边界 | `paper/nwd_mechanism_positioning.md`；Zotero父条目Z8FC6KJQ/MDFG3JPK及PDF附件MCP核实 | NWD原文两种应用范围在12/24epoch的AP差距2.6→0.9pp，支持不能短程绝对淘汰但不保证本项目恢复；当前完整NWD30仍在训练 | 不改损失/配置/C/mix/继续线，不追加推理/训练；新候选不能冒充原NWD或SLA复现，不据此宣称独立算法创新 |

| 2026-10-30 19:19（选择暂停等待训练） | `70b6884`后本次安排提交并push | 文献核对完成后，按用户授权选择暂停持续目标，保留唯一训练与a-nwd，并更新监控终态接续职责 | `outputs/p1_small_nwd_pilot30_s0/launch_receipt.json`；临时监控a-nwd | 包装30248/训练46428身份匹配；CSV连续5轮、正在第6/30轮，stderr为空，未结束 | 原训练及监控继续；正常时静默，终态独立验收/记录/清理监控；若无法恢复目标active状态则通知用户点击继续，不声称自动恢复 |

| 2026-10-30 20:11–20:14（NWD全量30轮中断） | `6f9d4d4`后本次中文提交并push | 两次核实包装/训练进程消失，归档原始日志/权重、Windows应用挂起事件与SHA256；同步A进度及命令入口 | `outputs/p1_small_nwd_interrupted_20261030_201452/interruption_report.json`；第3.33节；`paper/small_nwd30_interruption.md` | 仅13/30完整轮，stdout停于14轮181/1414；原回执无exit_code、status仍training；事件1002身份匹配原包装进程，根因未明 | 本次未完成，不能判断性能或后续线；不重启/续训/拼接/扩训，处置后删除a-nwd；目标仍paused，需用户点击继续 |

| 2026-10-31（用户终端重跑交接） | 本次中文文档提交并push | 用户确认手动中断，核实无训练进程、13个文件哈希后归档原输出/run，准备原入口完整30轮重跑命令 | `outputs/p1_small_nwd_manual_handoff_20261031/handoff.json`；`python -u scripts/run_small_nwd30_background.py`；第3.34节 | 保留所有中断证据，配置/源码/协议不变；本次未启动训练，后续由用户执行并通知验收 | 至少完整30轮后判断性能，不续训/拼接、不自动扩训；等待用户完成 |

| 2026-10-31（NWD完整30轮验收） | `23c2656`后本次中文提交并push | 核实用户终端30轮、exit0与完整val960/recall；独立重放来源/产物与两组预测；同步实验4.15、独立结果表和证据/命令入口 | `python scripts/audit_p1_small_nwd.py`；`outputs/p1_small_nwd_pilot30_s0/independent_audit.json`；第3.35节 | accepted=true；mAP50-95−0.2123pp、small−0.6698pp（−178），类别4升6降；训练8092.09秒 | 两条继续线均失败，保留原P1，不自动扩训/调参追试；原Gate3不变，尚未找到可扩大训练方向 |

| 2026-10-31（分配几何诊断准备） | 本次中文提交并push，推理前冻结 | 原NWD负面确认后核实TAL扩框池与DFL支持域，准备原框有点时优先框内点的单因素缓存重分配诊断 | `paper/assignment_geometry_protocol.md`；`scripts/audit_assignment_geometry.py` | 111tests；新增文件Pyright零错误/警告；FCOS已有PDF补入顶层RegionSSOD，尚未新增数据推理或训练 | 固定256增强输入后重放，证据不足则不改阈值；通过也仅准备完整30轮，不能当作学生性能收益 |

| 2026-10-31（分配几何初版未过线） | `cb365b4`后本次中文提交并push | 完成256个增强输入及32批原始预测缓存，重分配减少框外监督但丢失小目标 | `outputs/assignment_geometry_20261031/analysis.json`；`paper/assignment_geometry_results.md` | small框外质量占比7.9472%/4.6675%；候选small覆盖98.1151%/98.8676%，未达99%保护条件 | 不训练初版、不改阈值；若研究覆盖保护须新协议及新样本，原确认缓存不可重用为独立确认 |

| 2026-10-31（覆盖保护规则准备） | 本次中文提交并push，首次新规则计算前冻结 | 初版失败后新增整图TAL退回保护，固定排除旧256后的新128基础tile确认 | `paper/assignment_coverage_protocol.md`；`scripts/audit_assignment_coverage.py`；独立assignment模块 | 114tests，新增三文件Pyright零错误/警告；尚无新规则数据诊断或学生性能结果 | 不降低原诊断线，新增最多128增强前向；过线后另登记30轮，不自动扩训 |

| 2026-10-31（覆盖保护新确认未过线） | `833cd83`后本次中文提交并push | 新128基础tile/2350正框确认，整图保护保留目标但作用空间未达线 | `outputs/assignment_coverage_20261031/analysis.json`；`paper/assignment_coverage_results.md` | 原有效目标保留100%，框外质量降低48.0992%<50%，16/128图退回原TAL；114tests/Pyright通过 | 不降低门槛、不继续换样本或训练这两项分配候选；继续寻找其他单因素，目标未完成 |

| 2026-10-31（回归置信度缓存诊断准备） | 本次中文提交并push，统计前冻结 | 停止分配路线后，考察原c对伪框回归监督的分配，不拟合/改标签 | `paper/regression_confidence_protocol.md`；`scripts/audit_regression_confidence.py` | 117tests、两个新文件Pyright零错误/警告；尚未计算该加权质量统计 | 零推理重放原缓存；诊断通过后才登记唯一30轮，不能当作学生性能或创新性证据 |

| 2026-10-31（回归置信度30轮准备） | 本次中文提交并push，训练前冻结 | 缓存诊断通过并重放，隔离box/DFL原c权重，准备唯一全量30轮与完整评估/验收 | `paper/regression_confidence30_protocol.md`；`scripts/run_p1_regression_confidence.py` | 缓存small加权IoU+0.0472/+0.0526，类别控制同向；122tests/五文件Pyright通过；未开始学生训练 | 提交后预检/AMP零步工程检查，通过才单次启动与临时监控；至少完整30轮后判断性能，不自动扩训 |

| 2026-10-31 12:31（回归置信度30轮启动） | `5f47671` + 本次启动记录，均push | 工程预检/真实AMP255梯度有限且0优化步后，WMI单次启动唯一30轮，创建临时监控a | `outputs/p1_regression_confidence_pilot30_s0/{preflight,engineering_check,launch_receipt,status}.json`；第3.42节 | 23807来源、11310tile；包装21156/训练10196身份核实，实际args配对；12:32第1轮179/1414，stderr空 | 只跟进原进程，完整30轮/val960/recall及独立验收后判断；不改源码/并行/自动扩训，终态清理监控 |

| 2026-10-31 12:37（回归置信度工程失败） | 本次中文提交并push | 首轮验证因误要求训练专用pseudo_weights退出，保存原始失败证据及哈希 | 第3.43节；`paper/regression_confidence30_failure.md`；`outputs/p1_regression_confidence_pilot30_s0/failure_report.json` | 原进程均停止、exit1/status failed，无CSV/权重；未完成30轮，不作性能判断 | 清理监控a；不自动重试，先补验证兼容与实际验证工程检查，再另行冻结用户运行入口 |

| 2026-10-31（验证兼容修复准备） | 本次中文提交并push后冻结 | 隔离eval原P1损失与train元数据保护，增加独立valfix配置/显式入口及实际验证工程检查 | 第3.44节；`paper/regression_confidence30_valfix_protocol.md` | 旧失败六文件源码快照已核对保存；124项单元测试通过，待执行新预检和实际验证工程检查 | 仅零优化器步骤检查；通过后交用户独立30轮命令，不自动重试，不据工程检查判断性能 |

| 2026-10-31（验证兼容真实工程验收） | `84817b2` + 本次中文结果记录，均push | 真实YOLO11s AMP前反传及标准2图训练期验证调用通过，原P1损失精确一致 | 第3.44节；`outputs/p1_regression_confidence_valfix_pilot30_s0/{preflight,engineering_check}.json` | 255梯度有限、0优化器步骤；124tests/Pyright通过；23809来源与原主表核对通过，无新训练 | 已交独立30轮用户命令；等完整训练后按固定线验收，未认定值得更大训练，持续目标未完成 |

| 2026-10-31（回归权重份额预登记） | 本次中文提交并push，首次统计前冻结 | 不推理不训练，只核对全量正伪框原c的尺度/类别组成及监督强度混淆 | 第3.45节；`paper/regression_confidence_mass_protocol.md` | 131tests及新增两文件Pyright通过，未启动学生或更改冻结训练变量 | 提交后首次统计并重放；仅框级代理，不据此调权或判断性能 |

| 2026-10-31（全量权重份额结果） | `245501b`预登记 + 本次中文结果提交，均push | 35067伪框只读统计及重放，复核RUPL/UTv2原文与Zotero附件 | 第3.45节；`paper/regression_confidence_mass_results.md` | small框数份额58.9329%→sum(c)份额51.4685%，car+4.9497pp；不是实际损失/梯度或性能结论 | 不改原c及继续线，不自动训练；等用户完整30轮后验收，目标尚未达成 |

| 2026-10-31（小训练启动责任更新） | 本次中文提交并push，启动前更新责任说明 | 用户明确30轮小训练由助手启动；发现13:28首轮中断，保留9文件/六源码快照后准备独立单次启动 | 第3.46节；`outputs/p1_regression_confidence_valfix_interrupted_20261031_133202/` | 原包装20616/训练1540已停止，KeyboardInterrupt，无exit_code/完整epoch，未改训练变量和继续线 | 提交后重新预检并启动唯一30轮、建临时监控；不续训/拼接/自动扩训，目标active未完成 |

| 2026-10-31 13:36（助手独立启动修正版30轮） | `12b6886` + 本次中文启动记录，均push | 归档用户主动中断后从COCO单次独立启动，建临时监控a | 第3.46节；`outputs/p1_regression_confidence_valfix_pilot30_s0/{os_dispatch,launch_receipt,preflight,status}.json` | 23809来源、实际args/元数据配对通过；包装22020/训练22512身份核实；13:37首轮346/1414、stderr空 | 只跟进该运行，完整30轮/val960/recall后独立验收；不自动扩训，目标active未完成 |

| 2026-10-31 13:41（首轮真实验证路径通过） | 本次中文记录并push | 核实完整epoch1及35批验证、CSV有限值，随后已进入epoch2 | 第3.46节；`outputs/p1_regression_confidence_valfix_pilot30_s0/first_epoch_validation_check.json` | 三项验证loss正常、stderr0字节，越过旧工程崩溃点；仅流程兼容证据 | 原监控继续静默跟进，至少完整30轮再判断性能，不改源码/科学变量或扩大预算 |

| 2026-10-31 15:48（修正版完整30轮验收） | 本次中文提交并push | 完成唯一valfix30轮、548原图val960/recall及独立预测重放，记录尺寸/类别/耗时取舍 | 第3.47节；`paper/regression_confidence30_results.md`；`outputs/p1_regression_confidence_valfix_pilot30_s0/independent_audit.json` | exit0、连续1–30轮、23809来源、131tests/原主表/新增15行数字核对；small+0.9444pp（+251）、AP−0.1755pp，类别1升9降，两条继续线均未达 | 不扩大本变体训练、不调权追试；中文提交push后清理监控a；持续目标保持paused，后续需用户点击继续 |

| 2026-10-31（覆盖保护30轮新探索授权） | `dbc8e76`中文提交并push | 按用户低门槛探索要求更新旧48.10%诊断的停止安排，准备唯一完整30轮；Zotero整理到Research/YOLO-RegionSSOD | 第3.48节；`paper/assignment_coverage30_protocol.md`；`scripts/run_p1_coverage_guard.py` | 原128tile诊断100%覆盖，未改规则/重采样；141tests/原表核对通过 | 原Gate3/旧实验结论不改 |

| 2026-10-31 16:07（覆盖保护30轮实际启动） | `dbc8e76` + 本次中文启动记录，均push | 23809来源预检、真实AMP及两图验证通过后助手单次启动；创建临时监控30，等待期核对FCOS/TOOD | 第3.48节；`outputs/p1_coverage_guard_pilot30_s0/{launch_receipt,startup_identity_check,engineering_check}.json` | 包装15432/训练30956身份与实际args配对；首轮batch增长，stderr空；255梯度有限、0工程优化步 | 跟进原训练，至少完整30轮后验收判断，终态清理监控；当前未找到已验证扩训方向 |

| 2026-10-31（等待期准备条件式100轮入口） | 本次中文提交并push | 小训练运行期间准备未来用户配对命令与独立验收，不更改冻结输入/源码 | 第3.49节；`scripts/run_coverage_guard_full100.py`；`paper/coverage_guard_full100_protocol.md` | 已测plan-only及未完成30轮时拒绝启动；原训练连续两轮验证正常，进入第三轮 | 100轮尚不可启动，必须等完整30轮正向验收；不作性能或目标完成判断 |

| 2026-10-31 16:24（按用户提醒暂停等待） | 本次中文记录并push | 选择暂停目标并停止额外只读观察会话49603，保留原训练及临时监控30 | 第3.50节；原launch_receipt及监控30 | 包装15432/训练30956仍存活；无训练源码或参数变更 | 终态验收/记录/清理后通知；工具不能自动恢复active，需如实提醒点击继续 |

| 2026-10-31 18:42（覆盖保护完整30轮验收） | 本次中文结果提交并push | 核实exit0、连续30轮、548原图val960/recall及独立预测重放，完成已授权终态收尾 | 第3.51节；paper/coverage_guard30_results.md；outputs/p1_coverage_guard_pilot30_s0/independent_audit.json | 23809来源，AP−0.0652pp、small+0.1016pp（+27），类别5升5降，未达较低探索线 | 不扩大此变体训练、不按val改规则；收尾后删除监控30，目标保持paused，后续需点击继续 |

| 2026-10-31（综合盘点与正式主线交接完成） | a65cc05及本次中文验收记录，均push | 盘点12组完整30轮，逐目标/误检/F1/IoU0.75重放，独立交接原B2/P1正式100轮 | 第3.52节；paper/training_investment_review_20261031.md；outputs/training_investment_20261031/handoff_preflight.json | 155tests；新增7行数据及原主表检查通过；两臂各37772来源预检exit0，无正式输出/新训练 | 交接已完成，正式两臂由用户顺序启动；不追加候选小训或自动大训，原历史判定及目标paused状态保留 |

| 2026-10-31（正式B2误中断恢复交接完成） | 49d114a及本次验收记录，均push | 核实63轮/last.pt及进程停止，保留原13文件，补显式检查点恢复和兼容验收 | 第3.53节；paper/paper1_full100_recovery.md；归档engineering/check.json及handoff_check.json | 真实恢复初始化255项优化器/EMA/scaler通过且0步骤；159tests及原主表通过；实际resume-check退出0，确认64→100轮，原证据未变 | 大训练剩余37轮由用户--resume启动；未自动续训，原exit_code未知仍保留，恢复事实单列 |

| 2026-11-01（正式B2 100轮验收） | 本次中文结果提交并push | 核实恢复工作进程已停/真实exit0，完整100轮/原63轮前缀、548原图val960/预测重放及来源审计；交接P1 | 第3.54节；paper/b2_full100_results.md；outputs/paper1_b2_formal100_s0/independent_audit.json | AP50-95=25.8264%、small=39.2858%；相对B1同轮数AP+0.9429pp；学生5.9305小时；P1预检37773来源通过 | B2已完成，P1正式100轮未启动，由用户运行；不扩seed/模块，不跨轮数混比，恢复事实及目标paused状态保留 |
| 2026-11-01（P1/B2正式100轮配对验收） | `e62e456`及本次中文收尾提交，均push | 双方完整100轮/548原图val960/实际args/冻结来源及预测重放通过；新增逐目标/误检/类别/操作点只读盘点及学生seed1隔离入口 | 第3.55节；paper/p1_b2_full100_results.md；outputs/paper1_full100_effects_s0/analysis.json | P1−B2：AP50-95+0.0528pp、small+2.1259pp（净+565），未匹配预测+1221、F1+0.1800pp，类别7升3降，学生耗时+13.49%；167tests、24表格行/557分析来源通过 | 值得用户补学生seed1两臂各100轮；实际两臂各37803来源预检exit0，无新训练/推理或输出；不延长seed0/叠模块，完成后优先收束 |
| 2026-11-01 17:47（B2学生seed1临时监控） | 本次中文提交并push | 核实用户既有B2训练包装/工作PID及创建时间/命令，创建15分钟临时心跳b2-1 | 第3.56节；outputs/paper1_b2_formal100_s1/monitor_initial_check_20261101.json | 包装10476/训练31640存活，第1轮批次增长、stderr为空；无完整epoch，不判效果 | 正常进度静默；终态独立验收、记录并提醒用户运行P1；失败不重试；收尾删除监控，不自动启动P1 |
| 2026-11-01 23:18（B2学生seed1正式100轮验收） | 本次中文提交并push | 唯一B2完整结束后独立验收，更新论文/证据及用户P1命令 | 第3.57节；paper/b2_seed1_full100_results.md；outputs/paper1_b2_formal100_s1/independent_audit.json | exit0、原两进程停止、连续100轮、37803来源/实际args/548图预测重放通过；AP50-95=25.6683%、small=39.3611%（10461）；P1只读37803来源预检exit0且无输出 | B2不重训；仅提醒用户运行同seed1 P1，预留7–8小时；终态删除b2-1，不自动启动训练或恢复目标；单臂不认定方法收益 |

| 2026-11-02 09:55（P1学生seed1临时监控） | 本次中文提交并push | 核实用户既有P1包装/训练进程身份、父子关系及首轮日志，创建15分钟临时心跳p1-1 | 第3.58节；outputs/paper1_p1_formal100_s1/monitor_initial_check_20261102.json | 包装4692/训练45248身份一致且存活；首轮197/1414批、stderr为空，尚无完整epoch | 正常训练静默；完整终态后独立验收同seed及两seed正式100轮结果并盘点下一步，提交push后删除监控；失败不重试，不自动扩训或恢复目标 |

| 2026-11-02 14:01（P1学生seed1恢复交接） | `e54eb61`及本次中文记录，均push | 核实VS Code渲染进程oom后训练在第56轮KeyboardInterrupt、原两进程停止，归档21文件；新增独立P1恢复/验收入口，剩余45轮由用户启动 | 第3.59节；paper/paper1_p1_seed1_recovery.md；outputs/paper1_p1_formal100_s1_recovery_20261102/{engineering_verified/check,handoff_check}.json | 连续1–55轮/13602.5秒、last.pt epoch54、原退出码未知；零批次零步骤验证P1及255优化器/EMA/AMP/调度恢复、原证据未变；179tests/原表/差异检查及实际37803来源check-only exit0通过 | 等用户从第56轮续到100轮；p1-1已更新并启用，等待resume回执；完成后独立验收同seed和两seed正式配对，不自动开训或扩训 |

| 2026-11-02（项目级科研技能安装） | 本次中文提交并push | 按用户指定仓库安装 ARS-Codex 3.24.0 到 `.agents/skills/academic-research-suite/`，固定上游 a1556557887f345aac99bdc6027e008e91fef236，保留来源与许可证，README补充使用入口 | `.agents/skills/academic-research-suite/{SKILL,INSTALLATION}.md` | 上游2657文件Git blob哈希全部匹配，项目179tests通过；仅完成安装，科研工作流未执行，无新训练或评估 | 后续显式使用 `$academic-research-suite`；既有实验预算、证据规则与论文路径继续适用 |

| 2026-11-02 17:35（P1 seed1恢复终态及两seed正式100轮收束） | 本次中文结果提交并push | 核实恢复包装/工作停止及真实exit0，独立验收原55轮前缀/连续100轮/37803来源/548图预测，重验B2及两seed配对；新增只读缓存取舍分析 | 第3.60节；paper/p1_b2_seed1_full100_results.md；outputs/paper1_full100_effects_s1/analysis.json及validation_report.md | seed1 AP50-95+0.3351pp、small+2.4194pp（净+643），未匹配预测+1435、F1+0.4031pp、成本+31.82%；两seed均值AP+0.1939pp/small+2.2726pp；179tests、新表数字/来源及原表检查通过 | 本轮主线训练收束，优先写作与已有机制分析；Gate3/4不追认为通过，必要论文缺口另行排序；中文提交push后删除p1-1，不自动扩训或恢复持续目标 |

| 2026-11-02（ARS plan：第一篇结构与证据盘点） | 本次中文规划记录，提交并push | 按用户“还是第一篇，你看下是不是可以写了”“要取长补短，尽量把我的成果放大”核对既有正文、正式两seed原始汇总及四臂验收记录；本轮完成规划，不生成全文或启动训练 | paper/p1_b2_seed1_full100_results.md；paper/experiments.md 4.21；outputs/paper1_p1_formal100_s1/two_seed_comparison.json | 重算配对AP50-95 +0.1939±0.1996pp、small recall +2.2726±0.2075pp；四臂accepted且各100轮；原表137来源/8组recall检查通过。引言及主表仍以30轮为中心，100轮结果已有但尚未整合为主叙事 | 可开始完整初稿：建议以正式100轮双seed覆盖收益为主结果，30轮消融作机制证据，保留误检/成本/类别退化及恢复限制；先重组正文、补讨论与结论，再优先补AP_small/完整PR评估。现有recall缓存仅conf≥0.25，不能直接充当完整AP/PR；官方口径另核。B0/B1第二seed、预算/新split与同协议相关方法比较按最终主张排序，不自动铺满矩阵；贡献定位仅为建议，未替作者确认新主张，Gate3/4不变，未评定具体期刊录用或投稿就绪 |

| 2026-11-02（AP_small/完整PR/等精度补评估） | `e3e20fb`、`988a89d`、`ddfebe8`及本次中文结果提交，均push | 按用户授权只评估四正式冻结权重，固定precision网格并补全部曲线/逐类；工程预检与绑定/类别轴修正保留原证据，新增文献MCP检索及Zotero归档 | 第3.61节；paper/small_pr_{protocol,results,literature}.md；outputs/paper1_small_pr_20261102/bound_v2/ | 190tests；1119输入哈希/四臂绑定/全部缓存重算通过，原mAP精确重放；AP_small−0.2460/+0.6129pp，P≥70%时small均值+0.7036pp、P≥90%为−0.2032pp；Zotero五篇归档，其中新增STOD、四篇有PDF | 完成本轮补评估，不新训练；正文保留精度区间、类别退化与cap/推理路径限制；三个新机制方向仅为待证假设，Gate3/4及原失败探索不追认通过；脚本/结果/复现入口中文提交push |

| 2026-11-02（H2零训练诊断及H1尺度操作点跨拟合） | `d943420`、`83b7435`、`db23e85`、`9464463`、`4a1ae9b`、`383b4af`、`7d3aa55`及本次结果文档中文提交，push | 从三方向选H2，先用四冻结权重缓存核对高精度逐GT增失、候选去向、FP组成、共同GT几何与Oracle容量；H2弱化后用5 split×2折校准/评估测试尺度条件阈值 | 第3.62节；paper/{high_precision_diagnostic_results,scale_operating_point_results}.md；outputs/paper1_high_precision_diagnostic_20261102/；outputs/paper1_scale_operating_point_20261102/ | 201tests；90%下711/752中有695/730完整缓存已匹配但低于阈值，共同GT几何未恶化；scale策略对两臂提高约4.4–5.4pp但P1独有优势仅seed1通过，总体继续线False | 完成诊断，零推理/零训练；不建议NWD/回归/逐边界训练，暂不给训练指令；将全局阈值/排序边界写入第一篇，未来需无GT质量信号—动作合同及至少30配对epoch；所有负向split、失败日志和审计保留 |

| 2026-11-02（下一改进点建议：类别歧义感知分类监督） | 本次中文规划记录提交并push | 用户询问如何扩大第一篇成果、是否换改进点；复核既有诊断和负向探索，建议从几何/操作点切换到训练期分类可靠性，先检查教师top1/top2或完整类别分布是否提供超出原最高分c的信息 | paper/high_precision_diagnostic_results.md；paper/scale_operating_point_results.md；src/regionssod/training/p1.py；Humble Teacher CVPR2021官方摘要 | IoU0.50/precision90%下P1错类FP占1535/1525个FP中的1036/1030（约67.5%），B2也近似，不能据此认定P1独有因果；原c缩放整个正anchor类别BCE向量，现有缓存只保存最高类别/分数，不能恢复类别歧义；软标签已有Humble Teacher先例 | 仅提出优先候选，未实现或启动推理/训练。先准备少量训练区域教师完整类别分数诊断，隐藏GT仅审计；若信号有效，保持原正伪框/回归/分配/网络等固定，设计一项分类监督变量的完整30轮配对并给用户执行指令；核对新颖性与监督强度混杂，不把原c换名或软标签本身宣称首创。现有稿件继续整理，新候选不能覆盖原结果 |

| 2026-11-02（类别歧义诊断与原P1叠加实现） | 本次中文提交并push | 继续用户已授权假设，冻结样本/门槛后诊断，通过后补齐原35067框全类别分数，独立分类目标实现和两臂30轮入口 | 第3.64节；paper/candidates/class_ambiguity/{diagnostic_results,student_protocol,validation}.md | 确认AUROC .7916→.8214、log-loss约−6.8%；6471tile/91336候选身份一致且最大框/分数差0；零学生优化步骤 | 交用户运行两臂各30轮，自动val960/AP_small/PR/等精度与审计；原P1及原Gate保留，不把教师诊断当学生增益、不自动扩训 |

| 2026-11-02 20:44（类别分数配对临时监控） | 本次中文记录并push | 用户已启动两臂各30轮，按要求建立每15分钟临时心跳p1；正常推进静默，完成后审计效果并删除 | outputs/p1_class_ambiguity_pair30_s0/monitor_initial_check_20261102.json | reference首轮242→381批持续增长，实际args与原配置一致；进程创建时间/父子链相符但CIM命令行为空，完整命令身份未核实 | 等两臂完整30轮和自动AP_small/PR评估，按冻结五项线判定；失败保留证据并报告，不自动重训/扩训；终态收尾删除p1 |

| 2026-11-03（类别分数配对30轮终态验收） | 本次中文提交并push | 两臂完整30轮/exit0、548图评估及独立数值重放通过，补全双IoU PR、逐类/成本与论文讨论 | 第3.65节；paper/candidates/class_ambiguity/results30.md；outputs/p1_class_ambiguity_pair30_s0/terminal_review/ | P90 small +0.8203pp，但AP−0.2631pp/AP_small−0.4162pp/P70−0.5418pp；五项线仅两项通过；原P1保留 | 不扩训/不重跑，保留正负结果及机制限制；收尾删除临时心跳p1，通知用户当前没有新训练指令 |

| 2026-11-03（原P1互补方向研究与竞争负项准备） | 本次中文提交并push | ARS+paper-search/Zotero；先固定协议，后重放教师/六学生缓存，独立实现与强度控制 | 第3.66节；paper/candidates/competitive_negative/；outputs/p1_competitive_negative_diagnostic_20261103/ | 真实top2占歧义错类94.67%/89.86%，但错误top2动作超过一半；30,565绑定/PR精确重放，预检通过，277tests | 完成探索诊断和三臂30轮工程准备，零训练/零新增真实图像推理；用户决定执行，失败停止不调val；原P1和所有负结果保留；独立区域split为优先补证据项 |

| 2026-11-03（用户明确基线/P1/改进主对照） | 本次中文提交并push | 增加新B2完整30轮，主三臂给绝对值及三项差值，强度控制另列可选第四组 | 第3.67节；paper/candidates/competitive_negative/main_comparison_protocol.md；run_p1_main_comparison.py | 实际旧三臂未启动；新B2冻结教师/27667伪框/11095tile绑定及零步骤预检；工程测试见validation | 准备完成，未训练；当前用新入口，保留旧合同与所有证据；不跨30/100轮叠加增益 |

| 2026-11-03 10:06（主三臂启动核对与监控） | 本次中文记录并push | 用户已启动后核对真实命令/父子链、39,828输入哈希及B2实际args，建立p1每15分钟临时心跳 | 第3.68节；outputs/p1_competitive_negative_main30_s0/monitor_initial_check_20261103.json | B2首轮持续增长；baseline/reference/candidate三臂，无控制组；临时监控ACTIVE | 正常静默，完整30轮及自动评估后独立audit并比较绝对值/三项增量；异常保留证据、不重跑；终态报告/中文提交push后删除p1 |

| 2026-11-03 14:08（主三臂工程失败收尾） | 本次中文记录并push | B2／P1连续30轮exit0；candidate首批硬正目标断言exit1，未进入统一评估；保留证据并收尾临时监控 | 第3.69节；paper/candidates/competitive_negative/main_comparison_failure_20261103.md；outputs/p1_competitive_negative_main30_s0/terminal_failure_20261103_140708_a233c4/ | 39,828冻结输入全部一致，17原文件快照一致，四权重哈希保存，无训练进程；candidate完整轮次0／无权重 | 工程失败不判性能；三臂差值和七护栏NOT_EVALUATED，机制NOT_TESTED；不自动修复／重跑／追加推理，文档commit并push后删除p1 |

| 2026-11-03（竞争负项首批定位） | 本次中文记录并push | 先协议，单批CUDA AMP零优化步骤复现，缓存dtype控制重放与回归测试 | 第3.70节；paper/candidates/competitive_negative/debug_results_20261103.md；outputs/p1_competitive_negative_debug_20261103_v1/ | 649伪前景无类别绑定错；1个TAL质量8.7088e-9转FP16为0触发过强断言，同预测原P1损失正常；286tests，参数及监测原文件未变 | 完成工程定位，非训练性能；原实现／合同和失败证据保留，不自动重训；后续独立修订需分开类别合法性和数值检查并保护strength正类 |

| 2026-11-03（AMP修复与B2／P1复用准备） | 本次中文记录并push | 独立V2数值修复、缓存／梯度／单批CUDA及只调度candidate的来源审计入口 | 第3.71节；paper/candidates/competitive_negative_v2/；outputs/p1_competitive_negative_v2_validation_20261103/ | 1988前景缓存通过；真实8图AMP通过，分配／box／DFL不变、正类改动0／优化0；315tests；复用预检通过 | 未启动训练；只准备新改进组30轮命令，原B2／P1权重及旧失败证据保留；完成后统一评估三方法与三项差值；七护栏不变、机制NOT_TESTED、不自动100轮 |

| 2026-11-03 14:38（V2改进组启动核对与监控） | 本次中文记录并push | 用户启动后核对真实进程／参数／39,864来源及复用权重，建立每15分钟临时p1-v2 | 第3.72节；outputs/p1_competitive_negative_main30_v2_s0/monitor_initial_check_20261103_143811_980833.json | candidate首批成功，1下溢锚点保持原P1、两项正类改动均0；第1轮持续推进；B2／P1未重训 | 正常静默，完整30轮及自动三权重评估后只读audit／比较绝对值和三项增量；失败保留不重试；终态报告／提交push后删除p1-v2 |

| 2026-11-03（V2三方法终态比较） | 本次中文记录并push | V2连续30轮exit0，复用本次B2／P1，统一548图val960与新日志只读audit通过 | 第3.73节；paper/candidates/competitive_negative_v2/{results30,terminal_validation}.md；outputs/p1_competitive_negative_main30_v2_s0/terminal_audit_20261103_171004_802028.log | P1−B2的AP+0.3349pp／small+2.2012pp；V2−P1 AP−0.1628pp／AP_small−0.2279pp／P90+0.3650pp／FP+7.8971%；七线仅2过，30份机制快照保护正类 | 停止V2，保留原P1及旧失败；无strength机制NOT_TESTED、同split／teacher／seed0开发val不外推；无新训练／推理，提交push后删除p1-v2；优先准备独立区域split证据 |

| 2026-11-03（V2失效原因与原P1瓶颈零训练诊断） | 本次中文提交并push | 先协议，三权重548图缓存／教师开放区域重放；同FP／K、FP／FN、共同GT几何、截断和GT-only容量探针 | 第3.74节；paper/diagnostics/competitive_negative_v2_bottleneck/；outputs/p1_v2_bottleneck_20261103/；diagnose_p1_v2_bottleneck.py --audit | conf.25 small+99，但同FP−137／同K−84；P1 P90 FN低分同类60.03%、高分FP错类69.91%；缓存／JSON／数组audit exit0，318tests；零训练／零新推理 | 保留原P1及旧失败；优先验证人工正锚点可信竞争负监督与等强度控制，仅计划、无训练入口；不认定唯一因果／新颖性，不调val、不建监控；独立区域split仍为补证据优先项 |

| 2026-11-03（人工GT可信竞争负监督工程实现与三臂准备） | 从干净bd1435d开始；本次中文提交并push | ARS先工程合同；独立来源token／候选2倍负项／同锚点等强度控制；新reference／candidate／control各30轮入口、自动统一评估与只读audit | 第3.75节；paper/candidates/reliable_competitor_negative/；scripts/{validate,run}_reliable_competitor_negative.py；outputs/reliable_competitor_negative_engineering/validation_final.json | 一个8图AMP batch仅一次forward；参数SHA不变、优化步0；TAL／box／DFL／正目标精确不变；1341人工锚点动作／497伪锚点保护／26零质量无动作；L1误差8.68e-8；完整806tests通过、真实预检39906来源exit0 | 工程通过，学生性能／机制NOT_TESTED；不训练／val推理／隐藏GT／新论文／空监控；默认新参考，冻结8性能+2机制线，任一失败停、不调val／重试／恢复／100轮；旧P1及V2失败保留；精确PowerShell与6–9小时预算见独立协议 |

| 2026-11-04（人工GT可信竞争负监督三臂终态） | 源码91cc57d；本次中文报告提交并push | 三新学生完整30轮exit0；统一548图val960、COCO/双IoU PR/固定FP分解，原进程停后只读audit通过 | 第3.76节；paper/candidates/reliable_competitor_negative/{results30,terminal_validation}.md；outputs/rcn30_terminal_audit_20261104_0235.log | 候选−P1 AP−0.3502pp／AP_small−0.1088pp／conf.25 small−3.1682pp；P90+1.1890pp／FP−22.5377%／错类FP−25.5983%；对控制AP−0.2034pp／AP_small+0.2063pp；5/10线通过，39906输入/57产物/数组/决定重放一致 | 性能4/8、机制1/2，按合同停止，原P1主线与旧V2失败保留；工程通过不等性能条件通过／机制建立；不调val／重试／恢复／扩训；只读缓存验收，无新增训练/推理/论文；提交push并删除rcn-30 |

| 2026-11-04（正目标几何重排研究与零训练准备） | 起点b1db01e；本次中文提交并push | ARS原文/作者代码核验、最多三方向取舍、先行协议后重放48批缓存；仅复用既有8图像素一次训练后forward；独立新loss/控制/三臂准备 | 第3.77节；paper/diagnostics/positive_quality_transport/；paper/candidates/positive_quality_transport/；scripts/run_positive_quality_transport.py --check-only | 实际AMP人工small逆序100/126、大差32/126、搬移8.7506%；框外质量增加反证保留；参考/TAL/boxDFL/保护项逐位一致，候选控制附加L1匹配；935tests通过；零学生训练、零optimizer/新loader/val推理 | 工程完成，性能NOT_TESTED/机制NOT_ESTABLISHED；新11线与三臂各30轮已冻结，预计6–12小时且开销未测；不覆盖旧失败，不自动训练/恢复/100轮，实际启动才建监控 |

| 2026-11-04 10:48（正目标三臂工程失败终态） | 训练源码060dc0f；本次只记状态并push | 用户10:45:44启动父21196，10:47:26完成预检并dispatch reference23212；10:47:57 worker exit1，父/worker均停止；临时监控捕获并执行一次只读audit | outputs/positive_quality_transport_three30_s0/ 原封印证据；外部pqt_failure_audit_20261104_1048.log及pqt_failure_terminal_receipt_20261104_1048.json | reference在val数据集构造调用复用的ReliableCompetitorTrainer.build_dataset，零参数super不接受PositiveQualityTrainer实例，TypeError；0完整epoch/无CSV或best；candidate/control未启动；audit进程exit0且terminal_evidence_verified=true，但accepted=false、performance=NOT_ACCEPTED | 这是Trainer集成工程失败，不是30轮性能失败或11门槛判定；先前缓存/935tests/预检未覆盖该真实Trainer验证路径。保持原源码/合同/失败目录，未自动修复/重试/恢复；需独立修复与新目录方案后再决定手动执行；临时观察终态删除，无源码修改故不重复pytest |

| 2026-11-04（V2独立Trainer集成修复与零训练复验） | 基点d6b95d1；本次中文提交并push | 在新模块中修正train/val派发，继承冻结loss/model，不改科学变量与11门槛；先保存工程协议，再用实际安装包三mode合成train/val及两份既有完整head缓存验证 | paper/candidates/positive_quality_transport_v2/；scripts/{validate,run}_positive_quality_transport_v2.py；outputs/positive_quality_transport_v2_engineering/ | 原TypeError反例复现，六合成batch正常、mode绑定正确，训练构造体AST等价；初始化/训练后缓存AMP完整loss/局部head梯度检查通过；完整1088tests通过；新真实batch/检测模型forward/optimizer/学生循环/真实val推理均0 | 仅集成修复；未验证完整Trainer setup/权重加载/AMP bootstrap/优化/性能/机制；旧失败封印、源码、合同保护，新预检加入旧失败绑定与合成派发证据；check-only exit0后由用户手动新三臂各30轮，不重跑旧入口/恢复/自动重试/扩100轮，不创建空监控 |

| 2026-11-05（正目标几何重排V2三臂负结果与ARS方法投入复盘） | 学生源码5966f58；本次中文复盘提交并push（提交号见git log） | 确认三新学生各完整30轮exit0/548图val960；原终态audit仅一次已通过；ARS先独立协议再重放封印预测，独立反方/源码机制/近邻文献核对 | 第3.78节；paper/diagnostics/positive_quality_negative_review/{protocol,results}.md；scripts/analyze_positive_quality_negative.py --audit；outputs/positive_quality_negative_review_20261105/ | 候选−P1 AP−.3145pp/AP_small−.3453pp/conf.25 small−.9444pp/P90+.5418pp；对控制AP−.1566pp/P90−.0301pp；同FP少160 small/同K少103；错类FP−285但重复+270；任去一类AP_small仍负；64项绑定/新统计精确重放/完整pytest通过 | 11线4过，停止原重排，不混V1零epoch工程失败；原P1/失败封印/冻结源码合同全部保护。跨对象可靠排序仅值得来源可行性审查：对象token不能证明安全背景，无法证明则停；多尺度/独立教师不优先，不调系数/筛类别/重试/恢复/100轮。新训练/forward/val推理/隐藏训练GT读取均0，无新监控；原临时v2已删除 |

| 2026-11-05（完整人工区域内跨对象排序A来源可行性审查） | 起点56e242f；本次中文审查提交并push（提交号见git log） | ARS先行协议；只投影合法人工清单/标签/manifest与构造参数，源码与安装增强审读、纯合成不可识别反例及独立反方核查 | 第3.79节；paper/diagnostics/reliable_region_ranking_feasibility/{protocol,results}.md；scripts/audit_reliable_region_ranking_feasibility.py --audit；outputs/reliable_region_ranking_feasibility_20261105/ | 6471人工tile/105058保留框与11310训练清单身份一致；面积比.09931134–.10027778、可见门槛.35；保留标签无法恢复被过滤足迹，固定侵蚀无背景保证；6484绑定/新审查精确重放、完整1093tests通过 | G1 PASS、G2/G3 NOT_ESTABLISHED，STOP_BEFORE_RANKING_DIAGNOSTICS，停止A；不代表实际负例不存在或新学生性能失败。不读隐藏标签/用隐藏计数补齐、不做排序统计/梯度/loss/forward/训练，不切换其他方向；原P1/失败封印/冻结源码合同保护，无新监控 |
| 2026-11-05（合法人工实例跨尺度观测B的ARS审查） | 起点708b45c；本次中文审查提交并push（提交号见git log） | 先行协议、源码/原文机制核对、仅合法metadata几何盘点与独立反方；不读取标签/像素 | 第3.80节；paper/diagnostics/human_multiscale_observation_review/{protocol,results}.md；outputs/human_multiscale_observation_review_20261105/ | 6471人工tile长边131–548，首次640下采样0；全图960/tile640目标线性比例.35543–.411；11项绑定/新几何字节一致重放；文档链接与diff检查 | STOP_GENERIC_B_NO_DISTINCT_SIGNAL；人工对应不同于旧可靠性诊断且可增约束，真实新上下文保留，但近邻差异/独立信号未成立，不升级loss/训练；不是性能负结果或跨尺度普遍无效。原P1/失败证据/冻结合同保护，无隐藏计数值提取/模型forward/val/训练/新监控 |
| 2026-11-05（原P1之后的ARS新机制方向筛选） | 起点aa1d621；本次中文研究提交并push（提交号见git log） | 先行协议，CABB/GFL原文定向核验、合法人工文本哈希/严格等式盘点、合成反例与独立反方三检查点；反方追加计算另存执行前声明 | 第3.81节；paper/diagnostics/next_mechanism_review/{protocol,adversarial_supplement,results}.md；outputs/next_mechanism_review_20261105/ | 公开保留事件给出输入标注框面积上界；同均值/同唯一众数仍可有不同联合相容质量，完整精确DFL分支代数一致；6471文件/105058框、4145严格触边/88横纵触边（非真实cut）；6483+4绑定/两结果字节一致重放；文档检查 | CONDITIONAL_ZERO_TRAINING_AUDIT_ONLY：只推荐有界定位分布约束的来源/工程审查，不认定原创性/性能或放行训练；强同界CABB点/模式/MAP及因子分布收缩解释保留。公开选择事件含模拟生成GT衍生侧信息，未新读隐藏GT/计数或manifest；无模型/训练/val/旧audit/新监控，未改源码/原P1/失败证据/冻结合同；正式仅文档提交 |

### 10.1 新会话开始流程

2026-10-30原图上下文诊断执行前的Git预检发现，历史`.gitignore`的`data/`
也屏蔽了`src/regionssod/data`四个源码文件，导致代码无法从Git完整复现。
本次将生成数据忽略限定为根目录`/data/`，保留原源码内容并补入版本控制；
新增忽略规则回归测试。两次预检均在GPU/输出创建前拒绝执行，未消耗推理或训练预算。
此项是复现缺陷修复，不更改split、裁片算法或已冻结的输入文件。

1. 进入 `D:\Python\YOLO-RegionSSOD`，运行 `git status --short --branch`。
2. 按顺序阅读：`AGENTS.md`、本文件、`docs/experiment_plan_8weeks.md`、
   `docs/baseline_protocol.md`、当前周对应的配置和日志。
3. 核对 `data/`、`runs/` 是否只是本地忽略产物，不能从 Git 状态推断实验已经完成。
4. 核对实际权重、`experiment.json`、`results.csv`、`metrics.json` 后再引用指标。
5. 修改前确认当前脏文件是否来自用户或其他会话；不要回滚未授权的改动。
6. 结束时更新本文件的进度日志，并在最终回复中明确“已完成/未完成/仅烟测”。

### 10.2 结果引用规则

- `data/`、`runs/`、`weights/` 被 `.gitignore` 排除，不代表它们不存在。
- 20 图、1 epoch、320 分辨率的数值只能作为链路证据。
- 只有固定 split、固定配置、完整 val 和可追溯权重的结果才能进入论文表。
- Ultralytics mAP 与 VisDrone 官方 AP 不能无说明混用。
- 任何外部确认集的结论都必须先完成版本审计、类别映射和固定评估命令。

| 2026-11-02（现有成果保护与第一篇整理） | 本次中文提交并push；原基点`c1ecf6c` | 使用ARS结构/写作合同、paper-search与Zotero MCP整理现有证据；增加独立成果快照、代码分轨、100轮正文与讨论/结论；候选按原P1叠加单一分类变量设计 | 第3.63节；paper/{README,paper1_writeup_plan,research_tracks,preservation,literature_map}.md；scripts/preserve_paper1.py；outputs/paper1_preservation/20261102_baseline | 58,707产物文件约4.20 GiB，归档及原件逐项哈希一致；核心文献已在BMQTT9Y7集合；新候选未实现/训练，未创造新效果数字 | 保留旧P1为独立主线；先教师全类别分数诊断再决定候选训练；当前无训练命令，Gate3/4不变；验证记录随本次提交保存 |

| 2026-11-05（有界保留事件监督独立实现与零训练终态） | 起点2dcece9；本次中文实现/证据提交并push（提交号见git log） | ARS先工程/科学合同，合法来源外集、增强实例/累计clip运输、同集合TAL/CIoU/DFL/MAP、正确Trainer集成；扩大真实增强流及独立反方复核 | 第3.82节；paper/candidates/bounded_retention/results.md；outputs/bounded_retention_engineering_20261105/；audit_bounded_retention_terminal.py --audit | 105058原行→105057native对象/1重复alias；02真实首批AMP有限；03的U名义负距另立native-event协议修订；v2第二batch精确top636.8081高于8400网格最大y636，7候选全空；25绑定/24快照/终态字节重放通过，完整1285tests通过 | STOP_BEFORE_STUDENT_TRAINING：工程覆盖门失败，三臂0epoch/无性能评估，最终学生配置未冻结；不改E/丢对象/clamp/新head绕过，不读取隐藏训练GT，不自动重试；原P1/失败/两版本合同源码快照保护；README/baseline同步，无训练监控需清理 |

| 2026-11-05（原P1编码语义复核与V3字面U方向停止） | 起点f87cf6f；本次中文提交并push（提交号见git log） | ARS核对原生DFL/CIoU，先冻结新编码合同；独立v3移除额外E可表示性门，31专项/真实扩大增强流/反方及编码union审查 | 第3.83节；paper/candidates/bounded_retention/p1_encoding_results.md；outputs/bounded_retention_p1_encoding_review_20261105/；audit_bounded_retention_p1_encoding.py --audit | 旧E坐标不改且7候选均恢复支撑；前12batch通过，第13batch U-bottom约[637.5464,640]无字面原子/32候选空，E均合法；连续outer见证及DFL均值可行；1316tests，31绑定/28快照/终态字节重放通过 | 停止当前离散质量方向，三臂0epoch；纠正旧E门过强而非回写旧STOP；编码union有信息损失/强物理点控制未认证，不以换事件强行开训；不否定保留信息或连续定位。无隐藏GT/旧质量缓存/模型batch推理/训练监控，原P1和两旧版及新V3证据保护 |

### 10.3 连续候选本轮执行记录

| 日期 | 版本 | 工作 | 证据路径 | 验证 | 状态 |
|---|---|---|---|---|---|
| 2026-11-05（连续密度合同先行冻结） | 起点92aa1b8；本次中文提交并push | ARS独立反方和来源几何审查选择normalized log-linear连续密度，最强同界全局point-DFL控制，进入新独立实现 | 第3.84节；continuous_review_contract.md；continuous_density_contract.md；continuous_numeric_addendum.md | 定义合理性不等于工程通过；全known原P1保留，U单点密度极限不另作否决；测度/积分/全局控制及真实集成仍需验证 | 学生0epoch、无val推理；通过全门才直接顺序3×30，旧源码/合同/失败不改，不读隐藏GT |
| 2026-11-05（连续数值容量与采样合同） | 延续93ee7c9 | 独立完整loss与旧E/U反例通过；正式完整3U矩形积分容量检查失败，保持目标另冻结后验采样计算合同 | 第3.85节；continuous_capacity_protocol.md；continuous_sampling_contract.md；curved_3u_capacity_01源码快照/receipt | 全局point gap0；19999cells/111.35秒积分残差0.4247；不把附加确定积分要求写成连续机制否决；有理见证/float几何表示问题单独修正 | V5待独立实现/验证，真实Trainer/增强/AMP及全库检查未过，三臂未启动；无隐藏GT、无验证推理、无新训练监控 |
| 2026-11-05（V5连续候选独立实现） | 延续77f228f；本次中文提交并push | 连续域/解析归一化/完整曲面后验采样/最强连续点DFL/独立Trainer实现；冻结E观测语义及双阶段真实检查清单 | 第3.86节；continuous_sampling_validation.md；continuous_observation_semantics.md；posterior_capacity_01/；suite_01/ | 4例3U容量通过/9.80秒；全库1715 passed/133.38秒；旧E/U反例不改坐标通过；独立审查缺口已修正 | 真实集成待执行，学生0epoch，无val推理；全门通过后直接顺序3×30；允许有依据的既有方法适配贡献，不把全新理论证明设为训练门槛 |

| 2026-11-05（V5真实失败与独立V6修订） | 5780870实现；66f3bfc新合同；本次中文实现提交并push | V5实际初始化/loader与20batch通过，第21batch暴露额外E-canvas门；冻结后独立V6修正域/积分，不改E | 第3.87节；native_e_revision_contract.md；posterior_integration_01/；v5_stop_audit.json；suite_v6_01/ | 51输入/48快照一致；反例正测度见证；V6完整1862 tests通过/98.78秒 | 真实V6待执行，学生0epoch，无val推理；旧代码/失败不改；允许已有方法合理适配贡献，全门通过后顺序3×30 |

| 2026-11-05（完整V6增强流及AMP门拆分） | a877b4b实现；1648327先冻结AMP合同 | 开启/关闭mosaic各1414batch通过，实际1次前向/7项head审计后原P1缩放梯度合并门停止；准备独立AMP重放和只读训练/评估工具 | 第3.88节；native_e_integration_01/result.json；amp_audit_revision_contract.md | 2744.809秒；原压力batch=closed1316/3U；未完成全模型有限反向 | 三臂未启动，0optimizer/0val推理；不改V6目标，不把初始AMP溢出直接视为机制失败，独立诊断待执行 |

| 2026-11-05（V6联合工程通过及三臂冻结） | f4a5346；本次中文提交并push | 一次真实AMP前向、固定序列共同scale1024、完整参数反向/原P1等价/同控制区分通过；保留旧STOP及路径核验失败，独立启动V2完成最终绑定 | 第3.89节；continuous_amp_results.md；three_arm30_contract.md；three_arm30_launch_addendum.md；native_amp_audit_01/；suite_launch_03/；freeze_20261105_v2/ | 2042 passed/126.04秒；24004输入只读核验通过；0 optimizer/0 val推理 | 工程放行、合同冻结；即将直接顺序原P1/候选/强控制各30轮，实际启动另记；无性能或机制成立结论 |

| 2026-11-05（实际顺序三臂启动） | 5f96181已push | 冻结输入后直接启动原P1→V6候选→同界最强连续点控制，各30轮；设临时只读进程监控 | 第3.90节；freeze_20261105_v2/launch_identity.json；continuous_three30_s0/reference_process.json；monitors/continuous_three30_s0/health.json（完整路径见正文） | 调度PID30104/P1 PID45600/监控PID18428；实际观察到P1第1轮batch | 运行中；后两臂未派发，三臂未完成，不作性能判断；禁止重跑/恢复/并行/自动杀worker |

| 2026-11-06（连续集合三臂工程失败收尾） | 起点122174e；本次中文报告提交并push | 候选第10轮共享全局点求解触发固定容量上限；只读终态/参考单臂审计、成本复盘与监控清理 | 第3.91节；paper/candidates/bounded_retention/three_arm30_terminal_failure.md；outputs/bounded_retention_continuous_terminal_20261106/ | 原P1 30轮exit0/1.922h，候选9轮后exit1/20.706h，控制未启动；24004输入/61封印产物通过，audit accepted=false/terminal_evidence_verified=true；两臂args一致 | STOP_ENGINEERING_POINT_RESOURCE_LIMIT；无最终548图val960，无性能/机制结论；不重试/恢复/提高限额，旧源码/合同/证据保护，进程自然结束，临时监控脚本已删除 |

| 2026-11-06（容量修复零训练与成本审查终态） | c24603d/f47d205等先冻结合同；本次中文实现与结果提交并push | 新独立同轴联合上界/必要源投影/分支提示；保留全部失败和未知源码回退；修复诊断监控后完成固定32批 | 第3.92节；paper/candidates/bounded_retention/point_capacity_review_results.md；outputs/bounded_retention_point_review_20261106/{synthetic_03,suite_04,real_03}/ | 8例最终压力通过；2087 tests；32批五臂全loss/全参数反向与状态恢复通过；完整loss旧/新0.9465，新耗时+5.65%；描述性三臂外推64.63h | 局部同类容量反例修复，整体2倍提速门失败；真实窗口无3U/后期覆盖，不能保证原故障消除；停止本轮扩展，不接入生产、不提高限额、不启动训练；临时诊断父/子进程均退出，无计划任务 |

| 2026-11-06（真实3U与第9轮EMA零训练压力门停止） | 943c9e9/65426a5/257052a先冻结合同及复核/重放补充；本次中文实现与结果提交push | 独立压力入口/只读资格checker；修复回调导入RNG造成的历史重放偏差，原失败保留 | 第3.93节；paper/candidates/bounded_retention/point_pressure_results.md；outputs/bounded_retention_pressure_20261106/ | 完整2828批及固定压力批身份/张量SHA一致；P1同批loss0.2904s/反向4.7054s，梯度有限；候选loss阶段120s超时；2137tests通过；374绑定0变化 | STOP_INITIAL_3U_LOSS_DEADLINE；第9轮case/32批未执行，0完整case，独立qualification STOP；不能区分point/积分/采样/CUDA等待，不能认定机制无效；未接入新入口/冻结三臂/训练，父子诊断进程已退出，无计划任务，旧证据保护 |

| 2026-11-06（120秒超时定位与合同审查） | 7e9686c/8612dd1/ccf2134/b45f95e先记录合同/包装修订/执行错误；本次中文实现与结果提交push | 保存预测、原P1逐位校准、同界候选/控制顺序trace；所有包装失败及误启动保留 | 第3.94节；paper/candidates/bounded_retention/loss_timeout_diagnostic_results.md；outputs/bounded_retention_timeout_20261106/ | 2147tests；两臂2590点输入/证书相同；3U2233锚点约101–104s/max15cells，控制loss108.391s；候选120s截断于第7次采样 | DIAGNOSTIC_ONLY：共享CPU精确几何为主成本，非点cap/显式CUDA主导；120s额外整批门很紧但不放宽，候选完成时间未知，无训练资格/开训，无新模型前向/反向/optimizer；旧源码和全部失败保留 |

## 11. 风险与退出条件

| 风险 | 触发信号 | 处理 |
|---|---|---|
| B2 无收益 | B2 pilot 低于 B1 | 检查隐藏 GT 密度、教师 recall、阈值和伪标签覆盖；不继续加模块 |
| P1 只增加伪标签数量 | 框数增加但 precision 下降 | 报告 precision/coverage，按尺寸和类别拆分，不宣称质量提升 |
| 主方法只在 10% 有效 | 5%/20% 无同向收益 | 如实报告预算敏感性，检查协议差异，不扩大结论 |
| 显存不足 | 640/batch 8 OOM | 所有对照统一降低 batch；不要只给主方法加梯度累积 |
| 指标不可比 | Ultralytics 与官方 AP 不一致 | 明确评测工具；正式表尽量用官方工具，保留差异说明 |
| 时间不够 | 第 6 周仍未过 Gate 4 | 停止扩训，转诊断论文或缩小到单一机制，不继续堆模块 |

## 12. 关键文件地图

| 文件 | 用途 |
|---|---|
| `README.md` | 环境、快速开始和 baseline 命令 |
| `AGENTS.md` | 后续会话的入口规则和当前范围 |
| `docs/experiment_plan_8weeks.md` | 科学问题、固定协议和完整实验设计 |
| `docs/baseline_protocol.md` | B0/B1/B2 的可执行命令和审计规则 |
| `docs/paper/README.md` | 方法、实验分析初稿及论文表格的写作入口 |
| `configs/train_full.yaml` | B0 完整监督配置 |
| `configs/train_partial_05.yaml` | 5% 部分监督配置 |
| `configs/train_partial_10.yaml` | 10% 主实验配置 |
| `configs/train_partial_20.yaml` | 20% 部分监督配置 |
| `configs/train_naive_10.yaml` | B2 自训练配置 |
| `configs/pseudo_p1_10.yaml`、`configs/train_p1_10.yaml` | P1 阈值与 30 epoch pilot 配置 |
| `src/regionssod/data/partial.py` | tile、隐藏 GT 和 manifest 生成 |
| `src/regionssod/pseudo/generator.py` | 伪标签生成和列表合并 |
| `src/regionssod/training/runner.py` | 训练、seed 和元数据记录 |
| `src/regionssod/training/p1.py` | 元信息增强传递、置信度分类权重和 uncertain-ignore 损失 |
| `src/regionssod/pseudo/scale.py` | 按原 tile 面积分层、候选/置信度/ignore 导出 |
| `src/regionssod/evaluation/runner.py` | 完整原图 val 和 `metrics.json` |
| `src/regionssod/evaluation/size_recall.py` | 固定操作点完整 val 分尺寸 recall，非官方 AP_small |
| `scripts/run_p1_pilot.py` | 本机已冻结输入的 pilot 训练/验证/Gate 3 顺序执行脚本 |
| `scripts/watch_p1.py` | 仅监听本机的实时进度页，3 秒刷新；不控制训练 |
| `scripts/build_paper_tables.py` | 从冻结评估产物生成论文结果表，`--check` 核对表格数字 |
| `tests/` | 数据、配置、伪标签和路径回归测试 |

## 13. 非本论文内容

`docs/RGB_IR_Reliability_Experiment_Design.md` 记录的是 M3FD/LLVIP 的 RGB-IR
局部可靠性融合方案。它目前不是本论文的实验部分，也不应该在本仓库中复用
VisDrone split 或 RegionSSOD 权重。未来若要启动该方向，应另建独立仓库和独立
数据/结果目录。
