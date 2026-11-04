# 正目标几何重排：实际工程验证记录

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate
- Origin Date: 2026-11-04，Asia/Shanghai
- Version Label: positive_quality_transport_v1
- Verification Status: 工程通过；学生性能NOT_TESTED，机制NOT_ESTABLISHED

## 代码差异与冻结保护

新增`src/regionssod/training/positive_quality_transport.py`，只包装原P1的assigner和BCE。
分配输出不改，分类BCE单独接收新目标；原分母、box/DFL继续使用原TAL张量。
显式来源运输直接复用冻结SourceDataset/validate_batch_provenance，增强算法未复制或改写。
人工按本图增强local GT index分组，重复mosaic token不跨实例合并。
旧P1、旧类别/V2/人工竞争负监督及回归/分配源码、协议和产物无修改。
新三臂配置仅名称/positive_quality_transport.mode不同，其余与原P1逐项一致。

新增三个诊断/工程脚本、一个独立三臂runner及三套测试；脚本来源清单绑定实际包和新源码。
未启动训练；没有新学生checkpoint/optimizer，未创建空监控。

## 预算与来源

先核对已有缓存，48批训练后张量做CPU重放。为补人工来源缺口，
按独立先行协议仅用已有8图缓存像素做一次冻结原P1训练后前向。
新DataLoader batch0，新模型forward1，BN eval且参数/buffer state哈希相同；
optimizer步骤0，学生训练循环0，val新增推理0。后续所有工程使用保存张量，不再前向。

实际工程重放两个完整cache：旧COCO初始化batch SHA
`8b239178522f0aa176195d5e7e781d0139efc4dffa7a070c4230ddcf355c4db9`；
新训练后同batch SHA `d4b7769984655f7c58f41ff33b4f5aac305d8a518e5bc6e9524c864c7b5633bd`。
训练后权重为原P1 30轮best；便利batch不代表全部训练数据或独立确认。

## 完整loss/实际CUDA AMP结果

实际运行环境：Python yolo Conda、torch2.10.0+cu128、Ultralytics8.4.172，RTX4070Ti。
完整P1和新reference/candidate/control均在相同缓存预测和batch上运行，
autograd只针对保存的score/distribution head张量，不通过真实模型参数。

| 检查 | 两cache结果 |
|---|---|
| reference完整损失/score与distribution梯度 | 与原P1逐位一致 |
| 三臂原TAL输出/分配对象/目标框/分母 | 原样/逐位一致 |
| box/DFL及distribution head梯度 | 与原P1逐位一致 |
| 所有伪来源/负类/未触发位置目标与score梯度 | 与原P1逐位一致 |
| candidate逐对象真类目标值多重集合 | 精确一致，无新增值 |
| 三臂有效目标支持 | 精确一致，零目标/FP16下溢保持零 |
| control逐对象总质量/值域 | 在声明float32容差内；不保多重集合 |
| control逐对象目标搬移L1最大绝对误差 | 1.78814e-7 |
| 实际FP16附加head梯度L1最大绝对误差 | 9.53674e-7（检查rtol=.02/atol=2e-6） |
| eval复用训练criterion但无来源sidecar | 显式回原P1，loss逐位一致 |

实际训练后人工small 126个有效多锚点对象中100个有逆序、32个大差逆序，
small搬移8.7506%，通过先定工程投入线。旧FP32代理为33个大差逆序，二者不混写。
工程统计只列有效对象129，来源完整人工small是131。

| 训练后同batch | candidate | control |
|---|---:|---:|
| 目标搬移L1 | 111.572260618 | 111.572259753 |
| 最大同对象有符号质量误差 | 0 | 3.57628e-7 |
| 改变真类目标的锚点数 | 866 | 1167 |
| box loss | 1.38024759 | 1.38024759 |
| cls loss | 1.18240392 | 1.15839589 |
| DFL loss | 1.05830300 | 1.05830300 |

原P1/reference cls loss同为1.09147084。候选/控制改变分类损失与最终梯度，
不能把相同目标总量/附加梯度L1解释成总BCE/参数梯度相同。
人工small框外质量35.3208→38.6970的反证保留，未加仅框内/CIoU规则追认。

## 合成与来源工程覆盖

本轮新测试覆盖：同对象multiset/总量/支持、控制搬移及附加BCE梯度、reference/无动作，
conf=1伪对象保护、重复token的local GT绑定、单锚点/平IoU/零目标/FP16下溢，
极端finite logit±100、NaN/Inf/越界质量、CUDA AMP有限梯度、确定性且不消耗训练RNG。
三臂runner专项104项覆盖11条门槛独立失败、不可达/非有限/缺失点、FP全类型、
已有目录拒绝、缺证据/来源漂移、worker凭据/前臂条件及只读失败终态边界等。
旧来源运输的mosaic/裁剪/关闭mosaic等价性证据与拒绝非法来源测试继续有效；
本轮没有为重复增强实测再加载真实batch，不能称每一种增强都已重新实测。

控制初稿未匹配扰动L1，回执独立保存到
`outputs/positive_quality_transport_engineering_unmatched_control/`，未进入任何学生实验。
修订时曾发现FP16 no-op分支赋到float32 control目标的dtype错误，已在缓存重放修复，
无新增forward。辅助几何占位统计也已替换为实际anchor/CIoU，不引用初稿占位值。
最终控制使用稳定凸组合，避免FP32中`1+(1e-30-1)`抹掉有效小目标；active端点
约束只消除float32舍入越界，不改零支持。新增极端值反例后重新完整loss重放与全套测试，
前一有效回执另存`outputs/positive_quality_transport_engineering_before_stable_convex/`。
这些是准备阶段工程问题，不是学生训练失败或学生性能证据。

独立原生反方三次检查：初次要求补实际AMP dtype路径与扰动控制，补齐后认可交付
手动三臂探索；最后纠正TOOD归一化数值/锚点位置区别及“全部覆盖不退”的过强表述。
本地多角色审阅不是跨模型独立验证，不升级为科学有效性认证。

## 完成检查及限制

完整 `python -m pytest`：**935 passed in 17.46s**；专项104项通过。
两诊断独立`--audit`均exit0/accepted=true；完整loss验证accepted=true，
`checks_actual_amp_path=true`。最终新三臂`--check-only`真实exit0/accepted=true，
核验40,242项绑定、11310训练图、105058人工框/35067伪框/52117ignore、548图val人口，
返回training=false/forward=false/created_files=false。检查时四个新训练输出/run目录均不存在。
预检不建模型/数据集，不创建训练目录；实际训练→统一评估→终态审计仅已准备，
没有真实执行，不能声称编排端到端训练已验证。

学生性能、语义机制、跨split稳定性、实际训练开销均未验证。
局部logit/目标匹配不能代替参数梯度/共享特征/后续TAL动态证据；
控制目标直方图与作用锚点数不同，存在残余混杂。
回执字段`signed_gradient_sum_max_error`实际测量的是目标值变化和的误差，
支持固定系数下梯度和的数学推论，不是直接测量FP16实际autograd梯度和。
distribution梯度精确仅指保存的head输出张量，未反传模型参数或共享特征。
控制的凸插值软化/反序方向及未来独立学生状态漂移仍是可能解释，
三臂只能检验实用收益和必要控制，不能充分证明纯几何因果。
不能把工程通过、缓存质量或按定义提高的加权IoU写成方法有效或论文贡献成立。
