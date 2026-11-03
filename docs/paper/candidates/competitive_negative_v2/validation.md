# AMP修订与复用调度验证（2026-11-03）

后续终态：V2完整30轮及统一三方法评估已经完成，见[结果](results30.md)和
[独立终态验证](terminal_validation.md)。以下保留执行前工程验证范围，不替代终态记录。

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate
- Verification Status: VERIFIED（工程范围）；学生性能UNVERIFIED
- Version Label: competitive_negative_amp_safe_v2_reuse30

## 修复及保留

新实现`src/regionssod/training/competitive_negative_v2.py`、独立配置和新入口。
原P1、完整类别分数变体、旧竞争负项源码／测试／配置／冻结合同均未改。
原失败run／日志／回执／17份快照与4份已完成权重保留；复用入口重新核对原输入及来源。
未启动任何正式训练或推理评估，未读取隐藏GT、未在val选参数。

修复不改变原P1的BCE目标dtype转换、目标张量、分母、TAL、box／DFL或正类权重。
类别绑定从原始分配类别和对象ID检查，原目标只允许单类非零／零质量，不接受软标签；
实际BCE质量为0时V2动作权重为1。strength用分配类别保护正类。

## 验证证据

| 检查 | 结果 |
|---|---|
| 已捕获前景缓存 | 1988前景／649伪前景；1个下溢目标正常保留 |
| semantic与strength缓存重放 | 均通过；各19个active锚点，正常非零目标与旧实现权重逐元素一致 |
| 真实CUDA AMP单批（8图） | 旧实现同错误；V2正常、loss有限，1个下溢锚点保持原P1处理 |
| 同预测与原P1比较 | 分配逐张量一致，box／DFL完全一致 |
| 正目标与分配正类权重变化 | 均0，目标张量不变 |
| 优化／反向／参数变化 | 0步／0次／参数哈希一致 |
| 来源及环境 | 原COCO初始化、实际增强与同种子；torch2.10.0+cu128／Ultralytics8.4.172 |
| 全套pytest | **315 passed in 8.78s** |
| 实际只读复用预检 | 通过；仅candidate列入train_arms，原B2／P1各连续1–30轮且exit0、权重绑定通过 |

单批和缓存均为已用开发数据，只是工程验证；历史失败首批没有张量，未建立逐位身份。
setup加载既有val元数据／框架AMP自检，不迭代val评估，不产出学生性能数字。
新增测试覆盖：零质量／FP16下溢、正常权重等价、真错类／软目标仍拒绝、
负梯度L1强度匹配与正类保护、原损失分配／回归／未触发梯度不变、deepcopy，
以及复用权重／args／CSV／源码篡改拒绝、只派发candidate、失败不进入评估、
三项差值／类别退化完整显示、错误权重绑定拒绝、合成终态报告审计及篡改检测。
合成调度／评估输出仅测试流程，没有真实推理或性能含义。

真实回执：`outputs/p1_competitive_negative_v2_validation_20261103/validation.json`；
对应控制台为`outputs/p1_competitive_negative_v2_validation_20261103_console.log`；
pytest日志为`outputs/p1_competitive_negative_v2_pytest_20261103.log`。
原缓存张量SHA256：`8a1aa278e75756ac5e5315248235e0c60b33f6ab943d76cf57ba01700fb41bb6`。
初次预检回执`outputs/p1_main_comparison_v2_preflights/20261103_142614_045380.json`；
调度／逐类报告补测完成后，按最终源码再次预检通过，新回执
`outputs/p1_main_comparison_v2_preflights/20261103_143240_749031.json`，未覆盖初版。
原17份证据及4份已完成权重另行复核，21／21一致；正式V2输出和候选run均不存在。

## 执行准备与限制

新入口只允许candidate worker，原B2／P1不可被调度重训；新输出／run已存在就拒绝。
复用预检把原preflight、退出码、args／experiment／CSV／best／last和旧失败证据绑定到新回执。
候选完整30轮后，才统一评估三个best，逐权重／预测绑定和全部AP／双IoU曲线重放。
新主表与decision自动回放，明确披露复用及源码commit差异。

正常目标不变；零有效质量锚点不做额外衰减是明确的数值边界修订，尚无完整epoch性能证据。
三臂比较仍限同split／同教师／单学生seed，开发val已多次查看。
七项性能线保留；没有strength实验，不作独立语义因果结论，也不自动扩大100轮。
精确预检／单组训练与终态audit命令见[执行入口](README.md)。
