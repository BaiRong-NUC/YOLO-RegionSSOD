# 连续集合V6零优化工程终态

2026-11-05。尚无学生性能结果。本报告不改写V1–V5及V6首轮失败。

## 已完成证据

- V6原完整检查：`outputs/bounded_retention_continuous_20261105/native_e_integration_01/`。
  mosaic开启/关闭各1414batch几何检查通过，11310 train/548 val loader实际建立。
  原STOP receipt SHA256：`39c03aa3a4243a0f3169abe174c4701d964f905df749c208148fc0bccee7d259`。
- 独立AMP检查：`outputs/bounded_retention_continuous_20261105/native_amp_audit_01/`。
  PID22280自然结束，727.922秒；一次显式压力batch前向、0 optimizer、0 val推理。
  receipt SHA256：`60ef98ace4414b1636baa04afd56d1488d934f93edf833c16c4e74c92a6b6504`。
- 两阶段2828batch重放统计一致；仍为原预定首次最大3U压力batch：closed index1316。
  旧7项实际head数值审计逐项一致；来源、源码及快照校验通过。
- 最终全库：`suite_amp_02/`，`python -m pytest`为**2037 passed/137.98秒**；
  代码输入哈希前后相同。此前`suite_amp_01/`的2034 passed保留。

## 实际AMP结果

| 臂 | 未缩放box | 未缩放cls | 未缩放dfl |
|---|---:|---:|---:|
| 原P1 | 10.50291157 | 30.58978271 | 8.40673065 |
| 候选 | 10.75401097 | 30.60271263 | 8.50402174 |
| 强控制 | 10.75401097 | 30.60271263 | 8.37613974 |

以上是该压力batch的工程loss，不是性能结果。三臂未缩放loss有限；共同scale依次
65536、32768、16384、8192、4096、2048未通过，**第一共同有限scale为1024**。
此时三臂head及全部可训练参数梯度有限，参数覆盖完整；候选实际`.backward()`亦通过。
候选与控制去缩放head梯度L1差为0.23438087879912928，CIoU和分类项完全一致。
原reference包装与真正P1DetectionLoss的loss/梯度均相同；前后模型参数哈希相同。
真实criterion只读诊断预览三臂均通过，文件的epoch字段不代表完成了学生训练。

因此旧65536非有限条件在本压力batch上来自AMP缩放溢出，而非未缩放loss非有限。
诊断scale不是训练超参数；实际训练继续使用原生默认动态GradScaler。
有限梯度不证明FP16没有欠流，也不保证整个训练轨迹永不触及数值容量。

联合验收器曾误要求旧STOP拥有`all_stream_geometry_checked`：该字段只在旧检查
全部反向成功后写入。现直接检查实际2828条有序几何记录及末位置closed/1413，
同时保留旧固定SHA、STOP状态、增强参数、来源和数值审计要求；没有伪造旧通过标记。

原P1源码SHA256仍为`5aadf46f534449a28ca7313763d3cd163663355d2004cd104ad9251ecee18efb`。
公开保留事件仍是模拟生成时GT衍生侧信息；source outer为保守外集。105058公开行对应
105057原生对象及1个完全相同duplicate alias；possible-U计数不等于真实截断/有效训练数。
贡献可为CABB/GFL相关思想的有依据适配；本工程通过没有证明原创性、性能或联合相关性。

来源、运输、连续目标/强控制、实际Trainer/loader/AMP和最终测试已联合通过；
下一步按[三臂冻结合同](three_arm30_contract.md)执行，不能把本报告写成已完成训练。
