# RCN三臂30轮：独立只读终态验证（2026-11-04）

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate
- Origin Date: 2026-11-04 (Asia/Shanghai)
- Verification Status: VERIFIED（冻结产物/确定性缓存）；学生性能条件未通过；机制NOT_ESTABLISHED
- Version Label: reliable_competitor_negative_terminal_v1

## 执行与回执

2026-11-04 02:35（Asia/Shanghai）先确认原--train父54292与三个worker均已停止，
原终态stage=completed/outcome=stop，三worker真实exit0，再执行：

```powershell
& C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe -B scripts/run_reliable_competitor_negative.py --audit
```

独立audit真实exit0／accepted=true，结果与自动training_audit/evaluation_audit/decision精确一致。
audit stdout/stderr保存在封印目录之外的新`outputs/rcn30_terminal_audit_20261104_0235.log`，
没有在原output/run增加文件；原57项精确产物集合和哈希保护不变。
新`outputs/rcn30_terminal_receipt_20261104.json`保存日志及原manifest/decision哈希；
没有覆盖或修复原产物。日志SHA`e1be746aae181e4776cdd1b9469775648f5e55d7a997ceced0866e6943e2f1e3`。

## 验证范围

| 项目 | 核对结果 |
|---|---|
| 训练 | reference/candidate/control各连续1–30轮、CSV有限、exit0，未恢复/重试 |
| 进程身份 | 启动、reference→candidate→control切换已核PID/创建时间/父子/claim；终态均停止 |
| 来源 | 39906项输入及代码/环境/数据/教师/初始化绑定通过；preflight SHA见结果报告 |
| 实际args | 三臂完整args与冻结expected_args精确一致，除name/save_dir的共同条件一致 |
| 源码 | 三experiment均91cc57d及reliable_competitor_negative_v1，各mode/config准确 |
| 权重预测 | 三新best/last与seal绑定，评估manifest严格指向各自best，预测/weight SHA一致 |
| 评估 | 548完整val960，38759 GT/small26577；COCO及双IoU micro PR精确重算 |
| 数组 | 全部micro_iou0.5/0.75 NPZ与COCO PR数组逐元素一致 |
| 固定分解 | conf.25/IoU.50 wrong-class及全FP类型逐图重算一致，与PR FP总数相同 |
| 判定 | 根comparison/decision全部正负值与10门槛重放一致，5通过，stop |
| 新推理/优化 | 本次audit 0模型推理/0训练步骤，只有CPU缓存计算 |

Ultralytics总体AP仅绑定原validator输出及文件哈希，**未从缓存独立重算其内部AP**。
COCO-style AP/PR数值和数组已精确重放；随机训练没有重跑，不能把VERIFIED写成训练逐位可复现或泛化成立。
标准源依赖/输入哈希不等于整个GPU驱动/所有CUDA二进制/系统状态均被记录。
工程契约由训练前806项测试和单batch验证支持，未重算全部训练batch梯度。

## 结果与收尾

工程通过；学生三臂性能已验证但8项性能只通过4项，2项机制只通过1项，语义机制未建立。
AP50–95−0.3502pp、AP_small−0.1088pp、conf.25 small−3.1682pp；
P90 small+1.1890pp、严格P90+0.0941pp、FP−22.5377%及错类FP−25.5983%均保留。
相对控制AP−0.2034pp／AP_small+0.2063pp；完整全部类别、曲线点与成本见[results30.md](results30.md)。
旧V2失败判定、P1主线、冻结现有候选README/合同/工程报告均未改，新增终态文档独立保存。
本轮只改报告和主入口状态，无源码修改，未重复训练前已通过的806项pytest；git diff --check通过。
更新主计划第3.76/10节、主README及baseline当前入口，中文commit并push只提交本次报告修改。
按授权在本轮全部终态处置后删除临时监控rcn-30，无自动重试/恢复/扩训。
11/11解释谬误检查及限制详见结果报告；没有新增采用论文、Zotero导入或未核实阅读声明。
