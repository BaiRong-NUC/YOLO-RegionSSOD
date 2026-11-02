# P1学生seed1在55轮后的显式恢复

2026-11-02，用户报告VS Code窗口崩溃并要求继续训练。
窗口弹窗报告oom，训练工作进程13:43:11记录KeyboardInterrupt后停止。
原包装未记录真实退出码，保留为未知；不能把VS Code窗口错误直接解释为
整机64GB内存耗尽或CUDA显存不足，具体因果链尚未确定。

## 固定范围与证据

- 原CSV连续完整1–55轮，累计13602.5秒；第56轮未完成。
- last.pt的零起始epoch=54、学生seed1、原目标100轮。
  优化器255项、EMA updates=10496、AMP scaler存在，权重/优化器有限。
  SHA256：`19c2aa987aa21465190c86e9f48a6b005006679501d7769265d19741e5fe6160`。
- 21个原始文件复制并逐项核对哈希，保存在
  `outputs/paper1_p1_formal100_s1_recovery_20261102/`。
  原日志、回执、status、CSV、args、experiment和best/last及冻结源码均保留。
- 独立入口`scripts/resume_paper1_p1_seed1.py`只允许这一捕获的中断，
  使用原P1Trainer、soft_weights/uncertain_ignore，恢复原优化器、EMA、scaler及学习率计划。
  原seed0/seed1入口、训练源码、配置和既有冻结协议不改。
- yolo环境检查点副本已完成零训练步骤验证：
  `outputs/paper1_p1_formal100_s1_recovery_20261102/engineering_verified/check.json`。
  实测start_epoch=55、总100、scheduler.last_epoch=54；
  255项优化器状态及参数组、EMA权重/10496次更新、AMP scaler恢复一致，
  P1训练器和两项选项正确，训练批次/优化器步骤均为0，21个原始文件未变。
  首次副本检查因隔离目录派生的name不同而拒绝，证据保留在engineering；
  验证器只对副本的name/save_dir匹配隔离路径，实际续训仍仅允许model/resume变化。
- 从第56轮重算，只补到100轮的45轮，不是再训练100轮。
  原55轮CSV字节前缀保持，原experiment保留COCO初始化事实。
  不承诺与未中断训练逐位一致；正式结果须披露seed1 P1的恢复。

## 用户在独立Git Bash执行

```bash
source /d/MiniConda/etc/profile.d/conda.sh
conda activate yolo
cd /d/Python/YOLO-RegionSSOD
python scripts/resume_paper1_p1_seed1.py --check-only
python -u scripts/resume_paper1_p1_seed1.py --resume
```

助手仅检查及零优化器步骤验证，剩余45轮由用户启动。
建议从Windows开始菜单单独打开Git Bash运行，避免绑定到刚崩溃的VS Code窗口。
不要重新执行原`--train`，不要用默认YOLO训练器直接恢复P1。
按原55轮平均约247秒/轮估算，剩余约3.1小时，含验证可预留3.5–4小时，
实际负载及后段关闭mosaic会影响速度。

```bash
tail -n 20 -F outputs/paper1_p1_formal100_s1/resume_stdout.log
```

日志和回执用`resume_`前缀，不覆盖原stdout/stderr/launch_receipt/status；
`recovery.json`记录捕获和恢复实现哈希，`resume_setup.json`记录真实起始轮次、
P1训练器及优化器/EMA/scaler恢复核对，重复恢复默认拒绝。
失败保留证据，不自动重试。

## 完成后独立验收

```bash
python scripts/resume_paper1_p1_seed1.py --audit
```

该入口按原548原图val960/固定操作点recall及保存预测重放验收，
要求真实恢复exit0和工作进程停止；保留原55轮字节前缀，
学生耗时按13602.5秒加恢复段累计时间计算。
重验同seed1 B2并生成paired_comparison和仅含正式100轮的two_seed_comparison。
原seed1入口的`--audit`按未中断协议检查，恢复后应使用本页独立入口，
不能跳过恢复事实或把原退出码补写为0。
两seed统计须标明seed0 B2和seed1 P1分别恢复；与旧30轮分开，
不作跨split泛化、显著性或官方AP_small主张。

原临时监控p1-1在准备期间暂停，交接后改为等待用户产生resume回执；
没有回执时不启动任务。用户启动后按新回执PID/创建时间/命令核实，
使用resume状态、日志和真实退出码，正常推进静默。
恢复失败或完整训练及评估验收收尾后删除监控，不自动重试或扩训。
