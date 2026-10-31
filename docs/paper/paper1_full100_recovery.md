# B2正式100轮的显式检查点恢复

2026-10-31，用户报告误中断并要求检查是否可恢复。
原B2于19:13:36启动，23:01:05在第64轮初期收到KeyboardInterrupt；
原PID8248已停止，没有其他正式训练进程。
原包装未记录真实退出码，继续保留为未知，不据中断status补写exit1。

## 已核实与保存

- CSV连续完整1–63轮，累计13621.5秒。
- `last.pt`保存epoch=62（零起始，即完成63轮）、原总目标100轮；
  优化器255项状态、EMA（updates=11683）及AMP scaler均存在，
  EMA张量有限；继续前还检查优化器状态有限。
- checkpoint SHA256：
  `d44cfb626e7b82bbec5c1f8081d0cde6416d46d30ffb709475347a6f1cdf0f8e`。
- 恢复从64/100开始，只补37轮；第64轮尚未保存的batch需要重新计算。
- 原13个文件已逐个复制并核对哈希，包括best/last、CSV、args、
  experiment、原日志/进程/阶段/冻结清单及原入口源码。
  归档`outputs/paper1_b2_formal100_recovery_20261031`，原始训练证据不删除。

这是一次有来源的checkpoint续训，不是拼接两次从COCO重新开始的训练。
现根据用户新请求增加显式恢复例外，原“禁止自动续训/覆盖”仍保留：
默认`--train`继续拒绝已有目录，恢复需要独立`--resume`。
仅放行这次B2的已捕获中断，P1不能使用该恢复入口。

## 用户操作

在Git Bash与yolo环境、仓库根目录先执行恢复检查：

```bash
source /d/MiniConda/etc/profile.d/conda.sh
conda activate yolo
cd /d/Python/YOLO-RegionSSOD
python scripts/run_paper1_full100.py --arm b2 --resume-check
```

确认输出`completed_epochs: 63`、`resume_epoch: 64`、`total_epochs: 100`后，
在同一终端执行：

```bash
python -u scripts/run_paper1_full100.py --arm b2 --resume
```

助手只做恢复检查和零优化器步骤的工程验证，不启动剩余37轮。
按最后5个完整epoch约178.68秒/轮估算，剩余训练约1.84小时，另加完整评估/I/O；
速度会随机器负载、关闭mosaic等阶段变化，不作为完成时刻保证。

新的进度写入：

```bash
tail -f outputs/paper1_b2_formal100_s0/resume_stdout.log
```

原`stdout.log`/`stderr.log`保留；新错误写入`resume_stderr.log`。
前63轮CSV原始字节前缀必须保持，后37轮使用原框架正常追加。
恢复时检查实际model/resume仅指向已存last.pt，其余参数与原args完全一致，
实际start_epoch/总epoch、优化器恢复及EMA计数写入`resume_setup.json`。
原experiment.json保持首次COCO初始化记录，恢复事实由独立recovery.json补充。

恢复完成后自动执行原val960/recall与独立验收；验收同时检查原始归档、
恢复源码、原冻结输入、真实新进程exit0和连续1–100轮。
CSV的time在恢复时会重新计时，因此报告总训练时间为
原63轮13621.5秒 + 恢复段累计时间，避免漏算已有时间。
原入口源码的变更仅允许与归档原哈希对应的这次恢复功能，
不能借恢复忽略训练源码、数据或配置变化。

B2验收正常通过后，按原计划运行P1：

```bash
python -u scripts/run_paper1_full100.py --arm p1 --train
```

不自动开启P1、不反复恢复；如果恢复再次失败，保留新证据并单独处理。

## 科研解释与限制

框架恢复权重、优化器、EMA、AMP与原进度/学习率计划。
现有checkpoint没有完整保存Python/NumPy/PyTorch及数据加载器的全部随机状态，
中断轮未保存batch也不会接着运行；因此不承诺与未中断训练逐位相同。
论文应标明B2在第63轮后恢复，P1是否连续训练另列；不能把中断隐藏为完全连续运行。
这不使结果自动失效，也不能用来消除所有训练随机性。
该恢复只是继续同一100轮计划，性能仍须训练完并按完整原图指标判断。

真实工程检查：在独立临时目录使用检查点副本调用当前yolo环境的原生恢复流程，
在训练初始化结束回调中主动停止，0个训练batch、0个优化器步骤。
已核对start_epoch=63、总100轮、scheduler.last_epoch=62，
255项优化器状态逐张量与检查点一致、EMA updates=11683、scaler状态一致；
原中断目录文件哈希未变。记录见归档engineering/check.json。
