# 覆盖保护：有条件的用户100轮配对入口

2026-10-31，在覆盖保护30轮训练等待期间准备。
**当前只完成交接代码准备，30轮结果未知，不表示已建议或启动大训练。**
本文件不修改正在运行的小训练及其冻结协议。

## 启动条件与范围

只有完整30轮exit0、终态独立验收accepted=true且worth_larger_training=true，
入口重新核对原始来源/产物和预测重放后，才允许执行100轮。
不存在完整结果、候选未达本轮探索线或原训练仍活跃时拒绝启动。
不把工程检查、文件存在或手改成功标志当作通过。

最小正式对照为完整P1和覆盖保护候选各独立100轮，固定学生seed0、
原split seed0/固定B1教师/伪标签/11310tile/COCO初始化，训练640、
SGD、batch8、close_mosaic10；两个模型从头训练，不从30轮权重续训。
两臂差异仅为分配规则与run名称；epochs均为100。
本脚本不自动扩大到其他seed/预算/教师、B0/B2或部署。
论文仍需按结论选择正式B2及重复实验，不以这两臂代替全部投稿证据。

## 命令及职责

当前仅可查看无GPU/无输出的计划：

```powershell
python scripts/run_coverage_guard_full100.py --plan-only
```

下列为**通过30轮后才生效的条件命令**，不是当前启动指令。
到时由用户执行，助手不自动启动。每一步成功后再执行下一步；
在PowerShell中不要把多行粘贴理解成失败自动停止。

```powershell
python scripts/run_coverage_guard_full100.py --arm p1 --check-only
python -u scripts/run_coverage_guard_full100.py --arm p1 --train
python scripts/run_coverage_guard_full100.py --arm p1 --audit
python scripts/run_coverage_guard_full100.py --arm coverage_guard --check-only
python -u scripts/run_coverage_guard_full100.py --arm coverage_guard --train
python scripts/run_coverage_guard_full100.py --arm coverage_guard --audit
```

顺序运行，不并行。预计至少约两次100轮的GPU成本，实际小时数在30轮
完成后用已测耗时给出，不把当前估计当作承诺。
已有任何对应输出/run/val/recall目录即拒绝重启或覆盖；
中断后保留证据，先检查，不自动resume。

## 输出与验收

run为runs/partial10_{p1,coverage_guard}_full100_s0；
独立记录位于outputs/{p1,coverage_guard}_full100_s0。
配置在执行时确定性生成到忽略目录并留档，Git保存生成入口和本协议。
每臂自动执行完整548图val960及固定操作点recall，
保存权重、参数、源码/输入哈希、预测和阶段状态。

进程结束后--audit检查连续完整1–100轮、实际参数、训练元数据、
完整原图协议、权重与产物哈希及保存预测匹配重放。
两臂独立验收完成后生成paired_comparison.json。
独立验收不做新推理，不创建新训练。
不存在独立重复seed时不得声称显著性，recall不等于官方AP_small。
100轮正向与否均如实报告；旧Gate3与30轮探索线不追改。
