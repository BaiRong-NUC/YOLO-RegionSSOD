# 共享精确几何缓存验证记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；Mode: validate。
- Date: 2026-11-06；Version: geometry_cache_v1；Verification Status: FINAL_THREE_ARM_30_FROZEN_NOT_STARTED。
- 合同79fc3aa，执行前守卫补充2cbb638，源码57c091b/41f5126；起点7f9babf。
- 仅零训练工程资格与成本，不是学生性能/原创性/机制成立证据；未启动训练。

## 实现与不变项

独立bounded_retention_point_geometry_cache只对同一原生NativeEDomain的精确Fraction cell缓存
v2 contract_bounds结果，最多1024条LRU；不缓存分数、随机采样或梯度。
沿用同一v2 code object，只在私有globals绑定缓存函数，不改变原模块或其他调用。
所有锚点、搜索遍历/tie、point/gap/完整diagnostics、20000cells/1e-5，以及连续目标/强控制不变。
域变异明确拒绝；子类走原函数，实例动态几何方法不支持。缓存引用环可由GC回收，非立即释放承诺。

独立同模型代理静态审查发现的变异分流/子类覆盖及期限/证据根问题，在真实数值执行前修正。
保留旧脚本、失败和门；没有重放旧失败训练。新checker继承原压力全部守卫，新增来源绑定和
实测loss/backward≤120秒检查。父轮询退出后的时间窗口也核验，不能利用轮询间隙放行晚完成。

## 测试与固定预测诊断

- 首批缓存等价性15项通过，后扩展变异/非法参数等情况。
- suite_01：2191 passed/118.43秒。
- 守卫补充后的suite_02：2217 passed/114.67秒，exit0。通过后才执行真实数值。
- 固定trace_04 manifest SHA、原始request/args/P1产物、两份旧trace及全部历史绑定。
  P1校准显式比较dtype/shape/uint8原始字节，包括有符号零；无新检测模型前向/反向/optimizer。

实际命令：

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' -B -u scripts/diagnose_bounded_retention_cached_loss.py --output outputs/bounded_retention_cache_20261106/trace_01
```

父47456；P1 worker24944、候选37028、控制8340，顺序独立进程，均exit0。
每臂各2590点的support身份、全部metadata及point/gap/cells与各自旧trace逐项相同；
7份domain JSON也逐字节相同。旧trace未保存的完整证书字段不在实际记录比对范围内。
候选/控制完整loss均完成；P1严格校准通过。所有绑定源码/输入未变化。

| 同输入带记录器耗时 | 原trace_04 | 缓存trace_01 |
|---|---:|---:|
| 候选完整loss | 120秒TIMEOUT | 58.047秒 |
| 控制完整loss | 108.391秒 | 48.438秒 |
| 候选2590点累计 | 104.339秒 | 42.877秒 |
| 控制2590点累计 | 100.982秒 | 42.800秒 |
| 候选2233个3U点 | 104.217秒 | 42.847秒 |
| 控制2233个3U点 | 100.904秒 | 42.770秒 |

2233个3U点来自同一对象/增强实例的锚点池，不是2233个对象或有效监督暴露。
每臂缓存域末值合计3889 hits/207 misses（94.95%），最大207条；不可累计相加每行的累计计数。
候选完成10次后验8.862秒，包含partition0.923秒；support包含point，均不得重复加总。
完整criterion CPU时间候选56.953秒/控制47.750秒，仍主要为CPU计算。
同步记0受计时分辨率影响，不宣称CUDA成本为零。

这是一批固定保存预测的描述性比较。旧候选未完成，不能报其完整loss加速倍数；
不将单批推广为整轮提速，不追改旧32批2倍速度门失败，不替代真实Trainer/全参数反向资格。
独立同模型只读复核再次解析全部JSONL、核验域文件和310项绑定，结论一致；非跨模型/盲审。
完整只读汇总在outputs/bounded_retention_cache_20261106/analysis_01/report.json。

## 真实压力门

固定预测门通过后，实际启动：

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' -B -u scripts/check_bounded_retention_cached_pressure.py --output outputs/bounded_retention_cache_20261106/audit_01
```

父61424、worker9604，2026-11-06T14:00:27.511601Z（北京时间22:00:27）启动。
继承两条1414批完整增强流、closed1316、初始与第9轮EMA、固定32批、34cases/102loss反向。
执行已完成，父/worker退出0，terminal=PASSED，耗时1142.55秒；34次显式前向、102次完整loss及
全参数反向，全部来源未变、每个case和最终参数/BN状态恢复。无训练/optimizer/验证推理。

| 真实3U压力权重 | 候选loss | 候选反向 | 控制loss | 候选3U监督锚点 |
|---|---:|---:|---:|---:|
| 初始 | 79.997秒 | 0.679秒 | 67.006秒 | 10 |
| 第9轮EMA | 83.828秒 | 0.710秒 | 69.194秒 | 10 |

两case候选/控制的TAL、几何和CIoU/分类相同，梯度不同；3U锚点不是独立对象数。
独立qualification_01却停止于`request metadata changed: obs`。定位到可选original_area_cap
的63个NaN占位：dtype、shape、字节完全相同，np.array_equal默认返回False。
保留停止报告，先以4c256ca冻结[逐字节审计修订](cache_metadata_audit_revision.md)，
再新增独立验收器，不修改数值执行、旧检查器或重新运行压力实验。
对象数组补充a83c1f4明确保留合法physical对象数组的原内容比较，避免比较指针或错误拒绝合法输入。
全库suite_03为2235 passed/164.00秒，源码未变。新版只读qualification_02已完整通过，
核验34cases/102arms/255可训练参数张量；initial/epoch9各10个正权重3U锚点，输入和状态绑定完整。
它没有执行任何forward/loss/backward，梯度有限性依赖已保存执行回执，不冒称重新计算。
验收报告SHA为21dc0f55a66cbf2dadf2d1e078def951ceeeb0477452a9760f427d7142188a88。
独立同模型静态复核确认旧数值/来源/时限守卫保留；非跨模型或盲审。
9dedeb7已冻结[cached_training_contract](cached_training_contract.md)，开始独立生产接线验证；
完整训练资格仍待真实生产路径及最终预检，不启动训练。

## 独立训练入口测试

新增cached目标/Trainer、三臂配置、launcher及真实生产接线检查器，尚未执行训练。
suite_04记录2308通过/2失败：合成flat-score样例可在收缩前认证可行最大点，
因此“每个2U样例必须命中缓存”的测试前提不成立。失败日志和完整源码快照保留。
修订只检查实例缓存闭包确已接通并与EMA隔离；实际缓存命中仍由专测及真实3U接线硬门验证。
相关75项通过后，suite_05全库2310 passed（174.34秒），274项绑定源码前后未变。
同模型独立只读复核确认缓存闭包解释和launcher旧守卫保留，非跨模型/盲审。
真实六用例生产接线及最终冻结预检仍待执行；此处不是训练许可。

## 真实生产接线

739c130提交并push后执行独立入口，未修改运行中源码：

```powershell
& 'C:/Users/BaiRong-NUC/.conda/envs/yolo/python.exe' -B -u scripts/check_bounded_retention_cached_wiring.py --output outputs/bounded_retention_cache_20261106/wiring_01
```

2026-11-06T15:09:14Z，父56680/worker25468启动；15:21:37Z最终done，父/子exit0、terminal PASSED。
三臂均通过实际factory→setup_model→set_model_attributes→dataloader→model.loss路径；
11310训练图/1414批、548验证图的loader构造通过，候选/控制完整合法来源绑定一致。
每臂重放初始/第9轮固定真实压力输入，共6次前向/完整loss/全参数反向；
optimizer_steps=0，validation_inference_calls=0，training_started=false。

| 臂 | 权重 | 完整loss秒 | 全参数反向秒 | 缓存hits/misses |
|---|---|---:|---:|---:|
| 原P1 | 初始 | 0.2704 | 4.4648 | — |
| 原P1 | 第9轮 | 0.0232 | 0.0305 | — |
| 候选 | 初始 | 80.7062 | 0.5567 | 3889/207 |
| 候选 | 第9轮 | 81.5177 | 0.6964 | 4088/199 |
| 强控制 | 初始 | 68.1062 | 0.3390 | 3889/207 |
| 强控制 | 第9轮 | 67.4281 | 0.3176 | 4088/199 |

预测/loss/items/TAL/几何/完整证书与已认证压力路径逐位相同；每case两head及255参数张量
的257项梯度完整统计/SHA一致且有限。两权重下候选/控制各有214项梯度SHA不同，
仅支持监督信号有区别，不证明检测收益。每case参数/BN及最终状态恢复；验证loss回到原P1，
EMA在lazy criterion前创建，实际保存/加载不含criterion/缓存闭包。诊断scale1024不进入训练。
六case保存回执和源码证明调用计数守卫通过，但counter未单独落盘；不声称独立重算该值。

result SHA：9d71b472fb91880769e576cc720f5d0e7fabd02a996cf2b90b6e7bb237c9f0dd；
terminal SHA：91d7899993ed667d3663c0a4b28379a61455b4415e92cff1ba46d3073c9927df。
同模型只读复核wiring_review_01通过，核验1043项非原始数据绑定/307源码快照；
23719合法输入与旧freeze绑定一致，未重读隐藏GT或重新执行模型。非跨模型、非盲审。
最长已完成阶段81.615秒，未提高120秒或数值限额；所有诊断进程正常退出。

## 最终冻结与用户启动指令

最终全库suite_05为2310 passed；实际生产wiring及独立复核通过后，生成
`outputs/bounded_retention_cached_freeze_20261106/freeze.json`，共24986项输入，
SHA256为`5c15221d671cbcaa12270744b039b510a5edf0000478329dbf285c471282983e`。
完整真实入口`--check`已exit0（83.45秒），回执在同目录preflight_01.json及process.json。
合并摘要遇冲突即拒绝；未复制旧freeze的整张清单或旧suite指针，合法输入先分类再核验。
最终许可仅适用于本机冻结输入/运行时/代码/三臂设置，所有旧失败不改写。

```powershell
Set-Location 'D:\Python\YOLO-RegionSSOD'
& 'C:\Users\BaiRong-NUC\.conda\envs\yolo\python.exe' -B -u scripts/run_bounded_retention_cached.py --train --spec outputs/bounded_retention_cached_freeze_20261106/freeze.json
```

此命令**未执行**。按最新指令仅交付，当前没有新训练PID、无临时训练监控任务。
运行后依次原P1→候选→同界最强连续点DFL控制，各全新COCO初始化、完整30轮；
不复用旧参考结果/第9轮权重、不并行、不重试/恢复、不自动降低batch。
输出`outputs/bounded_retention_cached_three30_s0/`目前不存在，必须保持全新；
调度及各臂身份由入口写入输出回执，每30分钟内置临时监控只记录进度，终态自动结束。
全部三臂结束后统一548图full-image val960及原合同完整AP/AP_small/召回/同K/同FP/
重复框/定位诊断；完整30轮前不判断性能。科学合同为three_arm30_contract.md，
独立工程冻结为cached_training_contract.md（9dedeb7）；不修改两个历史合同的状态文字。
可以再次只读预检，将上述`--train`替换为`--check`；预检拒绝时停止，不删目录/改摘要绕过。

## 当前成本边界

第9轮权重下预先固定的16个open批，候选完整loss均值5.149秒/中位数3.176秒，
控制均值1.519秒；16个closed批候选均值1.618秒/中位数0.316秒，控制均值0.427秒。
这组数据不是随机总体样本；不能与不同权重的旧32批直接声称匹配提速。
按20轮open/10轮closed、1414批机械外推前向+loss+诊断反向，只是约数天计算量的成本提示，
不是实际训练ETA，还不包括加载、优化器、验证和I/O，也不覆盖训练中分布变化。
固定保存预测中共享点求解的局部提速已证明；整轮明显提速尚未证明，旧2倍速度门失败保留。
完整阶段均值/中位数/最大值及外推限制见outputs/bounded_retention_cache_20261106/analysis_02/report.json。

## 解释约束检查

Simpson：按臂/阶段/维数分开；生态谬误：工程成本不等于检测质量；Berkson/collider：
固定既往失败批不代表全部批；基础率：不外推3U频率；均值回归：不挑快重复；
幸存者偏差：旧失败保留；多重搜索：未搜seed/容量/误差；分析分叉：合同/补充先冻；
相关因果：同输入等价缓存仅支持局部成本归因；反向因果：不从提速推断机制/性能。
11项覆盖，无统计显著性或性能检验。
