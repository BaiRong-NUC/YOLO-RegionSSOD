# 论文复现结果总览

| 论文复现卡 | 基线结构图 | 论文方法结构图 | 本机复现进度 |
|---|---|---|---|
| [S³OD](s3od/README.md) | [SSOD baseline](s3od/ssod-baseline-architecture.svg) | [S³OD](s3od/s3od-architecture.svg) | 已核对作者代码与评测协议，本机指标待测 |

每复现一篇论文，在这里新建 `docs/reproduce/<论文简称>/`，并在上表登记。子目录的 `README.md` 记录评测协议、论文报告值、本机复现值和差异。作者基线图与论文方法图放在同一子目录，各保存 SVG 原图和 PNG 预览，沿用相同的画布、层位置和编号。对应的作者代码、适配代码和绘图脚本位于仓库根目录的 `reproduce/<论文简称>/`。
