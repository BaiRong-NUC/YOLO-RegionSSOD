# ESOD 作者实现

[复现卡与两张 SVG](../../docs/reproduce/esod/)记录 VisDrone val 全量权重重评、基线前向测速及复跑命令。

| 路径 | 用途 |
|---|---|
| [upstream](upstream/) | 作者 `alibaba/esod` 的固定提交 `bde3571bd7db697e441eef0278cd425e888ea026` |
| [运行入口](../../scripts/reproduce/esod/) | 本机 COCO 重评、耗时记录与作者内部指标交叉核对 |
| [绘图脚本](figures/draw_pair.py) | 逐层绘制 YOLOv5m 基线与 ESOD 的结构 SVG |
| `local/` | 本机权重、独立评测环境、预测和逐图耗时 |
