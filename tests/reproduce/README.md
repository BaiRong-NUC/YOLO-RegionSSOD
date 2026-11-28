# 外部论文复现测试

按 `tests/reproduce/<论文简称>/` 存放本项目为该论文编写的测试，例如核对作者划分、输入格式和评测指标。简称与[复现脚本目录](../../scripts/reproduce/)一致。

从仓库根目录执行 `python -m pytest tests/reproduce/s3od` 可验证 S³OD 的尺度自适应阈值分组。测试使用合成旋转框，结果见 [S³OD 复现卡](../../docs/reproduce/s3od/)。

近年推理实验包含 [DAASD 网格密度阈值](daasd/test_slicing.py)、[ASAHI 横竖图边缘覆盖](asahi/test_grid.py) 与 [ASA-NMS 对作者抑制索引](asanms/test_suppression.py)。最后一项需安装 Numba 并初始化该篇子模块；单元测试不作为精度复现证据。
