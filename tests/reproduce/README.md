# 外部论文复现测试

按 `tests/reproduce/<论文简称>/` 存放本项目为该论文编写的测试，例如核对作者划分、输入格式和评测指标。简称与[复现脚本目录](../../scripts/reproduce/)一致。

从仓库根目录执行 `python -m pytest tests/reproduce/s3od` 可验证 S³OD 的尺度自适应阈值分组。测试使用合成旋转框，结果见 [S³OD 复现卡](../../docs/reproduce/s3od/)。
