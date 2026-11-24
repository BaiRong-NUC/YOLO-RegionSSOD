import ast
import runpy
from pathlib import Path

import numpy as np
import sklearn.mixture
import torch


UPSTREAM = Path(__file__).resolve().parent / "upstream"


def load_sat():
    """载入作者公开配置及 SAT 函数，供独立于旋转框框架的评测使用。"""
    config = runpy.run_path(str(UPSTREAM / "configs/dota1.5/s3od_dota1.5_1percent_test.py"))
    settings = config["semi_wrapper"]["train_cfg"]

    # 保持作者函数原样，只隔离同一文件中依赖 MMDetection/MMRotate 的其他代码。
    source = UPSTREAM / "s3od/models/utils/bbox_utils.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    names = {"thr_select_policy", "get_adaptive_size2", "filter_invalid_with_adaptive_thr_size2"}
    module.body = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    helpers = {"np": np, "torch": torch, "skm": sklearn.mixture}
    exec(compile(module, str(source), "exec"), helpers)
    return settings, helpers["get_adaptive_size2"], helpers["filter_invalid_with_adaptive_thr_size2"]
