"""DAASD 原文第 3 节：整图密度分档与边界对齐的切片窗口。"""

import numpy as np


def select_tier(boxes, width, height):
    # 每个检测框按中心计入唯一网格；平均值包含空单元格。
    centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
    columns = (centers[:, 0] * 4 / width).astype(int).clip(0, 3)
    rows = (centers[:, 1] * 3 / height).astype(int).clip(0, 2)
    counts = np.bincount(rows * 4 + columns, minlength=12)
    if counts.max() > 30 or counts.mean() > 15:
        return 'high', 480, 0.3
    if counts.max() > 15 or counts.mean() > 8:
        return 'medium', 640, 0.2
    return 'low', 800, 0.1


def starts(length, size, overlap):
    # 边缘切片向图内平移，不重复最后一块；小于窗口时只看一次。
    end = max(0, length - size)
    return [*range(0, end, round(size * (1 - overlap))), end]
