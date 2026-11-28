"""郝川艳等，2025：按原图长边在 6 块与 12 块之间选择。"""

from math import floor


def intervals(length, count, overlap):
    # 式（2）末尾加一后取整；两个端点对齐原图，均摊中间的取整误差。
    size = min(length, floor(length / (count - (count - 1) * overlap)) + 1)
    return [(offset, offset + size) for index in range(count)
            for offset in [round(index * (length - size) / (count - 1))]]


def grid(width, height, overlap=0.15):
    columns, rows = (3, 2) if max(width, height) <= 1818 else (4, 3)
    if height > width:
        columns, rows = rows, columns
    return [(left, top, right, bottom) for top, bottom in intervals(height, rows, overlap)
            for left, right in intervals(width, columns, overlap)]
