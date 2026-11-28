"""把作者逐类逐框的空间距离判定编译为 CPU 循环，保留置信度不变。"""

import numpy as np
from numba import njit


@njit
def keep_indices(boxes, scores, labels, limit):
    # 按全局分数遍历但仅抑制同类：等价于各类独立处理后取全局 top-k。
    order = np.argsort(-scores, kind='mergesort')
    suppressed = np.zeros(len(scores), dtype=np.bool_)
    kept = []
    for rank, best in enumerate(order):
        if suppressed[best]:
            continue
        kept.append(best)
        if len(kept) >= limit:
            break
        left, top, right, bottom = boxes[best]
        scale = np.float32(0.1) * np.hypot(right-left, bottom-top)
        threshold = np.float32(1) - np.exp(-scores[best] * np.float32(0.1))
        for pos in range(rank + 1, len(order)):
            other = order[pos]
            if suppressed[other] or labels[other] != labels[best]:
                continue
            x1, y1, x2, y2 = boxes[other]
            distance = np.hypot(x1-left, y1-top) + np.hypot(x2-right, y2-bottom)
            if scale > 0 and scores[other] * np.exp(-distance/scale) >= threshold:
                suppressed[other] = True
    return np.array(kept, dtype=np.int64)
