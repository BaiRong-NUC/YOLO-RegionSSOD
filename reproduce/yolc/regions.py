"""将 YOLC 的 16×10 密度网格、八连通选区和外扩规则迁移到 YOLO 分数图。"""

import cv2
import numpy as np


def cluster_windows(probability, width, height):
    # YOLO 的 P3 类别分数代替 CenterNet 热图。还原到原图后按作者 findclusters 的灰度阈值二值化。
    heat = 1 - cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    gray = (255 * heat / max(float(heat.max()), 1e-8)).astype(np.uint8)
    active = gray <= (10 / 11 * 255)
    cell_w, cell_h = width // 16, height // 10
    counts = active[:cell_h * 10, :cell_w * 16].reshape(10, cell_h, 16, cell_w).sum(axis=(1, 3)).T
    grid = np.zeros((16, 10), dtype=np.uint8)
    grid.flat[counts.flatten().argsort()[-15:]] = 1
    _, _, stats, _ = cv2.connectedComponentsWithStats(grid.T.copy(), connectivity=8)
    regions = sorted(stats[1:], key=lambda row: int(row[2]) * int(row[3]), reverse=True)[:2]
    windows = []
    for col, row, cols, rows, _ in regions:
        left, top, box_w, box_h = col * cell_w, row * cell_h, cols * cell_w, rows * cell_h
        if box_w * box_h >= 96**2:
            windows.append((max(0, int(left - 0.1 * box_w)), max(0, int(top - 0.1 * box_h)),
                            min(width - 1, int(left + 1.1 * box_w)), min(height - 1, int(top + 1.1 * box_h))))
    return windows


def relocate(windows, width, height, seed):
    # 安慰剂对照：只移动位置，不改该图裁片的数量、尺寸或计算预算。
    rng = np.random.default_rng(seed)
    shifted = []
    for left, top, right, bottom in windows:
        box_w, box_h = right - left, bottom - top
        start_x, start_y = int(rng.integers(width - box_w + 1)), int(rng.integers(height - box_h + 1))
        shifted.append((start_x, start_y, start_x + box_w, start_y + box_h))
    return shifted
