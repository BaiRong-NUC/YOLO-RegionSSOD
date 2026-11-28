"""独立实现 ASAHI 的网格对照，不把整图分支、训练和后处理混入消融。"""

import os
from pathlib import Path

import torch
from torchvision.ops import batched_nms

from reproduce.asahi.grid import grid
from reproduce.daasd.slicing import starts

LOCAL = Path(__file__).resolve().parents[3] / 'reproduce/asahi/local'


def make_detector(weight):
    from ultralytics import YOLO

    model = YOLO(str(weight))

    def predict(image, arm):
        height, width = image.shape[:2]
        if arm == 'baseline':
            tiles = [(left, top, min(left + 512, width), min(top + 512, height))
                     for top in starts(height, 512, 0.15) for left in starts(width, 512, 0.15)]
        elif arm == 'method':
            tiles = grid(width, height)
        else:
            raise ValueError(arm)
        # 两臂都保留整图通道；逐块推理，计时包含所有前向及跨视图合并。
        pieces = []
        for left, top, right, bottom in [(0, 0, width, height), *tiles]:
            view = image[top:bottom, left:right]
            pred = model.predict(view, imgsz=512, conf=0.001, iou=0.7, rect=False, device=0,
                                 half=False, max_det=500, verbose=False, save=False)[0].boxes.data.clone()
            pred[:, :4] += pred.new_tensor([left, top, left, top])
            pieces.append(pred)
        boxes = torch.cat(pieces)
        keep = batched_nms(boxes[:, :4], boxes[:, 4], boxes[:, 5], 0.5)[:500]
        predict.slice_info = (arm, len(tiles))
        return boxes[keep].cpu().numpy()

    return predict


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(LOCAL))
    from scripts.reproduce.paired_inference import run_pairs

    run_pairs('asahi', make_detector)
