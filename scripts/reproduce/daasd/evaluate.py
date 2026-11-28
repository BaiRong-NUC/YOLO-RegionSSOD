"""在固定 YOLO11s 权重上隔离 DAASD 的切片收益与分档收益。"""

import os
from pathlib import Path

import torch
from torchvision.ops import batched_nms

from reproduce.daasd.slicing import select_tier, starts

LOCAL = Path(__file__).resolve().parents[3] / 'reproduce/daasd/local'


def make_detector(weight):
    from ultralytics import YOLO

    model = YOLO(str(weight))

    def forward(image, confidence):
        return model.predict(image, conf=confidence, iou=0.7, imgsz=640, rect=False, max_det=500,
                             device=0, half=False, verbose=False, save=False)[0].boxes.data

    def predict(image, arm):
        height, width = image.shape[:2]
        if arm == 'whole':
            predict.slice_info = ('whole', 0)
            return forward(image, 0.001).cpu().numpy()
        tier, size, overlap = 'fixed', 640, 0.2
        if arm == 'adaptive':
            # 密度扫描计入延迟，但扫描框不进入最终输出。
            tier, size, overlap = select_tier(forward(image, 0.1).cpu().numpy(), width, height)
        elif arm != 'fixed':
            raise ValueError(arm)
        pieces = []
        for top in starts(height, size, overlap):
            for left in starts(width, size, overlap):
                boxes = forward(image[top:top + size, left:left + size], 0.001).clone()
                boxes[:, :4] += boxes.new_tensor([left, top, left, top])
                pieces.append(boxes)
        # 映回原图后再做类内 NMS；两种切片策略的后处理完全相同。
        boxes = torch.cat(pieces)
        keep = batched_nms(boxes[:, :4], boxes[:, 4], boxes[:, 5], 0.5)[:500]
        predict.slice_info = (tier, len(pieces))
        return boxes[keep].cpu().numpy()

    return predict


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(LOCAL))
    from scripts.reproduce.paired_inference import run_pairs

    run_pairs('daasd', make_detector, ('whole', 'fixed', 'adaptive'))
