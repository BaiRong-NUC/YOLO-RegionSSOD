"""固定三尺度金字塔，单独验证 SNIP 的推理尺度筛框。"""

import math

import cv2
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO

from scripts.reproduce.paired_inference import MAX_DET, run_pairs

# 作者公开配置中的短边、最长边、原图上有效框面积平方根范围。
SCALES = ((1400, 2000, -1, 90), (800, 1280, 32, 180), (480, 512, 75, -1))


def make_detector(weight):
    model = YOLO(weight)

    def predict(image, arm):
        height, width = image.shape[:2]
        views = []
        for short, long, lower, upper in SCALES:
            factor = min(short / min(height, width), long / max(height, width))
            scaled_h, scaled_w = round(height * factor), round(width * factor)
            padded_h, padded_w = math.ceil(scaled_h / 32) * 32, math.ceil(scaled_w / 32) * 32
            resized = cv2.resize(image, (scaled_w, scaled_h))
            padded = cv2.copyMakeBorder(resized, 0, padded_h-scaled_h, 0, padded_w-scaled_w,
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
            prediction = model.predict(padded, imgsz=(padded_h, padded_w), rect=False, device=0,
                                       verbose=False, conf=0.001, iou=0.7, max_det=MAX_DET)[0]
            boxes = prediction.boxes.data.clone()
            # 先还原原图像素，再应用作者的严格下界和闭合上界。
            boxes[:, 0:4:2] = boxes[:, 0:4:2].clamp(0, scaled_w) * (width / scaled_w)
            boxes[:, 1:4:2] = boxes[:, 1:4:2].clamp(0, scaled_h) * (height / scaled_h)
            area = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
            keep = area > 0
            if arm == 'method':
                if lower > 0:
                    keep &= area > lower**2
                if upper > 0:
                    keep &= area <= upper**2
            views.append(boxes[keep])
        boxes = torch.cat(views)
        keep = batched_nms(boxes[:, :4], boxes[:, 4], boxes[:, 5], 0.5)[:MAX_DET]
        return boxes[keep].cpu().numpy()

    return predict


if __name__ == '__main__':
    run_pairs('snip', make_detector)
