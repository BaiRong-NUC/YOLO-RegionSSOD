"""在两份成熟 YOLO11s 权重上隔离 YOLC 密集区选择带来的增量。"""

import os
import zlib
from pathlib import Path

import torch

from reproduce.yolc.regions import cluster_windows, relocate

LOCAL = Path(__file__).resolve().parents[3] / 'reproduce/yolc/local'


def make_detector(weight):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionPredictor

    class HeadMap(DetectionPredictor):
        capture = False

        def postprocess(self, preds, img, orig_imgs):
            if self.capture:
                # 640 方形输入的首层 80×80 个位置对应 stride=8；去掉补边，只在真实图像内选区。
                output = preds[0] if isinstance(preds, (tuple, list)) else preds
                rows, cols = img.shape[-2] // 8, img.shape[-1] // 8
                heat = output[0, 4:, :rows * cols].amax(dim=0).reshape(rows, cols)
                height, width = orig_imgs[0].shape[:2]
                gain = min(cols / width, rows / height)
                used_w, used_h = max(1, round(width * gain)), max(1, round(height * gain))
                left, top = (cols - used_w) // 2, (rows - used_h) // 2
                self.probability = heat[top:top + used_h, left:left + used_w].cpu().numpy()
            return super().postprocess(preds, img, orig_imgs)

    model = YOLO(str(weight))

    def forward(view):
        return model.predict(view, predictor=HeadMap, imgsz=640, rect=False, device=0, conf=0.001, iou=0.7,
                             max_det=500, save=False, verbose=False)[0].boxes.data

    def predict(image, arm):
        height, width = image.shape[:2]
        HeadMap.capture = arm != 'whole'
        coarse = forward(image)
        if arm == 'whole':
            predict.slice_info = (arm, 0)
            return coarse.cpu().numpy()
        windows = cluster_windows(model.predictor.probability, width, height)
        if arm == 'random':
            windows = relocate(windows, width, height, zlib.crc32(image[::80, ::80].tobytes()))
        HeadMap.capture = False
        keep = torch.ones(len(coarse), dtype=torch.bool, device=coarse.device)
        fine = []
        for left, top, right, bottom in windows:
            # 作者用局部预测替换完全落在选区内的整图框，不删除跨边界的原框。
            keep &= ~((coarse[:, 0] >= left) & (coarse[:, 2] <= right) &
                      (coarse[:, 1] >= top) & (coarse[:, 3] <= bottom))
            boxes = forward(image[top:bottom, left:right]).clone()
            boxes[:, :4] += boxes.new_tensor([left, top, left, top])
            fine.append(boxes)
        boxes = torch.cat([coarse[keep], *fine])
        predict.slice_info = (arm, len(windows))
        return boxes[boxes[:, 4].argsort(descending=True)[:500]].cpu().numpy()

    return predict


if __name__ == '__main__':
    os.environ.setdefault('YOLO_CONFIG_DIR', str(LOCAL))
    from scripts.reproduce.paired_inference import run_pairs

    run_pairs('yolc', make_detector, ('whole', 'random', 'guided'))
