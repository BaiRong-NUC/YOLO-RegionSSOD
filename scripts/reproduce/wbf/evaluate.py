"""比较相同的原图和镜像双视图，在原图坐标合并时采用硬 NMS 或作者 WBF。"""

import sys

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO

from scripts.reproduce.paired_inference import MAX_DET, ROOT, run_pairs

sys.path.insert(0, str(ROOT / 'reproduce/wbf/upstream'))
from ensemble_boxes import weighted_boxes_fusion


def make_detector(weight):
    model = YOLO(weight)
    def predict(image, arm):
        height, width = image.shape[:2]
        outputs = model.predict([image, cv2.flip(image, 1)], imgsz=640, rect=False, device=0, verbose=False,
                                conf=0.001, iou=0.7, max_det=MAX_DET)
        views = [output.boxes.data.cpu().numpy().copy() for output in outputs]
        # 镜像结果还原到原图坐标，两臂用同一批候选、同一类别和匹配阈值。
        views[1][:, [0, 2]] = width - views[1][:, [2, 0]]
        views = [rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])] for rows in views]
        if arm == 'baseline':
            rows = torch.from_numpy(np.concatenate(views))
            kept = batched_nms(rows[:, :4], rows[:, 4], rows[:, 5], 0.55)
            return rows[kept[:MAX_DET]].numpy()
        scale = np.array([width, height, width, height])
        boxes, scores, labels = weighted_boxes_fusion([rows[:, :4] / scale for rows in views],
            [rows[:, 4] for rows in views], [rows[:, 5] for rows in views], iou_thr=0.55, skip_box_thr=0.001)
        return np.column_stack((boxes * scale, scores, labels))[:MAX_DET]
    return predict


if __name__ == '__main__':
    run_pairs('wbf', make_detector)
