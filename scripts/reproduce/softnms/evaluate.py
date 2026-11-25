"""配对验证 YOLO11s 对拥挤目标使用高斯 Soft-NMS 后的小目标精度。"""

import numpy as np
from numba import njit
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.ops import xywh2xyxy

from scripts.reproduce.paired_inference import MAX_DET, run_pairs


@njit
def gaussian_keep(boxes, scores):
    # 各类独立降分；坐标沿用 YOLO 连续边界，统一面积公式，无像素端点 +1。
    scores = scores.copy()
    kept = []
    for _ in range(min(len(scores), MAX_DET)):
        best = scores.argmax()
        score = scores[best]
        if score < 0.001:
            break
        kept.append((best, score))
        scores[best] = 0
        left, top, right, bottom = boxes[best]
        area = (right - left) * (bottom - top)
        for index in range(len(scores)):
            if scores[index] < 0.001:
                continue
            x1, y1, x2, y2 = boxes[index]
            intersection = max(0, min(right, x2) - max(left, x1)) * max(0, min(bottom, y2) - max(top, y1))
            if intersection > 0:
                iou = intersection / (area + (x2 - x1) * (y2 - y1) - intersection)
                scores[index] *= np.exp(-iou * iou / 0.5)
    return kept


class CandidatePredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        # 两臂取相同的原始候选框，先按置信度过滤，再只切换去重方式。
        output = preds[0] if isinstance(preds, (tuple, list)) else preds
        rows = output[0].T
        scores, labels = rows[:, 4:].max(dim=1)
        valid = scores > 0.001
        scores, labels = scores[valid], labels[valid]
        boxes = xywh2xyxy(rows[valid, :4])
        if self.mode == 'baseline':
            keep = batched_nms(boxes, scores, labels, 0.7)[:MAX_DET]
            result = torch.cat((boxes[keep], scores[keep, None], labels[keep, None]), dim=1)
        else:
            corners, confidence, classes = (value.cpu().numpy() for value in (boxes, scores, labels))
            found = []
            for category in np.unique(classes):
                indices = np.flatnonzero(classes == category)
                for local, score in gaussian_keep(corners[indices], confidence[indices]):
                    found.append((*corners[indices[local]], score, category))
            kept = sorted(found, key=lambda box: box[4], reverse=True)[:MAX_DET]
            result = boxes.new_tensor(kept).reshape(-1, 6)
        return self.construct_results([result], img, orig_imgs)


def make_detector(weight):
    model = YOLO(weight)
    def predict(image, arm):
        if model.predictor is None:
            CandidatePredictor.mode = arm
        else:
            model.predictor.mode = arm
        prediction = model.predict(image, predictor=CandidatePredictor, imgsz=640, rect=False,
                                   device=0, verbose=False)[0]
        return prediction.boxes.data.cpu().numpy()
    return predict


if __name__ == '__main__':
    run_pairs('softnms', make_detector)
