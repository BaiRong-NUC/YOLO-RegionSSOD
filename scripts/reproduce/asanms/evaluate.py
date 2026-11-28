"""相同 YOLO11s 原始候选，比较标准 IoU NMS 与 ASA-NMS。"""

import os
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[3] / 'reproduce/asanms/local'


def make_detector(weight):
    import torch
    from torchvision.ops import batched_nms
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionPredictor
    from ultralytics.utils.ops import xywh2xyxy
    from reproduce.asanms.suppression import keep_indices

    class Predictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs):
            # 两臂在同一张 640 输入图的检测头输出上分叉，不用已经做过硬 NMS 的框。
            rows = (preds[0] if isinstance(preds, (list, tuple)) else preds)[0].T
            scores, labels = rows[:, 4:].max(dim=1)
            valid = scores > 0.001
            boxes = xywh2xyxy(rows[valid, :4])
            scores, labels = scores[valid], labels[valid]
            if self.arm == 'baseline':
                keep = batched_nms(boxes, scores, labels, 0.7)[:500]
            else:
                keep = keep_indices(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy(), 500)
            result = torch.cat((boxes[keep], scores[keep, None], labels[keep, None]), dim=1)
            return self.construct_results([result], img, orig_imgs)

    model = YOLO(str(weight))

    def predict(image, arm):
        Predictor.arm = arm
        return model.predict(image, predictor=Predictor, imgsz=640, rect=False, device=0,
                             half=False, verbose=False, save=False)[0].boxes.data.cpu().numpy()

    return predict


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(LOCAL))
    from scripts.reproduce.paired_inference import run_pairs

    run_pairs('asanms', make_detector)
