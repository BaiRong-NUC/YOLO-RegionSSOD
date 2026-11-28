"""从一张 VisDrone 验证图中追踪一辆整图漏检、切片找回的小车。"""

from functools import partial
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
from draw.draw_car import draw_crop

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reproduce.sahi.evaluate_visdrone import CONFIDENCE, IMAGE_SIZE, MAX_DET, prepare_image

sys.path.insert(0, str(ROOT / 'reproduce/sahi/upstream'))

from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

STEM = '0000001_02999_d_0000005'
WEIGHT = ROOT / 'cache/2026-11-24-RegionSSOD/worktree/runs/partial10_b2_formal100_s0/weights/best.pt'


def car_predictions(result):
    # 先对全部类别应用原协议的每图上限，再看 car。
    ranked = sorted(result.object_prediction_list, key=lambda box: box.score.value, reverse=True)
    cars = [box for box in ranked[:MAX_DET] if box.category.name == 'car']
    return torch.tensor([box.bbox.to_xyxy() for box in cars]).reshape(-1, 4), [box.score.value for box in cars]


@torch.inference_mode()
def main():
    data = ROOT / 'data/visdrone'
    image, labels = prepare_image(data / f'images/val/{STEM}.jpg', data / f'original_annotations/val/{STEM}.txt')
    cars = torch.tensor([[left, top, left + width, top + height] for category, (left, top, width, height) in labels if category == 4], dtype=torch.float32)
    model = UltralyticsDetectionModel(model_path=str(WEIGHT), confidence_threshold=CONFIDENCE,
                                      image_size=IMAGE_SIZE, device='cuda:0', fuse=True)
    model.model.predict = partial(model.model.predict, max_det=MAX_DET, rect=False, iou=0.7)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    full, _ = car_predictions(get_prediction(rgb, model, verbose=0))
    sliced, scores = car_predictions(get_sliced_prediction(
        rgb, model, slice_height=640, slice_width=640, overlap_height_ratio=0.25,
        overlap_width_ratio=0.25, perform_standard_pred=True, postprocess_type='NMS',
        postprocess_match_metric='IOU', postprocess_match_threshold=0.5, verbose=0))

    # 按标注顺序选第一辆满足条件的小车，排除一个切片框同时匹配两辆车的歧义。
    full_hits = box_iou(cars, full) >= 0.5
    overlaps = box_iou(cars, sliced)
    unique_hits = (overlaps >= 0.5) & ((overlaps >= 0.5).sum(0) == 1)
    areas = (cars[:, 2:] - cars[:, :2]).prod(1)
    eligible = torch.where((areas < 32**2) & ~full_hits.any(1) & unique_hits.any(1))[0]
    if not len(eligible):
        raise ValueError('本图没有满足选例条件的小车')
    index = int(eligible[0])
    target = cars[index]
    recovered = int(torch.where(unique_hits[index])[0][0])

    # 重放同一整图前向，保留检测头解码出的所有框，随后按原顺序过筛。
    predictor = model.model.predictor
    tensor = predictor.preprocess([image])
    raw = predictor.inference(tensor)
    raw = raw[0] if isinstance(raw, (list, tuple)) else raw
    boxes = ops.xywh2xyxy(raw[0, :4].T).clone()
    ops.scale_boxes(tensor.shape[2:], boxes, image.shape[:2])
    overlap = box_iou(target[None], boxes.cpu())[0]
    confidence, classes = raw[0, 4:].cpu().max(0)
    car_id = list(model.category_names).index('car')
    near = overlap >= 0.5
    correct = near & (classes == car_id)
    passed = correct & (confidence > CONFIDENCE)
    # 放宽的仅是输出个数，用候选索引把去重和最终 500 框截断区分开。
    _, kept = non_max_suppression(raw.clone(), CONFIDENCE, 0.7, max_det=boxes.shape[0], return_idxs=True)
    kept = kept[0].cpu().long().flatten()
    final = kept[:MAX_DET]
    replay = boxes.cpu()[final[classes[final] == car_id]]
    np.testing.assert_allclose(replay.numpy(), full.numpy(), atol=0.002, rtol=0)
    print('图名、原图高宽、输入高宽：', STEM, image.shape[:2], tuple(tensor.shape[2:]))
    gain = min(IMAGE_SIZE / size for size in image.shape[:2])
    print('标注 xyxy、缩放后宽高：', target.tolist(), ((target[2:] - target[:2]) * gain).tolist())
    print('整图最高 IoU：', float(box_iou(target[None], full).max()) if len(full) else 0)
    print('切片框、分数、IoU：', sliced[recovered].tolist(), scores[recovered], float(overlaps[index, recovered]))
    print('总候选、位置对、位置类别都对、过分数、过去重、进前500：',
          len(boxes), int(near.sum()), int(correct.sum()), int(passed.sum()), int(passed[kept].sum()), int(passed[final].sum()))
    best = int(overlap.argmax())
    # 检测头按 P3、P4、P5 顺序展平；从候选索引倒推实际来自哪一格。
    offset = 0
    for level, stride in enumerate(model.model.model.model[-1].stride.tolist(), start=3):
        rows, cols = (int(size / stride) for size in tensor.shape[2:])
        if best < offset + rows * cols:
            row, col = divmod(best - offset, cols)
            print('原始候选索引、来源、格点行列（从0起）：', best, f'P{level}', row, col)
            break
        offset += rows * cols
    print('位置最接近的原始框：', boxes[best].tolist(), float(overlap[best]), model.model.names[int(classes[best])], float(confidence[best]))
    print('该候选的 car 分数：', float(raw[0, 4 + car_id, best]))
    if (classes == car_id).any():
        print('类别选 car 的原始框中最高 IoU：', float(overlap[classes == car_id].max()))
    if correct.any():
        print('合格位置和类别中的最高分：', float(confidence[correct].max()))
    draw_crop(image, target, boxes[best], sliced[recovered])


if __name__ == '__main__':
    main()
