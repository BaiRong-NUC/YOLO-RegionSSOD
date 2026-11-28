"""用第一步选定的小车核对 SAHI 切片来源，画出两路网络输入。"""

import base64
from functools import partial
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reproduce.sahi.evaluate_visdrone import CONFIDENCE, IMAGE_SIZE, MAX_DET, prepare_image
sys.path.insert(0, str(ROOT / 'reproduce/sahi/upstream'))
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.slicing import get_slice_bboxes


def closest_car(result, target, offset=(0, 0)):
    # 与整图评测一样，先取全部类别的前 500 框，再核对 car 与标注的 IoU。
    ranked = sorted(result.object_prediction_list, key=lambda item: item.score.value, reverse=True)
    cars = [item for item in ranked[:MAX_DET] if item.category.name == 'car']
    if not cars:
        raise ValueError('本案例视图缺少可核对的车框')
    boxes = torch.tensor([item.bbox.to_xyxy() for item in cars]) + torch.tensor(offset * 2)
    overlaps = box_iou(target[None], boxes)[0]
    best = int(overlaps.argmax())
    return boxes[best], cars[best].score.value, float(overlaps[best])


def input_panel(predictor, image, target, origin, title, overlap):
    # 从推理器预处理后的 RGB 张量取像素，两栏截取相同大小并按相同倍率显示。
    tensor = predictor.preprocess([image])[0]
    pixels = tensor.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    height, width = image.shape[:2]
    gain = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    pad_x = (IMAGE_SIZE - round(width * gain)) // 2
    pad_y = (IMAGE_SIZE - round(height * gain)) // 2
    left, top, right, bottom = (target.numpy() - np.tile(origin, 2)) * gain
    left, right = left + pad_x, right + pad_x
    top, bottom = top + pad_y, bottom + pad_y
    start_x, start_y = round((left + right) / 2) - 24, round((top + bottom) / 2) - 24
    crop = pixels[start_y:start_y + 48, start_x:start_x + 48]
    assert crop.shape == (48, 48, 3)
    encoded = base64.b64encode(cv2.imencode('.png', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))[1]).decode()
    return f'''<text y="22" font-size="17">{title}</text>
    <svg y="36" width="240" height="240" viewBox="0 0 48 48">
      <image width="48" height="48" image-rendering="pixelated" href="data:image/png;base64,{encoded}"/>
      <rect x="{left-start_x}" y="{top-start_y}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#ffe45e" stroke-width="0.42"/>
    </svg>
    <text y="303">标注宽高　{right-left:.2f} × {bottom-top:.2f} px</text>
    <text y="328">最贴近的车框　IoU {overlap:.4f}</text>'''


@torch.inference_mode()
def main():
    stem = '0000001_02999_d_0000005'
    data = ROOT / 'data/visdrone'
    annotation = data / f'original_annotations/val/{stem}.txt'
    image, labels = prepare_image(data / f'images/val/{stem}.jpg', annotation)
    left, top, width, height, _, category, *_ = map(int, annotation.read_text().splitlines()[64].split(',')[:8])
    assert category == 4 and (category, [left, top, width, height]) in labels
    target = torch.tensor([left, top, left + width, top + height], dtype=torch.float32)
    weight = ROOT / 'cache/2026-11-24-RegionSSOD/worktree/runs/partial10_b2_formal100_s0/weights/best.pt'
    model = UltralyticsDetectionModel(model_path=str(weight), confidence_threshold=CONFIDENCE,
                                      image_size=IMAGE_SIZE, device='cuda:0', fuse=True)
    model.model.predict = partial(model.model.predict, max_det=MAX_DET, rect=False, iou=0.7)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    options = dict(slice_height=IMAGE_SIZE, slice_width=IMAGE_SIZE, overlap_height_ratio=0.25, overlap_width_ratio=0.25)
    tiles = get_slice_bboxes(*image.shape[:2], **options)
    # 这辆车只完整落在一块中，可以直接检查该块的输出是否一直保留到跨视图去重之后。
    [tile] = [box for box in tiles if box[0] <= left and box[1] <= top and box[2] >= left + width and box[3] >= top + height]
    x0, y0, x1, y1 = tile
    whole = closest_car(get_prediction(rgb, model, verbose=0), target)
    single = closest_car(get_prediction(rgb[y0:y1, x0:x1], model, verbose=0), target, (x0, y0))
    merged = closest_car(get_sliced_prediction(rgb, model, **options, perform_standard_pred=True,
                         postprocess_type='NMS', postprocess_match_metric='IOU', postprocess_match_threshold=0.5, verbose=0), target)
    torch.testing.assert_close(single[0], merged[0], atol=0.002, rtol=0)
    assert abs(single[1] - merged[1]) < 1e-5 and whole[2] < 0.5 <= merged[2]
    print(f'{len(tiles)} 块，完整含车的块 {tile}。单块原图坐标 {single[0].tolist()}，分数 {single[1]:.4f}。')
    print(f'整图 / 单块 / 合并 IoU：{whole[2]:.4f} / {single[2]:.4f} / {merged[2]:.4f}')
    predictor = model.model.predictor
    full_panel = input_panel(predictor, image, target, (0, 0), '整图缩到 640 后', whole[2])
    tile_panel = input_panel(predictor, image[y0:y1, x0:x1], target, (x0, y0), '原图裁出 640 方块后', single[2])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="568" height="386" viewBox="0 0 568 386">
    <title>同一辆车在整图输入与切片输入中各占多少像素</title>
    <rect width="100%" height="100%" fill="white"/><g fill="#17384b" font-size="15" font-family="Microsoft YaHei, sans-serif">
    <g transform="translate(24,8)">{full_panel}</g><g transform="translate(304,8)">{tile_panel}</g>
    <text x="24" y="371">两栏各取 48 × 48 个输入像素，同倍率放大。黄框为人工标注。</text></g></svg>'''
    # 正式图放在 svgs 下，同时更新旧对话链接指向的同名文件。
    for destination in (HERE / 'svgs/inputs.svg', HERE / 'inputs.svg'):
        destination.write_text(svg, encoding='utf-8')


if __name__ == '__main__':
    main()
