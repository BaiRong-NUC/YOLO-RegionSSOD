"""把人工标注、整图候选和切片预测叠加在同一块原图上。"""

import base64
from pathlib import Path

import cv2


def draw_crop(image, target, whole_candidate, prediction):
    # 两栏复用同一块真实像素，只改变叠加的预测框，便于看清位置差异。
    left, top, right, bottom = map(int, target.tolist())
    origin_x, origin_y = max(0, left - 45), max(0, top - 32)
    crop = image[origin_y:bottom + 32, origin_x:right + 45]
    height, width = crop.shape[:2]
    encoded = base64.b64encode(cv2.imencode('.png', crop)[1]).decode()

    def frame(box, color, dash=''):
        x1, y1, x2, y2 = box.tolist()
        return f'<rect x="{x1-origin_x}" y="{y1-origin_y}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" {dash}/>'

    panels = []
    for offset, box, title, color in [(0, whole_candidate, '整图 · 位置最接近的候选', '#ff8876'),
                                      (width + 4, prediction, '切片 · 最终检出的车', '#40e4ed')]:
        reference = frame(target, '#ffd54a')
        detected = frame(box, color, 'stroke-dasharray="3 2"')
        panels.append(f'<g transform="translate({offset},14)"><text y="-4" font-size="6" fill="#17384b">{title}</text><use href="#crop"/>{reference}{detected}</g>')
    total = 2 * width + 4
    folder = Path(__file__).resolve().parents[1] / 'output'
    folder.mkdir(exist_ok=True)
    (folder / 'car.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total*4}" height="{(height+14)*4}" viewBox="0 0 {total} {height+14}"><title>同一辆车，黄实线是人工标注，虚线是模型的框</title><rect width="100%" height="100%" fill="white"/><defs><image id="crop" width="{width}" height="{height}" image-rendering="pixelated" href="data:image/png;base64,{encoded}"/></defs>{"".join(panels)}</svg>', encoding='utf-8')
