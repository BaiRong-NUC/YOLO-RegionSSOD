"""固定作者 YOLC 权重，在 VisDrone 上配对比较整图与局部放大。"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[3]
AUTHOR = ROOT / 'reproduce/yolc/upstream'
LOCAL = AUTHOR.parent / 'local'
sys.path.insert(0, str(AUTHOR))

from eval_yolc import inference, init_detector


def score(truth, detections, image_ids):
    # 沿用作者标注、类别和 maxDets，直接按维度取均值以明确框数上限。
    evaluator = COCOeval(truth, truth.loadRes(detections), 'bbox')
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = [100, 300, 1000]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval['precision']
    recall = evaluator.eval['recall']
    values = dict(AP=precision[:, :, :, 0, -1], AP50=precision[0, :, :, 0, -1],
                  AP_small=precision[:, :, :, 1, -1], AR_small=recall[:, :, 1, -1])
    return {name: float(scores[scores >= 0].mean() * 100) for name, scores in values.items()}


def run_arm(model, truth, images, image_ids, crops):
    # 两组都走作者 eval_yolc.inference，保留相同的候选区计算、读盘与粗检测过程。
    def detect(image_id):
        name = truth.imgs[image_id]['file_name']
        return inference(model, model, str(images / name), name, saved_crop=crops, crop=bool(crops))

    for image_id in image_ids[:3]:
        detect(image_id)
    durations = []
    predictions = []
    category_ids = truth.getCatIds()
    for index, image_id in enumerate(image_ids, start=1):
        torch.cuda.synchronize()
        start = perf_counter()
        results = detect(image_id)
        torch.cuda.synchronize()
        durations.append(1000 * (perf_counter() - start))
        for category_id, boxes in zip(category_ids, results):
            for left, top, right, bottom, confidence in boxes:
                # 跳过作者为避免空数组加入的全零占位框。
                if confidence > 0:
                    predictions.append(dict(image_id=image_id, category_id=category_id,
                                            score=float(confidence), bbox=list(map(float, (
                                                left, top, right - left, bottom - top)))))
        if index % 100 == 0:
            print(f'k={crops}  {index}/{len(image_ids)}', flush=True)
    return dict(crops=crops, images=len(image_ids), **score(truth, predictions, image_ids),
                mean_ms=float(np.mean(durations)), p95_ms=float(np.percentile(durations, 95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=ROOT / 'data/visdrone/images/val')
    parser.add_argument('--checkpoint', type=Path, default=LOCAL / 'yolc.pth')
    parser.add_argument('--limit', type=int, help='仅用于工程烟测')
    parser.add_argument('--order', nargs='+', type=int, choices=[0, 2], default=[0, 2, 2, 0])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    truth = COCO(str(AUTHOR / 'annotations/VisDrone2019-DET_val_coco.json'))
    image_ids = truth.getImgIds()[:args.limit]
    if not image_ids or any(not (args.images / truth.imgs[key]['file_name']).is_file() for key in image_ids):
        parser.error('指定划分的图像缺失')
    model = init_detector(str(AUTHOR / 'configs/yolc.py'), str(args.checkpoint), device='cuda:0')
    record = dict(checkpoint=str(args.checkpoint), gpu=torch.cuda.get_device_name(), runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 作者在聚类函数内无条件写调试图，关闭写图以免污染工作区和计时。
    with patch('cv2.imwrite', return_value=True):
        for crops in args.order:
            record['runs'].append(run_arm(model, truth, args.images, image_ids, crops))
            temporary = args.output.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
            temporary.replace(args.output)
            print(record['runs'][-1], flush=True)


if __name__ == '__main__':
    main()
