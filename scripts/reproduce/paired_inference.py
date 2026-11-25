"""在同一套 VisDrone 图像、标注和 YOLO11s 权重上评测推理阶段的改动。"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scripts.reproduce.sahi.evaluate_visdrone import MAX_DET, ROOT, evaluate, read_validation


def run_pairs(paper, make_detector):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--limit', type=int, choices=range(1, 1611))
    args = parser.parse_args()
    # Ultralytics 接收 BGR，在预解码时保留原通道，避免 test-dev 同时存两套像素。
    images, truth = read_validation(ROOT / 'data/visdrone', args.limit, args.split, rgb=False)
    local = ROOT / 'reproduce' / paper / 'local'
    local.mkdir(parents=True, exist_ok=True)
    result_file = local / f'{args.split}-{len(images)}.json'
    reports = []
    for seed in (0, 1):
        weight = ROOT / f'cache/2026-11-24-RegionSSOD/worktree/runs/partial10_b2_formal100_s{seed}/weights/best.pt'
        with weight.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        detector = make_detector(weight)
        # 交叉执行顺序；计时覆盖预处理、前向、合并及预测回传，数据提前解码。
        for arm in (('baseline', 'method') if seed == 0 else ('method', 'baseline')):
            for image in images[:5]:
                detector(image, arm)
            detections, durations = [], []
            for image_id, image in enumerate(images, 1):
                torch.cuda.synchronize()
                start = perf_counter()
                boxes = detector(image, arm)
                torch.cuda.synchronize()
                durations.append((perf_counter() - start) * 1000)
                for left, top, right, bottom, score, label in boxes:
                    detections.append(dict(image_id=image_id, category_id=int(label) + 1, score=float(score),
                                           bbox=list(map(float, (left, top, right-left, bottom-top)))))
            report = dict(seed=seed, arm=arm, images=len(images), sha256=digest, mean_ms=np.mean(durations),
                          p95_ms=np.percentile(durations, 95), **evaluate(truth, detections))
            reports.append(report)
            temporary = result_file.with_suffix('.tmp')
            temporary.write_text(json.dumps(reports, indent=2), encoding='utf8')
            temporary.replace(result_file)
            print(report, flush=True)
