"""在同一套 VisDrone 图像、标注和 YOLO11s 权重上评测推理阶段的改动。"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import torch

from scripts.reproduce.sahi.evaluate_visdrone import MAX_DET, ROOT, evaluate, read_validation


def run_pairs(paper, make_detector, arms=('baseline', 'method'), *, checkpoints=None, splits=('val', 'test')):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=splits, default=splits[0])
    parser.add_argument('--limit', type=int, choices=range(1, 1611))
    args = parser.parse_args()
    # Ultralytics 接收 BGR，在预解码时保留原通道，避免 test-dev 同时存两套像素。
    images, truth = read_validation(ROOT / 'data/visdrone', args.limit, args.split, rgb=False)
    local = ROOT / 'reproduce' / paper / 'local'
    local.mkdir(parents=True, exist_ok=True)
    result_file = local / f'{args.split}-{len(images)}.json'
    reports = []
    weights = checkpoints or {seed: ROOT / f'cache/2026-11-24-RegionSSOD/worktree/runs/partial10_b2_formal100_s{seed}/weights/best.pt'
                              for seed in (0, 1)}
    for index, (label, weight) in enumerate(weights.items()):
        with weight.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        detector = make_detector(weight)
        # 交叉执行顺序；计时覆盖预处理、前向、合并及预测回传，数据提前解码。
        for arm in (arms if index % 2 == 0 else reversed(arms)):
            for image in images[:5]:
                detector(image, arm)
            detections, durations, strata = [], [], []
            for image_id, image in enumerate(images, 1):
                torch.cuda.synchronize()
                start = perf_counter()
                boxes = detector(image, arm)
                torch.cuda.synchronize()
                durations.append((perf_counter() - start) * 1000)
                # 可选的逐图分层与耗时在计时窗口外记录，供跨臂按相同图像配对。
                if hasattr(detector, 'slice_info'):
                    strata.append((*detector.slice_info, durations[-1]))
                for left, top, right, bottom, score, category in boxes:
                    detections.append(dict(image_id=image_id, category_id=int(category) + 1, score=float(score),
                                           bbox=list(map(float, (left, top, right-left, bottom-top)))))
            report = dict(arm=arm, images=len(images), sha256=digest, mean_ms=np.mean(durations),
                          p95_ms=np.percentile(durations, 95), **evaluate(truth, detections))
            report['checkpoint' if checkpoints else 'seed'] = label
            if strata:
                report[getattr(detector, 'trace_key', 'by_image_tier_slices_ms')] = strata
            reports.append(report)
            temporary = result_file.with_suffix('.tmp')
            temporary.write_text(json.dumps(reports, indent=2), encoding='utf8')
            # Windows 上编辑器或进度读取器可能短暂占用结果文件；只重试原子替换，不重跑实验。
            for attempt in range(12):
                try:
                    temporary.replace(result_file)
                    break
                except PermissionError:
                    if attempt == 11:
                        raise
                    sleep(0.25)
            print({key: value for key, value in report.items() if not key.startswith('by_image_')}, flush=True)
