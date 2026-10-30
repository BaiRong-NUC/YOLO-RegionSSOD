"""Small, fixed-sample, horizontal-flip audit; hidden GT is diagnostic only."""
from __future__ import annotations
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
from audit_paper1_gaps import DATA, PSEUDO, overlaps, match_iou, size
from regionssod.data.annotations import Tile, YoloBox, clip_box_to_tile, label_path_for_image, read_label_file, replace_images_component
from run_p1_scale_seed_pair import ROOT, digest, read

OUT = ROOT / 'outputs/cross_view_flip_20261030'
SEED, SAMPLE = 20261030, 256


def support(original, flipped, threshold):
    """Greedily assign distinct same-class view boxes by descending IoU."""
    iou = overlaps(original, flipped)
    same = np.asarray([p.class_id for p in original])[:, None] == np.asarray([p.class_id for p in flipped])[None, :]
    a, b = np.where(same & (iou >= threshold))
    used_a, used_b = set(), set()
    for k in np.argsort(-iou[a, b], kind='stable'):
        i, j = int(a[k]), int(b[k])
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    return used_a


def box(item):
    return YoloBox(item['class_id'], *item['xywhn'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true', help='Reuse saved flipped predictions without inference')
    args = parser.parse_args()
    if OUT.exists() and not args.replay:
        raise FileExistsError(f'Output exists: {OUT}')
    if args.replay and not (OUT / 'flip_candidates.jsonl').is_file():
        raise FileNotFoundError('Saved flip predictions missing')
    if args.replay:
        receipt = read(OUT / 'analysis.json')
        if digest(OUT / 'flip_candidates.jsonl') != receipt['flip_candidates_sha256']:
            raise ValueError('Saved view predictions changed')
        for filename, expected in receipt['input_sha256'].items():
            if Path(filename).resolve() != Path(__file__).resolve() and digest(filename) != expected:
                raise ValueError(f'Audit input changed: {filename}')
    records = [json.loads(x) for x in (DATA / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    chosen = sorted(random.Random(SEED).sample(range(len(records)), SAMPLE))
    records = [records[i] for i in chosen]
    candidates = {r['source_image']: r for r in map(json.loads, (PSEUDO / 'candidates.jsonl').read_text(encoding='utf-8').splitlines())}
    metadata = {r['source_image']: r for r in map(json.loads, (PSEUDO / 'metadata.jsonl').read_text(encoding='utf-8').splitlines())}
    source, split, settings = [read(p) for p in (DATA / 'source.json', DATA / 'summary.json', PSEUDO / 'summary.json')]
    weights = Path(settings['weights'])
    if digest(weights) != settings['teacher_sha256'] or ultralytics.__version__ != settings['ultralytics_version']:
        raise ValueError('Teacher or Ultralytics version differs from frozen P1')
    inputs = [DATA / 'manifest.jsonl', DATA / 'source.json', DATA / 'summary.json', PSEUDO / 'candidates.jsonl', PSEUDO / 'metadata.jsonl', PSEUDO / 'summary.json', weights, Path(__file__).resolve()]
    OUT.mkdir(parents=True, exist_ok=True)
    model = None if args.replay else YOLO(str(weights))
    names = ['baseline', 'small025'] + [f'small025_flip{int(t*10):02d}_conf{int(c*100):02d}' for c in (.05, .25, .50) for t in (.3, .5)]
    totals = {name: Counter() for name in names}
    saved = {r['source_image']: r for r in map(json.loads, (OUT / 'flip_candidates.jsonl').read_text(encoding='utf-8').splitlines())} if args.replay else {}
    flips = []
    start = time.perf_counter()
    for offset in range(0, len(records), 8):
        batch = records[offset:offset+8]
        images = []
        for record in batch:
            image = Path(record['unlabeled_tiles'][0]['image'])
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise ValueError(f'Cannot read {image}')
            inputs.append(image)
            images.append(np.ascontiguousarray(pixels[:, ::-1]))
        results = [None] * len(batch) if args.replay else model.predict(source=images, imgsz=960, conf=.05, iou=.7, max_det=500, device='0', batch=8, verbose=False)
        if len(results) != len(batch):
            raise ValueError('Incomplete inference batch')
        for record, result in zip(batch, results, strict=True):
            entry = record['unlabeled_tiles'][0]
            image = entry['image']
            row = candidates[image]
            tile = Tile(*entry['tile'])
            mapped = saved[image]['candidates'] if args.replay else []
            if not args.replay:
                if list(result.orig_shape) != row['shape']:
                    raise ValueError('Tile shape mismatch')
                for cls, conf, coords in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.xywhn.cpu().numpy(), strict=True):
                    x, y, w, h = map(float, coords)
                    mapped.append({'class_id': int(cls), 'confidence': float(conf), 'xywhn': [1-x, y, w, h]})
            flips.append({'source_image': image, 'candidates': mapped})
            if image not in metadata:
                continue
            label = label_path_for_image(Path(record['source_image']), Path(source['images_train']), Path(source['labels_train']))
            inputs.append(label)
            gt = [g for b in read_label_file(label) if (g := clip_box_to_tile(b, tile, *record['source_size'], split['min_box_visibility'])) is not None]
            if len(gt) != entry['visible_gt_boxes_hidden_from_training']:
                raise ValueError('Hidden GT reconstruction mismatch')
            positive_path = replace_images_component(Path(metadata[image]['image'])).with_suffix('.txt')
            inputs.append(positive_path)
            positive = read_label_file(positive_path)
            locked = {g for _, g in match_iou(positive, gt)}
            candidates025 = [box(c) for c in row['candidates'] if c['state'] == 'ignore' and c['scale'] == 'small' and c['confidence'] >= .25]
            # Match among the eligible candidates, not against frozen positives.
            scenarios = {'baseline': [], 'small025': candidates025}
            for confidence in (.05, .25, .50):
                view_boxes = [box(c) for c in mapped if c['confidence'] >= confidence]
                for threshold in (.3, .5):
                    selected = support(candidates025, view_boxes, threshold)
                    scenarios[f'small025_flip{int(threshold*10):02d}_conf{int(confidence*100):02d}'] = [p for i, p in enumerate(candidates025) if i in selected]
            residual_ids = [i for i in range(len(gt)) if i not in locked]
            residual = [gt[i] for i in residual_ids]
            for name, added in scenarios.items():
                pairs = match_iou(added, residual)
                counts = totals[name]
                counts.update(tiles=1, gt=len(gt), small_gt=sum(size(g, tile) == 'small' for g in gt), positives=len(positive), baseline_tp=len(locked), baseline_small_tp=sum(size(gt[g], tile) == 'small' for g in locked), added=len(added), added_tp=len(pairs), added_small_tp=sum(size(residual[g], tile) == 'small' for _, g in pairs))
    output = {'sample_seed': SEED, 'sample_size': SAMPLE, 'sample_indices': chosen, 'scope': 'random all tiles; metrics restricted to existing P1 accepted tiles', 'flip': 'horizontal, same teacher; original P1 candidates; flip conf 0.05, imgsz 960, iou 0.7, max_det 500', 'matching': 'same class IoU >=0.50, descending IoU greedy; positive GT matches locked; all sizes jointly', 'elapsed_seconds': time.perf_counter()-start, 'ultralytics': ultralytics.__version__, 'teacher_sha256': digest(weights), 'input_sha256': {str(p): digest(p) for p in inputs}, 'metrics': {k: dict(v) for k,v in totals.items()}}
    if not args.replay:
        (OUT / 'flip_candidates.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in flips), encoding='utf-8')
    output['flip_candidates_sha256'] = digest(OUT / 'flip_candidates.jsonl')
    (OUT / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'elapsed_seconds': output['elapsed_seconds'], 'metrics': output['metrics']}, indent=2))

if __name__ == '__main__':
    main()
