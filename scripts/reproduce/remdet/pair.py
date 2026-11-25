"""用作者的 YOLOv8s 对照配置验证单独替换 RemNet 骨干的效果。"""

import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
import torch
from mmengine import Config
from mmengine.runner import Runner
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from scripts.reproduce.esod.evaluate_visdrone import ROOT, VISDRONE_NAMES, read_visdrone

AUTHOR = ROOT / 'reproduce/remdet/upstream'
LOCAL = AUTHOR.parent / 'local'
DATA = ROOT / 'data/remdet/pair64'
sys.path.insert(0, str(AUTHOR))
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules


def prepare(split):
    # 保存固定抽样和原始图名；忽略区只重编码一次，避免两臂的数据差异。
    output = DATA / f'{split}.json'
    if output.exists():
        return output
    folder = DATA / split
    folder.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / 'data/visdrone/images' / split).glob('*.jpg'))
    assert len(paths) == (6471 if split == 'train' else 548)
    paths = sorted(random.Random(2026).sample(paths, 64)) if split == 'train' else paths
    result = dict(info={}, categories=[dict(id=index, name=name) for index, name in enumerate(VISDRONE_NAMES, 1)],
                  images=[], annotations=[])
    for index, path in enumerate(paths, 1):
        label = ROOT / 'data/visdrone/original_annotations' / split / path.with_suffix('.txt').name
        image, boxes, masked = read_visdrone(path, label)
        destination = folder / path.name
        if masked:
            if not cv2.imwrite(str(destination), image):
                raise OSError(destination)
        else:
            destination.write_bytes(path.read_bytes())
        height, width = image.shape[:2]
        result['images'].append(dict(id=index, file_name=path.name, height=height, width=width))
        for category, box in boxes:
            result['annotations'].append(dict(id=len(result['annotations'])+1, category_id=category, bbox=box,
                                              image_id=index, area=box[2]*box[3], iscrowd=0))
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result), encoding='utf8')
    temporary.replace(output)
    return output


def configuration(arm, seed):
    suffix = '_remdet' if arm == 'method' else ''
    cfg = Config.fromfile(str(AUTHOR / f'config_remdet/yolov8/yolov8_s{suffix}-300e_visdrone.py'))
    cfg.work_dir = str(LOCAL / f'{arm}-s{seed}')
    cfg.randomness = dict(seed=seed)
    cfg.env_cfg.mp_cfg.mp_start_method = 'spawn'
    loader = cfg.train_dataloader
    loader.update(batch_size=4, num_workers=0, persistent_workers=False)
    # 两个作者配置只变骨干。小样本统一使用等比缩放、补边、随机翻转和相同优化预算。
    loader.dataset.update(data_root=str(DATA), ann_file='train.json', data_prefix=dict(img='train'),
                          pipeline=[*cfg.pre_transform, *cfg.train_pipeline_stage2[2:4], *cfg.last_transform[-2:]])
    cfg.optim_wrapper.optimizer.batch_size_per_gpu = 4
    cfg.optim_wrapper.optimizer.lr = 0.01
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=31)
    cfg.default_hooks.param_scheduler.update(max_epochs=30, warmup_mim_iter=16, warmup_epochs=1)
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=30, save_optimizer=False,
                                       save_param_scheduler=False)
    cfg.default_hooks.pop('visualization', None)
    cfg.visualizer.vis_backends = []
    cfg.custom_hooks = []
    cfg.val_dataloader = cfg.val_cfg = cfg.val_evaluator = None
    cfg.test_dataloader = cfg.test_cfg = cfg.test_evaluator = None
    return cfg


def score(cfg, checkpoint, output):
    truth = COCO(str(prepare('val')))
    # 作者推理入口从 test_dataloader 读取同一份预处理配置。
    pipeline = [step for step in cfg.test_pipeline if step.type != 'LoadAnnotations']
    cfg.test_dataloader = dict(dataset=dict(type=cfg.dataset_type, metainfo=dict(classes=VISDRONE_NAMES),
                                            pipeline=pipeline))
    model = init_detector(cfg, str(checkpoint), device='cuda:0')
    images = [(key, cv2.imread(str(DATA / 'val' / row['file_name']))) for key, row in truth.imgs.items()]
    for _, image in images[:5]:
        inference_detector(model, image)
    detections, durations = [], []
    for key, image in images:
        torch.cuda.synchronize()
        start = perf_counter()
        instances = inference_detector(model, image).pred_instances.cpu().numpy()
        torch.cuda.synchronize()
        durations.append((perf_counter()-start)*1000)
        for box, label, conf in zip(instances.bboxes, instances.labels, instances.scores):
            left, top, right, bottom = box
            detections.append(dict(image_id=key, category_id=int(label)+1, score=float(conf),
                                   bbox=list(map(float, (left, top, right-left, bottom-top)))))
    if detections:
        predicted = truth.loadRes(detections)
    else:
        predicted = COCO()
        predicted.dataset = {**truth.dataset, 'annotations': []}
        predicted.createIndex()
    evaluator = COCOeval(truth, predicted, 'bbox')
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate()
    evaluator.accumulate()
    stats = dict(images=len(images), detections=len(detections), mean_ms=np.mean(durations),
                 p95_ms=np.percentile(durations, 95))
    for area, name in enumerate(('AP', 'AP_small', 'AP_medium', 'AP_large')):
        precision = evaluator.eval['precision'][:, :, :, area, -1]
        stats[name] = float(precision[precision>=0].mean()*100)
    output.write_text(json.dumps(stats, indent=2), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['baseline','method','public'], required=True)
    parser.add_argument('--seed', type=int, choices=[0,1], default=0)
    args = parser.parse_args()
    register_all_modules()
    if args.arm == 'public':
        cfg = Config.fromfile(str(AUTHOR/'config_remdet/remdet/remdet_tiny-300e_coco.py'))
        score(cfg, LOCAL/'remdet_tiny.pth', LOCAL/'public-val.json')
        return
    prepare('train')
    cfg = configuration(args.arm, args.seed)
    folder = Path(cfg.work_dir)
    checkpoint = folder / 'epoch_30.pth'
    if not checkpoint.exists():
        # 预编译 MMCV 在 Windows 上无需本机 MSVC，日志只记录实际使用的运行时。
        runtime = dict(PyTorch=torch.__version__, CUDA=torch.version.cuda, GPU=torch.cuda.get_device_name())
        with patch('mmengine.runner.runner.collect_env', return_value=runtime):
            Runner.from_cfg(cfg).train()
    score(cfg, checkpoint, folder / 'val.json')


if __name__ == '__main__':
    main()
