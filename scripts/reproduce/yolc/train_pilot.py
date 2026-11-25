"""用固定的 VisDrone 小样本独立训练 YOLC，供不同种子的局部放大对照使用。"""

import argparse
import hashlib
import os
import random
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import torch

ROOT = Path(__file__).resolve().parents[3]
AUTHOR = ROOT / 'reproduce/yolc/upstream'
LOCAL = AUTHOR.parent / 'local'
PILOT = LOCAL / 'pilot16'
sys.path.insert(0, str(AUTHOR))

from gen_crop import convert_to_cocodetection


def prepare_data():
    # 两个训练种子共享同一份抽样；训练原图和原标注通过硬链接交给作者的裁片脚本。
    source = ROOT / 'data/visdrone'
    training = PILOT / 'VisDrone2019-DET-train'
    images = sorted((source / 'images/train').glob('*.jpg'))
    selected = random.Random(2026).sample(images, 16)
    for image in selected:
        annotation = source / 'original_annotations/train' / image.with_suffix('.txt').name
        for original, folder in [(image, 'images'), (annotation, 'annotations')]:
            target = training / folder / original.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(original, target)
    if set((training / 'images').glob('*')) != {training / 'images' / img.name for img in selected}:
        raise ValueError('训练目录与预先抽取的 16 张图不一致')
    annotation_file = PILOT / 'VisDrone2019-DET_train_coco_1crop.json'
    if not annotation_file.exists():
        # 保留训练裁片，只屏蔽作者在聚类时写出的调试热图。
        write_image = cv2.imwrite

        def save_training_image(filename, pixels):
            return True if Path(filename).name.startswith('binary_heatmap_') else write_image(filename, pixels)

        torch.set_num_threads(2)
        with patch('cv2.imwrite', side_effect=save_training_image):
            convert_to_cocodetection(str(PILOT), str(PILOT))
    return annotation_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    backbone = LOCAL / 'hrnetv2_w48-d2186c55.pth'
    if not hashlib.sha256(backbone.read_bytes()).hexdigest().startswith('d2186c55'):
        raise ValueError('HRNet 骨干权重校验失败')
    annotations = prepare_data()
    # 小样本训练控制分辨率和重复次数；两个种子均取第 30 轮，不按验证集挑权重。
    settings = [f'data.train.dataset.ann_file={annotations.as_posix()}',
                f'data.train.dataset.img_prefix={(PILOT / "VisDrone2019-DET-train-crop/images").as_posix()}',
                f'model.backbone.init_cfg.checkpoint={backbone.as_posix()}',
                'data.train.times=1', 'data.samples_per_gpu=2', 'data.workers_per_gpu=0',
                'data.train.dataset.pipeline.3.crop_size=(512,320)',
                'data.train.dataset.pipeline.4.img_scale=(512,320)',
                'runner.max_epochs=30', 'lr_config.warmup_iters=20', 'optimizer.lr=0.00125',
                'checkpoint_config.interval=30', 'checkpoint_config.save_optimizer=False',
                'checkpoint_config.create_symlink=False', 'log_config.interval=16']
    command = [str(AUTHOR / 'train.py'), str(AUTHOR / 'configs/yolc.py'), '--seed', str(args.seed),
               '--no-validate', '--work-dir', str(PILOT / f'seed{args.seed}'), '--cfg-options', *settings]
    # 使用预编译算子；旧版 MMCV 收集环境日志时会尝试启动本地 MSVC。
    runtime = {'PyTorch': torch.__version__, 'CUDA': torch.version.cuda, 'GPU': torch.cuda.get_device_name()}
    with patch.object(sys, 'argv', command), patch('mmdet.utils.collect_env', return_value=runtime):
        runpy.run_path(command[0], run_name='__main__')


if __name__ == '__main__':
    main()
