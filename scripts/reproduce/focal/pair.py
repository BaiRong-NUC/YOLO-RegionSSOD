"""在 RetinaNet 上比较论文表 1b 推荐的平衡交叉熵和聚焦损失配置。"""

import argparse
from unittest.mock import patch

import torch
from mmengine import Config

from scripts.reproduce.remdet.pair import AUTHOR, DATA, ROOT, Runner, prepare, register_all_modules, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', choices=[0, 1], type=int, required=True)
    parser.add_argument('--arm', choices=['baseline', 'method'], required=True)
    args = parser.parse_args()
    register_all_modules()
    prepare('train')
    cfg = Config.fromfile(str(AUTHOR / 'config_remdet/retinanet/retinanet_r50_fpn_1x_visdrone.py'))
    local = ROOT / 'reproduce/focal/local'
    folder = local / 'table1b' / f'{args.arm}-s{args.seed}'
    cfg.work_dir = str(folder)
    cfg.randomness = dict(seed=args.seed)
    cfg.model.backbone.init_cfg.checkpoint = str(local / 'resnet50-0676ba61.pth')
    # 采用原文各自推荐的正负样本权重，避免用论文已知较弱的交叉熵配置充当基线。
    gamma, alpha = (2.0, 0.25) if args.arm == 'method' else (0.0, 0.75)
    cfg.model.bbox_head.loss_cls.update(gamma=gamma, alpha=alpha)
    cfg.model.test_cfg.update(score_thr=0.001, max_per_img=300)
    cfg.test_pipeline = [dict(type='LoadImageFromFile'), dict(type='Resize', scale=(640, 640), keep_ratio=True),
                         dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]
    pipeline = [cfg.test_pipeline[0], dict(type='LoadAnnotations', with_bbox=True), cfg.test_pipeline[1],
                dict(type='RandomFlip', prob=0.5), dict(type='PackDetInputs')]
    cfg.train_dataloader.update(batch_size=4, num_workers=0, persistent_workers=False, batch_sampler=None)
    cfg.train_dataloader.dataset.update(data_root=str(DATA), ann_file='train.json', data_prefix=dict(img='train'),
                                       filter_cfg=None, pipeline=pipeline)
    cfg.env_cfg.mp_cfg.mp_start_method = 'spawn'
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=31)
    cfg.optim_wrapper.optimizer.lr = 0.0025
    cfg.optim_wrapper.clip_grad = dict(max_norm=10, norm_type=2)
    cfg.param_scheduler = [dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=16),
                           dict(type='MultiStepLR', by_epoch=True, begin=0, end=30, milestones=[22, 27], gamma=0.1)]
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=30, save_optimizer=False,
                                        save_param_scheduler=False)
    cfg.default_hooks.pop('visualization', None)
    cfg.visualizer.vis_backends = []
    cfg.val_dataloader = cfg.val_cfg = cfg.val_evaluator = None
    cfg.test_dataloader = cfg.test_cfg = cfg.test_evaluator = None
    checkpoint = folder / 'epoch_30.pth'
    if not checkpoint.exists():
        runtime = dict(PyTorch=torch.__version__, CUDA=torch.version.cuda, GPU=torch.cuda.get_device_name())
        with patch('mmengine.runner.runner.collect_env', return_value=runtime):
            Runner.from_cfg(cfg).train()
    score(cfg, checkpoint, folder / 'val.json')


if __name__ == '__main__':
    main()
