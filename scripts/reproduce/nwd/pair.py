"""在固定 YOLOv8s 小样本训练上单独检验 NWD 框回归损失。"""

import argparse
from unittest.mock import patch

import torch

from scripts.reproduce.remdet.pair import ROOT, Runner, configuration, prepare, register_all_modules, score


class WassersteinLoss(torch.nn.Module):
    def __init__(self, loss_weight):
        super().__init__()
        self.loss_weight = loss_weight

    def capture_strides(self, module, inputs, assigned):
        # 头部在回归前按步长除框；按原分配器的正样本顺序取出步长，恢复输入像素。
        mask = assigned['fg_mask_pre_prior'].bool()
        self.strides = inputs[2][None, :, 2].expand_as(mask)[mask].unsqueeze(1)

    def forward(self, pred, target, weight):
        delta = (pred - target) * self.strides
        center = (delta[:, :2] + delta[:, 2:]) / 2
        half_size = (delta[:, 2:] - delta[:, :2]) / 2
        distance = (center.square().sum(1) + half_size.square().sum(1) + 1e-7).sqrt()
        # 作者指数归一化的常数为 12.8 像素，保留 YOLO 原有的正样本权重和求和方式。
        return self.loss_weight * ((1 - torch.exp(-distance / 12.8)) * weight.flatten()).sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=[0, 1])
    seed = parser.parse_args().seed
    register_all_modules()
    prepare('train')
    cfg = configuration('baseline', seed)
    folder = ROOT / f'reproduce/nwd/local/seed{seed}'
    cfg.work_dir = str(folder)
    checkpoint = folder / 'epoch_30.pth'
    if not checkpoint.exists():
        runtime = dict(PyTorch=torch.__version__, CUDA=torch.version.cuda, GPU=torch.cuda.get_device_name())
        with patch('mmengine.runner.runner.collect_env', return_value=runtime):
            runner = Runner.from_cfg(cfg)
            head = runner.model.bbox_head
            loss = WassersteinLoss(head.loss_bbox.loss_weight)
            head.loss_bbox = loss
            head.assigner.register_forward_hook(loss.capture_strides)
            runner.train()
    score(cfg, checkpoint, folder / 'val.json')


if __name__ == '__main__':
    main()
