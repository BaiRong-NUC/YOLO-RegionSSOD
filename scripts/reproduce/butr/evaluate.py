"""BUTR 作者 VisDrone 权重：同一解码器，固定或按图选择查询数量。"""

import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

PAPER = Path(__file__).resolve().parents[3] / 'reproduce/butr'


def make_detector(weight):
    sys.path.insert(0, str(PAPER / 'upstream'))
    from src.core import YAMLConfig

    scale = weight.stem.split('_')[1].upper()
    config = YAMLConfig(str(PAPER / f'upstream/configs/models/BUTR-{scale}-VisDrone.yml'))
    config.yaml_cfg['HGNetv2']['pretrained'] = False
    checkpoint = torch.load(weight, weights_only=True, map_location='cpu')
    model = config.model
    model.load_state_dict(checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model'])
    model = model.deploy().cuda()
    post = config.postprocessor.deploy().cuda()
    transform = Compose([Resize((800, 800)), ToTensor()])
    dynamic = model.decoder._compute_budget_expected_count

    def fixed(count):
        # 仅替换预算计算；top-k、解码器与输出头继续调用原作者实现。
        return lambda logits, **kwargs: logits.new_full((logits.shape[0],), count, dtype=torch.long)

    budgets = {'fixed250': fixed(250), 'fixed500': fixed(500), 'dynamic': dynamic}

    @torch.inference_mode()
    def predict(image, arm):
        model.decoder._compute_budget_expected_count = budgets[arm]
        height, width = image.shape[:2]
        inputs = transform(Image.fromarray(image[:, :, ::-1])).unsqueeze(0).cuda()
        outputs = model(inputs)
        queries = outputs['pred_logits'].shape[1]
        predict.slice_info = (arm, queries)
        labels, boxes, scores = post(outputs, inputs.new_tensor([[width, height]]))
        # 作者训练沿用 VisDrone 的类别编号 1..10，0 和 11 不参加十类评测。
        keep = (labels[0] >= 1) & (labels[0] <= 10)
        return torch.cat((boxes[0, keep], scores[0, keep, None], labels[0, keep, None]-1), dim=1).cpu().numpy()

    predict.trace_key = 'by_image_arm_queries_ms'
    return predict


if __name__ == '__main__':
    os.environ.setdefault('YOLO_CONFIG_DIR', str(PAPER / 'local'))
    from scripts.reproduce.paired_inference import run_pairs

    # 权重在 train+val 上训练，只能在 test-dev 做无交叉的评测。
    run_pairs('butr', make_detector, ('fixed250', 'dynamic', 'fixed500'), splits=('test',),
              checkpoints={scale: PAPER / f'local/butr_{scale.lower()}_visdrone.pth' for scale in ('S', 'L')})
