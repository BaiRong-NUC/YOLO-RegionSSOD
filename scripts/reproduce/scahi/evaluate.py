"""同一批 VisDrone 视图比较作者的框级 F0 与稠密 F2 融合。"""

import os
import sys
import json
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[3] / 'reproduce/scahi'


def make_detector(weight):
    # 单独加载作者源码和锁定版 SAHI，避免污染旧 SAHI 复现实验。
    sys.path[:0] = [str(PAPER / 'local/deps'), str(PAPER / 'upstream')]
    import sahi
    from scahi.fusion.run import clip_xywh, f0_image
    from scahi.fusion.scalespace import CachedLevel, FusionParams
    from scahi.fusion.scalespace_torch import fuse_image_f2
    from scahi.infer.cache_mz import LEVELS, raw_predict_level
    from scahi.infer.raw_head import load_unfused

    if sahi.__version__ != '0.12.8':
        raise RuntimeError('请先按复现卡安装 sahi==0.12.8')
    model = load_unfused(weight, 'cuda')
    # 直接读取作者预先冻结的两行，防止把其他模型调出的阈值带进本机配对。
    frozen = json.loads((PAPER / 'upstream/configs/ext/frozen_rows_ext.json').read_text(encoding='utf8'))
    rows = {row['id']: row['fusion'] for row in frozen['rows'] if 'fusion' in row}
    baseline = rows['E_yolo11s_mz1280_nms-ios_o2m']
    method = rows['E_yolo11s_mz1280_F2_o2m']
    if any(baseline[key] != method[key] for key in ('levels', 'branch', 'overlap')):
        raise ValueError('两臂必须使用相同的视图和预测头')
    params = FusionParams(**method['params'])
    overlap = method['overlap']

    def predict(image, arm):
        height, width = image.shape[:2]
        levels = []
        # 使用作者冻结的 YOLO11s 1280 行；限制切片批大小，适配本机显存。
        for name in method['levels']:
            raw = raw_predict_level(model, image, LEVELS[name], overlap, batch=1)
            levels.append(CachedLevel(name, raw['o2m_xyxy'], raw['o2m_probs'], raw['o2m_std'],
                                      raw['o2m_tile'], raw['tiles'], raw['gains'], name == 'w1280', overlap))
        if arm == 'baseline':
            detections = f0_image(levels, width, height, baseline['branch'], baseline['variant'], baseline['params'])
        else:
            detections = fuse_image_f2(levels, width, height, params, nc=len(model.names))
        # 作者输出是原图 xywh、类别从 1 开始；项目公共评测接收 xyxy 和零起始类别。
        rows = clip_xywh(detections, width, height).nlargest(500, 'score')
        return np.column_stack((rows.x, rows.y, rows.x + rows.w, rows.y + rows.h,
                                rows.score, rows.category_id - 1))

    return predict


if __name__ == '__main__':
    # 先指定已有的实验缓存区，避免第三方库在仓库根目录自动创建配置目录。
    os.environ.setdefault('YOLO_CONFIG_DIR', str(PAPER / 'local'))
    from scripts.reproduce.paired_inference import run_pairs

    run_pairs('scahi', make_detector)
