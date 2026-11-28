import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def test_matches_author_per_class():
    pytest.importorskip('numba')
    from reproduce.asanms.suppression import keep_indices

    path = Path(__file__).resolve().parents[3] / 'reproduce/asanms/upstream/ASA_NMS.py'
    if not path.is_file():
        pytest.skip('作者子模块未初始化')
    spec = importlib.util.spec_from_file_location('author_asa', path)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    # 制造近乎重复、同位置异类、小幅错位和低分候选，对比两个实现保留的原始索引。
    rng = np.random.default_rng(42)
    boxes = rng.normal([12, 11, 24, 30], 2, size=(120, 4)).astype('f')
    scores = rng.uniform(.001, 1, len(boxes)).astype('f')
    classes = rng.integers(0, 4, len(boxes))
    reference = author.distance_nms_indices_adaptive(torch.tensor(boxes), torch.tensor(scores), torch.tensor(classes))
    for limit in (3, 500):
        expected = sorted(reference.tolist(), key=lambda index: -scores[index])[:limit]
        assert keep_indices(boxes, scores, classes, limit).tolist() == expected
