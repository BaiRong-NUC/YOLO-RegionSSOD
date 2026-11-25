"""覆盖 SAHI 配对实验使用的非默认 COCO 评测上限与小目标指标。"""

from pycocotools.coco import COCO
import pytest

from scripts.reproduce.sahi.evaluate_visdrone import evaluate


def test_small_object_after_first_hundred_detections():
    # 第 101 个候选才命中小目标，误用 COCO 默认的 100 框上限会得到零分。
    gt = COCO()
    gt.dataset = {
        "info": {}, "images": [{"id": 1, "height": 64, "width": 64}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [0, 0, 8, 8], "area": 64, "iscrowd": 0}],
    }
    gt.createIndex()
    misses = [{"image_id": 1, "category_id": 1, "bbox": [30, 30, 8, 8], "score": 0.99}] * 100
    hit = {"image_id": 1, "category_id": 1, "bbox": [0, 0, 8, 8], "score": 0.9}
    metrics = evaluate(gt, [*misses, hit])
    assert metrics["AP50_small"] == pytest.approx(100 / 101)
    assert metrics["AP_small"] == pytest.approx(metrics["AP50_small"])
    assert metrics["AP_large"] is None
    assert evaluate(gt, [])["AP_small"] == 0
