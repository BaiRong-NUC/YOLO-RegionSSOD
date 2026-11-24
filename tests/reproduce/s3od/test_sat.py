import pytest
import torch

from reproduce.s3od.sat import load_sat


def test_sat_size_boundary_in_published_config():
    """验证同一个伪框在估计阈值与使用阈值时如何分组。"""
    settings, estimate_thresholds, filter_boxes = load_sat()

    # 每行是 [中心 x, 中心 y, 宽, 高, 角度, 置信度]。每组放 102 个框，
    # 超过配置要求的 100 个样本，让两组分别用 GMM 估计阈值。
    proposals = torch.tensor([
        [0, 0, side, side, 0, score]
        for side, scores in [(30, (0.4, 0.8)), (32, (0.8, 1.0))]
        for score in scores
        for repeat in range(51)
    ])
    labels = torch.zeros(len(proposals), dtype=torch.long)
    thresholds = estimate_thresholds(
        [proposals], [labels], vaild_len=settings["cls_thr_vaild_len"],
        percent=settings["cls_thr_percent"], min=settings["cls_thr_min"],
        max=settings["cls_thr_max"], small_size=settings["small_size"],
    )
    assert thresholds == pytest.approx({"small": 0.6, "nomal": 0.9}, abs=1e-5)

    # 三个框的面积分别为 784、900、1024。统一给 0.75 分，
    # 使保留结果只取决于该框使用小目标阈值 0.6，还是普通目标阈值 0.9。
    boxes = torch.tensor([[0, 0, side, side, 0] for side in (28, 30, 32)])
    scores = torch.full((3,), 0.75)
    # 作者训练调用省略 small_size，因此筛框时沿用函数默认值 800。
    kept, _ = filter_boxes(boxes, labels[:3], scores, thresholds)
    assert kept[:, 2].tolist() == [28]

    # 对照调用显式传入估计阈值时的分界 1024，只有面积为 900 的框改变结果。
    kept, _ = filter_boxes(boxes, labels[:3], scores, thresholds, small_size=settings["small_size"])
    assert kept[:, 2].tolist() == [28, 30]
