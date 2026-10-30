import math
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch
from ultralytics.utils.loss import BboxLoss

from regionssod.training.nwd import (
    NWDDetectionLoss, SmallNWDBboxLoss, estimate_normalizer, nwd_loss_pixels,
)
from regionssod.training.p1 import P1DetectionLoss
from test_p1 import loss_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from screen_p1_small_nwd import qualifies


def test_nwd_formula_translation_shape_and_pixel_scaling():
    target = torch.tensor([[0., 0., 4., 4.]])
    shifted = target + torch.tensor([[3., 4., 3., 4.]])
    assert nwd_loss_pixels(shifted, target, 10).item() == pytest.approx(1-math.exp(-.5))
    assert torch.allclose(nwd_loss_pixels(shifted, target, 10),
                          nwd_loss_pixels(shifted*2, target*2, 20))
    expanded = torch.tensor([[-1., -1., 5., 5.]])
    assert nwd_loss_pixels(expanded, target, 10).item() == pytest.approx(
        1-math.exp(-math.sqrt(2)/10))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_exact_match_and_disjoint_containment_gradients_remain_finite(dtype):
    pred = torch.tensor([[0, 0, 4, 4], [20, 20, 24, 24], [-2, -2, 6, 6]],
                        dtype=dtype, requires_grad=True)
    target = torch.tensor([[0, 0, 4, 4]] * 3, dtype=dtype)
    losses = nwd_loss_pixels(pred, target, 12)
    assert losses.dtype == torch.float32 and losses[0].item() == 0
    losses.sum().backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.equal(pred.grad[0], torch.zeros(4, dtype=dtype))
    assert (pred.grad[1:].abs().sum(dim=1) > 0).all()


def bbox_inputs(*, large=False, stride=8.):
    target = torch.tensor([[[0., 0., 80., 80.]]] if large else [[[16., 16., 24., 24.]]]) / stride
    predicted = (target + 2 / stride).requires_grad_()
    return (torch.zeros(1, 1, 64, requires_grad=True), predicted,
            (target[0, :, :2] + target[0, :, 2:]) / 2, target, torch.ones(1, 1, 1),
            torch.tensor(1.), torch.ones(1, 1, dtype=torch.bool),
            torch.tensor([640., 640.]), torch.tensor([[stride]]))


def test_only_small_ciou_term_changes_dfl_and_large_boxes_are_exactly_preserved():
    for large in (False, True):
        args = bbox_inputs(large=large)
        base, dfl = BboxLoss(16)(*args)
        changed, changed_dfl = SmallNWDBboxLoss(16, normalizer=12)(*args)
        assert torch.equal(dfl, changed_dfl)
        assert torch.equal(base, changed) if large else not torch.equal(base, changed)


def test_stride_is_restored_before_nwd_and_denominator_is_not_renormalized():
    values = [SmallNWDBboxLoss(16, normalizer=12)(*bbox_inputs(stride=s))[0]
              for s in (8., 16., 32.)]
    assert torch.allclose(values[0], values[1], atol=1e-6)
    assert torch.allclose(values[0], values[2], atol=1e-6)
    args = list(bbox_inputs())
    loss, _ = SmallNWDBboxLoss(16, normalizer=12)(*args)
    args[5] = torch.tensor(2.)
    half, _ = SmallNWDBboxLoss(16, normalizer=12)(*args)
    assert torch.allclose(half, loss/2)


@pytest.mark.parametrize("empty,mix", [(False, 0), (False, .5), (True, .5)])
def test_full_p1_assignment_classification_and_dfl_are_preserved(empty, mix):
    model, preds, batch = loss_fixture()
    batch["bboxes"][:, 2:] = .25
    batch["pseudo_weights"] = torch.tensor([.3, 1.])
    if empty:
        batch = {key: value[:0] for key, value in batch.items()}
    model.nwd_options = {"normalizer": 12., "mix": mix, "small_side": 32.}
    original, _ = P1DetectionLoss(model)(preds, batch)
    modified, _ = NWDDetectionLoss(model)(preds, batch)
    assert torch.equal(original[1:], modified[1:])
    if mix == 0 or empty:
        assert torch.equal(original, modified)
    else:
        assert not torch.equal(original[0], modified[0])
    modified.sum().backward()
    assert all(torch.isfinite(preds[key].grad).all() for key in ("boxes", "scores"))


def test_normalizer_uses_only_open_boxes_in_resized_pixel_units(tmp_path):
    image = tmp_path / "labeled/images/train/a.jpg"
    label = tmp_path / "labeled/labels/train/a.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8))
    # Resize ratio 3.2: side 4 -> 12.8 pixels; large box is excluded.
    label.write_text("0 .5 .5 .02 .04\n0 .5 .5 .5 .5\n", encoding="utf-8")
    summary = estimate_normalizer([str(image)])
    assert summary["small_boxes"] == 1 and summary["normalizer"] == pytest.approx(12.8)
    assert not (tmp_path / "unlabeled").exists()


def test_screen_requires_fixed_ap_gain_and_nonnegative_small_recall():
    assert qualifies({"mAP50_95": .30, "small_recall": 0})
    assert not qualifies({"mAP50_95": .299, "small_recall": 1})
    assert not qualifies({"mAP50_95": .5, "small_recall": -.01})
