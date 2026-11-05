"""Uniform revised U semantics, including preserved real geometry regression."""
import numpy as np
import pytest
import torch
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors

from regionssod.training.bounded_retention_observation import build_observation
from regionssod.training.bounded_retention_objective import support_geometry as visible_support
from regionssod.training.bounded_retention_objective_v2 import support_geometry, NativeEventLoss, current_unknown
from regionssod.training.p1 import P1DetectionLoss
from test_bounded_retention_objective import fixture


def observed_regression():
    # Public legal-label/augmentation witness from integration_03, not hidden GT.
    physical = build_observation(["0.6359649109", "0.01096491252", "0.01754386105", "0.02192982504"],
        source_ratio_upper=10.069343278363734, tile_size_lower=131, tile_size_upper=548)
    obs = {"physical": physical, "original_unknown": physical.unknown,
        "affine": np.array([331.88758850097656, 331.88758850097656, 377.2526969909668, 52.00620663166046]),
        "affine_bounds": np.array([[331.8875885009764, 331.88758850097673],
            [331.8875885009764, 331.88758850097673], [377.2526969909666, 377.25269699096697],
            [52.006206631660454, 52.00620663166047]]),
        "augmentation_clip_window_bounds": np.array([[0., 0.], [45.26473999023437, 45.26473999023438],
                                                    [640., 640.], [640., 640.]])}
    box = torch.tensor([585.4102783203125, 52.00620651245117, 591.2327880859375, 59.28444290161133])
    return obs, box


def test_real_witness_unknown_nominal_negative_is_not_clamped_or_assumed_empty():
    obs, box = observed_regression()
    anchor, stride = torch.tensor([[588., 52.]]), torch.tensor([[8.]])
    native = TaskAlignedAssigner(stride=[8., 16., 32.])
    assert native.select_candidates_in_gts(anchor, box[None, None], torch.ones(1, 1, 1)).bool().all()
    old = visible_support(obs, box, anchor, stride, (640, 640))
    new = support_geometry(obs, box, anchor, stride, (640, 640))
    assert new["target"][0, 1] < 0 and not old["valid"].any()
    assert new["values"][new["valid"][0]].flatten().tolist() == [0]
    # bin1 (top44) is excluded by actual cumulative crop at45.2647, even
    # though an ideal uncropped example could admit it. No rule is relaxed.
    assert (new["geometry"][0, :, [0, 2, 3]] == box[[0, 2, 3]]).all()
    assert new["geometry"][0, 0, 1] == 52.


@pytest.mark.parametrize("x", [580., 600.])
def test_negative_exact_side_remains_illegal(x):
    obs, box = observed_regression()
    support = support_geometry(obs, box, torch.tensor([[x, 52.]]), torch.tensor([[8.]]), (640, 640))
    assert not support["valid"].any()


def test_terminal_known_top_remains_unrepresentable_after_canvas_crop():
    physical = build_observation(["0.9457831324", "0.7951807227", "0.1084337352", "0.06626506089"],
        source_ratio_upper=10.069343278363734, tile_size_lower=131, tile_size_upper=548)
    obs = {"physical": physical, "original_unknown": physical.unknown,
        "affine": np.array([331.88758850097656, 331.88758850097656, 45.365108489990234, 383.893795132637]),
        "affine_bounds": np.array([[331.8875885009764, 331.88758850097673],
            [331.8875885009764, 331.88758850097673], [45.36510848999021, 45.365108489990256],
            [383.89379513263685, 383.8937951326372]]),
        "augmentation_clip_window_bounds": np.array([[0., 0.], [45.26473999023437, 45.26473999023438],
                                                    [640., 640.], [640., 640.]])}
    box = torch.tensor([341.264892578125, 636.80810546875, 377.252685546875, 640.])
    assert current_unknown(obs, 640, 640)[0].tolist() == [False, False, True, False]
    anchors, strides = make_anchors([torch.empty(1, 1, 640 // s, 640 // s) for s in (8, 16, 32)],
                                   [8, 16, 32], .5)
    anchors *= strides
    assert anchors[:, 1].max() < box[1]
    mask = TaskAlignedAssigner(stride=[8, 16, 32]).select_candidates_in_gts(
        anchors, box[None, None], torch.ones(1, 1, 1)).reshape(-1).bool()
    assert mask.sum() == 7
    support = support_geometry(obs, box, anchors[mask], strides[mask], (640, 640))
    assert (support["target"][:, 1] < 0).all()
    assert not support["valid"].any()


@pytest.mark.parametrize("mode", ["reference", "candidate", "control"])
def test_revised_full_known_is_original_p1(mode):
    model, preds, batch = fixture()
    batch["retention_observations"][0]["original_unknown"][:] = False
    model.bounded_retention_mode = mode
    a, b = NativeEventLoss(model)(preds, batch)[0], P1DetectionLoss(model)(preds, batch)[0]
    assert torch.equal(a, b)
    ga = torch.autograd.grad(a.sum(), (preds["boxes"], preds["scores"]), retain_graph=True)
    gb = torch.autograd.grad(b.sum(), (preds["boxes"], preds["scores"]))
    assert all(torch.equal(x, y) for x, y in zip(ga, gb))


def test_revised_candidate_control_use_identical_assignments_ciou_and_distinct_dfl():
    model, preds, batch = fixture()
    a = NativeEventLoss(model)
    model.bounded_retention_mode = "control"
    b = NativeEventLoss(model)
    aa, la, _ = a.get_assigned_targets_and_loss(preds, batch)
    ab, lb, _ = b.get_assigned_targets_and_loss(preds, batch)
    assert all(torch.equal(x, y) for x, y in zip(aa, ab))
    assert torch.equal(la[:2], lb[:2]) and la[2] < lb[2]
    assert not torch.equal(torch.autograd.grad(la.sum(), preds["boxes"], retain_graph=True)[0],
                           torch.autograd.grad(lb.sum(), preds["boxes"])[0])
