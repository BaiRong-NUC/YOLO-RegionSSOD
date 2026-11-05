"""Complete head-loss checks with synthetic observations, no detector training."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from regionssod.training.bounded_retention import UnsupportedObservation
from regionssod.training.bounded_retention_objective import BoundedRetentionLoss, current_unknown, support_geometry, partial_dfl
from regionssod.training.p1 import P1DetectionLoss


class SyntheticOracle:
    """Exact synthetic left/top=.5, unknown right/bottom>=1."""
    visible_edge_bounds = np.array([[.5, .5], [.5, .5], [1., 1.], [1., 1.]])
    def compatible_ranges(self, ranges):
        r, b = np.maximum(ranges[:, 2, 0], 1), np.maximum(ranges[:, 3, 0], 1)
        return (ranges[:, 2, 1] >= r) & (ranges[:, 3, 1] >= b) & ((r - .5) * (b - .5) <= .25 / .35)


def record():
    return {"physical": SyntheticOracle(), "original_unknown": np.array([False, False, True, True]),
            "affine": np.array([32., 32., 0., 0.]), "source_object_id": 20, "occurrence_id": 0,
            "affine_bounds": np.repeat(np.array([32., 32., 0., 0.])[:, None], 2, 1),
            "augmentation_clip_window_bounds": np.repeat(np.array([0., 0., 64., 64.])[:, None], 2, 1)}


def fixture():
    model = torch.nn.Module()
    model.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1)))
    model.model = [SimpleNamespace(stride=torch.tensor([8.]), nc=2, reg_max=16)]
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    model.bounded_retention_mode = "candidate"
    gen = torch.Generator().manual_seed(29)
    preds = {"boxes": torch.randn(1, 64, 64, generator=gen, requires_grad=True),
             "scores": torch.randn(1, 2, 64, generator=gen, requires_grad=True),
             "feats": [torch.zeros(1, 1, 8, 8)]}
    batch = {"batch_idx": torch.tensor([0.]), "cls": torch.tensor([[0.]]),
             "bboxes": torch.tensor([[.375, .375, .25, .25]]), "pseudo_weights": torch.ones(1),
             "ignore_bboxes": torch.empty(0, 4), "ignore_batch_idx": torch.empty(0),
             "object_is_human": torch.tensor([True]), "object_id": torch.tensor([20]),
             "obs": {"occurrence_id": np.array([0])},
             "retention_observations": [record()]}
    return model, preds, batch


@pytest.mark.parametrize("mode", ["reference", "candidate", "control"])
def test_complete_all_known_loss_and_gradients_exactly_match_p1(mode):
    model, preds, batch = fixture()
    batch["retention_observations"][0]["original_unknown"][:] = False
    model.bounded_retention_mode = mode
    left, right = BoundedRetentionLoss(model), P1DetectionLoss(model)
    one = left(preds, batch)[0]
    two = right(preds, batch)[0]
    assert torch.equal(one, two)
    a = torch.autograd.grad(one.sum(), [preds["boxes"], preds["scores"]], retain_graph=True)
    b = torch.autograd.grad(two.sum(), [preds["boxes"], preds["scores"]])
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_full_candidate_control_share_tal_ciou_weights_and_differ_only_event_dfl():
    model, preds, batch = fixture()
    left = BoundedRetentionLoss(model)
    model.bounded_retention_mode = "control"
    right = BoundedRetentionLoss(model)
    a, loss_a, _ = left.get_assigned_targets_and_loss(preds, batch)
    b, loss_b, _ = right.get_assigned_targets_and_loss(preds, batch)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(loss_a[:2], loss_b[:2])
    assert loss_a[2] < loss_b[2]
    assert left.diagnostics["effective_anchors"] > 0
    ga = torch.autograd.grad(loss_a.sum(), [preds["boxes"], preds["scores"]], retain_graph=True)
    gb = torch.autograd.grad(loss_b.sum(), [preds["boxes"], preds["scores"]])
    assert all(torch.isfinite(x).all() for x in ga + gb)
    assert torch.equal(ga[1], gb[1])
    assert not torch.equal(ga[0], gb[0])


def test_horizontal_flip_changes_unknown_side_without_inventing_unknown():
    obs = record()
    obs["affine"] = np.array([-32., 32., 64., 0.])
    obs["affine_bounds"] = np.repeat(obs["affine"][:, None], 2, 1)
    unknown, mapping = current_unknown(obs, 64, 64)
    assert unknown.tolist() == [True, False, False, True]
    assert [(a, b) for a, b, _, _ in mapping] == [(2, 0), (3, 3)]


def test_known_canvas_crop_resolves_structural_unknown_but_retains_original_record():
    obs = record()
    obs["affine"] = np.array([64., 64., 0., 0.])
    obs["affine_bounds"] = np.repeat(obs["affine"][:, None], 2, 1)
    obs["augmentation_clip_window_bounds"][2:] = 63.
    original = copy.deepcopy(obs["original_unknown"])
    assert not current_unknown(obs, 64, 64)[0].any()
    assert np.array_equal(obs["original_unknown"], original)


def test_empty_event_is_not_turned_into_native_clamped_label():
    model, preds, batch = fixture()
    class EmptyOracle:
        visible_edge_bounds = SyntheticOracle.visible_edge_bounds
        def compatible_ranges(self, ranges):
            return np.zeros(len(ranges), dtype=bool)
    batch["retention_observations"][0]["physical"] = EmptyOracle()
    with pytest.raises(UnsupportedObservation, match="no legal discrete"):
        BoundedRetentionLoss(model)(preds, batch)


def test_support_geometry_retains_noninteger_exact_distance():
    box = torch.tensor([16.25, 16.5, 32., 32.])
    anchors = torch.tensor([[20., 20.]])
    support = support_geometry(record(), box, anchors, torch.tensor([[8.]]), (64, 64))
    assert support["valid"].any()
    assert torch.all(support["distances"][0, :, 0] == (20 - 16.25) / 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_partial_exact_edges_use_native_dfl_precision_and_gradient(dtype):
    from ultralytics.utils.loss import DFLoss
    support = support_geometry(record(), torch.tensor([16.25, 16.5, 32., 32.]),
        torch.tensor([[20., 20.]]), torch.tensor([[8.]]), (64, 64))
    logits = torch.randn(4, 16, generator=torch.Generator().manual_seed(8), dtype=dtype, requires_grad=True)
    actual = partial_dfl(logits, support, 0, "candidate")
    known = (~support["mask"]).nonzero().flatten()
    expected_known = DFLoss(16)(logits[known], support["target"][0, known][None].clone()).squeeze() * len(known) / 4
    ga = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    ge = torch.autograd.grad(expected_known, logits)[0]
    assert torch.equal(ga[known], ge[known])
    assert torch.isfinite(actual) and torch.isfinite(ga).all()


def test_partial_assignment_oom_is_not_retried_with_wrong_batch_identity(monkeypatch):
    model, preds, batch = fixture()
    criterion = BoundedRetentionLoss(model)
    calls = []
    def out_of_memory(*args):
        calls.append(1)
        raise RuntimeError("CUDA out of memory synthetic")
    monkeypatch.setattr(criterion.assigner, "_forward", out_of_memory)
    with pytest.raises(RuntimeError, match="out of memory synthetic"):
        criterion(preds, batch)
    assert len(calls) == 1


def left_record(clip_left):
    class CaptureOracle:
        visible_edge_bounds = np.array([[0., 0.], [.5, .5], [.5, .5], [1., 1.]])
        def compatible_ranges(self, ranges):
            self.last = ranges.copy()
            return np.ones(len(ranges), dtype=bool)
    obs = record()
    obs["physical"] = CaptureOracle()
    obs["original_unknown"] = np.array([True, False, False, False])
    obs["affine"] = np.array([32., 32., 20., 0.])
    obs["affine_bounds"] = np.repeat(obs["affine"][:, None], 2, 1)
    obs["augmentation_clip_window_bounds"][0] = clip_left
    return obs


def test_middle_crop_resolution_does_not_depend_only_on_final_canvas():
    obs = left_record(6.)
    obs["affine"][2] = 4.
    obs["affine_bounds"][2] = 4.
    assert 0 < obs["affine"][2] < 64
    assert not current_unknown(obs, 64, 64)[0].any()
    assert obs["original_unknown"][0]


def test_same_final_affine_different_crop_history_has_different_support():
    def event(obs):
        return support_geometry(obs, torch.tensor([20., 16., 36., 32.]),
            torch.tensor([[28., 20.]]), torch.tensor([[8.]]), (64, 64))
    a, b = event(left_record(6.)), event(left_record(1.))
    at_four = b["geometry"][0, :, 0] == 4
    assert b["valid"][0, at_four].all()
    assert not a["valid"][0, at_four].any()
    boundary = left_record(4.)
    event(boundary)
    assert np.isneginf(boundary["physical"].last[:, 0, 0]).any()


def test_known_geometry_does_not_roundtrip_through_distances():
    obs = record()
    box = torch.tensor([1e-6, 16.5, 32., 32.])
    support = support_geometry(obs, box, torch.tensor([[20., 20.]]), torch.tensor([[8.]]), (64, 64))
    assert torch.all(support["geometry"][..., 0] == box[0])


@pytest.mark.parametrize("change", ["missing", "pseudo", "object", "occurrence"])
def test_consumer_rejects_missing_or_misbound_human_evidence(change):
    model, preds, batch = fixture()
    if change == "missing":
        batch["retention_observations"] = [None]
    elif change == "pseudo":
        batch["object_is_human"][:] = False
    elif change == "object":
        batch["object_id"][:] = 21
    else:
        batch["obs"]["occurrence_id"][:] = 1
    with pytest.raises(ValueError, match="evidence|identity mismatch"):
        BoundedRetentionLoss(model)(preds, batch)


def test_zero_iou_conflict_cannot_revive_illegal_partial_anchor():
    model, _, _ = fixture()
    criterion = BoundedRetentionLoss(model)
    criterion.observations = {(0, 0): record()}
    assigner = criterion.assigner
    positives = torch.tensor([[[1., 0.], [1., 1.], [1., 1.]]])
    assigner.legal_assignment = positives.bool()
    overlaps = torch.zeros_like(positives)
    indices, fg, final = assigner.select_highest_overlaps(positives, overlaps, 3, overlaps)
    assert indices.tolist() == [[0, 1]]
    assert fg.bool().all()
    assert not (final.bool() & ~assigner.legal_assignment).any()
