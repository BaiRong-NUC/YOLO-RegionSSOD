"""Independent P1 encoding audit; synthetic tensors and published STOP witness only.

No detector, weights, data loader, raw labels, hidden GT or optimizer is used.
Native DFL encodes q=clip(d, 0, 14.99); this is NOT a claim that its decoded
four-edge prediction can realize the supplied physical/nominal geometry.
V3 absence is a dependency failure, never a skipped or fabricated pass.
"""
from contextlib import nullcontext
import copy
import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F
import ultralytics
from ultralytics.utils.loss import BboxLoss, DFLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, dist2bbox, TaskAlignedAssigner

from regionssod.training.bounded_retention import UnsupportedObservation
from regionssod.training.bounded_retention_observation import build_observation
from regionssod.training.bounded_retention_objective_v2 import support_geometry as v2_support
from regionssod.training.p1 import P1DetectionLoss


@pytest.fixture
def v3():
    # Deliberately no importorskip: the independently owned API must exist.
    return importlib.import_module("regionssod.training.bounded_retention_objective_v3")


def logits(dtype=torch.float32, shape=(4, 16), device="cpu"):
    return torch.randn(shape, generator=torch.Generator().manual_seed(105),
                       dtype=dtype).to(device).requires_grad_()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_installed_native_dfl_is_clipped_encoding_not_raw_feasibility(dtype):
    assert ultralytics.__version__ == "8.4.172"
    anchor = torch.tensor([[8., 8.]])
    box = torch.tensor([[1., 8.80810546875, 208., 12.]])
    saved_box = box.clone()
    raw = bbox2dist(anchor, box)
    assert raw[0, 1] < 0 and raw[0, 2] > 14.99
    encoded = bbox2dist(anchor, box, 15)
    assert torch.equal(encoded, raw.clamp(0, 14.99))
    assert torch.equal(box, saved_box)
    pred = logits(dtype)
    argument = raw.clone()
    direct = DFLoss(16)(pred, argument)
    native_pipeline = DFLoss(16)(pred, encoded.clone())
    assert torch.equal(argument, encoded)  # DFLoss itself encodes in place.
    assert torch.equal(direct, native_pipeline)
    gd = torch.autograd.grad(direct.sum(), pred, retain_graph=True)[0]
    gp = torch.autograd.grad(native_pipeline.sum(), pred)[0]
    assert torch.equal(gd, gp) and torch.isfinite(gd).all()
    assert raw[0, 1] < 0 and raw[0, 2] == 200


def test_native_bboxloss_ciou_keeps_original_geometry_while_dfl_encodes():
    anchor = torch.tensor([[8., 8.]])
    target = torch.tensor([[[1., 8.80810546875, 208., 12.]]])
    saved = target.clone()
    pred_box = torch.tensor([[[0., 5., 40., 15.]]], requires_grad=True)
    pred = logits(shape=(1, 1, 64))
    ciou, dfl = BboxLoss(16)(pred, pred_box, anchor, target,
        torch.ones(1, 1, 1), torch.tensor(1.), torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([256., 256.]), torch.ones(1, 1))
    original_ciou = (1 - bbox_iou(pred_box[0], target[0], xywh=False, CIoU=True)).sum()
    assert torch.equal(ciou, original_ciou)
    raw = bbox2dist(anchor, target)
    expected_dfl = DFLoss(16)(pred.reshape(-1, 16), raw.clone()).sum()
    assert torch.equal(dfl, expected_dfl)
    encoded_geometry = dist2bbox(raw.clamp(0, 14.99), anchor, xywh=False)
    changed_ciou = (1 - bbox_iou(pred_box[0], encoded_geometry[0], xywh=False, CIoU=True)).sum()
    assert not torch.equal(ciou, changed_ciou)
    assert torch.equal(torch.autograd.grad(ciou, pred_box, retain_graph=True)[0],
                       torch.autograd.grad(original_ciou, pred_box)[0])
    assert torch.equal(torch.autograd.grad(dfl, pred, retain_graph=True)[0],
                       torch.autograd.grad(expected_dfl, pred)[0])
    assert torch.equal(target, saved)


@pytest.mark.parametrize("known_count", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("mode", ["candidate", "control"])
def test_partial_exact_encoding_loss_and_grad_match_native_primitive(v3, known_count, dtype, mode):
    # Independent finite event fixture: includes negative E, overflow E and a
    # noninteger E as the number of supplied sides increases. No geometry oracle
    # is claimed by this algebra-only fixture.
    mask = torch.arange(4) >= known_count
    atoms = torch.stack([torch.zeros(4 - known_count, dtype=torch.long),
                         torch.ones(4 - known_count, dtype=torch.long)])
    raw = torch.tensor([[-.10101318359375, 20.25, .375, 2.]])
    support = {"mask": mask, "target": raw, "values": atoms,
               "valid": torch.ones(1, 2, dtype=torch.bool)}
    saved = raw.clone()
    pred = logits(dtype)
    actual = v3.partial_dfl(pred, support, 0, mode)
    native_e = DFLoss(16)(pred[~mask], raw[:, ~mask].clone()).squeeze() * known_count
    atom_logp = F.log_softmax(pred.float(), -1)[mask.nonzero().flatten()[None], atoms].sum(1)
    event = torch.logsumexp(atom_logp, 0) if mode == "candidate" else atom_logp.max()
    expected = (native_e - event) / 4
    assert torch.equal(actual, expected)
    actual_grad = torch.autograd.grad(actual, pred, retain_graph=True)[0]
    native_grad = torch.autograd.grad(native_e / 4, pred, retain_graph=True)[0]
    assert torch.equal(actual_grad[~mask], native_grad[~mask])
    # Compare against the actual native four-side primitive too, not only
    # against the selected-E reduction used by partial_dfl.
    native_four = DFLoss(16)(pred, raw.clone()).sum()
    native_four_grad = torch.autograd.grad(native_four, pred)[0]
    assert torch.equal(actual_grad[~mask], native_four_grad[~mask])
    assert torch.isfinite(actual_grad).all() and torch.equal(raw, saved)


def stopped_e_witness():
    # Copied verbatim from native_event_integration_01/result.json. Tests do not
    # open that receipt or its referenced human label path.
    physical = build_observation(["0.9457831324", "0.7951807227", "0.1084337352", "0.06626506089"],
        source_ratio_upper=10.069343278363734, tile_size_lower=131, tile_size_upper=548)
    record = {"physical": physical, "original_unknown": physical.unknown,
        "source_object_id": 142895, "occurrence_id": 51,
        "affine": np.array([331.88758850097656, 331.88758850097656, 45.365108489990234, 383.893795132637]),
        "affine_bounds": np.array([[331.8875885009764, 331.88758850097673],
            [331.8875885009764, 331.88758850097673], [45.36510848999021, 45.365108489990256],
            [383.89379513263685, 383.8937951326372]]),
        "augmentation_clip_window_bounds": np.array([[0., 0.], [45.26473999023437, 45.26473999023438],
                                                     [640., 640.], [640., 640.]])}
    box = torch.tensor([341.264892578125, 636.80810546875, 377.252685546875, 640.])
    anchors = torch.tensor([[348., 636.], [356., 636.], [364., 636.], [372., 636.],
                            [344., 632.], [360., 632.], [376., 632.]])
    strides = torch.tensor([[8.], [8.], [8.], [8.], [16.], [16.], [16.]])
    return record, box, anchors, strides


def test_real_stopped_e_stays_636_808_and_never_becomes_unknown(v3):
    record, box, anchors, strides = stopped_e_witness()
    saved_box, saved_mask = box.clone(), record["original_unknown"].copy()
    saved_affine, saved_clip = record["affine_bounds"].copy(), record["augmentation_clip_window_bounds"].copy()
    saved_certificate = copy.deepcopy(record["physical"].certificate)
    old = v2_support(record, box, anchors, strides, (640, 640))
    new = v3.support_geometry(record, box, anchors, strides, (640, 640))
    assert not old["valid"].any() and new["valid"].any()
    assert new["mask"].tolist() == [False, False, True, False]
    assert (new["target"][:, 1] < 0).all()
    assert torch.equal(new["target"], old["target"])
    assert torch.equal(new["geometry"], old["geometry"])
    assert (new["geometry"][..., 1] == 636.80810546875).all()
    assert (new["geometry"][..., ~new["mask"]] == box[~new["mask"]]).all()
    assert TaskAlignedAssigner(stride=[8., 16., 32.]).select_candidates_in_gts(
        anchors, box[None, None], torch.ones(1, 1, 1)).bool().all()
    for row in new["valid"].any(1).nonzero().flatten():
        before = new["target"].clone()
        value = v3.partial_dfl(logits(), new, row, "candidate")
        assert torch.isfinite(value) and torch.equal(new["target"], before)
    assert torch.equal(box, saved_box)
    np.testing.assert_array_equal(record["original_unknown"], saved_mask)
    np.testing.assert_array_equal(record["affine_bounds"], saved_affine)
    np.testing.assert_array_equal(record["augmentation_clip_window_bounds"], saved_clip)
    assert record["physical"].certificate == saved_certificate


def known_head_fixture(case, device="cpu"):
    # Module with head metadata only, never a detector constructor/forward.
    model = torch.nn.Module()
    model.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1, device=device)))
    model.model = [SimpleNamespace(stride=torch.tensor([8.], device=device), nc=2, reg_max=16)]
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    size = 32
    dtype = torch.float16 if device == "cuda" else torch.float32
    pred = {"boxes": logits(dtype, (1, 64, size * size), device),
            "scores": logits(dtype, (1, 2, size * size), device),
            "feats": [torch.zeros(1, 1, size, size, device=device, dtype=dtype)]}
    if case == "negative":
        # Give native TAL a positive-quality tiny-box prediction. Arbitrary
        # random wide predictions can yield no positives, vacuously bypassing
        # precisely the negative E encoding this regression must exercise.
        with torch.no_grad():
            distributions = pred["boxes"].view(1, 4, 16, -1)
            distributions.fill_(-10)
            distributions[:, :, :2] = 0
            distributions[:, 1, 0] = 10
    xyxy = torch.tensor([120., 252.80810546875, 128., 256.] if case == "negative"
                        else [0., 0., 256., 256.], device=device)
    xywh = torch.cat(((xyxy[:2] + xyxy[2:]) / 2, xyxy[2:] - xyxy[:2])) / 256
    record, _, _, _ = stopped_e_witness()
    record["original_unknown"][:] = False
    batch = {"batch_idx": torch.tensor([0.], device=device), "cls": torch.tensor([[0.]], device=device),
        "bboxes": xywh[None], "pseudo_weights": torch.ones(1, device=device),
        "ignore_bboxes": torch.empty(0, 4, device=device), "ignore_batch_idx": torch.empty(0, device=device),
        "object_is_human": torch.tensor([True], device=device),
        "object_id": torch.tensor([142895], device=device), "obs": {"occurrence_id": np.array([51])},
        "retention_observations": [record]}
    return model, pred, batch


@pytest.mark.parametrize("case", ["negative", "overflow"])
@pytest.mark.parametrize("mode", ["reference", "candidate", "control"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA AMP tensor audit requires CUDA"))])
def test_all_known_complete_loss_and_gradients_match_p1_with_active_encoding(v3, case, mode, device):
    model, pred, batch = known_head_fixture(case, device)
    model.bounded_retention_mode = mode
    native, revised = P1DetectionLoss(model), v3.P1EncodingLoss(model)
    context = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
    with context:
        assigned, _, _ = native.get_assigned_targets_and_loss(pred, batch)
        actual = revised(pred, batch)[0]
        expected = native(pred, batch)[0]
    assert torch.equal(actual, expected)
    fg, _, target, anchors, strides = assigned
    raw = bbox2dist(anchors, target / strides)
    assert fg.any(), "encoding comparison must have assigned positives"
    if case == "negative":
        assert (raw[fg] < 0).any(), "must actually encode a negative assigned E"
    else:
        assert (raw[fg] > 14.99).any(), "must actually encode an overflow assigned E"
    ga = torch.autograd.grad(actual.sum(), (pred["boxes"], pred["scores"]), retain_graph=True)
    gb = torch.autograd.grad(expected.sum(), (pred["boxes"], pred["scores"]))
    assert all(torch.equal(a, b) and torch.isfinite(a).all() for a, b in zip(ga, gb))


def test_empty_u_event_still_rejects_instead_of_using_encoded_e_as_witness(v3):
    record, box, anchors, strides = stopped_e_witness()
    class EmptyOracle:
        visible_edge_bounds = record["physical"].visible_edge_bounds
        def compatible_ranges(self, ranges):
            return np.zeros(len(ranges), dtype=bool)
    record["physical"] = EmptyOracle()
    support = v3.support_geometry(record, box, anchors, strides, (640, 640))
    assert not support["valid"].any()
    with pytest.raises(UnsupportedObservation, match="empty discrete support"):
        v3.partial_dfl(logits(), support, 0, "candidate")


def test_v3_does_not_relax_accumulated_clip_or_existing_u_atoms(v3):
    from test_bounded_retention_native_event import observed_regression
    record, box = observed_regression()
    args = (record, box, torch.tensor([[588., 52.]]), torch.tensor([[8.]]), (640, 640))
    old, new = v2_support(*args), v3.support_geometry(*args)
    for key in ("mask", "target", "values", "valid", "geometry"):
        assert torch.equal(old[key], new[key])
    assert new["values"][new["valid"][0]].flatten().tolist() == [0]


def test_frozen_p1_v1_v2_source_and_contracts_remain_unchanged():
    # Pins from the preserved native_event_integration_01 input hashes.
    root = Path(__file__).resolve().parents[1]
    frozen = {
        "src/regionssod/training/p1.py": "5aadf46f534449a28ca7313763d3cd163663355d2004cd104ad9251ecee18efb",
        "src/regionssod/training/bounded_retention_objective.py": "cd88b076084baf6a0947753b60f2f8f41c469294479a915e86e58cf03a96a49b",
        "src/regionssod/training/bounded_retention_objective_v2.py": "6796146c0b6025d5c339bf11c3e1bd3f5bddfcb927b57fc5d4eca18a537a8512",
        "src/regionssod/training/bounded_retention_observation.py": "a75aee530428f0de030c71834b266341912ef097f016353d499b4ff7f10d698e",
        "src/regionssod/training/bounded_retention_transport.py": "0855c0637367f6b11459046b53000e4104bad8c0ab8b749499a961d7b8d57494",
        "src/regionssod/training/bounded_retention_source.py": "6b9bb7344f800809c19e7269f5ffdcd761e5bd630f48c8d2958f60bd31fcfeaf",
        "src/regionssod/training/bounded_retention_training.py": "f0bd229ff668e628cfac0d927fcbd336790a57044098581bd6529baf7118a405",
        "src/regionssod/training/bounded_retention_training_v2.py": "7bf352dc4ab203b4cdefec4972fec077c2576fd4c9befa66381c7092d8f0abe9",
        "docs/paper/candidates/bounded_retention/consumer_contract.md": "3018f46dc748aafdcb985f9de51a09df6abbdd6b54b6634455b7898d4fa1d59a",
        "docs/paper/candidates/bounded_retention/native_event_addendum.md": "bbbdc3434e0154f188589e563d668c525a0a2117f27872bf8a1d33fa39773674",
    }
    for path, expected in frozen.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
