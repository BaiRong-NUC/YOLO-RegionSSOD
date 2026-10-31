from pathlib import Path
import sys

import torch
from ultralytics.utils.tal import TaskAlignedAssigner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_assignment_geometry import SupportedPointAssigner, strict_inside, summarize


def test_strict_supported_pool_and_point_free_fallback_do_not_move_targets():
    points = torch.tensor([[4., 4.], [12., 4.], [20., 4.]])
    boxes = torch.tensor([[[3., 3., 5., 5.], [7., 2., 9., 6.], [0., 0., 0., 0.]]])
    valid = torch.tensor([[[True], [True], [False]]])
    original = boxes.clone()
    base = TaskAlignedAssigner(stride=[8, 16, 32]).select_candidates_in_gts(points, boxes, valid)
    changed = SupportedPointAssigner(stride=[8, 16, 32]).select_candidates_in_gts(points, boxes, valid)
    assert torch.equal(changed[0, 0], torch.tensor([True, False, False]))
    assert torch.equal(changed[0, 1], base[0, 1])
    assert not changed[0, 2].any()
    assert torch.equal(boxes, original)


def test_true_boundary_points_are_not_inside_and_both_coordinates_matter():
    points = torch.tensor([[1., 1.], [0., 1.], [1., 2.], [1., 3.]])
    boxes = torch.tensor([[[0., 0., 2., 2.]]])
    assert torch.equal(strict_inside(points, boxes), torch.tensor([[[True, False, False, False]]]))


def test_diagnostic_never_passes_without_sample_coverage_or_outside_mass():
    assert not summarize([])["worth_30_epoch_confirmation"]
    rows = []
    for group in (0, 128):
        for gt in range(110):
            values = {"anchors": 1, "score_mass": .5, "outside_anchors": 0, "outside_mass": 0.}
            rows.append({"image": group, "gt": gt, "small": True, "baseline": values,
                         "candidate": dict(values)})
    result = summarize(rows)
    assert not result["worth_30_epoch_confirmation"]
    assert all(g["checks"]["small_active_at_least_100"] for g in result["groups"].values())
