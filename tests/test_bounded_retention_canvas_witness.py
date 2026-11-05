"""Independent D-contract review of the disclosed source181722 witness.

No report, label, model, or loader is opened. The frozen old domain reproduces
the failure; the production V6 domain preserves E and the source certificate.
"""
from fractions import Fraction as Q
from itertools import product

import pytest

from regionssod.training.bounded_retention_observation import build_ideal_observation, build_observation
from regionssod.training.bounded_retention_continuous_domain import ContinuousDomain
from regionssod.training.bounded_retention_domain_v6 import NativeEDomain


@pytest.fixture(scope="module")
def disclosed():
    physical = build_observation(
        ["0.2919708029", "0.7892335766", "0.5839416058", "0.3959854024"],
        source_ratio_upper=10.069343278363734, tile_size_lower=131,
        tile_size_upper=548, min_visibility=.35, retention_tolerance=1e-12,
    )
    record = {
        "physical": physical, "original_unknown": [True, False, False, False],
        "affine_bounds": [
            [-627.9640579223637, -627.9640579223628],
            [627.9640579223628, 627.9640579223637],
            [249.36527264118158, 249.36527264118232],
            [-304.64127218723314, -304.6412721872328],
        ],
        "augmentation_clip_window_bounds": [
            [-1.1368683772161605e-13, 1.1368683772161605e-13],
            [-1e-323, 1e-323],
            [639.9999999999999, 640.0000000000001],
            [639.9999999999998, 640.0000000000002],
        ],
    }
    box = [-1.52587890625e-05, 66.63676452636719,
           249.36524963378906, 315.30133056640625]
    return record, box


def test_disclosed_witness_has_positive_measure_without_changing_E(disclosed):
    record, box = disclosed
    frozen = ContinuousDomain(record, box, (640, 640))
    assert frozen._empty_D and frozen.intervals == ()
    assert frozen.point((300,)) is None
    assert frozen.classify(((300, 301),)) == "outside"
    d = NativeEDomain(record, box, (640, 640))
    assert d.active_sides == (2,)  # source left becomes output right
    assert d.d_bounds == ((Q(0), Q(640)),)
    assert d.classify(((300, 301),)) == "inside"
    assert any(i.contains(Q(300)) and i.contains(Q(301)) for i in d.intervals)
    for u in (Q(300), Q(601, 2), Q(301)):
        w = d.point((u,))
        assert w is not None
        assert w.source.ratio >= d.physical.effective_retention
        assert w.source.input_area <= w.source.area_cap
        assert w.source.visible_area / w.source.input_area == w.source.ratio
        assert all(w.xyxy[j] == Q(box[j]) for j in (0, 1, 3))
        assert w.xyxy[0] == -Q(1, 65536)
        for axis, aw in enumerate((w.source.x, w.source.y)):
            a, b = aw.input_left, aw.input_right
            visible_a, visible_b = max(Q(0), min(Q(1), a)), max(Q(0), min(Q(1), b))
            center, size = d.physical.intervals[axis], d.physical.intervals[axis + 2]
            assert center.lo <= (visible_a + visible_b) / 2 <= center.hi
            assert size.lo <= visible_b - visible_a <= size.hi
            assert aw.visible_width == visible_b - visible_a
            assert 0 < b - a <= d.physical.source_ratio_upper


@pytest.mark.parametrize("cell,status", [
    (((0, 0),), "inside"), (((-1, 0),), "mixed"),
    (((-2, -1),), "outside"), (((640, 641),), "mixed"),
    (((641, 642),), "outside"),
])
def test_canvas_still_restricts_U_only(disclosed, cell, status):
    record, box = disclosed
    d = NativeEDomain(record, box, (640, 640))
    assert d.classify_D(cell) == status


def test_zero_width_still_excluded_and_known_axis_keeps_its_values(disclosed):
    record, box = disclosed
    d = NativeEDomain(record, [0, *box[1:]], (640, 640))
    assert d.classify_D(((0, 0),)) == "outside"
    assert d.classify_D(((0, 1),)) == "mixed"
    d = NativeEDomain(record, [640, *box[1:]], (640, 640))
    assert d._empty_D and not d.intervals
    d = NativeEDomain(record, [box[0], -2, box[2], -1], (640, 640))
    assert d.classify(((300, 301),)) == "inside"
    w = d.point((300,))
    assert w.xyxy[1] == -2 and w.xyxy[3] == -1
    d = NativeEDomain(record, [box[0], -1, box[2], -1], (640, 640))
    assert d._empty_D and not d.intervals


def test_reflection_keeps_upper_outside_E_and_closed_U_canvas_face(disclosed):
    record, box = disclosed
    mirrored = dict(record)
    affine = [tuple(map(Q, pair)) for pair in record["affine_bounds"]]
    clips = [tuple(map(Q, pair)) for pair in record["augmentation_clip_window_bounds"]]
    affine[0] = (-affine[0][1], -affine[0][0])
    affine[2] = (640 - affine[2][1], 640 - affine[2][0])
    clips[0], clips[2] = ((640 - clips[2][1], 640 - clips[2][0]),
                          (640 - clips[0][1], 640 - clips[0][0]))
    mirrored["affine_bounds"], mirrored["augmentation_clip_window_bounds"] = affine, clips
    known_right = Q(640) + Q(1, 16384)  # next float32 above 640
    d = NativeEDomain(mirrored, [640 - Q(box[2]), Q(box[1]), known_right, Q(box[3])],
                        (640, 640))
    assert d.active_sides == (0,) and d.d_bounds == ((Q(0), Q(640)),)
    assert d.classify(((339, 340),)) == "inside"
    assert d.classify_D(((640, 640),)) == "inside"
    assert d.point((340,)).xyxy[2] == known_right


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("sx,sy", [(10, 10), (-10, 10), (10, -10), (-10, -10)])
def test_old_valid_domain_equivalence(dimension, sx, sy):
    xywh = {
        1: (Q(3, 4), Q(1, 2), Q(1, 2), Q(1, 5)),
        2: (Q(1, 2), Q(1, 2), Q(1), Q(1, 5)),
        3: (Q(1, 2), Q(3, 4), Q(1), Q(1, 2)),
    }[dimension]
    p = build_ideal_observation(xywh)
    affine = (sx, sy, 20, 20)
    record = {"physical": p, "original_unknown": p.unknown,
              "affine_bounds": tuple((Q(v), Q(v)) for v in affine),
              "augmentation_clip_window_bounds": ((0, 0), (0, 0), (64, 64), (64, 64))}
    ends = [sorted(((xywh[a] - xywh[a + 2] / 2) * affine[a] + 20,
                    (xywh[a] + xywh[a + 2] / 2) * affine[a] + 20)) for a in (0, 1)]
    box = [ends[0][0], ends[1][0], ends[0][1], ends[1][1]]
    old, new = (cls(record, box, (64, 64)) for cls in (ContinuousDomain, NativeEDomain))
    assert not old._empty_D and not old._empty_C
    assert old.dimension == new.dimension == dimension
    for field in ("box", "mask", "active_sides", "d_bounds", "bounds", "search_bounds", "intervals"):
        assert getattr(old, field) == getattr(new, field)
    for point in product(*((Q(0), (a + b) / 2, Q(64)) for a, b in old.bounds)):
        assert old.point(point) == new.point(point)
    cell_options = [((a - 1, a + 1), (a, b), (b - 1, b + 1)) for a, b in old.bounds]
    for cell in [old.bounds, old.d_bounds, *product(*cell_options)]:
        assert old.classify_D(cell) == new.classify_D(cell)
        assert old.certify_cell(cell) == new.certify_cell(cell)
        assert old.feasible_witness(cell) == new.feasible_witness(cell)
