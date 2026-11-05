"""Pure synthetic continuous-domain certificates; no labels, model or loader."""
from fractions import Fraction as Q
from itertools import product
import math

import pytest

from regionssod.training.bounded_retention_observation import build_ideal_observation, build_observation
from regionssod.training.bounded_retention_continuous_domain import (
    ContinuousDomain, Interval1D, active_mask, _merge, _source_projection,
)


def record(p, affine=(10, 10, 20, 20), clip=(0, 0, 64, 64)):
    return {"physical": p, "original_unknown": p.unknown,
            "affine_bounds": tuple((Q(v), Q(v)) for v in affine),
            "augmentation_clip_window_bounds": tuple((Q(v), Q(v)) for v in clip)}


def single(*, affine=(10, 10, 20, 20), clip=(0, 0, 64, 64)):
    p = build_ideal_observation([Q(3, 4), Q(1, 2), Q(1, 2), Q(1, 5)])
    r = record(p, affine, clip)
    box = [25, 24, 30, 26] if affine[0] > 0 else [30, 24, 35, 26]
    return ContinuousDomain(r, box, (64, 64))


def three():
    p = build_ideal_observation([Q(1, 2), Q(3, 4), Q(1), Q(1, 2)])
    return ContinuousDomain(record(p), [20, 25, 30, 30], (64, 64))


def verify_joint(domain, witness):
    assert witness is not None
    assert witness.source.ratio >= domain.physical.effective_retention
    assert witness.source.input_area <= witness.source.area_cap
    assert witness.source.visible_area / witness.source.input_area == witness.source.ratio
    assert domain.point(witness.point) is not None
    for j, active in enumerate(domain.mask):
        if not active:
            assert witness.xyxy[j] == domain.box[j]


def test_one_u_exact_projection_and_point_certificates():
    d = single()
    assert d.mask == (False, False, True, False)
    assert d.active_sides == (2,)
    assert d.d_bounds == ((Q(25), Q(64)),)
    assert d.bounds == d.search_bounds == ((Q(30), Q(275, 7)),)
    assert d.intervals == (Interval1D(Q(30), Q(275, 7)),)
    for v in (Q(30), Q(32), Q(275, 7)):
        verify_joint(d, d.point((v,)))
    assert d.point((Q(30)-Q(1, 10**30),)) is None
    assert d.point((Q(275, 7)+Q(1, 10**30),)) is None
    assert d.classify(((30, 32),)) == "inside"
    assert d.classify(((40, 42),)) == "outside"
    assert d.classify(((29, 31),)) == "mixed"
    verify_joint(d, d.feasible_witness(((29, 31),)))


def test_source_projection_cache_depends_on_source_not_output_transform():
    before = _source_projection.cache_info().hits
    a, b = single(), single(affine=(-10, 10, 40, 20))
    assert _source_projection.cache_info().hits > before
    assert b.active_sides == (0,)
    assert b.intervals == (Interval1D(Q(145, 7), Q(30)),)
    for i in (0, 1, 2, 3, 4):
        x = Q(30)+(Q(275, 7)-30)*i/4
        assert a.point((x,)) is not None
        assert b.point((60-x,)) is not None


def test_interval_affine_and_uncertain_clip_have_exact_real_image():
    d = single()
    r = record(d.physical)
    r["affine_bounds"] = ((Q(9), Q(11)), (Q(10), Q(10)), (Q(19), Q(21)), (Q(20), Q(20)))
    r["augmentation_clip_window_bounds"] = ((Q(0), Q(0)), (Q(0), Q(0)),
                                             (Q(33), Q(34)), (Q(64), Q(64)))
    d = ContinuousDomain(r, [25, 24, 30, 26], (64, 64))
    assert d.intervals == (Interval1D(Q(28), Q(34)),)
    for v in (28, Q(67, 2), 34):
        verify_joint(d, d.point((v,)))
    assert d.point((Q(34)+Q(1, 10**30),)) is None


def test_strict_output_area_is_retained_as_open_interval_endpoint():
    d = single()
    r = record(d.physical)
    # Nominal E is supplied independently of the existential physical E.
    d = ContinuousDomain(r, [32, 24, 33, 26], (64, 64))
    assert d.intervals == (Interval1D(Q(32), Q(275, 7), False, True),)
    assert d.point((32,)) is None
    assert d.classify(((32, 33),)) == "mixed"
    assert d.classify(((32, 32),)) == "outside"
    verify_joint(d, d.feasible_witness(((32, 33),)))


def test_union_merging_preserves_gaps_open_points_and_no_double_counting():
    assert _merge([Interval1D(Q(0), Q(2)), Interval1D(Q(1), Q(3)),
                   Interval1D(Q(5), Q(5))]) == (Interval1D(Q(0), Q(3)), Interval1D(Q(5), Q(5)))
    a, b = Interval1D(Q(0), Q(1), True, False), Interval1D(Q(1), Q(2), False, True)
    assert _merge([a, b]) == (a, b)
    assert _merge([a, b, Interval1D(Q(1), Q(1))]) == (Interval1D(Q(0), Q(2)),)


def test_three_unknown_cell_has_joint_inside_certificate():
    d = three()
    cell = ((Q(19), Q(20)), (Q(30), Q(31)), (Q(30), Q(31)))
    cert = d.certify_cell(cell)
    assert d.active_sides == (0, 2, 3) and d.intervals is None
    assert cert.status == "inside" and len(cert.axis_proofs) == 2
    assert cert.axis_proofs[0].ratio_lower*cert.axis_proofs[1].ratio_lower == Q(25, 36)
    for point in product(*[tuple(a+(b-a)*i/3 for i in range(4)) for a, b in cell]):
        verify_joint(d, d.point(point))
    # Independently check the actual interpolated witness, not just sample membership.
    point = (Q(193, 10), Q(307, 10), Q(302, 10))
    for axis, proof in enumerate(cert.axis_proofs):
        mixed = [Q(0)]*4
        for corner, witness in proof.corners:
            weight = Q(1)
            for x, c, (lo, hi) in zip(point, corner, cell):
                weight *= (x-lo)/(hi-lo) if c == hi else (hi-x)/(hi-lo)
            for j, value in enumerate((witness.input_left, witness.input_right,
                                       witness.visible_left, witness.visible_right)):
                mixed[j] += weight*value
        assert (mixed[3]-mixed[2])/(mixed[1]-mixed[0]) >= proof.ratio_lower
        source_point = [None]*4
        source_point[axis], source_point[axis+2] = mixed[:2]
        queries = d._ranges(tuple((v, v) for v in point))
        for j in (axis, axis+2):
            assert queries[j][0] <= source_point[j] <= queries[j][1]


def test_three_u_negative_scale_has_same_joint_inside_bound():
    p = three().physical
    d = ContinuousDomain(record(p, affine=(-10, 10, 40, 20)), [30, 25, 40, 30], (64, 64))
    cell = ((29, 30), (40, 41), (30, 31))
    cert = d.certify_cell(cell)
    assert cert.status == "inside"
    assert cert.axis_proofs[0].ratio_lower*cert.axis_proofs[1].ratio_lower == Q(25, 36)
    verify_joint(d, d.feasible_witness(cell))
    for point in product(*cell):
        verify_joint(d, d.point(point))


def test_interval_parameter_multi_u_inside_certificate_keeps_one_branch():
    p = three().physical
    r = record(p)
    r["affine_bounds"] = ((9, 11), (9, 11), (19, 21), (19, 21))
    d = ContinuousDomain(r, [20, 25, 30, 30], (64, 64))
    cell = ((18, 19), (31, 32), (31, 32))
    cert = d.certify_cell(cell)
    assert cert.status == "inside"
    for point in product((Q(18), Q(37, 2), Q(19)), (Q(31), Q(63, 2), Q(32)),
                         (Q(31), Q(63, 2), Q(32))):
        verify_joint(d, d.point(point))


def test_instance_point_cache_reuses_complete_certificate_and_can_be_disabled():
    d = single()
    first = d.point((32,))
    assert first is d.point((Q(32),))
    assert d._point_cached.cache_info().hits >= 1
    uncached = ContinuousDomain(record(d.physical), [25, 24, 30, 26], (64, 64), point_cache_size=0)
    assert uncached.point((32,)) == first
    assert uncached.point((32,)) is not uncached.point((32,))
    assert _source_projection.cache_info().maxsize == 32768


def test_three_unknown_coupling_is_not_product_of_marginal_allowlists():
    d = three()
    for p in ((15, 30, 30), (20, 35, 30), (20, 30, 35)):
        verify_joint(d, d.point(p))
    assert d.point((15, 35, 35)) is None
    assert d.classify(((14, 15), (35, 36), (35, 36))) == "outside"
    assert d.classify(((15, 20), (30, 35), (30, 35))) == "mixed"
    verify_joint(d, d.feasible_witness(((15, 20), (30, 35), (30, 35))))


def test_resolved_original_unknown_remains_free_in_source_witness():
    p = build_ideal_observation([Q(1, 2), Q(3, 4), Q(1), Q(1, 2)])
    r = record(p, affine=(10, 10, 0, 0), clip=(Q(1, 10), 0, 64, 64))
    d = ContinuousDomain(r, [Q(1, 10), 5, 10, 10], (64, 64))
    assert d.active_sides == (2, 3)
    w = d.point((15, 14))
    verify_joint(d, w)
    assert w.source.x.input_left == 0 and w.xyxy[0] == Q(1, 10)
    assert w.source.ratio == Q(10, 27)


def test_thin_nonclipped_branches_survive_when_both_clipped_core_empty():
    p = build_observation(["0.5", "0.5", "0.9999999999", "0.2"])
    d = ContinuousDomain(record(p), [20, 24, 30, 26], (64, 64))
    assert d.active_sides == (0, 2)
    verify_joint(d, d.point((19, Q(30)-Q(1, 10**9))))
    verify_joint(d, d.point((Q(20)+Q(1, 10**9), 31)))
    assert d.point((19, 31)) is None
    assert d.classify(((18, 19), (31, 32))) == "outside"
    # No inference of inside from different clip-branch witnesses.
    assert d.classify(((19, 21), (29, 31))) == "mixed"


def test_clip_face_is_not_discarded_or_falsely_interpolated_across():
    p = build_ideal_observation([Q(3, 4), Q(3, 4), Q(1, 2), Q(1, 2)])
    r = record(p, clip=(0, 0, 35, 64))
    d = ContinuousDomain(r, [25, 25, 30, 30], (64, 64))
    assert d.classify(((34, 35), (30, 31))) == "mixed"
    assert d.classify(((35, 35), (30, 31))) == "inside"
    verify_joint(d, d.feasible_witness(((35, 35), (30, 31))))


def test_D_is_independent_of_retention_and_has_negative_distance_tails():
    d = single()
    assert d.classify_D(((26, 63),)) == "inside"
    assert d.classify(((26, 29),)) == "outside"
    assert d.classify(((40, 63),)) == "outside"
    # No anchor/reg_max exists in the domain API: integration owns encoding.
    assert d.d_bounds[0][1] == 64


def test_exact_resolve_helper_matches_domain_without_extra_ulp_gate():
    p = single().physical
    r = record(p, clip=(0, 0, 30, 64))
    assert active_mask(r) == (False,)*4
    d = ContinuousDomain(r, [25, 24, 30, 26], (64, 64))
    assert d.mask == active_mask(r) and d.bounds == ()
    assert d.classify(()) == "inside"
    verify_joint(d, d.point(()))
    r["augmentation_clip_window_bounds"] = ((0, 0), (0, 0),
                                             (math.nextafter(30., math.inf),)*2, (64, 64))
    assert active_mask(r) == (False, False, True, False)
    assert ContinuousDomain(r, [25, 24, 30, 26], (64, 64)).mask == active_mask(r)


def test_marginal_search_box_keeps_both_thin_branches_and_full_D_separate():
    p = build_observation(["0.5", "0.5", "0.9999999999", "0.2"])
    d = ContinuousDomain(record(p), [20, 24, 30, 26], (64, 64))
    assert d.d_bounds == ((Q(0), Q(64)), (Q(0), Q(64)))
    assert d.bounds[0][1] > 20 and d.bounds[1][0] < 30
    assert d.bounds[0][1] < 21 and d.bounds[1][0] > 29
    for point in ((Q(19), Q(30)-Q(1, 10**9)), (Q(20)+Q(1, 10**9), Q(31))):
        assert all(lo <= x <= hi for x, (lo, hi) in zip(point, d.bounds))
        verify_joint(d, d.point(point))


def test_empty_D_has_well_formed_search_box_and_explicit_outside():
    d = ContinuousDomain(record(single().physical), [70, 24, 30, 26], (64, 64))
    assert d.intervals == ()
    assert d.classify(d.bounds) == d.classify_D(d.d_bounds) == "outside"
    assert d.feasible_witness(d.bounds) is None


def test_actual_known_source_has_zero_dimensional_D_and_retains_E():
    p = build_ideal_observation([Q(1, 2), Q(1, 2), Q(1, 5), Q(1, 5)])
    d = ContinuousDomain(record(p), [24, 24, 26, 26], (64, 64))
    assert active_mask(record(p)) == (False,)*4
    assert d.active_sides == () and d.intervals is None
    assert d.classify(()) == "inside"
    assert d.point(()).xyxy == (Q(24), Q(24), Q(26), Q(26))


def test_invalid_inputs_fail_visibly_without_becoming_empty_certificates():
    d = single()
    with pytest.raises(ValueError, match="reversed"):
        d.classify(((32, 31),))
    with pytest.raises(ValueError, match="finite"):
        d.point((float("nan"),))
    r = record(d.physical)
    r["affine_bounds"] = ((-1, 1), *r["affine_bounds"][1:])
    with pytest.raises(ValueError, match="SCALE_CROSSES_ZERO"):
        ContinuousDomain(r, [25, 24, 30, 26], (64, 64))


def test_published_v3_bottom_gap_has_continuous_source_certificate():
    # Already supplied public witness constants; no receipt/label is opened.
    p = build_observation(["0.3067885117", "0.9569190608", "0.07571801671", "0.08616187839"],
                          source_ratio_upper=10.069343278363734, tile_size_lower=131, tile_size_upper=548)
    r = {"physical": p, "original_unknown": p.unknown,
         "affine_bounds": [[-496.50463104248075, -496.5046310424802],
                           [496.5046310424802, 496.50463104248075],
                           [600.9212662577627, 600.9212662577631],
                           [141.04174208641032, 141.04174208641072]],
         "augmentation_clip_window_bounds": [[-1.1368683772161605e-13, 1.1368683772161605e-13],
                                              [-1e-323, 1e-323],
                                              [639.9999999999999, 640.0000000000001],
                                              [639.9999999999998, 640.0000000000002]]}
    d = ContinuousDomain(r, [429.80218505859375, 594.7666015625, 467.39654541015625,
                             637.54638671875], (640, 640))
    assert d.active_sides == (3,)
    assert len(d.intervals) == 1
    assert Q(637) < d.intervals[0].lo < Q(638) and d.intervals[0].hi == 640
    for bottom in (638, 639, 640):
        verify_joint(d, d.point((bottom,)))
    assert d.point((636,)) is None
    assert d.classify(((638, 640),)) == "inside"
