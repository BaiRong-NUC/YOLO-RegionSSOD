"""Independent synthetic review: consumer gates and exact-domain counterexamples.

No detector, dataset, label, checkpoint or integration-script main is opened.
Gate stubs exercise rejection only; they do not certify numerical algorithms.
"""
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest
import torch

from regionssod.training import bounded_retention_objective_v4 as objective
from regionssod.training.bounded_retention import UnsupportedObservation
from regionssod.training.bounded_retention_continuous_domain import (
    ContinuousDomain, FREE, _axis_options, _branches, _forward, _planes, _vertices,
)
from regionssod.training.bounded_retention_continuous_numerics import (
    DFLScore, NumericalCertificationError, integrate_domain, optimize_point,
)
from regionssod.training.bounded_retention_observation import (
    build_ideal_observation, build_observation,
)


def record(physical, affine=(1, 1, 0, 0), clip=(0, 0, 2, 2)):
    return {"physical": physical, "original_unknown": physical.unknown,
            "affine_bounds": tuple((Q(x), Q(x)) for x in affine),
            "augmentation_clip_window_bounds": tuple((Q(x), Q(x)) for x in clip)}


def support_inputs():
    p = build_ideal_observation([Q(3, 4), Q(1, 2), Q(1, 2), Q(1, 5)])
    return (record(p), torch.tensor([.5, .4, 1., .6]),
            torch.tensor([[.5, .5]]), torch.tensor([[1.]]), (2, 2),
            torch.zeros(1, 4, 16, requires_grad=True))


@pytest.mark.parametrize("qualified,gap", [(False, Q(0)), (True, Q(1, 1000))])
def test_point_qualification_and_global_gap_are_both_required(monkeypatch, qualified, gap):
    monkeypatch.setattr(objective, "optimize_point", lambda *a, **k: SimpleNamespace(
        point=(Q(1),), gap=gap, diagnostics={"qualified": qualified}))
    with pytest.raises(UnsupportedObservation, match="global continuous point"):
        objective.continuous_support(*support_inputs())


@pytest.mark.parametrize("numerator_ok,residual,denominator_ok", [
    (False, 0., True), (True, 1e-3, True), (True, 0., False),
])
def test_integral_qualification_and_residual_are_consumed(
        monkeypatch, numerator_ok, residual, denominator_ok):
    inputs = support_inputs()
    support = objective.continuous_support(*inputs)
    monkeypatch.setattr(objective, "integrate_domain", lambda domain, score: SimpleNamespace(
        log_integral=score.logits.sum()*0, relative_residual_bound=residual,
        diagnostics={"qualified": numerator_ok}))
    monkeypatch.setattr(objective, "normalizer", lambda logits, *a: (
        logits.sum()*0+1, {"qualified": denominator_ok}))
    with pytest.raises(UnsupportedObservation, match="residual|normalizer"):
        objective.partial_dfl(inputs[-1][0], support, 0, "candidate")


def test_fraction_witness_certificate_does_not_transfer_to_nearest_float():
    inputs = support_inputs()
    p = inputs[0]["physical"]
    d = ContinuousDomain(record(p, affine=(1, 1, Q(1, 2**54), 0)),
                         [.5, .4, 1., .6], (2, 2))
    score = DFLScore(-torch.arange(16).double()[None], [.5], [1], 1)
    certificate = optimize_point(d, score)
    assert certificate.point == (Q(1)+Q(1, 2**54),) and certificate.gap == 0
    assert d.point(certificate.point) is not None
    # A consumer may retain Q and budget rounding, or recertify a representable
    # replacement. Simply casting does not produce another membership proof.
    assert float(certificate.point[0]) == 1.
    assert d.point(tuple(map(float, certificate.point))) is None


def test_flat_score_has_distinct_joint_witnesses_requiring_an_explicit_tie_rule():
    p = build_ideal_observation([Q(1, 2), Q(1, 4), Q(1), Q(1, 2)])
    d = ContinuousDomain(record(p, (10, 10, 20, 20), (0, 0, 64, 64)),
                         [20, 20, 30, 25], (64, 64))
    score = DFLScore(torch.zeros(3, 16), [20]*3, [-1, -1, 1], 1)
    earlier, later = (Q(10, 7), Q(20), Q(30)), (Q(20), Q(20), Q(30))
    assert d.point(earlier) is not None and d.point(later) is not None
    assert earlier < later and score.value_at(earlier) == score.value_at(later)
    # Keep the counterexample independent of the eventual chosen tie policy.
    certificate = optimize_point(d, score)
    assert d.point(certificate.point) is not None and certificate.gap == 0


def branch_cut_hints(domain):
    """Review-only splitting prototype; every child still needs certification.

    Retain ALL projected polygon vertex coordinates before merging branches.
    These are hints, not a decomposition into feasible Cartesian products.
    """
    intervals, unknown, maximum, tau = domain._key
    cuts = []
    for side in domain.active_sides:
        original = domain._mapping[side]
        axis = original % 2
        needed = tau / max(w.ratio for _, w in _axis_options(
            domain._key, 1-axis, (FREE,)*4))
        points = {Q(0), Q(1)}
        edge = domain.physical.edge_intervals[original]
        points.update((edge.lo, edge.hi))
        for branch in _branches((unknown[axis], unknown[axis+2])):
            planes, left, right = _planes(intervals[axis], intervals[axis+2],
                (unknown[axis], unknown[axis+2]), maximum, branch)
            a, b, c = (right[j]-left[j] for j in range(3))
            vertices = _vertices(planes+((-needed-a, needed-b, c),))
            points.update(v[0 if original < 2 else 1] for v in vertices)
        transport = domain._transport(side)
        output = {x for g in points for x in _forward((g, g), *transport)}
        output.update(x for pair in transport[1:] for x in pair)
        cuts.append(tuple(sorted(output)))

    def hint(cell):
        choices = [(abs(x-(lo+hi)/2)/(hi-lo), j, x)
                   for j, (lo, hi) in enumerate(cell) if hi > lo
                   for x in cuts[j] if lo < x < hi]
        if choices:
            _, j, x = min(choices)
            return j, x
        return None
    return hint


def test_branch_hints_find_mass_without_faking_residual_qualification():
    p = build_observation(["0.5", "0.5", "0.9999999999", "0.2"])
    d = ContinuousDomain(record(p, (10, 10, 20, 20), (0, 0, 64, 64)),
                         [20, 24, 30, 26], (64, 64))
    thin = ((Q(18), Q(19)), (Q(30)-Q(9, 10**10), Q(30)-Q(6, 10**10)))
    assert d.classify(thin) == "inside"
    assert (thin[0][1]-thin[0][0])*(thin[1][1]-thin[1][0]) == Q(3, 10**10)
    # Instance-only hint injection: no production domain/numerics edits.
    d.split_hint = branch_cut_hints(d)
    score = DFLScore(torch.zeros(2, 16, requires_grad=True), [20, 20], [-1, 1], 1)
    try:
        result = integrate_domain(d, score, max_cells=127)
    except NumericalCertificationError as exc:
        # Hints expose positive interior mass but are not sufficient to bound
        # every mixed cell. Resource exhaustion must remain unqualified.
        from decimal import Decimal
        assert exc.diagnostics["status"] == "RESOURCE_LIMIT"
        assert not exc.diagnostics["qualified"]
        assert Decimal(exc.diagnostics["shifted_mass_lower"]) > 0
        assert exc.diagnostics["relative_residual_bound"] > 1e-4
        return
    assert result.diagnostics["qualified"]
    assert result.relative_residual_bound <= 1e-4
    assert torch.isfinite(result.log_integral)
    assert torch.isfinite(torch.autograd.grad(result.log_integral, score.logits)[0]).all()


def test_closed_cell_can_intersect_source_only_on_a_zero_measure_face():
    p = build_observation(["0.5", "0.5", "0.9999999999", "0.2"])
    d = ContinuousDomain(record(p, (10, 10, 20, 20), (0, 0, 64, 64)),
                         [20, 24, 30, 26], (64, 64))
    # L in [18,19] forces source left<0, hence visible left=0. Joint xywh
    # constraints imply R_source<=min(width.hi, 2*center.hi)<1. Thus the
    # following closed cell intersects C only where R=ceiling, a 1D face
    # with zero 2D Lebesgue mass. Calling it pointwise outside is incorrect.
    ceiling = 20+10*min(p.intervals[2].hi, 2*p.intervals[0].hi)
    assert ceiling < 30
    cell = ((Q(18), Q(19)), (ceiling, Q(30)))
    assert d.classify(cell) == "mixed"
    assert d.point((Q(18), ceiling)) is not None
    assert d.point((Q(19), ceiling)) is not None
    assert d.point((Q(18), (ceiling+30)/2)) is None


def test_validation_trainer_delegation_uses_real_mro_without_reinitialization(monkeypatch):
    from regionssod.training.bounded_retention_training_v4 import ContinuousDensityTrainer
    from regionssod.training.p1 import P1Trainer
    sentinel = object()
    calls = []

    def parent(self, path, mode, batch):
        calls.append((type(self), path, mode, batch))
        return sentinel

    monkeypatch.setattr(P1Trainer, "build_dataset", parent)
    trainer = object.__new__(ContinuousDensityTrainer)  # No Trainer/model/data init.
    assert trainer.build_dataset("synthetic-unused", "val", 1) is sentinel
    assert calls == [(ContinuousDensityTrainer, "synthetic-unused", "val", 1)]


def assert_fast_axis_matches_reference(center, size, unknown, cap, branch, left=FREE, right=FREE):
    from regionssod.training.bounded_retention_continuous_axis import axis_max
    from regionssod.training.bounded_retention_continuous_domain import _axis, _axis_reference
    assert _axis is axis_max and _axis_reference.__wrapped__ is not axis_max.__wrapped__
    args = center, size, unknown, cap, branch, left, right
    actual, reference = axis_max.__wrapped__(*args), _axis_reference.__wrapped__(*args)
    assert (actual is None) == (reference is None), args
    if actual is None:
        return False
    assert actual.ratio == reference.ratio, (args, actual, reference)
    a, b = actual.input_left, actual.input_right
    assert isinstance(a, Q) and isinstance(b, Q) and isinstance(actual.ratio, Q)
    assert 0 < actual.visible_width <= actual.input_width <= cap
    assert -cap <= a and b <= 1+cap
    assert left[0] <= a <= left[1] and right[0] <= b <= right[1]
    assert actual.visible_left == max(Q(0), a)
    assert actual.visible_right == min(Q(1), b)
    planes, _, _ = _planes(center, size, unknown, cap, branch)
    assert all(x*a+y*b <= z for x, y, z in planes)
    return True


@pytest.mark.parametrize("branch", [(False, False), (False, True), (True, False), (True, True)])
def test_constant_count_axis_random_fraction_differential(branch):
    import random
    from regionssod.training.bounded_retention_observation import Interval
    rng = random.Random(20261105+2*branch[0]+branch[1])
    feasible = 0
    for case in range(384):
        unknown = tuple(clipped or bool(rng.randrange(2)) for clipped in branch)
        if case % 2:
            # Construct positive witnesses, then widen independent observation
            # and query intervals. Every third cap is deliberately tightened.
            a = -Q(rng.randrange(17), 8) if branch[0] else Q(rng.randrange(16), 32)
            b = 1+Q(rng.randrange(17), 8) if branch[1] else Q(rng.randrange(17, 33), 32)
            l, r = max(Q(0), a), min(Q(1), b)
            c, w = (l+r)/2, r-l
            center = Interval(c-Q(rng.randrange(5), 32), c+Q(rng.randrange(5), 32))
            size = Interval(w-Q(rng.randrange(5), 32), w+Q(rng.randrange(5), 32))
            cap = b-a+Q(rng.randrange(5), 8)
            if case % 3 == 0:
                cap = max(Q(1, 32), cap/2)
            left = (a-Q(rng.randrange(5), 8), a+Q(rng.randrange(5), 8))
            right = (b-Q(rng.randrange(5), 8), b+Q(rng.randrange(5), 8))
        else:
            # Independent ranges include incompatible observations and queries,
            # subunit M, zero size, and domains outside the initial tile.
            center = Interval(*sorted(Q(rng.randrange(-4, 21), 16) for _ in range(2)))
            size = Interval(*sorted(Q(rng.randrange(-4, 21), 16) for _ in range(2)))
            cap = Q(rng.randrange(1, 33), 8)
            left = tuple(sorted(Q(rng.randrange(-32, 33), 8) for _ in range(2)))
            right = tuple(sorted(Q(rng.randrange(-32, 33), 8) for _ in range(2)))
        if case % 11 == 0:
            left = FREE
        if case % 13 == 0:
            right = FREE
        feasible += assert_fast_axis_matches_reference(center, size, unknown, cap, branch, left, right)
    assert 50 < feasible < 384, "both feasible and infeasible cases must be exercised"


@pytest.mark.parametrize("branch,cap,left,right,expected", [
    ((True, True), Q(1), FREE, FREE, Q(1)),
    ((True, True), Q(999, 1000), FREE, FREE, None),
    ((True, False), Q(3, 2), (Q(-1), Q(-1)), FREE, Q(1, 3)),
    ((True, False), Q(1), (Q(-1), Q(-1)), FREE, None),
    ((False, True), Q(3, 2), FREE, (Q(2), Q(2)), Q(1, 3)),
    ((False, True), Q(1), FREE, (Q(2), Q(2)), None),
    ((False, False), Q(1), FREE, FREE, Q(1)),
    ((False, False), Q(1), (Q(1, 2), Q(1, 2)), (Q(1, 2), Q(1, 2)), None),
])
def test_axis_cap_contact_and_strict_positive_area(branch, cap, left, right, expected):
    from regionssod.training.bounded_retention_continuous_axis import axis_max
    from regionssod.training.bounded_retention_observation import Interval
    center = size = Interval(Q(0), Q(1))
    assert_fast_axis_matches_reference(center, size, (True, True), cap, branch, left, right)
    result = axis_max(center, size, (True, True), cap, branch, left, right)
    assert (None if result is None else result.ratio) == expected


def test_unclipped_axis_cap_and_joint_center_size_constraints_are_not_dropped():
    from regionssod.training.bounded_retention_continuous_axis import axis_max
    from regionssod.training.bounded_retention_observation import Interval
    center, size = Interval(Q(0), Q(1)), Interval(Q(1, 2), Q(1))
    assert axis_max(center, size, (False, False), Q(1, 4), (False, False)) is None
    center, size = Interval(Q(3, 4), Q(3, 4)), Interval(Q(1, 2), Q(1, 2))
    # The unique visible pair is (.5,1), incompatible with this left query.
    assert axis_max(center, size, (False, False), Q(1), (False, False), (Q(0), Q(1, 4))) is None
    assert_fast_axis_matches_reference(center, size, (False, False), Q(1), (False, False))


def test_fast_axis_large_fraction_rounding_band_and_empty_queries():
    from regionssod.training.bounded_retention_continuous_axis import axis_max
    from regionssod.training.bounded_retention_observation import Interval
    p = build_observation(["0.5", "0.5", "0.9999999999", "0.2"])
    for branch in _branches((True, True)):
        for left, right in ((FREE, FREE), ((Q(-1, 10), Q(0)), (Q(1), Q(2))),
                            ((Q(0), Q(1, 10**10)), (Q(1)-Q(1, 10**10), Q(2)))):
            assert_fast_axis_matches_reference(p.intervals[0], p.intervals[2],
                (True, True), p.source_ratio_upper, branch, left, right)
    broad = Interval(Q(0), Q(1))
    for query in ((Q(1), Q(0)), (float("inf"), float("inf")),
                  (-float("inf"), -float("inf"))):
        for branch in _branches((True, True)):
            assert axis_max(broad, broad, (True, True), Q(2), branch, query, FREE) is None
    with pytest.raises(ValueError, match="contradicts"):
        axis_max(broad, broad, (False, True), Q(2), (True, False))
    with pytest.raises(ValueError, match="positive"):
        axis_max(broad, broad, (True, True), Q(0), (False, False))


def test_control_keeps_fraction_weights_when_geometry_rounds_outside_C():
    inputs = list(support_inputs())
    inputs[0] = record(inputs[0]["physical"], affine=(1, 1, Q(1, 2**54), 0))
    inputs[2] = torch.tensor([[1., .5]])
    inputs[-1] = (-50*torch.arange(16).double()).repeat(1, 4, 1).requires_grad_()
    support = objective.continuous_support(*inputs)
    certificate = support["point_certificates"][0]
    assert certificate.point == (Q(1)+Q(1, 2**54),)
    assert support["geometry"][0, 0, 2] == 1.
    assert support["geometry_rounding_error"][0] == (Q(1, 2**54),)
    assert support["domain"].point(certificate.point) is not None
    assert support["domain"].point((1.,)) is None
    assert torch.equal(support["geometry"][0, 0, ~support["mask"]],
                       inputs[1][~support["mask"]].double())
    loss = objective.partial_dfl(inputs[-1][0], support, 0, "control")
    grad = torch.autograd.grad(loss, inputs[-1], retain_graph=True)[0]
    # q_1=2^-54 survives although the absolute coordinate rounds to 1. Using
    # rounded geometry would make this gradient positive (q_1 would be zero).
    assert grad[0, 2, 1] < 0
    expected = (torch.softmax(inputs[-1][0, 2], -1)[1]-2**-54)/4
    torch.testing.assert_close(grad[0, 2, 1], expected, rtol=1e-14, atol=0)
    support["geometry"][0, 0, 2] = 1.25
    changed = objective.partial_dfl(inputs[-1][0], support, 0, "control")
    changed_grad = torch.autograd.grad(changed, inputs[-1])[0]
    assert torch.equal(loss, changed) and torch.equal(grad, changed_grad)


@pytest.mark.parametrize("negative", [False, True])
def test_production_split_hint_is_exact_cover_preserving_and_transport_aware(negative):
    p = build_observation(["0.5", "0.5", "0.9999999999", "0.2"])
    r = record(p, (10, 10, 20, 20), (0, 0, 64, 64))
    if negative:
        r["affine_bounds"] = ((Q(-11), Q(-9)), (Q(10), Q(10)),
                              (Q(39), Q(41)), (Q(20), Q(20)))
        r["augmentation_clip_window_bounds"] = ((Q(0), Q(1)), (Q(0), Q(0)),
                                                 (Q(63), Q(64)), (Q(64), Q(64)))
    d = ContinuousDomain(r, [20, 24, 30, 26], (64, 64))
    assert d._split_cuts is None
    witness = d.feasible_witness(d.bounds)
    before = d.point(witness.point)
    hint = d.split_hint(d.bounds)
    assert hint is not None and hint == d.split_hint(d.bounds)
    assert d.point(witness.point) == before
    axis, cut = hint
    assert isinstance(cut, Q) and d.bounds[axis][0] < cut < d.bounds[axis][1]
    children = [list(d.bounds), list(d.bounds)]
    children[0][axis] = (d.bounds[axis][0], cut)
    children[1][axis] = (cut, d.bounds[axis][1])
    import math
    volume = lambda box: math.prod(hi-lo for lo, hi in box)
    assert sum(map(volume, children)) == volume(d.bounds)
    # Every child still receives ordinary certification; hints are not proofs.
    assert all(d.classify(child) in {"inside", "mixed", "outside"} for child in children)
    for j, side in enumerate(d.active_sides):
        transport = d._transport(side)
        assert all(x in d._split_cuts[j] for pair in transport[1:] for x in pair if x not in FREE)
        for source_boundary in (Q(0), Q(1)):
            assert all(x in d._split_cuts[j] for x in _forward((source_boundary,)*2, *transport))
    assert d.split_hint(tuple((x, x) for x in witness.point)) is None


def test_production_hint_retains_thin_mixed_leaves_and_posterior_both_branches(monkeypatch):
    from regionssod.training.bounded_retention_continuous_sampling import (
        _partition, sample_posterior, SamplingCertificationError,
    )
    p = build_observation(["0.5", "0.5", "0.9999999999", "0.2"])
    d = ContinuousDomain(record(p, (10, 10, 20, 20), (0, 0, 64, 64)),
                         [20, 24, 30, 26], (64, 64))
    score = DFLScore(torch.zeros(2, 16), [20, 20], [-1, 1], 1)
    with monkeypatch.context() as patch:
        patch.setattr(d, "split_hint", None)
        with pytest.raises(SamplingCertificationError) as caught:
            _partition(d, score, 63)
        assert caught.value.diagnostics["status"] == "PARTITION_RESOURCE_LIMIT"
        assert not caught.value.diagnostics["qualified"]
    leaves, _, diagnostics = _partition(d, score, 255)
    assert diagnostics["acceptance_lower_bound"] >= 1/8
    assert diagnostics["mixed_cells"] > 0 and diagnostics["inside_cells"] > 0
    # Check admitted points on BOTH thin branches against the retained cover.
    for point in ((Q(19), Q(30)-Q(1, 10**9)), (Q(20)+Q(1, 10**9), Q(31))):
        assert d.point(point) is not None
        assert any(all(lo <= x <= hi for x, (lo, hi) in zip(point, leaf.bounds)) for leaf in leaves)
    result = sample_posterior(d, score, seed=20261105, max_cells=255)
    assert result.diagnostics["qualified"] and result.diagnostics["accepted"] == 256
    assert result.diagnostics["q_mean_error_bound"] <= 1e-10
    # Flat posterior must exercise both arms of this positive-measure union.
    # This is branch coverage for one fixed seed, not a statistical bias proof.
    assert result.q_mean[0, 1:].sum() > 0
    assert result.q_mean[1, 11:].sum() > 0


def test_nominal_E_pairs_with_physical_E_without_full_box_transport_equality():
    p = build_ideal_observation([Q(3, 4), Q(1, 2), Q(1, 2), Q(1, 5)])
    native = torch.tensor([.5, .4, 1., .6], dtype=torch.float32)
    changed = native.clone()
    changed[0] = torch.nextafter(native[0], torch.tensor(1.))
    a = ContinuousDomain(record(p), native.tolist(), (2, 2))
    b = ContinuousDomain(record(p), changed.tolist(), (2, 2))
    wa, wb = a.point((Q(5, 4),)), b.point((Q(5, 4),))
    assert wa is not None and wb is not None and wa.source == wb.source
    assert wb.xyxy[0] == Q(changed[0].item()) != wb.source.x.input_left
    assert wb.source.x.input_left == Q(1, 2)
    assert wb.source.visible_area / wb.source.input_area == wb.source.ratio
    assert wb.source.ratio >= p.effective_retention
    assert b.certificate["transported_output_sides"] == (2,)
    assert not b.certificate["full_box_transport_inverse"]
