"""Pure synthetic continuous numerics; no dataset, detector, or training."""
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from types import SimpleNamespace
import math
import time

import numpy as np
import pytest
import torch

from regionssod.training.bounded_retention_continuous_numerics import (
    CAP, DecimalBounds, DFLScore, NumericalCertificationError,
    certified_linear_exp, certify_rectangle, integrate_domain,
    integrate_interval, integrate_rectangle, optimize_point,
)


class IntervalDomain:
    def __init__(self, intervals):
        self.intervals = tuple(SimpleNamespace(lo=Q(a), hi=Q(b), left_closed=lc, right_closed=rc)
                               for a, b, lc, rc in intervals)
        self.bounds = ((min(x.lo for x in self.intervals), max(x.hi for x in self.intervals)),)

    def point(self, p):
        u, = p
        ok = any((u > i.lo or u == i.lo and i.left_closed)
                 and (u < i.hi or u == i.hi and i.right_closed) for i in self.intervals)
        return SimpleNamespace(point=tuple(p), xyxy=(u, Q(0), Q(100), Q(100))) if ok else None

    def classify(self, cell):
        lo, hi = cell[0]
        if any(lo >= i.lo and hi <= i.hi and self.point((lo,)) and self.point((hi,)) for i in self.intervals):
            return "inside"
        if all(hi < i.lo or lo > i.hi for i in self.intervals):
            return "outside"
        return "mixed"

    def feasible_witness(self, cell):
        lo, hi = cell[0]
        for i in self.intervals:
            a, b = max(lo, i.lo), min(hi, i.hi)
            if a <= b:
                result = self.point(((a + b) / 2,))
                if result is not None:
                    return result
        return None


class CoupledCornerDomain:
    """A non-factorizable union of m slabs, with exact boundary split hints."""
    intervals = None

    def __init__(self, dimension):
        self.bounds = tuple((Q(0), Q(1)) for _ in range(dimension))

    def point(self, p):
        inside = all(0 <= x <= 1 for x in p) and any(x <= Q(1, 2) for x in p)
        return SimpleNamespace(point=tuple(p), xyxy=tuple(p)) if inside else None

    def classify(self, cell):
        if any(hi <= Q(1, 2) for lo, hi in cell):
            return "inside"
        if all(lo > Q(1, 2) for lo, hi in cell):
            return "outside"
        return "mixed"

    def feasible_witness(self, cell):
        return self.point(tuple(lo for lo, hi in cell))

    def split_hint(self, cell):
        # Resolve broad half-cells first, then progressively shrink the
        # closed boundary faces. No face is mislabeled outside.
        for j, (lo, hi) in enumerate(cell):
            if lo < Q(1, 2) < hi:
                return j, Q(1, 2)
        for j, (lo, hi) in enumerate(cell):
            if lo == Q(1, 2) < hi:
                return j, (lo + hi) / 2
        return None


def score(z, dimension=1, offsets=None, signs=None, stride=1, grad=True):
    values = torch.as_tensor(z, dtype=torch.float64).reshape(dimension, 16).clone()
    values.requires_grad_(grad)
    return DFLScore(values, offsets or (0,) * dimension, signs or (1,) * dimension, stride)


def assert_qualified(result):
    assert result.diagnostics["qualified"]
    assert result.diagnostics["runtime_checked"]
    assert result.relative_residual_bound <= 1e-4
    assert result.diagnostics["log_value_error_bound"] <= 1e-4
    assert result.diagnostics["gradient_error_bound"] <= 1e-4


@pytest.mark.parametrize("va,vb,width", [(0, 0, 3), (0, 1, 1), (1, 0, 1),
    (Q(1, 3), Q(1, 3) + Q(1, 10**50), Q(1, 10**20)), (-700, -699, 2), (0, -1000, 1)])
def test_outward_exp_moments_enclose_independent_decimal_antiderivative(va, vb, width):
    out = certified_linear_exp(va, vb, width)
    with localcontext() as ctx:
        ctx.prec = 180
        def d(q):
            q = Q(q)
            return Decimal(q.numerator) / Decimal(q.denominator)
        a, b, w = d(va), d(vb), d(width)
        delta = b - a
        if delta == 0:
            exact_mass = w * a.exp()
            exact_right = exact_mass / 2
        else:
            exact_mass = w * (b.exp() - a.exp()) / delta
            exact_right = w * (b.exp() * (delta - 1) + a.exp()) / delta**2
        exact_left = exact_mass - exact_right
        for bounds, value in ((out.mass, exact_mass), (out.left, exact_left), (out.right, exact_right)):
            assert bounds.lo <= value <= bounds.hi


def test_decimal_basic_ops_enclose_fraction_without_tolerance():
    a, b = Q(1, 3), Q(-7, 11)
    da, db = DecimalBounds.exact(a), DecimalBounds.exact(b)
    for out, exact in ((da + db, a + b), (da - db, a - b), (da * db, a * b), (da / db, a / b)):
        assert Q(out.lo) <= exact <= Q(out.hi)
    with pytest.raises(NumericalCertificationError, match="zero"):
        da / DecimalBounds.fractions(-1, 1)


def test_constant_scores_have_exact_volume_and_uniform_kernel_moments():
    s = score(torch.zeros(16))
    result = integrate_rectangle(s, ((0, 1),))
    assert_qualified(result)
    assert result.log_integral.item() == 0
    gradient, = torch.autograd.grad(result.log_integral, s.logits)
    assert torch.equal(gradient[0, :2], torch.tensor([.5, .5], dtype=torch.float64))
    assert not gradient[0, 2:].any()
    cert = certify_rectangle(s, ((0, 1),))
    assert cert.mass.lo == cert.mass.hi == 1


def test_interval_union_integrates_without_double_counting_or_filling_gaps():
    domain = IntervalDomain([(0, 2, True, True), (1, 3, False, False), (4, 5, True, True)])
    s = score(torch.zeros(16))
    result = integrate_domain(domain, s)
    assert_qualified(result)
    assert result.log_integral.item() == pytest.approx(math.log(4), abs=1e-14)
    assert result.diagnostics["interval_count"] == 2


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_negative_and_upper_saturation_preserve_physical_interval_and_raw_shift(sign, dtype):
    z = torch.linspace(-3, 4, 16, dtype=dtype)[None].requires_grad_()
    s = DFLScore(z, (0,), (sign,), 1)
    tail = ((20, 40),) if sign > 0 else ((-40, -20),)
    result = integrate_rectangle(s, tail)
    assert_qualified(result)
    cap_weight = float(CAP - 14)
    expected = math.log(20) + float(z[0, 14].detach()) * (1 - cap_weight) + float(z[0, 15].detach()) * cap_weight
    assert result.log_integral.item() == pytest.approx(expected, abs=1e-13)
    low = ((-40, -20),) if sign > 0 else ((20, 40),)
    assert integrate_rectangle(s, low).log_integral.item() == pytest.approx(math.log(20) - 3)


def test_factorized_rectangle_value_and_gradient_match_one_axis_products():
    z = torch.linspace(-.7, 1.1, 48, dtype=torch.float64).reshape(3, 16).requires_grad_()
    s = DFLScore(z, (2, 5, -1), (-1, 1, 1), 2)
    bounds = ((Q(-1), Q(4)), (Q(2), Q(8)), (Q(0), Q(3)))
    result = integrate_rectangle(s, bounds)
    individual = sum(integrate_interval(s, j, lo, hi).log_integral for j, (lo, hi) in enumerate(bounds))
    assert_qualified(result)
    assert torch.allclose(result.log_integral, individual, atol=2e-14, rtol=0)
    g1, = torch.autograd.grad(result.log_integral, z, retain_graph=True)
    g2, = torch.autograd.grad(individual, z)
    assert torch.allclose(g1, g2, atol=2e-14, rtol=0)
    assert torch.allclose(g1.sum(1), torch.ones(3, dtype=torch.float64), atol=1e-14, rtol=0)


def test_autograd_gradcheck_crosses_multiple_dfl_pieces():
    z = torch.linspace(-.5, 1, 16, dtype=torch.float64)[None].requires_grad_()
    def f(logits):
        return integrate_rectangle(DFLScore(logits, (0,), (1,), 1), ((Q(-1, 3), Q(7, 3)),)).log_integral
    assert torch.autograd.gradcheck(f, (z,), eps=1e-6, atol=2e-7, rtol=1e-5)


def test_exact_point_global_uses_knots_and_saturation_not_only_interval_ends():
    z = torch.zeros(16); z[2] = 4
    s = score(z)
    d = IntervalDomain([(Q(1, 4), Q(15, 4), True, True)])
    result = optimize_point(d, s)
    assert result.point == (Q(2),) and result.gap == 0
    assert result.diagnostics["qualified"]
    assert s.value_at(result.point) == 4


def test_open_global_supremum_has_feasible_witness_and_certified_gap():
    s = score(torch.arange(16))
    domain = IntervalDomain([(0, 1, False, False)])
    result = optimize_point(domain, s)
    assert 0 < result.point[0] < 1 and domain.point(result.point)
    assert result.gap <= Q.from_float(1e-5)
    assert result.gap == 1 - s.value_at(result.point)


def test_point_cap_reports_feasible_bound_not_unproven_empty_domain():
    s = score(torch.arange(16))
    backing = IntervalDomain([(0, 1, False, False)])
    # An oracle without exact interval metadata still uses bounded B&B.
    domain = SimpleNamespace(bounds=backing.bounds, point=backing.point,
        classify=backing.classify, feasible_witness=backing.feasible_witness)
    with pytest.raises(NumericalCertificationError) as failure:
        optimize_point(domain, s, max_cells=1)
    d = failure.value.diagnostics
    assert d["status"] == "RESOURCE_LIMIT" and d["score_upper"] == 1
    assert d["score_lower"] > 0 and d["gap"] > 1e-5


def test_exact_point_ties_are_deterministic_among_visited_witnesses():
    d = IntervalDomain([(2, 3, True, True), (0, 1, True, True)])
    s = score(torch.zeros(16))
    assert optimize_point(d, s).point == optimize_point(d, s).point == (Q(0),)


class CountedIntervalDomain(IntervalDomain):
    def __init__(self, intervals):
        super().__init__(intervals)
        self.point_calls = 0

    def point(self, p):
        self.point_calls += 1
        return super().point(p)

    def classify(self, cell):
        raise AssertionError("analytic 1U path must not classify cells")

    def feasible_witness(self, cell):
        raise AssertionError("analytic 1U path must not search for witnesses")


@pytest.mark.parametrize("intervals,sign,offset,stride", [
    ([(0, 1, False, False)], 1, 0, 1),
    ([(-4, 0, False, True)], -1, 2, 2),
    ([(-100, -10, False, False)], 1, 0, 1),
    ([(20, 30, False, False)], 1, 0, 1),
    ([(0, Q(1, 2), True, True), (3, 4, True, False)], 1, 0, 1),
    ([(7, 7, True, True), (1, 2, False, True)], 1, 0, 1),
])
def test_analytic_1u_single_source_query_and_native_control_gap(intervals, sign, offset, stride):
    d = CountedIntervalDomain(intervals)
    s = score(torch.arange(16), offsets=(offset,), signs=(sign,), stride=stride)
    result = optimize_point(d, s, max_cells=1)
    assert d.point_calls == result.diagnostics["point_calls"] == 1
    assert result.diagnostics["method"] == "exact_1d_point" and result.diagnostics["qualified"]
    assert IntervalDomain.point(d, result.point) is not None
    closure_points = [x for item in d.intervals for x in s.breakpoints(0, item.lo, item.hi)]
    supremum = max(s.value_at((x,)) for x in closure_points)
    assert result.gap == supremum - s.value_at(result.point)
    assert 0 <= result.gap <= Q.from_float(1e-5)
    # Independent native continuous DFL evaluation, including fixed E and /4.
    def native_loss(x):
        distance = min(float(CAP), max(0., sign * (float(x) - offset) / stride))
        k = math.floor(distance)
        weight = distance - k
        logp = s.logits[0].log_softmax(0)
        return -(1 - weight) * logp[k] - weight * logp[k + 1]
    reference = min(float(native_loss(x).detach()) for x in closure_points)
    actual = float(native_loss(result.point[0]).detach())
    assert abs((actual + 2.75) / 4 - (reference + 2.75) / 4 - float(result.gap) / 4) < 1e-14


@pytest.mark.parametrize("include_singleton", [False, True])
def test_touching_open_intervals_preserve_excluded_knot_and_closed_singleton(include_singleton):
    intervals = [(1, 2, True, False), (2, 3, False, True)]
    if include_singleton:
        intervals.append((2, 2, True, True))
    d = CountedIntervalDomain(intervals)
    z = torch.zeros(16); z[2] = 10
    result = optimize_point(d, score(z))
    assert d.point_calls == 1
    if include_singleton:
        assert result.point == (Q(2),) and result.gap == 0
    else:
        assert result.point[0] < 2 and 0 < result.gap <= Q.from_float(1e-5)


def test_analytic_1u_global_competition_keeps_later_components_and_overlap():
    d = CountedIntervalDomain([(0, 3, True, False), (1, 4, False, True),
                              (10, 13, False, False), (5, 5, False, True)])
    z = torch.zeros(16); z[2] = 8; z[12] = 9
    result = optimize_point(d, score(z))
    assert result.point == (Q(12),) and result.gap == 0 and d.point_calls == 1
    assert result.diagnostics["nonempty_components"] == 3


def test_analytic_1u_inconsistent_final_witness_fails_without_dropping_component():
    d = CountedIntervalDomain([(0, 1, True, True)])
    d.point = lambda p: None
    with pytest.raises(NumericalCertificationError) as failure:
        optimize_point(d, score(torch.arange(16)))
    assert failure.value.diagnostics["status"] == "INCONSISTENT_INTERVAL_ORACLE"
    assert not failure.value.diagnostics["qualified"]


def test_missing_endpoint_closure_metadata_uses_generic_search():
    backing = IntervalDomain([(0, 1, False, False)])
    d = SimpleNamespace(bounds=backing.bounds, intervals=((Q(0), Q(1)),),
        point=backing.point, classify=backing.classify, feasible_witness=backing.feasible_witness)
    result = optimize_point(d, score(torch.arange(16)))
    assert result.diagnostics.get("method") != "exact_1d_point"
    assert 0 < result.point[0] < 1 and 0 < result.gap <= Q.from_float(1e-5)


@pytest.mark.parametrize("open_output_boundary", [False, True])
def test_production_synthetic_1u_domain_uses_one_final_joint_witness(open_output_boundary):
    from regionssod.training.bounded_retention_observation import build_ideal_observation
    from regionssod.training.bounded_retention_continuous_domain import ContinuousDomain
    physical = build_ideal_observation([Q(3, 4), Q(1, 2), Q(1, 2), Q(1, 5)])
    record = {"physical": physical, "original_unknown": physical.unknown,
        "affine_bounds": tuple((Q(x), Q(x)) for x in (10, 10, 20, 20)),
        "augmentation_clip_window_bounds": tuple((Q(x), Q(x)) for x in (0, 0, 64, 64))}
    box = [32, 24, 33, 26] if open_output_boundary else [25, 24, 30, 26]
    d = ContinuousDomain(record, box, (64, 64), point_cache_size=0)
    point = d.point
    calls = []
    def counted(p):
        calls.append(p)
        return point(p)
    d.point = counted
    z = -torch.arange(16) if open_output_boundary else torch.zeros(16)
    if not open_output_boundary:
        z[7] = 4
    s = score(z, offsets=(32 if open_output_boundary else 30,), stride=1 if open_output_boundary else Q(1, 2))
    result = optimize_point(d, s)
    assert len(calls) == 1 and result.diagnostics["point_calls"] == 1
    witness = point(result.point)
    assert witness is not None and witness.source.ratio >= physical.effective_retention
    for j in range(4):
        if not d.mask[j]:
            assert witness.xyxy[j] == Q(box[j])
    if open_output_boundary:
        assert result.point[0] > 32 and 0 < result.gap <= Q.from_float(1e-5)
    else:
        assert result.point == (Q(67, 2),) and result.gap == 0
        # Expose only closure-free interval pairs to force the original oracle
        # path on exactly the same source geometry and score.
        generic = SimpleNamespace(bounds=d.bounds, intervals=tuple((i.lo, i.hi) for i in d.intervals),
            point=d.point, classify=d.classify, feasible_witness=d.feasible_witness)
        calls.clear()
        reference = optimize_point(generic, s)
        assert reference.point == result.point and reference.gap == result.gap
        assert len(calls) > result.diagnostics["point_calls"]
        print({"synthetic_1U_source_queries": {"analytic": 1, "generic": len(calls)},
               "point": str(result.point[0]), "gap": str(result.gap)})


def test_analytic_1u_arbitrarily_steep_open_slope_uses_exact_gap():
    d = CountedIntervalDomain([(0, 1, True, False)])
    s = score(torch.arange(16, dtype=torch.float64) * 1e12)
    result = optimize_point(d, s)
    assert d.point_calls == 1 and result.point[0] < 1
    assert result.gap == Q.from_float(1e-5) / 2


@pytest.mark.parametrize("dimension", [2, 3])
def test_nonrectangular_coupled_slab_union_certifies_value_and_gradient(dimension):
    domain = CoupledCornerDomain(dimension)
    s = score(torch.zeros(dimension, 16), dimension)
    result = integrate_domain(domain, s, max_cells=4000)
    assert_qualified(result)
    exact = math.log(1 - .5**dimension)
    lo, hi = result.diagnostics["log_integral_bounds"]
    assert lo <= exact <= hi
    assert abs(result.log_integral.item() - exact) <= result.diagnostics["log_value_error_bound"] + 1e-15
    assert result.diagnostics["inside_cells"] > 1
    assert result.diagnostics["geometric_residual"] > 0
    gradient, = torch.autograd.grad(result.log_integral, s.logits)
    assert torch.isfinite(gradient).all()
    excluded = Q(1, 2**dimension)
    expected_q1 = (Q(1, 2) - excluded * Q(3, 4)) / (1 - excluded)
    for axis in range(dimension):
        for k, expected in ((0, 1 - expected_q1), (1, expected_q1)):
            lo, hi = result.diagnostics["gradient_bounds"][axis][k]
            assert Q.from_float(lo) <= expected <= Q.from_float(hi)
            assert abs(float(gradient[axis, k]) - float(expected)) <= result.diagnostics["gradient_error_bound"]


def test_low_global_decimal_context_does_not_shrink_error_certificate():
    s = score(torch.linspace(-.13, .81, 16))
    with localcontext() as ctx:
        ctx.prec = 3
        result = integrate_rectangle(s, ((Q(1, 7), Q(13, 7)),))
    assert_qualified(result)


def test_nonconstant_coupled_mass_and_gradient_have_independent_rectangle_union_reference():
    z = torch.zeros(2, 16, dtype=torch.float64)
    z[:, :2] = torch.tensor([[-.2, .3], [.4, -.1]], dtype=torch.float64)
    z.requires_grad_()
    s = DFLScore(z, (0, 0), (1, 1), 1)
    result = integrate_domain(CoupledCornerDomain(2), s)
    def integral(j, lo, hi):
        delta = z[j, 1] - z[j, 0]
        return torch.exp(z[j, 0] + delta * lo) * torch.expm1(delta * (hi - lo)) / delta
    reference = (integral(0, 0, .5) * integral(1, 0, 1) + integral(0, .5, 1) * integral(1, 0, .5)).log()
    ga, = torch.autograd.grad(result.log_integral, z, retain_graph=True)
    gr, = torch.autograd.grad(reference, z)
    assert abs(float(result.log_integral.detach() - reference.detach())) <= result.diagnostics["log_value_error_bound"]
    assert torch.max(torch.abs(ga - gr)) <= result.diagnostics["gradient_error_bound"] + 1e-15


def test_input_snapshot_and_nonintegral_sign_validation():
    source = torch.zeros(1, 16, dtype=torch.float64)
    s = DFLScore(source, (0,), (1,), 1)
    source[:] = 100
    assert integrate_rectangle(s, ((0, 1),)).log_integral.item() == 0
    with pytest.raises(ValueError, match="exactly"):
        DFLScore(source, (0,), (-1.2,), 1)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -1., 0., 1.])
def test_invalid_rectangle_tolerance_cannot_be_qualified(invalid):
    with pytest.raises(ValueError, match="tolerance"):
        integrate_rectangle(score(torch.zeros(16)), ((0, 1),), relative_tolerance=invalid)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, 1.5, True])
def test_search_requires_finite_integer_resource_cap(invalid):
    d = IntervalDomain([(0, 1, True, True)])
    s = score(torch.zeros(16))
    for search in (optimize_point, integrate_domain):
        with pytest.raises(ValueError):
            search(d, s, max_cells=invalid)


def test_nonempty_zero_measure_domain_is_an_explicit_failure():
    d = IntervalDomain([(1, 1, True, True)])
    s = score(torch.zeros(16))
    assert optimize_point(d, s).point == (Q(1),)
    with pytest.raises(NumericalCertificationError) as failure:
        integrate_domain(d, s)
    assert failure.value.diagnostics["status"] == "ZERO_VOLUME"


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_logits_rejected(invalid):
    z = torch.zeros(1, 16); z[0, 2] = invalid
    with pytest.raises(ValueError, match="finite"):
        DFLScore(z, (0,), (1,), 1)


def test_huge_finite_score_is_an_explicit_arithmetic_resource_failure():
    z = torch.zeros(16, dtype=torch.float64); z[0] = -1e300
    s = score(z)
    with pytest.raises(NumericalCertificationError) as failure:
        integrate_rectangle(s, ((-1, 0),))
    assert failure.value.diagnostics["status"] == "ARITHMETIC_RESOURCE_LIMIT"


def test_actual_three_unknown_curved_source_domain_reports_resource_limit():
    """Required full-domain pressure case; this is NOT a 3U integration pass.

    A successful global point plus a failed integral is the expected outcome.
    The 512-cell receipt must not be presented as a feasibility impossibility
    or as successful whole-domain integration at the 1e-4 production budget.
    """
    from regionssod.training.bounded_retention_observation import build_ideal_observation
    from regionssod.training.bounded_retention_continuous_domain import ContinuousDomain
    physical = build_ideal_observation([Q(1, 2), Q(9, 10), Q(1), Q(1, 5)])
    affine = np.array([64., 64., 64., 64.])
    record = {"physical": physical, "original_unknown": physical.original_unknown,
        "affine": affine, "affine_bounds": np.repeat(affine[:, None], 2, 1),
        "augmentation_clip_window_bounds": np.repeat(np.array([0., 0., 256., 256.])[:, None], 2, 1)}
    domain = ContinuousDomain(record, [64., 115.2, 128., 128.], (256, 256))
    assert domain.active_sides == (0, 2, 3)
    s = score(torch.zeros(3, 16), 3, offsets=(96, 96, 120), signs=(-1, 1, 1), stride=8)
    assert optimize_point(domain, s).gap == 0
    started = time.perf_counter()
    with pytest.raises(NumericalCertificationError) as failure:
        integrate_domain(domain, s, max_cells=512)
    diagnostics = failure.value.diagnostics
    assert diagnostics["status"] == "RESOURCE_LIMIT" and diagnostics["cells"] <= 512
    assert diagnostics["arithmetic_certified"] and not diagnostics["qualified"]
    assert Decimal(diagnostics["shifted_mass_upper"]) > Decimal(diagnostics["shifted_mass_lower"]) > 0
    assert diagnostics["relative_residual_bound"] > 1e-4
    # stdout can be exposed with pytest -s for the main session's receipt.
    print({"synthetic_curved_3U": diagnostics, "seconds": time.perf_counter() - started})
