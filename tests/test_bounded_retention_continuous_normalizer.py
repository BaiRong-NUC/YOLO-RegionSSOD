"""Synthetic analytic-D tests; mpmath comparisons are NOT outward certificates.

No source data, model, optimizer or training. The high-precision triangle
reference directly integrates left then right, independently of production's
anchor-centered triangle decomposition. Qualification must remain explicit.
"""
import json
import math
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
import torch

from regionssod.training.bounded_retention import UnsupportedObservation
from regionssod.training.bounded_retention_continuous_normalizer import integrate_interval_union, normalizer


def make_logits(dtype=torch.float64, device="cpu"):
    z = torch.randn(4, 16, generator=torch.Generator().manual_seed(271), dtype=torch.float64)
    return z.to(device=device, dtype=dtype).requires_grad_()


def call(z, mask, box=(1., 2., 7., 6.), anchor=(4., 4.), stride=.5, canvas=(8., 8.)):
    return normalizer(z, box, anchor, stride, torch.as_tensor(mask, dtype=torch.bool), canvas, audit=True)


def mp_knots(lo, hi, anchor, stride, sign):
    return sorted({lo, hi, *(anchor+sign*stride*k for k in [*range(15), mp.mpf(14.99)]
                             if lo < anchor+sign*stride*k < hi)})


def mp_score(z, x, anchor, stride, sign):
    q = min(max(sign*(x-anchor)/stride, mp.mpf(0)), mp.mpf(14.99))
    k = int(mp.floor(q))
    return z[k]+(q-k)*(z[k+1]-z[k])


def mp_interval(z, lo, hi, anchor, stride, sign):
    """Independent high-precision antiderivative, not torch moment code."""
    result = mp.mpf(0)
    cuts = mp_knots(lo, hi, anchor, stride, sign)
    for left, right in zip(cuts[:-1], cuts[1:]):
        q = sign*((left+right)/2-anchor)/stride
        if q < 0 or q > mp.mpf(14.99):
            result += (right-left)*mp.exp(mp_score(z, (left+right)/2, anchor, stride, sign))
        else:
            k = int(mp.floor(q))
            b = sign*(z[k+1]-z[k])/stride
            fleft = mp.exp(mp_score(z, left, anchor, stride, sign))
            result += fleft*(right-left) if b == 0 else fleft*mp.expm1(b*(right-left))/b
    return result


def mp_triangle(zleft, zright, width, anchor, stride, dps=55):
    """Numerical high-precision outer quadrature, analytic inner integral.

    D is exactly 0<=left<right<=width. Precision agreement is a test, not a
    certification of mpmath quadrature or the production float64 evaluator.
    """
    with mp.workdps(dps):
        zl, zr = [[mp.mpf(float(x)) for x in z] for z in (zleft, zright)]
        w, a, s = map(mp.mpf, (width, anchor, stride))
        cuts = sorted(set(mp_knots(mp.mpf(0), w, a, s, -1)
                          + mp_knots(mp.mpf(0), w, a, s, 1)))
        integral = mp.quad(lambda left: mp.exp(mp_score(zl, left, a, s, -1))
                           * mp_interval(zr, left, w, a, s, 1), cuts)
        return +mp.log(integral)


@pytest.mark.parametrize("mask,volume", [
    ([True, False, False, False], 7.),
    ([False, True, False, False], 6.),
    ([False, False, True, False], 7.),
    ([False, False, False, True], 6.),
    ([True, True, False, False], 42.),
    ([True, False, True, False], 32.),
    ([False, True, False, True], 32.),
    ([True, True, True, False], 192.),
    ([True, True, False, True], 224.),
])
@pytest.mark.parametrize("anchor", [(-4., -3.), (0., 0.), (4., 4.), (8., 8.), (12., 10.)])
def test_uniform_density_exact_volumes_all_axis_layouts_and_anchor_positions(mask, volume, anchor):
    z = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True)
    result, diagnostics = call(z, mask, anchor=anchor)
    assert result.item() == pytest.approx(math.log(volume), abs=3e-14)
    assert result.dtype == torch.float64
    assert diagnostics["unknown_count"] == sum(mask)
    normalized_log_z = result-sum(mask)*math.log(16)
    assert normalized_log_z.item() == pytest.approx(math.log(volume/16**sum(mask)), abs=3e-14)
    gradient = torch.autograd.grad(result, z)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient.sum(1), torch.tensor(mask, dtype=torch.float64), rtol=0, atol=2e-14)


@pytest.mark.parametrize("side", [0, 1, 2, 3])
def test_one_unknown_matches_high_precision_antiderivative_value_and_grad(side):
    z = make_logits()
    mask = [i == side for i in range(4)]
    box, anchor, stride, canvas = (1.3, .7, 17.4, 18.2), (3.1, 8.3), .7, (20., 20.)
    actual, _ = call(z, mask, box, anchor, stride, canvas)
    axis = side % 2
    lo, hi = (0., box[side+2]) if side < 2 else (box[side-2], canvas[axis])
    sign = -1 if side < 2 else 1
    values = z.detach().numpy()
    def reference(row):
        with mp.workdps(65):
            return mp.log(mp_interval([mp.mpf(float(x)) for x in row],
                          mp.mpf(lo), mp.mpf(hi), mp.mpf(anchor[axis]), mp.mpf(stride), sign))
    assert actual.item() == pytest.approx(float(reference(values[side])), abs=2e-13)
    grad = torch.autograd.grad(actual, z)[0]
    for bin_index in (0, 1, 7, 14, 15):
        plus, minus = values[side].copy(), values[side].copy()
        plus[bin_index] += 1e-5
        minus[bin_index] -= 1e-5
        with mp.workdps(65):
            fd = float((reference(plus)-reference(minus))/mp.mpf(2e-5))
        assert grad[side, bin_index].item() == pytest.approx(fd, abs=3e-11)
    assert torch.equal(grad[~torch.tensor(mask)], torch.zeros_like(grad[~torch.tensor(mask)]))


@pytest.mark.parametrize("anchor", [-1., .37, 1.5])
def test_triangle_nonconstant_direct_high_precision_reference(anchor):
    # width1.1 / stride.09 crosses many DFL knots; outside-anchor cases include
    # whole-domain bin0 on one side and a partly saturated opposite tail.
    z = make_logits()
    actual, _ = call(z, [True, False, True, False], box=(0., .1, 1.1, .9),
                     anchor=(anchor, .5), stride=.09, canvas=(1.1, 1.))
    reference = mp_triangle(z[0].detach().tolist(), z[2].detach().tolist(), 1.1, anchor, .09)
    assert actual.item() == pytest.approx(float(reference), abs=3e-12)


@pytest.mark.parametrize("mask", [[True, False, False, False], [True, False, True, False],
                                  [True, True, True, False]])
@pytest.mark.parametrize("flat", [False, True])
def test_autograd_for_interval_triangle_and_three_unknowns(mask, flat):
    z = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True) if flat else make_logits()
    assert torch.autograd.gradcheck(lambda scores: call(scores, mask)[0], (z,),
                                   eps=1e-6, atol=2e-8, rtol=2e-6, fast_mode=True)


def test_raw_score_convention_restores_offsets_and_softmax_constants_cancel():
    z = make_logits()
    mask = torch.tensor([True, True, True, False])
    offsets = torch.tensor([100., -80., 41., 999.], dtype=torch.float64)
    base, _ = call(z, mask)
    shifted, _ = call(z+offsets[:, None], mask)
    torch.testing.assert_close(shifted-base, offsets[mask].sum(), rtol=0, atol=1e-13)
    normalized = base-torch.logsumexp(z[mask], 1).sum()
    normalized_shift = shifted-torch.logsumexp((z+offsets[:, None])[mask], 1).sum()
    torch.testing.assert_close(normalized, normalized_shift, rtol=0, atol=1e-13)
    gradient = torch.autograd.grad(normalized, z)[0]
    torch.testing.assert_close(gradient.sum(1), torch.zeros(4, dtype=torch.float64), rtol=0, atol=2e-14)


def test_fp16_promotes_to_float64_and_preserves_backward():
    half = make_logits(torch.float16)
    full = half.detach().double().requires_grad_()
    mask = [True, False, True, True]
    a, _ = call(half, mask)
    b, _ = call(full, mask)
    assert a.dtype == torch.float64 and torch.equal(a, b)
    ga, gb = torch.autograd.grad(a, half)[0], torch.autograd.grad(b, full)[0]
    assert ga.dtype == torch.float16 and torch.equal(ga, gb.half())


def test_overflow_plateau_uses_binary64_cap_interpolation_not_last_bin():
    z = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        z[2, 15] = 10.
    result, _ = call(z, [False, False, True, False], box=(30., 1., 35., 3.),
                     anchor=(0., 2.), stride=1., canvas=(40., 4.))
    expected = math.log(10.)+(14.99-14)*10.
    assert result.item() == pytest.approx(expected, abs=2e-14)
    assert abs(result.item()-(math.log(10.)+10)) > .09
    grad = torch.autograd.grad(result, z)[0]
    assert grad[2, 14].item() == pytest.approx(15-14.99, abs=1e-14)
    assert grad[2, 15].item() == pytest.approx(14.99-14, abs=1e-14)


@pytest.mark.parametrize("value", [0., 1e-8, .099, .101, 1000.])
def test_zero_near_zero_and_steep_moments_are_finite(value):
    z = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        z[0] = torch.arange(16, dtype=torch.float64)*value
        z[2] = -torch.arange(16, dtype=torch.float64)*value
    result, _ = call(z, [True, False, True, False])
    gradient = torch.autograd.grad(result, z)[0]
    assert torch.isfinite(result) and torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient.sum(1), torch.tensor([1., 0., 1., 0.], dtype=torch.float64),
                               rtol=0, atol=2e-11)


def test_geometry_and_inputs_not_mutated_and_runtime_budget_explicit():
    z = make_logits()
    box, anchor, canvas = np.array([1., 2., 7., 6.]), np.array([4., 4.]), np.array([8., 8.])
    mask = np.array([True, False, True, False])
    saved = [x.copy() for x in (box, anchor, canvas, mask)]
    logits_copy = z.detach().clone()
    _, diagnostics = normalizer(z, box, anchor, .5, mask, canvas, audit=True)
    for a, b in zip((box, anchor, canvas, mask), saved):
        np.testing.assert_array_equal(a, b)
    assert torch.equal(z, logits_copy)
    assert json.loads(json.dumps(diagnostics)) == diagnostics
    assert diagnostics["qualified"] is True
    budget = diagnostics["error_budget"]
    assert budget["geometric_residual_mass"] == budget["quadrature_residual_mass"] == 0
    assert 0 < budget["relative_arithmetic_mass_bound"] < 1e-8
    assert 0 < budget["absolute_log_error_bound"] < 1e-8
    assert 0 < budget["logit_gradient_error_bound"] < 1e-8
    assert "not C" in diagnostics["arithmetic_scope"]


@pytest.mark.parametrize("mask", [[False]*4, [True]*4])
def test_all_known_and_four_unknown_are_not_silently_routed(mask):
    with pytest.raises(ValueError, match="1..3 U"):
        call(make_logits(), mask)


@pytest.mark.parametrize("box", [(8., 2., 8., 6.), (9., 2., 10., 6.), (1., 7., 7., 6.)])
def test_zero_volume_invalid_fixed_geometry_fails(box):
    with pytest.raises(UnsupportedObservation):
        call(make_logits(), [False, False, True, False], box=box)


def test_fixed_e_and_canvas_change_d_but_unused_nominal_u_does_not():
    z = make_logits()
    a, _ = call(z, [False, False, True, False], box=(1., 2., 7., 6.))
    b, _ = call(z, [False, False, True, False], box=(1., 2., 3., 6.))
    smaller, _ = call(z, [False, False, True, False], box=(2., 2., 7., 6.))
    assert torch.equal(a, b) and smaller < a


def test_named_result_also_supports_main_loss_attribute_access():
    result = call(make_logits(), [True, False, False, False])
    integral, diagnostics = result
    assert result.log_integral is integral and result.diagnostics is diagnostics


@pytest.mark.parametrize("change", ["nan_logits", "infinite_geometry", "zero_stride", "bad_canvas", "integer_mask"])
def test_invalid_inputs_fail_explicitly(change):
    logits = make_logits().detach()
    box, anchor, stride, mask, canvas = [1., 2., 7., 6.], [4., 4.], .5, [True, False, False, False], [8., 8.]
    if change == "nan_logits":
        logits[0, 0] = float("nan")
    elif change == "infinite_geometry":
        anchor[0] = float("inf")
    elif change == "zero_stride":
        stride = 0
    elif change == "bad_canvas":
        canvas[1] = -1
    elif change == "integer_mask":
        mask = [1, 0, 0, 0]
    with pytest.raises(ValueError):
        normalizer(logits, box, anchor, stride, mask, canvas)


def test_triangle_selected_gradients_against_independent_high_precision_difference():
    z = make_logits()
    result, _ = call(z, [True, False, True, False], box=(0., .1, 1.1, .9),
                     anchor=(.37, .5), stride=.8, canvas=(1.1, 1.))
    gradient = torch.autograd.grad(result, z)[0]
    for row, column in [(0, 0), (0, 1), (2, 0), (2, 1)]:
        plus, minus = z.detach().clone(), z.detach().clone()
        plus[row, column] += 1e-5
        minus[row, column] -= 1e-5
        with mp.workdps(55):
            high = mp_triangle(plus[0].tolist(), plus[2].tolist(), 1.1, .37, .8)
            low = mp_triangle(minus[0].tolist(), minus[2].tolist(), 1.1, .37, .8)
            fd = float((high-low)/mp.mpf(2e-5))
        assert gradient[row, column].item() == pytest.approx(fd, abs=2e-11)


@pytest.mark.parametrize("violation", ["logits", "coordinates", "tiny_stride", "large_stride"])
def test_outside_declared_arithmetic_envelope_fails(violation):
    z = make_logits().detach()
    anchor, stride = (4., 4.), .5
    if violation == "logits":
        z[0, 0] = 65537.
    elif violation == "coordinates":
        anchor = (2.**31, 4.)
    elif violation == "tiny_stride":
        stride = 2.**-21
    else:
        stride = 2.**31
    with pytest.raises(UnsupportedObservation, match="arithmetic envelope"):
        call(z, [True, False, True, False], anchor=anchor, stride=stride)


@pytest.mark.parametrize("sign", [-1, 1])
def test_declared_logit_envelope_edges_qualify_with_actual_bounds(sign):
    z = torch.full((4, 16), sign*65536., dtype=torch.float64, requires_grad=True)
    result, diagnostics = call(z, [True, True, True, False])
    assert result.item() == pytest.approx(sign*3*65536+math.log(192), abs=1e-10)
    assert diagnostics["qualified"]
    assert diagnostics["error_budget"]["absolute_log_error_bound"] <= 1e-8
    assert diagnostics["error_budget"]["logit_gradient_error_bound"] <= 1e-8
    torch.autograd.grad(result, z)  # runtime verification must retain caller's graph


@pytest.mark.parametrize("fault", ["value", "gradient"])
def test_runtime_enclosure_rejects_value_or_autograd_error(monkeypatch, fault):
    import regionssod.training.bounded_retention_continuous_normalizer as module
    original = module._analytic_normalizer

    def faulty(logits, *args):
        result = original(logits, *args)
        delta = logits.new_tensor(2e-7)
        if fault == "gradient":
            delta = delta*(logits[0, 0]-logits[0, 0].detach())
        return module.NormalizerResult(result.log_integral+delta, result.diagnostics)

    monkeypatch.setattr(module, "_analytic_normalizer", faulty)
    with pytest.raises(UnsupportedObservation, match="runtime value/gradient"):
        call(make_logits(), [True, False, True, False])


def test_no_grad_caller_keeps_no_grad_result_but_still_checks_derivatives():
    with torch.no_grad():
        result, diagnostics = call(make_logits(), [True, False, True, False])
    assert not result.requires_grad and diagnostics["qualified"]


def test_decimal_enclosures_contain_independent_high_precision_triangle_result():
    z = make_logits()
    result, diagnostics = call(z, [True, False, True, False], box=(0., .1, 1.1, .9),
                               anchor=(.37, .5), stride=.8, canvas=(1.1, 1.))
    # This is numerical validation of the enclosure implementation, NOT the
    # construction/proof of its bound (directed operations construct the bound).
    with mp.workdps(110):
        reference = mp_triangle(z[0].detach().tolist(), z[2].detach().tolist(), 1.1, .37, .8, dps=110)
        lo, hi = map(mp.mpf, diagnostics["decimal_log_integral_interval"])
        assert lo <= reference <= hi
        actual_error = abs(mp.mpf(result.item())-reference)
        assert actual_error <= mp.mpf(diagnostics["error_budget"]["absolute_log_error_bound"])


@pytest.mark.parametrize("slope", [0., 1e-20, -.1, .1, -1., 1., 1.0000000000000002, -30., 30.])
def test_outward_moments_cover_high_precision_integrals_across_branches(slope):
    from regionssod.training.bounded_retention_continuous_normalizer import _decimal_moments
    a, b = Fraction.from_float(.3), Fraction.from_float(.3)+Fraction.from_float(slope)
    moments = _decimal_moments(a, b)
    with mp.workdps(110):
        aa = mp.mpf(a.numerator)/a.denominator
        bb = mp.mpf(b.numerator)/b.denominator
        for n, interval in enumerate(moments):
            reference = mp.quad(lambda t: t**n * mp.exp(aa+(bb-aa)*t), [0, 1])
            assert mp.mpf(str(interval.lo)) <= reference <= mp.mpf(str(interval.hi))


@pytest.mark.parametrize("mask", [[True, False, False, False], [True, False, True, False],
                                  [True, True, True, False]])
@pytest.mark.parametrize("magnitude", [1., 65536.])
def test_fast_kernel_matches_runtime_audit_for_value_and_gradient(mask, magnitude):
    z = make_logits().detach().sign().mul(magnitude).requires_grad_()
    args = (z, [1., 2., 7., 6.], [4., 4.], .5, mask, [8., 8.])
    fast = normalizer(*args)
    audited = normalizer(*args, audit=True)
    assert torch.equal(fast.log_integral, audited.log_integral)
    gf = torch.autograd.grad(fast.log_integral, z)[0]
    ga = torch.autograd.grad(audited.log_integral, z)[0]
    assert torch.equal(gf, ga)
    assert fast.diagnostics["qualified"] and not fast.diagnostics["runtime_checked"]
    assert fast.diagnostics["arithmetic_mode"] == "fast_conditional_budget"
    assert "not formal" in fast.diagnostics["arithmetic_assumptions"][-1]
    assert audited.diagnostics["error_budget"]["absolute_log_error_bound"] < 1e-8
    assert fast.diagnostics["error_budget"]["absolute_log_error_bound"] == 1e-7


@pytest.mark.parametrize("audit", [False, True])
def test_minimum_segment_width_triangle_survives_without_mass_underflow(audit):
    width = 2.**-900
    z = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True)
    result = normalizer(z, [0., 2., width, 6.], [4., 4.], .5,
                        [True, False, True, False], [width, 8.], audit=audit)
    assert result.log_integral.item() == pytest.approx(2*math.log(width)-math.log(2), abs=1e-11)
    assert torch.isfinite(torch.autograd.grad(result.log_integral, z)[0]).all()
    with pytest.raises(UnsupportedObservation, match=r"2\^-900"):
        normalizer(z, [0., 2., width/2, 6.], [4., 4.], .5,
                   [True, False, True, False], [width/2, 8.], audit=audit)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for synthetic AMP kernel verification")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("magnitude", [1., 65504.])
def test_cuda_amp_synthetic_value_gradient_and_runtime_budget(dtype, magnitude):
    cpu = make_logits().detach().sign().mul(magnitude).to(dtype).requires_grad_()
    cuda = cpu.detach().cuda().requires_grad_()
    geometry = ([1., 2., 7., 6.], [4., 4.], .5, [True, True, True, False], [8., 8.])
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fast = normalizer(cuda, *geometry)
        audited = normalizer(cuda, *geometry, audit=True)
    reference = normalizer(cpu, *geometry, audit=True)
    assert fast.log_integral.dtype == torch.float64 and fast.log_integral.is_cuda
    torch.testing.assert_close(fast.log_integral.cpu(), reference.log_integral, rtol=0, atol=1e-8)
    assert torch.equal(fast.log_integral, audited.log_integral)
    gpu_grad = torch.autograd.grad(fast.log_integral, cuda)[0].cpu()
    cpu_grad = torch.autograd.grad(reference.log_integral, cpu)[0]
    # Cast-back gradients are not advertised as a float64 certificate. Test
    # their ordinary dtype accuracy independently of the audited FP64 kernel.
    tolerance = {torch.float16: 5e-4, torch.float32: 2e-7, torch.float64: 1e-8}[dtype]
    torch.testing.assert_close(gpu_grad, cpu_grad, rtol=0, atol=tolerance)
    assert audited.diagnostics["qualified"] and audited.diagnostics["runtime_checked"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA synthetic derivative audit")
@pytest.mark.parametrize("profile", ["random", "moment_boundary"])
def test_cuda_nonconstant_and_moment_branch_boundary_gradients(profile):
    cpu = make_logits()
    if profile == "moment_boundary":
        with torch.no_grad():
            cpu[:] = torch.arange(16, dtype=torch.float64)[None] * torch.tensor(
                [.0999999, .1000001, -.0999999, -.1000001], dtype=torch.float64)[:, None]
    cuda = cpu.detach().cuda().requires_grad_()
    geometry = ([1., 2., 7., 6.], [4., 4.], .5, [True, True, True, False], [8., 8.])
    reference = normalizer(cpu, *geometry, audit=True)
    audited = normalizer(cuda, *geometry, audit=True)
    torch.testing.assert_close(audited.log_integral.cpu(), reference.log_integral, rtol=0, atol=1e-10)
    gpu_grad = torch.autograd.grad(audited.log_integral, cuda)[0].cpu()
    cpu_grad = torch.autograd.grad(reference.log_integral, cpu)[0]
    torch.testing.assert_close(gpu_grad, cpu_grad, rtol=0, atol=1e-10)


@pytest.mark.parametrize("audit", [False, True])
def test_union_disjoint_overlap_touching_duplicates_and_point_mass(audit):
    intervals = [(Fraction(3), Fraction(4)), (0, 2), (1, 3), (0, 2), (6, 7), (9, 9)]
    saved = list(intervals)
    z = torch.zeros(16, dtype=torch.float64, requires_grad=True)
    result = integrate_interval_union(z, intervals, 1, 1, 1, audit=audit)
    assert result.log_integral.item() == pytest.approx(math.log(5), abs=2e-14)
    assert result.diagnostics["exact_union"] == [["0", "4"], ["6", "7"]]
    assert result.diagnostics["exact_union_length"] == "5"
    assert result.diagnostics["zero_measure_input_intervals"] == 1
    assert result.diagnostics["input_intervals"] == 6 and result.diagnostics["union_intervals"] == 2
    assert intervals == saved and json.loads(json.dumps(result.diagnostics)) == result.diagnostics
    assert torch.autograd.grad(result.log_integral, z)[0].sum().item() == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize("power", [60, 900])
@pytest.mark.parametrize("audit", [False, True])
def test_union_fraction_width_below_float_ulp_is_not_collapsed(power, audit):
    lo, width = Fraction(1), Fraction(1, 2**power)
    hi = lo+width
    assert float(lo) == float(hi)
    z = torch.linspace(-.7, 1.3, 16, dtype=torch.float64, requires_grad=True)
    result = integrate_interval_union(z, [(lo, hi)], 0, 1, 1, audit=audit)
    with mp.workdps(360):
        left, right = mp.mpf(lo.numerator)/lo.denominator, mp.mpf(hi.numerator)/hi.denominator
        reference = mp.log(mp_interval(list(map(mp.mpf, z.detach().tolist())), left, right,
                                        mp.mpf(0), mp.mpf(1), 1))
        assert result.log_integral.item() == pytest.approx(float(reference), abs=2e-12)
    assert result.diagnostics["exact_union_length"] == str(width)
    gradient = torch.autograd.grad(result.log_integral, z)[0]
    assert torch.isfinite(gradient).all() and gradient[1] > .999999999
    with pytest.raises(UnsupportedObservation, match=r"2\^-900"):
        integrate_interval_union(z, [(lo, lo+Fraction(1, 2**901))], 0, 1, 1, audit=audit)


@pytest.mark.parametrize("sign", [-1, 1])
def test_union_nonconstant_high_precision_value_and_all16_gradients(sign):
    intervals = [(Fraction(-2), Fraction(3, 10)), (Fraction(11, 10), Fraction(22, 10)),
                 (Fraction(33, 10), Fraction(201, 10))]
    anchor, stride = Fraction(2, 5), Fraction(7, 10)
    z = make_logits()[0].detach().requires_grad_()
    result = integrate_interval_union(z, intervals, anchor, stride, sign, audit=True)
    gradient = torch.autograd.grad(result.log_integral, z)[0]
    with mp.workdps(110):
        def qmp(q):
            return mp.mpf(q.numerator)/q.denominator
        a, s = qmp(anchor), qmp(stride)
        cells = [(qmp(lo), qmp(hi)) for lo, hi in intervals]
        values = list(map(mp.mpf, z.detach().tolist()))
        def reference(row):
            return mp.log(sum(mp_interval(row, lo, hi, a, s, sign) for lo, hi in cells))
        assert result.log_integral.item() == pytest.approx(float(reference(values)), abs=3e-13)
        for k in range(16):
            expected = mp.diff(lambda v: reference(values[:k]+[v]+values[k+1:]), values[k])
            assert gradient[k].item() == pytest.approx(float(expected), abs=3e-12)
    assert result.diagnostics["error_budget"]["geometric_residual_mass"] == 0


@pytest.mark.parametrize("sign,interval", [(1, (-10, -2)), (-1, (2, 10))])
def test_union_negative_distance_plateau_is_bin0(sign, interval):
    z = make_logits()[0].detach().requires_grad_()
    result = integrate_interval_union(z, [interval], 0, 1, sign, audit=True)
    assert result.log_integral.item() == pytest.approx(math.log(8)+z[0].item(), abs=2e-14)
    expected = torch.zeros_like(z)
    expected[0] = 1
    torch.testing.assert_close(torch.autograd.grad(result.log_integral, z)[0], expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("sign,interval", [(1, (20, 30)), (-1, (-30, -20))])
def test_union_overflow_plateau_preserves_cap_and_both_last_bins(sign, interval):
    z = make_logits()[0].detach().requires_grad_()
    result = integrate_interval_union(z, [interval], 0, 1, sign, audit=True)
    fraction = 14.99-14
    expected = math.log(10)+(1-fraction)*z[14].item()+fraction*z[15].item()
    assert result.log_integral.item() == pytest.approx(expected, abs=2e-14)
    gradient = torch.autograd.grad(result.log_integral, z)[0]
    assert gradient[14].item() == pytest.approx(1-fraction, abs=1e-14)
    assert gradient[15].item() == pytest.approx(fraction, abs=1e-14)


@pytest.mark.parametrize("audit", [False, True])
def test_union_raw_offset_and_existing_normalizer_c_equals_d(audit):
    z = make_logits()
    denominator = normalizer(z, [1., 2., 7., 6.], [4., 4.], .5,
                             [True, False, False, False], [8., 8.], audit=audit)
    numerator = integrate_interval_union(z[0], [(0, 7)], 4, Fraction(1, 2), -1, audit=audit)
    torch.testing.assert_close(denominator.log_integral, numerator.log_integral, rtol=0, atol=1e-14)
    event = denominator.log_integral-numerator.log_integral
    torch.testing.assert_close(torch.autograd.grad(event, z)[0], torch.zeros_like(z), rtol=0, atol=1e-14)
    shifted = integrate_interval_union(z[0]+100, [(0, 7)], 4, Fraction(1, 2), -1, audit=audit)
    assert (shifted.log_integral-numerator.log_integral).item() == pytest.approx(100, abs=1e-13)


def test_union_fast_mode_does_not_run_decimal_or_internal_autograd(monkeypatch):
    import regionssod.training.bounded_retention_continuous_normalizer as module
    def forbidden(*args, **kwargs):
        raise AssertionError("audit executed on fast path")
    monkeypatch.setattr(module, "_certified_interval", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    z = make_logits()[0].detach().requires_grad_()
    result = integrate_interval_union(z, [(0, 1), (3, 4)], 0, 1, 1)
    result.log_integral.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert result.diagnostics["qualified"] and not result.diagnostics["runtime_checked"]
    assert result.diagnostics["error_budget"]["absolute_log_error_bound"] == 1e-7


@pytest.mark.parametrize("fault", ["value", "inactive_bin_gradient"])
def test_union_audit_rejects_value_or_any_of_all16_gradients(monkeypatch, fault):
    import regionssod.training.bounded_retention_continuous_normalizer as module
    original = module._analytic_interval_union
    def faulty(logits, *args):
        value = original(logits, *args)
        error = value.new_tensor(2e-7)
        if fault == "inactive_bin_gradient":
            error = error*(logits[15]-logits[15].detach())
        return value+error
    monkeypatch.setattr(module, "_analytic_interval_union", faulty)
    with pytest.raises(UnsupportedObservation, match="runtime value/gradient"):
        integrate_interval_union(make_logits()[0], [(0, 1)], 0, 1, 1, audit=True)


@pytest.mark.parametrize("intervals", [[], [(1, 1)], [(0, 0), (2, 2)]])
def test_union_zero_total_measure_fails(intervals):
    with pytest.raises(UnsupportedObservation, match="zero Lebesgue"):
        integrate_interval_union(torch.zeros(16), intervals, 0, 1, 1)


@pytest.mark.parametrize("intervals", [[(2, 1)], [(0, float("nan"))], [(0, float("inf"))], [(0, 1, 2)]])
def test_union_invalid_intervals_fail(intervals):
    with pytest.raises(ValueError):
        integrate_interval_union(torch.zeros(16), intervals, 0, 1, 1)


def test_union_segment_limit_and_coordinate_limits_not_silently_qualified():
    z = torch.zeros(16)
    with pytest.raises(UnsupportedObservation, match="segment count"):
        integrate_interval_union(z, [(2*k, 2*k+1) for k in range(129)], -100, 1, 1)
    with pytest.raises(UnsupportedObservation, match="coordinate envelope"):
        integrate_interval_union(z, [(0, 2**31)], 0, 1, 1)


def test_union_no_grad_and_interval_objects():
    from types import SimpleNamespace
    z = make_logits()[0]
    with torch.no_grad():
        result = integrate_interval_union(z, [SimpleNamespace(lo=Fraction(1, 3), hi=Fraction(4, 3))],
                                          Fraction(1, 7), Fraction(1, 2), -1, audit=True)
    assert not result.log_integral.requires_grad and result.diagnostics["runtime_checked"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA synthetic 1U union AMP test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("thin", [False, True])
def test_union_cuda_amp_matches_cpu_value_and_gradient(dtype, thin):
    cpu = make_logits()[0].detach().to(dtype).requires_grad_()
    cuda = cpu.detach().cuda().requires_grad_()
    intervals = ([(Fraction(1), Fraction(1)+Fraction(1, 2**80))] if thin else
                 [(Fraction(-2), Fraction(1, 3)), (Fraction(7, 3), Fraction(30))])
    args = (intervals, Fraction(1, 7), Fraction(7, 10), 1)
    with torch.autocast("cuda", dtype=torch.float16):
        fast = integrate_interval_union(cuda, *args)
        audited = integrate_interval_union(cuda, *args, audit=True)
    reference = integrate_interval_union(cpu, *args, audit=True)
    assert fast.log_integral.dtype == torch.float64 and fast.log_integral.is_cuda
    assert torch.equal(fast.log_integral, audited.log_integral)
    torch.testing.assert_close(fast.log_integral.cpu(), reference.log_integral, rtol=0, atol=1e-10)
    gpu_gradient = torch.autograd.grad(fast.log_integral, cuda)[0].cpu()
    cpu_gradient = torch.autograd.grad(reference.log_integral, cpu)[0]
    torch.testing.assert_close(gpu_gradient, cpu_gradient, rtol=0,
                               atol={torch.float16: 5e-4, torch.float32: 2e-7, torch.float64: 1e-10}[dtype])
