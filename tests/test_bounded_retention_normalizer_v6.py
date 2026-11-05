"""V6 synthetic native-E normalization checks; no source data/model/training.

Independent high-precision references are reused from the preserved numerical
review tests, not from production kernels. Numerical quadrature comparisons
are cross-checks, not outward certificates. Decimal audits separately check
every promoted-logit gradient under the existing arithmetic assumptions.
"""
from fractions import Fraction as Q
import math

import mpmath as mp
import numpy as np
import pytest
import torch

from regionssod.training import bounded_retention_continuous_normalizer as legacy
from regionssod.training import bounded_retention_normalizer_v6 as v6
from regionssod.training.bounded_retention import UnsupportedObservation
from test_bounded_retention_continuous_normalizer import make_logits, mp_interval, mp_triangle


DELTA = 2.**-16


def interval_reference(row, lo, hi, anchor, stride, sign):
    """Independent 70-digit antiderivative and all16 high-precision derivatives."""
    with mp.workdps(70):
        values = tuple(mp.mpf(float(x)) for x in row)
        args = tuple(mp.mpf(float(x)) for x in (lo, hi, anchor, stride))
        def log_mass(values):
            return mp.log(mp_interval(values, *args, sign))
        value = float(log_mass(values))
        gradient = [float(mp.diff(lambda x, k=k: log_mass(values[:k]+(x,)+values[k+1:]), values[k]))
                    for k in range(16)]
    return value, torch.tensor(gradient, dtype=torch.float64)


def check_budget(result, audit):
    d = result.diagnostics
    assert d["qualified"] and d["runtime_checked"] == audit
    assert d["error_budget"]["geometric_residual_mass"] == 0
    assert d["error_budget"]["quadrature_residual_mass"] == 0
    for field in ("absolute_log_error_bound", "logit_gradient_error_bound"):
        if audit:
            assert d["error_budget"][field] <= 1e-8
        else:
            assert d["error_budget"][field] == 1e-7


@pytest.mark.parametrize("side", range(4))
@pytest.mark.parametrize("anchor", [(3.1, 4.3), (-2., 12.)])
@pytest.mark.parametrize("audit", [False, True])
def test_outside_e_single_u_independent_value_and_all_logit_gradients(side, anchor, audit):
    z = make_logits()
    mask = [j == side for j in range(4)]
    box = [1., 2., 7., 6.]
    box[side+2 if side < 2 else side-2] = 8+DELTA if side < 2 else -DELTA
    result = v6.normalizer(z, box, anchor, .3, mask, (8., 8.), audit=audit)
    expected, expected_grad = interval_reference(z[side].detach(), 0, 8, anchor[side % 2], .3,
                                                 -1 if side < 2 else 1)
    assert result.log_integral.item() == pytest.approx(expected, abs=2e-12)
    gradient = torch.autograd.grad(result.log_integral, z)[0]
    torch.testing.assert_close(gradient[side], expected_grad, rtol=0, atol=3e-12)
    assert torch.equal(gradient[~torch.tensor(mask)], torch.zeros_like(gradient[~torch.tensor(mask)]))
    check_budget(result, audit)


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("audit", [False, True])
def test_reported_negative_e_magnitude_and_mirror_keep_exact_input(mirror, audit):
    # Publicly supplied witness magnitudes only; nominal U is an arbitrary
    # placeholder, and no source object or on-disk training record is loaded.
    box = np.array([300., 66.6367645, 640+DELTA, 315.3013306] if mirror else
                   [-DELTA, 66.6367645, 300., 315.3013306], dtype=np.float64)
    side = 0 if mirror else 2
    mask = np.arange(4) == side
    anchor, canvas = np.array([252., 124.]), np.array([640., 640.])
    original = [x.copy() for x in (box, mask, anchor, canvas)]
    z = make_logits()
    original_logits = z.detach().clone()
    result = v6.normalizer(z, box, anchor, 8., mask, canvas, audit=audit)
    value, derivative = interval_reference(z[side].detach(), 0, 640, 252., 8., -1 if mirror else 1)
    assert result.log_integral.item() == pytest.approx(value, abs=2e-12)
    gradient = torch.autograd.grad(result.log_integral, z)[0]
    torch.testing.assert_close(gradient[side], derivative, rtol=0, atol=3e-12)
    for current, saved in zip((box, mask, anchor, canvas), original):
        np.testing.assert_array_equal(current, saved)
    assert torch.equal(z, original_logits)
    check_budget(result, audit)


MIXED_CASES = [
    pytest.param([False, False, True, True], [-DELTA, -2., 5., 6.], 64., id="two_high_u"),
    pytest.param([True, True, False, False], [1., 2., 8+DELTA, 10.], 64., id="two_low_u"),
    pytest.param([True, False, True, True], [0., -DELTA, 8., 6.], 256., id="x_triangle_bottom_u"),
    pytest.param([True, True, True, False], [0., 2., 8., 10.], 256., id="x_triangle_top_u"),
    pytest.param([False, True, True, True], [-DELTA, 0., 5., 8.], 256., id="y_triangle_right_u"),
    pytest.param([True, True, False, True], [1., 0., 8+DELTA, 8.], 256., id="y_triangle_left_u"),
    pytest.param([False, False, False, True], [-4., 2., -2., 6.], 6., id="fixed_x_entirely_left"),
    pytest.param([False, False, False, True], [9., 2., 12., 6.], 6., id="fixed_x_entirely_right"),
    pytest.param([False, False, True, False], [1., -4., 7., -2.], 7., id="fixed_y_entirely_above"),
]


@pytest.mark.parametrize("mask,box,volume", MIXED_CASES)
@pytest.mark.parametrize("audit", [False, True])
def test_mixed_axes_flat_volume_known_axis_factor_one(mask, box, volume, audit):
    z = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True)
    result = v6.normalizer(z, box, (4., 4.), 1., mask, (8., 8.), audit=audit)
    assert result.log_integral.item() == pytest.approx(math.log(volume), abs=2e-14)
    gradient = torch.autograd.grad(result.log_integral, z)[0]
    torch.testing.assert_close(gradient.sum(1), torch.tensor(mask, dtype=torch.float64), rtol=0, atol=2e-14)
    assert torch.equal(gradient[~torch.tensor(mask)], torch.zeros_like(gradient[~torch.tensor(mask)]))
    check_budget(result, audit)


@pytest.mark.parametrize("mask,box,volume", MIXED_CASES)
def test_nonconstant_mixed_axes_independent_reference_and_audited_gradients(mask, box, volume):
    z = make_logits()
    result = v6.normalizer(z, box, (4., 4.), 1., mask, (8., 8.), audit=True)
    expected = 0.
    interval_gradients = {}
    for low, high in ((0, 2), (1, 3)):
        if mask[low] and mask[high]:
            expected += float(mp_triangle(z[low].detach().tolist(), z[high].detach().tolist(), 8., 4., 1., dps=45))
        elif mask[low] or mask[high]:
            side = low if mask[low] else high
            lo, hi = (0., min(8., box[high])) if mask[low] else (max(0., box[low]), 8.)
            value, gradient = interval_reference(z[side].detach(), lo, hi, 4., 1., -1 if side == low else 1)
            expected += value
            interval_gradients[side] = gradient
    assert result.log_integral.item() == pytest.approx(expected, abs=3e-12)
    gradient = torch.autograd.grad(result.log_integral, z)[0]
    for side, expected_gradient in interval_gradients.items():
        torch.testing.assert_close(gradient[side], expected_gradient, rtol=0, atol=3e-12)
    assert torch.equal(gradient[~torch.tensor(mask)], torch.zeros_like(gradient[~torch.tensor(mask)]))
    check_budget(result, True)  # Independent Decimal enclosure checks all64.
    assert torch.autograd.gradcheck(lambda x: v6.normalizer(x, box, (4., 4.), 1., mask, (8., 8.)).log_integral,
                                   (z,), eps=1e-6, atol=2e-8, rtol=2e-6, fast_mode=True)


@pytest.mark.parametrize("side", range(4))
@pytest.mark.parametrize("beyond", [0., 1.])
@pytest.mark.parametrize("audit", [False, True])
def test_empty_single_u_intersection_is_rejected(side, beyond, audit):
    box = [1., 2., 7., 6.]
    box[side+2 if side < 2 else side-2] = -beyond if side < 2 else 8+beyond
    with pytest.raises(UnsupportedObservation, match="zero U volume"):
        v6.normalizer(make_logits(), box, (4., 4.), 1., [i == side for i in range(4)], (8., 8.), audit=audit)


@pytest.mark.parametrize("box", [[-2., 2., -2., 6.], [-2., 2., -4., 6.], [10., 2., 9., 6.]])
@pytest.mark.parametrize("audit", [False, True])
def test_nonpositive_fixed_axis_still_rejected(box, audit):
    with pytest.raises(UnsupportedObservation, match="nonpositive box extent"):
        v6.normalizer(make_logits(), box, (4., 4.), 1., [False, False, False, True], (8., 8.), audit=audit)


@pytest.mark.parametrize("mask", [[False]*4, [True]*4])
def test_all_known_and_four_u_still_rejected_by_component(mask):
    with pytest.raises(ValueError, match="1..3 U"):
        v6.normalizer(make_logits(), [-DELTA, 2., 7., 6.], (4., 4.), 1., mask, (8., 8.))


@pytest.mark.parametrize("mask", [[True, False, False, False], [False, True, False, False],
    [False, False, True, False], [False, False, False, True], [True, True, False, False],
    [True, False, True, False], [False, True, False, True], [True, True, True, False], [True, True, False, True]])
@pytest.mark.parametrize("audit", [False, True])
def test_original_valid_domain_value_gradient_and_budgets_unchanged(mask, audit):
    z = make_logits()
    geometry = ([1., 2., 7., 6.], [4., 4.], .5, mask, [8., 8.])
    old = legacy.normalizer(z, *geometry, audit=audit)
    new = v6.normalizer(z, *geometry, audit=audit)
    assert torch.equal(old.log_integral, new.log_integral)
    assert torch.equal(torch.autograd.grad(old.log_integral, z)[0], torch.autograd.grad(new.log_integral, z)[0])
    assert old.diagnostics == new.diagnostics


def test_legacy_normalizer_retains_frozen_canvas_gate():
    geometry = ([-DELTA, 2., 7., 6.], [4., 4.], .5, [False, False, True, False], [8., 8.])
    with pytest.raises(UnsupportedObservation, match="fixed E outside canvas"):
        legacy.normalizer(make_logits(), *geometry)
    assert v6.normalizer(make_logits(), *geometry).diagnostics["qualified"]


@pytest.mark.parametrize("audit", [False, True])
def test_fraction_union_public_api_is_preserved(audit):
    z = make_logits()[0].detach().requires_grad_()
    intervals = [(Q(3), Q(3)+Q(1, 2**80)), (Q(5), Q(6))]
    old = legacy.integrate_interval_union(z, intervals, 4, Q(1, 2), -1, audit=audit)
    new = v6.integrate_interval_union(z, intervals, 4, Q(1, 2), -1, audit=audit)
    assert torch.equal(old.log_integral, new.log_integral) and old.diagnostics == new.diagnostics
    assert torch.equal(torch.autograd.grad(old.log_integral, z)[0], torch.autograd.grad(new.log_integral, z)[0])


@pytest.mark.parametrize("audit", [False, True])
def test_e_coordinate_arithmetic_envelope_is_not_removed(audit):
    args = ([4., 4.], .5, [False, False, True, False], [8., 8.])
    accepted = v6.normalizer(make_logits(), [-2.**30, 2., 7., 6.], *args, audit=audit)
    check_budget(accepted, audit)
    with pytest.raises(UnsupportedObservation, match="arithmetic envelope"):
        v6.normalizer(make_logits(), [-2.**30-1, 2., 7., 6.], *args, audit=audit)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="synthetic CUDA AMP/audit needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("mask", [[False, False, True, False], [False, True, True, True]])
def test_cuda_amp_outside_e_fast_and_audit_match_cpu_promoted_values_and_gradients(dtype, mask):
    cpu = make_logits(dtype=dtype)
    cuda = cpu.detach().cuda().requires_grad_()
    geometry = ([-DELTA, 2., 7., 6.], [3.1, 4.3], .3, mask, [8., 8.])
    reference = v6.normalizer(cpu, *geometry, audit=True)
    with torch.autocast("cuda", dtype=torch.float16):
        fast = v6.normalizer(cuda, *geometry)
        audited = v6.normalizer(cuda, *geometry, audit=True)
    assert fast.log_integral.dtype == torch.float64 and fast.log_integral.is_cuda
    assert torch.equal(fast.log_integral, audited.log_integral)
    torch.testing.assert_close(fast.log_integral.cpu(), reference.log_integral, rtol=0, atol=1e-9)
    gf = torch.autograd.grad(fast.log_integral, cuda)[0]
    ga = torch.autograd.grad(audited.log_integral, cuda)[0]
    gc = torch.autograd.grad(reference.log_integral, cpu)[0]
    assert torch.equal(gf, ga)
    tolerance = {torch.float16: 5e-4, torch.float32: 2e-7, torch.float64: 1e-9}[dtype]
    torch.testing.assert_close(ga.cpu(), gc, rtol=0, atol=tolerance)
    known = ~torch.tensor(mask, device="cuda")
    assert torch.equal(ga[known], torch.zeros_like(ga[known]))
    check_budget(fast, False)
    check_budget(audited, True)
