"""V4 PRE-PRODUCTION mathematical review, NOT a source/Trainer qualification.

Only synthetic logits and explicitly specified domains are used. One-dimensional
integration is analytic on every DFL segment. The curved 2D domain has an exact
volume and global-point certificate; its nonconstant integral uses mpmath and
is explicitly a HIGH-PRECISION NUMERICAL CHECK, not a certified enclosure.
No production objective, oracle, model, data, validation or optimizer is used.
"""
import math

import mpmath as mp
import numpy as np
import pytest
import torch
from ultralytics.utils.loss import DFLoss


CAP = 14.99
OFFSET, SCALE = .2, .6


def synthetic_logits(first=0., second=1.):
    values = torch.full((16,), -3., dtype=torch.float64)
    values[:2] = torch.tensor([first, second], dtype=torch.float64)
    return values.requires_grad_()


def interpolation_weights(distance):
    q = min(max(float(distance), 0.), CAP)
    k = math.floor(q)
    weights = torch.zeros(16, dtype=torch.float64)
    weights[k], weights[k + 1] = k + 1 - q, q - k
    return weights


def energy(logits, distance):
    return -(interpolation_weights(distance) * logits.log_softmax(0)).sum()


def native_energy(logits, distance):
    return DFLoss(16)(logits[None], torch.tensor([[distance]], dtype=torch.float64)).squeeze()


def segments(intervals):
    """Disjoint raw-distance intervals, split at ALL encoding breakpoints.

    q-clipping does not collapse the physical measure of either plateau.
    A singleton is not silently widened or assigned positive Lebesgue mass.
    """
    ordered = sorted(intervals)
    for i, (lo, hi) in enumerate(ordered):
        if not math.isfinite(lo + hi) or lo >= hi:
            raise ValueError("positive finite interval length required; singleton needs explicit measure")
        if i and ordered[i - 1][1] > lo:
            raise ValueError("overlapping intervals double-count measure")
        cuts = [lo, *[x for x in (*range(15), CAP) if lo < x < hi], hi]
        yield from zip(cuts, cuts[1:])


def integral_1d(logits, intervals):
    """Analytic exponential-affine integral; differentiable in synthetic logits.

    I = h exp(-(A+B)/2) sinhc((B-A)/2). The small-argument branch has a
    remainder below 3e-30 for |(B-A)/2| < 1e-3, below float64 rounding here.
    This bounded toy implementation is not a production interval evaluator.
    """
    terms = []
    for lo, hi in segments(intervals):
        a, b = energy(logits, lo), energy(logits, hi)
        t = (b - a) / 2
        if abs(float(t.detach())) < 1e-3:
            sinhc = 1 + t**2 / 6 + t**4 / 120 + t**6 / 5040
        else:
            sinhc = torch.sinh(t) / t
        terms.append((hi - lo) * torch.exp(-(a + b) / 2) * sinhc)
    if not terms:
        raise ValueError("empty domain has no normalized energy")
    return torch.stack(terms).sum()


def volume(intervals):
    return sum(hi - lo for lo, hi in intervals)


def energy_set(logits, intervals):
    return -torch.log(integral_1d(logits, intervals) / volume(intervals))


def normalized_event(logits, event, domain):
    # In these fixtures C is explicitly a subset of independently fixed D.
    return torch.log(integral_1d(logits, domain)) - torch.log(integral_1d(logits, event))


def global_point_1d(logits, intervals):
    """Certified global minimum: affine pieces attain their minima at endpoints.

    Flat saturated intervals may have many minimizers; choose the first listed
    endpoint. No optimizer or fixed sample grid is used to claim optimality.
    """
    points = sorted({x for segment in segments(intervals) for x in segment})
    losses = torch.stack([energy(logits, x) for x in points])
    index = int(losses.detach().argmin())
    return losses[index], points[index], tuple(points)


def mp_integral_1d(logit_values, intervals):
    """Independent high-precision closed form, with exact endpoint conversion."""
    with mp.workdps(70):
        z = [mp.mpf(float(x)) for x in logit_values]
        normalizer = mp.log(sum(mp.exp(x) for x in z))
        c = [normalizer - x for x in z]
        total = mp.mpf(0)
        for lo, hi in segments(intervals):
            # Midpoint identifies the affine branch; coefficients are derived
            # directly, independently of the torch endpoint/sinhc formula.
            midpoint = (lo + hi) / 2
            if midpoint < 0:
                slope, intercept = mp.mpf(0), c[0]
            elif midpoint > CAP:
                q = mp.mpf(CAP)
                slope, intercept = mp.mpf(0), (15 - q) * c[14] + (q - 14) * c[15]
            else:
                k = math.floor(midpoint)
                slope = c[k + 1] - c[k]
                intercept = c[k] - k * slope
            lo, hi = mp.mpf(lo), mp.mpf(hi)
            if slope == 0:
                total += (hi - lo) * mp.exp(-intercept)
            else:
                total += mp.exp(-intercept - slope * lo) * (-mp.expm1(-slope * (hi - lo))) / slope
        return float(total)


@pytest.mark.parametrize("distance", [-3., -.01, 0., .37, 1., 5.8, 14.7, CAP, 15., 20.])
def test_continuous_piece_energy_and_gradient_match_native_dfl(distance):
    z = synthetic_logits(.4, 1.7)
    actual, expected = energy(z, distance), native_energy(z, distance)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
    ga = torch.autograd.grad(actual, z, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, z)[0]
    torch.testing.assert_close(ga, ge, rtol=0, atol=1e-14)


@pytest.mark.parametrize("flat", [False, True])
def test_exact_piece_integral_crosses_knots_gaps_and_saturated_tails(flat):
    z = (torch.zeros(16, dtype=torch.float64) if flat else
         torch.tensor([math.sin(i * .7) + i * .03 for i in range(16)], dtype=torch.float64)).requires_grad_()
    domain = [(-1.3, .7), (1.2, 3.8), (14.4, 16.2)]
    actual = integral_1d(z, domain)
    expected = mp_integral_1d(z.detach().tolist(), domain)
    assert float(actual.detach()) == pytest.approx(expected, rel=2e-14)
    if flat:
        assert float(actual.detach()) == pytest.approx(volume(domain) / 16, rel=2e-14)
    grad = torch.autograd.grad(-actual.log(), z)[0]
    # Independent central differences of the high-precision closed form.
    values = z.detach().numpy().copy()
    finite_difference = []
    for j in range(16):
        delta = np.zeros(16)
        delta[j] = 1e-5
        finite_difference.append((-math.log(mp_integral_1d(values + delta, domain))
                                  + math.log(mp_integral_1d(values - delta, domain))) / 2e-5)
    np.testing.assert_allclose(grad.numpy(), finite_difference, rtol=0, atol=6e-11)


def test_same_bin_support_different_interval_locations_changes_loss_and_gradient():
    z = synthetic_logits(0., 2.)
    left, right = [(.1, .3)], [(.7, .9)]
    a, b = energy_set(z, left), energy_set(z, right)
    # Both intervals touch exactly bins {0,1}, but every q shifts by .6.
    assert float((a - b).detach()) == pytest.approx(1.2, abs=1e-13)
    ga = torch.autograd.grad(a, z, retain_graph=True)[0]
    gb = torch.autograd.grad(b, z)[0]
    expected = torch.zeros(16, dtype=torch.float64)
    expected[:2] = torch.tensor([-.6, .6], dtype=torch.float64)
    torch.testing.assert_close(ga - gb, expected, rtol=0, atol=1e-13)
    domain = [(-1., 17.)]
    na, nb = normalized_event(z, left, domain), normalized_event(z, right, domain)
    assert float((na - nb).detach()) == pytest.approx(1.2, abs=1e-13)
    assert global_point_1d(z, left)[1] == .3
    assert global_point_1d(z, right)[1] == .9


def test_global_point_visits_all_components_and_dfl_knots_not_first_local_valley():
    z = torch.full((16,), -4., dtype=torch.float64)
    z[2], z[12] = 2., 5.  # inferior local valley and the global valley
    z.requires_grad_()
    intervals = [(-2., 3.25), (9.2, 16.5)]
    best, witness, checked = global_point_1d(z, intervals)
    assert witness == 12.
    assert {-2., 0., 1., 2., 3., 3.25, 9.2, 10., 11., 12., 13., 14., CAP, 16.5} == set(checked)
    assert best < native_energy(z, 2.)
    torch.testing.assert_close(best, native_energy(z, 12.), rtol=0, atol=1e-14)


@pytest.mark.parametrize("interval,encoded", [((-3., -1.), 0.), ((15.2, 19.), CAP)])
def test_saturated_plateaus_keep_raw_volume_and_feasible_point(interval, encoded):
    z = synthetic_logits(.5, -.7)
    point, witness, _ = global_point_1d(z, [interval])
    assert interval[0] <= witness <= interval[1]
    torch.testing.assert_close(point, native_energy(z, encoded), rtol=0, atol=1e-14)
    expected_mass = volume([interval]) * torch.exp(-native_energy(z, encoded))
    torch.testing.assert_close(integral_1d(z, [interval]), expected_mass, rtol=2e-14, atol=0)
    torch.testing.assert_close(energy_set(z, [interval]), point, rtol=0, atol=1e-14)


@pytest.mark.parametrize("center", [-.5, .37, 15.3])
def test_energy_singleton_limit_is_native_dfl_without_assigning_zero_volume_mass(center):
    z = synthetic_logits(0., 2.)
    target = native_energy(z, center)
    reference_grad = torch.autograd.grad(target, z, retain_graph=True)[0]
    for halfwidth in [1e-2, 1e-4, 1e-6]:
        value = energy_set(z, [(center - halfwidth, center + halfwidth)])
        grad = torch.autograd.grad(value, z, retain_graph=True)[0]
        if halfwidth == 1e-6:
            torch.testing.assert_close(value, target, rtol=0, atol=2e-11)
            torch.testing.assert_close(grad, reference_grad, rtol=0, atol=2e-11)
    with pytest.raises(ValueError, match="singleton needs explicit measure"):
        integral_1d(z, [(center, center)])


def test_energy_full_domain_is_not_constant_but_normalized_full_event_is_zero():
    domain = [(-2., 17.)]  # Fixed independently of each event and prediction.
    uniform = torch.zeros(16, dtype=torch.float64, requires_grad=True)
    nonuniform = synthetic_logits(2., -.4)
    assert float(energy_set(uniform, domain).detach()) == pytest.approx(math.log(16))
    changed = energy_set(nonuniform, domain)
    assert not torch.isclose(changed, energy_set(uniform, domain))
    assert torch.autograd.grad(changed, nonuniform)[0].abs().max() > .01
    for z in (uniform, nonuniform):
        value = normalized_event(z, domain, domain)
        assert value.item() == 0
        # Separate but identical integral graphs cancel mathematically. Their
        # backward accumulation can leave float64 roundoff, not a preference.
        torch.testing.assert_close(torch.autograd.grad(value, z)[0],
                                   torch.zeros_like(z), rtol=0, atol=2e-14)


def test_normalized_singleton_density_limit_includes_log_z_and_event_loss_diverges():
    z = synthetic_logits(.4, 1.7)
    domain, center = [(0., 1.)], .37
    log_z = integral_1d(z, domain).log()
    density_point = native_energy(z, center) + log_z
    density_gradient = torch.autograd.grad(density_point, z, retain_graph=True)[0]
    previous_event_loss = -math.inf
    for halfwidth in [1e-2, 1e-4, 1e-6]:
        event = [(center - halfwidth, center + halfwidth)]
        loss = normalized_event(z, event, domain)
        assert float(loss.detach()) > previous_event_loss
        previous_event_loss = float(loss.detach())
        corrected = loss + math.log(volume(event))
        if halfwidth == 1e-6:
            torch.testing.assert_close(corrected, density_point, rtol=0, atol=2e-11)
            grad = torch.autograd.grad(corrected, z, retain_graph=True)[0]
            torch.testing.assert_close(grad, density_gradient, rtol=0, atol=2e-11)
    assert abs(float(log_z.detach())) > .1
    native_grad = torch.autograd.grad(native_energy(z, center), z)[0]
    assert not torch.allclose(density_gradient, native_grad)


def test_density_normalization_has_a_gradient_effect_not_shared_by_plain_point_dfl():
    # On a fixed one-bin D, the common softmax normalizer cancels between C
    # and D. Plain DFL point/energy still has gradients on the other bins.
    z = synthetic_logits(.4, 1.7)
    domain, event = [(.2, .8)], [(.3, .4)]
    event_grad = torch.autograd.grad(normalized_event(z, event, domain), z)[0]
    energy_grad = torch.autograd.grad(energy_set(z, event), z)[0]
    point_grad = torch.autograd.grad(global_point_1d(z, event)[0], z)[0]
    torch.testing.assert_close(event_grad[2:], torch.zeros_like(z[2:]), rtol=0, atol=1e-14)
    torch.testing.assert_close(energy_grad[2:], z.softmax(0)[2:], rtol=0, atol=1e-14)
    torch.testing.assert_close(point_grad[2:], z.softmax(0)[2:], rtol=0, atol=1e-14)


def area_numerical_reference(zx, zy, dps=60):
    """HIGH-PRECISION NUMERICAL CHECK ONLY for a complete, explicit curved C.

    x,y in [0,1], (1+x)(1+y)<=2; actual DFL distances u=.2+.6*x,
    v=.2+.6*y. This has the SAME marginal ranges as D=[.2,.8]^2.
    y is integrated analytically; mpmath integrates x numerically. No cell
    intersection boolean, point sampling or local optimizer is used as proof.
    """
    with mp.workdps(dps):
        zx, zy = [[mp.mpf(float(z)) for z in values] for values in (zx, zy)]
        p = [[mp.exp(z) / sum(mp.exp(t) for t in values) for z in values] for values in (zx, zy)]
        offset, scale = mp.mpf(OFFSET), mp.mpf(SCALE)
        ax, ay = scale * (zx[0] - zx[1]), scale * (zy[0] - zy[1])
        c = -mp.log(p[0][0]) - mp.log(p[1][0]) + offset * (zx[0] - zx[1] + zy[0] - zy[1])
        def inner(t, moment=False):
            if ay == 0:
                return t*t/2 if moment else t
            if moment:
                return (1 - (1 + ay*t) * mp.exp(-ay*t)) / ay**2
            return -mp.expm1(-ay*t) / ay
        def integrate(mx=False, my=False):
            return mp.quad(lambda x: mp.exp(-ax*x) * (x if mx else 1)
                           * inner(2/(1+x)-1, my), [0, 1])
        raw = integrate()
        ex, ey, exy = integrate(mx=True)/raw, integrate(my=True)/raw, integrate(True, True)/raw
        means = [offset + scale*ex, offset + scale*ey]
        gradients = []
        for probs, mean in zip(p, means):
            gradients.append([probs[k] - ((1-mean) if k == 0 else mean if k == 1 else 0)
                              for k in range(16)])
        mass = scale**2 * mp.exp(-c) * raw
        vol = scale**2 * (2*mp.log(2)-1)  # exact analytic volume
        return {"mass": +mass, "volume": +vol, "energy": -mp.log(mass/vol),
                "gradient": gradients, "cross_gradient": -scale**2*(exy-ex*ey)}


def global_area_point(zx, zy):
    """Exact global certificate on this one-bin toy, not a local optimizer.

    The area inequality implies x+y+xy<=1, hence x+y<=1. C is contained
    in the triangle with vertices (0,0),(1,0),(0,1), and contains all three.
    An affine energy therefore has the same global minimum on C and that
    triangle: c+min(0,ax,ay). All witnesses below are actual members of C.
    """
    points = [(OFFSET, OFFSET), (OFFSET+SCALE, OFFSET), (OFFSET, OFFSET+SCALE)]
    losses = torch.stack([energy(zx, x) + energy(zy, y) for x, y in points])
    index = int(losses.detach().argmin())
    c = energy(zx, OFFSET) + energy(zy, OFFSET)
    ax, ay = SCALE*(zx[0]-zx[1]), SCALE*(zy[0]-zy[1])
    lower = c + torch.stack([c.new_zeros(()), ax, ay]).min()
    return losses[index], points[index], lower


def test_area_constant_integrand_has_exact_analytic_mass_and_event_probability():
    # This case DOES have an analytic answer; mpmath remains just its numerical check.
    reference = area_numerical_reference([0.]*16, [0.]*16)
    with mp.workdps(60):
        volume_exact = mp.mpf(SCALE)**2 * (2*mp.log(2)-1)
        assert mp.almosteq(reference["mass"], volume_exact/256, rel_eps=mp.mpf('1e-50'))
        assert mp.almosteq(reference["energy"], 2*mp.log(16), rel_eps=mp.mpf('1e-50'))
        domain_mass = mp.mpf(SCALE)**2/256
        event_loss = -mp.log(reference["mass"]/domain_mass)
        assert mp.almosteq(event_loss, -mp.log(2*mp.log(2)-1), rel_eps=mp.mpf('1e-50'))


def test_same_marginal_ranges_area_coupling_changes_mass_loss_and_cross_edge_gradient():
    zx, zy = synthetic_logits(0., 1.1), synthetic_logits(0., -.7)
    x, y = zx.detach().tolist(), zy.detach().tolist()
    result = area_numerical_reference(x, y, 60)
    lower_precision = area_numerical_reference(x, y, 35)
    # Agreement of two precisions is numerical evidence, NOT a proof enclosure.
    with mp.workdps(60):
        assert abs(result["mass"]-lower_precision["mass"]) < mp.mpf('1e-32')
    marginal = [(OFFSET, OFFSET+SCALE)]
    independent_mass = integral_1d(zx, marginal)*integral_1d(zy, marginal)
    independent_energy = energy_set(zx, marginal)+energy_set(zy, marginal)
    assert float(result["mass"]) < float(independent_mass.detach())
    assert abs(float(result["energy"])-float(independent_energy.detach())) > 1e-3
    assert float(result["cross_gradient"]) > 1e-3
    # The other edge's logits change this edge's gradient only under coupling.
    step = 1e-4
    plus, minus = y.copy(), y.copy()
    plus[1] += step
    minus[1] -= step
    rp, rm = area_numerical_reference(x, plus), area_numerical_reference(x, minus)
    cross_fd = float((rp["gradient"][0][1]-rm["gradient"][0][1])/(2*step))
    assert cross_fd == pytest.approx(float(result["cross_gradient"]), rel=2e-8, abs=1e-11)
    gx = torch.autograd.grad(independent_energy, zx)[0]
    changed_y = synthetic_logits(0., 2.)
    unchanged_gx = torch.autograd.grad(energy_set(zx, marginal)+energy_set(changed_y, marginal), zx)[0]
    assert torch.equal(gx, unchanged_gx)
    # Independent scalar finite differences also check the coupled first derivative.
    for axis in [0, 1]:
        xp, xm, yp, ym = x.copy(), x.copy(), y.copy(), y.copy()
        (xp if axis == 0 else yp)[1] += step
        (xm if axis == 0 else ym)[1] -= step
        fp, fm = area_numerical_reference(xp, yp), area_numerical_reference(xm, ym)
        fd = float((fp["energy"]-fm["energy"])/(2*step))
        assert fd == pytest.approx(float(result["gradient"][axis][1]), rel=2e-8, abs=1e-10)


def test_area_strong_point_is_global_feasible_and_not_independent_edge_extrema():
    zx, zy = synthetic_logits(0., 1.), synthetic_logits(0., 2.)
    best, witness, lower = global_area_point(zx, zy)
    torch.testing.assert_close(best, lower, rtol=0, atol=1e-14)  # zero analytic optimality gap
    assert witness == (OFFSET, OFFSET+SCALE)
    x, y = ((value-OFFSET)/SCALE for value in witness)
    assert (1+x)*(1+y) <= 2
    independent_point = energy(zx, OFFSET+SCALE)+energy(zy, OFFSET+SCALE)
    assert independent_point < best  # its (x,y)=(1,1) violates the area constraint
    expected = native_energy(zx, witness[0])+native_energy(zy, witness[1])
    torch.testing.assert_close(best, expected, rtol=0, atol=1e-14)
    gb = torch.autograd.grad(best, (zx, zy), retain_graph=True)
    ge = torch.autograd.grad(expected, (zx, zy))
    for actual, reference in zip(gb, ge):
        torch.testing.assert_close(actual, reference, rtol=0, atol=1e-14)
    numeric_set = area_numerical_reference(zx.detach().tolist(), zy.detach().tolist())
    assert float(numeric_set["energy"]) > float(best.detach())


def test_equal_global_point_values_do_not_certify_a_unique_gradient():
    zx, zy = synthetic_logits(0., 1.), synthetic_logits(0., 1.)
    left = energy(zx, OFFSET+SCALE)+energy(zy, OFFSET)
    right = energy(zx, OFFSET)+energy(zy, OFFSET+SCALE)
    lower = global_area_point(zx, zy)[2]
    torch.testing.assert_close(left, lower, rtol=0, atol=1e-14)
    torch.testing.assert_close(right, lower, rtol=0, atol=1e-14)
    gl = torch.autograd.grad(left, zx, retain_graph=True)[0]
    gr = torch.autograd.grad(right, zx)[0]
    assert (gl-gr).abs().max() > .59


def coordinate_integral_1d(logits, intervals, anchor, stride, sign=1):
    """Final-coordinate Lebesgue measure, not raw-distance measure.

    sign=-1 represents left/top; sign=+1 represents right/bottom. The absolute
    Jacobian is stride for either sign; geometry itself is never q-clipped.
    """
    distances = [tuple(sorted((sign*(lo-anchor)/stride, sign*(hi-anchor)/stride)))
                 for lo, hi in intervals]
    return stride * integral_1d(logits, distances)


@pytest.mark.parametrize("sign", [-1, 1])
def test_canvas_coordinate_integral_keeps_jacobian_and_negative_distance_measure(sign):
    z = synthetic_logits(.4, 1.7)
    interval, anchor, stride = [(1., 39.)], 7., 2.
    raw = [tuple(sorted((sign*(1-anchor)/stride, sign*(39-anchor)/stride)))]
    expected = stride * mp_integral_1d(z.detach().tolist(), raw)
    assert float(coordinate_integral_1d(z, interval, anchor, stride, sign).detach()) == pytest.approx(
        expected, rel=2e-14)
    assert min(raw[0]) < 0  # saturated portions still have physical-coordinate volume


def frozen_one_unknown_terms():
    """Synthetic positive box with three E sides and one unknown right edge.

    Canvas 200x100; E=(left20,top60,bottom96), anchor=(16,52), stride2.
    Thus D_right=(20,200], independent of C_right=[30.2,31.8]. E includes
    negative distances and overflow, and the D upper tail also saturates.
    """
    known_logits = torch.stack([synthetic_logits(.1, .7).detach(),
                                synthetic_logits(.4, -.2).detach(),
                                synthetic_logits(-.3, 1.2).detach()]).requires_grad_()
    unknown_logits = synthetic_logits(.2, .4)
    with torch.no_grad():
        unknown_logits[7], unknown_logits[8] = .6, 1.8
    raw_e = torch.tensor([[-2., -4., 22.]], dtype=torch.float64)
    native_e = DFLoss(16)(known_logits, raw_e.clone()).squeeze()*3
    domain, event, anchor, stride = [(20., 200.)], [(30.2, 31.8)], 16., 2.
    log_zd = coordinate_integral_1d(unknown_logits, domain, anchor, stride).log()
    log_zc = coordinate_integral_1d(unknown_logits, event, anchor, stride).log()
    raw_c = [((lo-anchor)/stride, (hi-anchor)/stride) for lo, hi in event]
    point, witness_d, _ = global_point_1d(unknown_logits, raw_c)
    return SimpleFrozenTerms(known_logits, unknown_logits, raw_e, native_e,
                            log_zd, log_zc, point, anchor+stride*witness_d)


class SimpleFrozenTerms:
    """Toy score decomposition only; never a production loss or training arm."""
    def __init__(self, known, unknown, raw_e, native_e, log_zd, log_zc, point, witness):
        self.known, self.unknown, self.raw_e = known, unknown, raw_e
        self.native_e, self.log_zd, self.log_zc = native_e, log_zd, log_zc
        self.point, self.witness = point, witness
        self.candidate = (native_e+log_zd-log_zc)/4
        self.control = (native_e+point)/4
        self.normalized_point_diagnostic = (native_e+point+log_zd)/4


def test_frozen_normalized_formula_preserves_native_e_and_plain_global_point_control():
    t = frozen_one_unknown_terms()
    assert t.witness == pytest.approx(31.8) and 20 < t.witness <= 200
    assert t.log_zd > t.log_zc
    # Restore actual L,T,R,B order to compare with the unmodified native /4.
    all_logits = torch.stack([t.known[0], t.known[1], t.unknown, t.known[2]])
    raw_four = torch.tensor([[-2., -4., (t.witness-16)/2, 22.]], dtype=torch.float64)
    native_point = DFLoss(16)(all_logits, raw_four.clone()).squeeze()
    torch.testing.assert_close(t.control, native_point, rtol=0, atol=1e-14)
    for score in (t.candidate, t.control, t.normalized_point_diagnostic):
        grad = torch.autograd.grad(score, t.known, retain_graph=True)[0]
        expected = torch.autograd.grad(native_point, t.known, retain_graph=True)[0]
        torch.testing.assert_close(grad, expected, rtol=0, atol=1e-14)
    gc = torch.autograd.grad(t.control, t.unknown, retain_graph=True)[0]
    gn = torch.autograd.grad(native_point, t.unknown, retain_graph=True)[0]
    torch.testing.assert_close(gc, gn, rtol=0, atol=1e-14)
    ga = torch.autograd.grad(t.candidate, t.unknown)[0]
    assert not torch.allclose(ga, gc)
    assert torch.equal(t.raw_e, torch.tensor([[-2., -4., 22.]], dtype=torch.float64))


def test_toy_decomposition_separates_normalization_from_aggregation_without_extra_arm():
    t = frozen_one_unknown_terms()
    normalization_shift = t.log_zd/4
    aggregation_shift = (-t.log_zc-t.point)/4
    torch.testing.assert_close(t.normalized_point_diagnostic-t.control,
                               normalization_shift, rtol=0, atol=1e-14)
    torch.testing.assert_close(t.candidate-t.normalized_point_diagnostic,
                               aggregation_shift, rtol=0, atol=1e-14)
    difference = t.candidate-t.control
    expected = normalization_shift+aggregation_shift
    torch.testing.assert_close(difference, expected, rtol=0, atol=1e-14)
    gd = torch.autograd.grad(difference, t.unknown, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, t.unknown)[0]
    torch.testing.assert_close(gd, ge, rtol=0, atol=1e-14)


def test_frozen_full_domain_cancels_only_unknown_term_and_keeps_native_e_over_four():
    t = frozen_one_unknown_terms()
    # Full C=D need not have zero total loss: the fixed E contribution remains.
    score = (t.native_e+t.log_zd-t.log_zd)/4
    torch.testing.assert_close(score, t.native_e/4, rtol=0, atol=1e-14)
    torch.testing.assert_close(torch.autograd.grad(score, t.unknown)[0],
                               torch.zeros_like(t.unknown), rtol=0, atol=1e-14)


def test_frozen_singleton_density_limit_keeps_native_e_plus_log_z_over_four():
    t = frozen_one_unknown_terms()
    center, halfwidth = 30.7, 1e-6
    interval = [(center-halfwidth, center+halfwidth)]
    log_zc = coordinate_integral_1d(t.unknown, interval, 16., 2.).log()
    corrected = (t.native_e+t.log_zd-log_zc+math.log(volume(interval)))/4
    expected = (t.native_e+native_energy(t.unknown, (center-16)/2)+t.log_zd)/4
    torch.testing.assert_close(corrected, expected, rtol=0, atol=2e-11)
    ga = torch.autograd.grad(corrected, (t.known, t.unknown), retain_graph=True)
    gb = torch.autograd.grad(expected, (t.known, t.unknown))
    for actual, reference in zip(ga, gb):
        torch.testing.assert_close(actual, reference, rtol=0, atol=2e-11)


def test_canvas_domain_with_opposite_unknown_edges_is_positive_box_triangle():
    # E top=2,bottom=6; U=(left,right) in canvas[0,8]. Positive width gives
    # D={0<=left<right<=8}, not the 8x8 product of marginal coordinate ranges.
    # Uniform logits make exp(-ell)=1/256 everywhere, including negative
    # encoded distances. The diagonal has zero Lebesgue measure.
    width = 8.
    z_domain = width**2/2/256
    z_wrong_marginal_product = width**2/256
    # C=[1,2]x[6,7] is entirely positive and has exact area1.
    z_event = 1/256
    event_score = math.log(z_domain)-math.log(z_event)
    assert event_score == pytest.approx(math.log(32), abs=1e-14)
    assert math.log(z_wrong_marginal_product/z_event)-event_score == pytest.approx(math.log(2))
    # A stricter source event must not redefine D and cancel its own loss.
    smaller_event_mass = .25/256
    assert math.log(z_domain/smaller_event_mass) > event_score > 0


def test_area_event_uses_full_canvas_domain_not_source_marginal_bounds():
    # E left=top=0, U right/bottom, canvas=.8x.8, anchor0/stride1. The review
    # event has source marginal range [.2,.8], but D is (0,.8]^2.
    zx, zy = synthetic_logits(0., 1.), synthetic_logits(0., 2.)
    coupled = area_numerical_reference(zx.detach().tolist(), zy.detach().tolist())
    canvas = [(0., OFFSET+SCALE)]
    source_marginals = [(OFFSET, OFFSET+SCALE)]
    zd = integral_1d(zx, canvas)*integral_1d(zy, canvas)
    smaller_zd = integral_1d(zx, source_marginals)*integral_1d(zy, source_marginals)
    assert zd > smaller_zd
    event_score = float(zd.detach().log())-math.log(float(coupled["mass"]))
    wrong_score = float(smaller_zd.detach().log())-math.log(float(coupled["mass"]))
    assert event_score > wrong_score > 0
    point, witness, lower = global_area_point(zx, zy)
    torch.testing.assert_close(point, lower, rtol=0, atol=1e-14)
    assert 0 < witness[0] <= OFFSET+SCALE and 0 < witness[1] <= OFFSET+SCALE
    # Extra normalized point is only a diagnostic: production control remains point.
    diagnostic = float((point+zd.log()).detach())
    assert diagnostic-float(point.detach()) == pytest.approx(float(zd.detach().log()))
