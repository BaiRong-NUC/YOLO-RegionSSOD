"""V6 native-E canvas normalizer COMPONENT; not a training qualification.

Frozen native_e_revision_contract.md: fixed nominal E remains unchanged and
may lie outside canvas. Only U coordinates are intersected with canvas while
requiring positive box extents. Copied from the preserved V4/V5 component;
all numerical helpers, arithmetic budgets and public interfaces are unchanged.

Return log integral_D exp(sum_U q(u)*logits), NOT exp(-DFL). Continuous U
uses float64 and binary64 cap14.99. Fixed E contributes only domain geometry.
Each axis is analytic: intervals use exponential moments, and opposite U edges
use an exact three-region decomposition of left<right. Fraction geometry
preserves binary input endpoints/DFL knots. An optional independent Decimal runtime
enclosure checks float64 value and logit-gradient error under declared library
rounding semantics. This component is not a full-loss/training qualification.
"""
from __future__ import annotations

from fractions import Fraction
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
import math
from typing import NamedTuple

import torch

from .bounded_retention import UnsupportedObservation


CAP = Fraction.from_float(14.99)
KNOTS = tuple(Fraction(k) for k in range(15)) + (CAP,)
MIN_INTERVAL_WIDTH = Fraction(1, 2**900)


class NormalizerResult(NamedTuple):
    """Tuple-unpackable result; named fields also support the full-loss caller."""

    log_integral: torch.Tensor
    diagnostics: dict


def _numbers(value, length, name):
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if tensor.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}]")
    numbers = tensor.tolist()
    if not all(math.isfinite(float(x)) for x in numbers):
        raise ValueError(f"{name} must be finite")
    return tuple(Fraction.from_float(float(x)) for x in numbers)


def _scalar(value, name):
    tensor = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
    if tensor.numel() != 1 or not math.isfinite(float(tensor.item())):
        raise ValueError(f"{name} must be one finite scalar")
    return Fraction.from_float(float(tensor.item()))


def _scores(z, coordinates, anchor, stride, sign):
    encoded = [min(max(sign*(x-anchor)/stride, Fraction(0)), CAP) for x in coordinates]
    indices = [q.numerator // q.denominator for q in encoded]
    k = torch.tensor(indices, device=z.device, dtype=torch.long)
    fraction = z.new_tensor([float(q-i) for q, i in zip(encoded, indices)])
    return z[k] + fraction*(z[k+1]-z[k])


def _segment_log_moments(a, b):
    """log mean exp(affine score), E[1-t], E[t], for t in [0,1].

    Analytic continuation near equal scores avoids removable singularities and
    preserves autograd at zero slope. Outside that branch, compute the small
    barycentric moment directly; 1-E[t] would lose it for very steep slopes.
    Series/transcendental rounding has no claimed rigorous runtime budget.
    """
    delta = b - a
    small_slope = delta.abs() < .1
    # Safe dummy inputs keep unselected torch.where branches finite in both
    # forward and backward, including zero slopes and very large logits.
    small_delta = torch.where(small_slope, delta, torch.zeros_like(delta))
    h = small_delta / 2
    sinhc = 1 + h**2/6 + h**4/120 + h**6/5040 + h**8/362880
    series_log_average = (a + b)/2 + torch.log(sinhc)
    series_mean_t = (.5 + small_delta/12 - small_delta**3/720 + small_delta**5/30240
                     - small_delta**7/1209600 + small_delta**9/47900160)
    magnitude = torch.where(small_slope, torch.ones_like(delta), delta.abs())
    denominator = -torch.expm1(-magnitude)
    log_average = torch.maximum(a, b) + torch.log(denominator) - torch.log(magnitude)
    small = 1/magnitude - torch.exp(-magnitude)/denominator
    large_left = torch.where(delta > 0, small, 1-small)
    large_right = torch.where(delta > 0, 1-small, small)
    return (torch.where(small_slope, series_log_average, log_average),
            torch.where(small_slope, 1-series_mean_t, large_left),
            torch.where(small_slope, series_mean_t, large_right))


def _positive_log(value):
    """Fraction geometry -> float log, with no silently vanished positive cells."""
    if value < MIN_INTERVAL_WIDTH:
        raise UnsupportedObservation("positive canvas segment/weight below declared 2^-900 arithmetic floor")
    number = float(value)
    if number <= 0 or not math.isfinite(number):
        raise UnsupportedObservation("positive canvas segment/weight not representable in float64")
    return math.log(number)


def _interval(z, lo, hi, anchor, stride, sign, diagnostics, weight=None):
    """Analytic log integral on one interval; optional positive linear weight.

    weight=(offset,slope) represents offset+slope*u; zero endpoints are legal.
    An empty interval contributes no term, not an epsilon-thickened interval.
    """
    if lo >= hi:
        return None
    cuts = sorted({lo, hi, *(anchor + sign*stride*k for k in KNOTS
                             if lo < anchor + sign*stride*k < hi)})
    scores = _scores(z, cuts, anchor, stride, sign)
    log_average, mean_left, mean_right = _segment_log_moments(scores[:-1], scores[1:])
    terms = log_average + z.new_tensor([_positive_log(r-l) for l, r in zip(cuts[:-1], cuts[1:])])
    if weight is not None:
        offset, slope = weight
        weights = [offset+slope*x for x in cuts]
        if min(weights) < 0:
            raise AssertionError("normalizer decomposition produced negative moment weight")
        logs = z.new_tensor([_positive_log(w) if w > 0 else -math.inf for w in weights])
        terms = terms + torch.logsumexp(torch.stack((logs[:-1]+torch.log(mean_left),
                                                     logs[1:]+torch.log(mean_right))), 0)
    diagnostics["analytic_segments"] += len(cuts)-1
    return torch.logsumexp(terms, 0)


def _analytic_normalizer(logits, nominal_box, anchor, stride, mask, canvas):
    """Return (raw_log_integral, JSON-serializable diagnostics).

    logits: floating tensor [4,16] in LTRB order. Geometry: box[4], anchor[2],
    positive stride scalar, bool mask[4], canvas[width,height]. Support 1..3 U;
    all-known belongs to original P1 and must not call this component.

    D uses all canvas U coordinates forming positive width/height with fixed E.
    No source retention/marginal/clip information enters D. Inputs are not
    mutated. The result is float64, on the logits device, with logit autograd.

    qualified=False means the missing floating arithmetic bound must not be
    mistaken for a passed production numerical gate. Analytic geometry has no
    quadrature/boundary residual, but that is distinct from arithmetic error.
    """
    if (not isinstance(logits, torch.Tensor) or logits.shape != (4, 16)
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError("finite floating logits[4,16] required")
    flags = torch.as_tensor(mask).detach().cpu()
    if flags.shape != (4,) or flags.dtype != torch.bool:
        raise ValueError("explicit bool mask[4] required")
    unknown = flags.tolist()
    count = sum(unknown)
    if not 1 <= count <= 3:
        raise ValueError("normalizer requires 1..3 U; all-known must use original P1")
    box = _numbers(nominal_box, 4, "nominal_box")
    anchors = _numbers(anchor, 2, "anchor")
    extents = _numbers(canvas, 2, "canvas")
    scale = _scalar(stride, "stride")
    if scale <= 0 or min(extents) <= 0:
        raise ValueError("positive stride and canvas required")
    z = logits.to(torch.float64)
    shift = z.detach().amax(1)
    shifted = z - shift[:, None]
    if not torch.isfinite(shifted).all():
        raise UnsupportedObservation("float64 logit centering overflow")
    diagnostics = {
        "convention": "raw_interpolated_logits_log_integral",
        "measure": "final_U_coordinate_Lebesgue",
        "unknown_count": count, "dtype": "float64", "analytic_segments": 0,
        "axes": [], "qualified": False,
        "qualification_reason": "no rigorous float64 exp/log/moment/autograd error bound",
        "error_budget": {"geometric_residual_mass": 0.0, "quadrature_residual_mass": 0.0,
                         "relative_arithmetic_mass_bound": None, "absolute_log_error_bound": None,
                         "logit_gradient_error_bound": None, "contract_relative_mass_tolerance": 1e-4},
        "max_abs_logit": float(z.detach().abs().max()),
        "max_logit_span": float((z.detach().amax(1)-z.detach().amin(1)).max()),
        "anchor_outside_canvas": any(a < 0 or a > w for a, w in zip(anchors, extents)),
    }
    axis_terms = []
    for axis, (a, width) in enumerate(zip(anchors, extents)):
        low, high = axis, axis+2
        if not unknown[low] and not unknown[high]:
            if box[low] >= box[high]:
                raise UnsupportedObservation("fixed E has nonpositive box extent")
            diagnostics["axes"].append("fixed")
            continue
        if unknown[low] and unknown[high]:
            split = min(max(a, Fraction(0)), width)
            center_left = _interval(shifted[low], Fraction(0), split, a, scale, -1, diagnostics)
            center_right = _interval(shifted[high], split, width, a, scale, 1, diagnostics)
            terms = []
            if center_left is not None and center_right is not None:
                terms.append(center_left+center_right)
            lower_triangle = _interval(shifted[low], Fraction(0), split, a, scale, -1,
                                       diagnostics, weight=(split, Fraction(-1)))
            upper_triangle = _interval(shifted[high], split, width, a, scale, 1,
                                       diagnostics, weight=(-split, Fraction(1)))
            if lower_triangle is not None:
                terms.append(shifted[high, 0]+lower_triangle)
            if upper_triangle is not None:
                terms.append(shifted[low, 0]+upper_triangle)
            axis_terms.append(torch.logsumexp(torch.stack(terms), 0))
            diagnostics["axes"].append("opposite_U_exact_triangle_moments")
        else:
            side = low if unknown[low] else high
            lo, hi = ((Fraction(0), min(width, box[high])) if unknown[low]
                      else (max(Fraction(0), box[low]), width))
            term = _interval(shifted[side], lo, hi, a, scale, -1 if side == low else 1, diagnostics)
            if term is None:
                raise UnsupportedObservation("canvas D has zero U volume")
            axis_terms.append(term)
            diagnostics["axes"].append("single_U_interval")
    active = flags.to(z.device)
    log_integral = torch.stack(axis_terms).sum() + shift[active].sum()
    if not torch.isfinite(log_integral):
        raise UnsupportedObservation("nonfinite analytic normalizer")
    return NormalizerResult(log_integral, diagnostics)


# Independent runtime arithmetic enclosure. Decimal exp/ln are correctly
# rounded (ROUND_HALF_EVEN); one adjacent Decimal in each direction encloses
# that rounding. Basic operations below explicitly use FLOOR/CEILING. This is
# an engineering enclosure under those library semantics, not a proof of the
# Python/torch/hardware stack. No mpmath comparison is used to produce a bound.
_DOWN = Context(prec=80, rounding=ROUND_FLOOR)
_UP = Context(prec=80, rounding=ROUND_CEILING)
_NEAR = Context(prec=80, rounding=ROUND_HALF_EVEN)
_ZERO = Decimal(0)
_ONE = Decimal(1)
ARITHMETIC_TOLERANCE = 1e-8
FAST_ARITHMETIC_TOLERANCE = 1e-7
MAX_ABS_LOGIT = 65536.
MAX_ABS_COORDINATE = float(2**30)
MIN_STRIDE = float(2**-20)


class _IV:
    """Closed Decimal interval; never use ambient Decimal precision/rounding."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        if isinstance(lo, _IV):
            self.lo, self.hi = lo.lo, lo.hi
        elif hi is not None:
            self.lo, self.hi = lo, hi
        else:
            q = Fraction(lo)
            a, b = Decimal(q.numerator), Decimal(q.denominator)
            self.lo, self.hi = _DOWN.divide(a, b), _UP.divide(a, b)

    def __add__(self, other):
        other = _IV(other)
        return _IV(_DOWN.add(self.lo, other.lo), _UP.add(self.hi, other.hi))

    __radd__ = __add__

    def __neg__(self):
        return _IV(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + -_IV(other)

    def __rsub__(self, other):
        return _IV(other) - self

    def __mul__(self, other):
        other = _IV(other)
        pairs = [(a, b) for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return _IV(min(_DOWN.multiply(a, b) for a, b in pairs),
                   max(_UP.multiply(a, b) for a, b in pairs))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _IV(other)
        if other.lo <= 0 <= other.hi:
            raise UnsupportedObservation("Decimal enclosure divides by a zero-containing interval")
        return self * _IV(_DOWN.divide(_ONE, other.hi), _UP.divide(_ONE, other.lo))

    def exp(self):
        return _IV(_NEAR.next_minus(_NEAR.exp(self.lo)), _NEAR.next_plus(_NEAR.exp(self.hi)))

    def ln(self):
        if self.lo <= 0:
            raise UnsupportedObservation("Decimal normalizer enclosure cannot certify positive mass")
        return _IV(_NEAR.next_minus(_NEAR.ln(self.lo)), _NEAR.next_plus(_NEAR.ln(self.hi)))


def _decimal_moments(a, b):
    """Enclose integral t**n exp(a+(b-a)t), n=0,1,2.

    Small slopes: positive exponential series at min(a,b), 24 terms and an
    explicit geometric upper bound on the remaining positive terms. Large
    slopes: exp at max(a,b), negative-exponential moment recurrences. All
    coefficients a,b and the branch decision are exact rational numbers.
    """
    v = abs(b-a)
    if v == 0:
        factor = _IV(a).exp()
        return factor, factor/2, factor/3
    if v <= 1:
        sums, term = [_IV(0), _IV(0), _IV(0)], _IV(1)
        for k in range(24):
            for n in range(3):
                sums[n] = sums[n] + term / (k+n+1)
            term = term * _IV(v) / (k+1)
        for n in range(3):
            tail = term / _IV(25+n) / (_IV(1)-_IV(v)/25)
            sums[n] = _IV(sums[n].lo, _UP.add(sums[n].hi, tail.hi))
        factor = _IV(min(a, b)).exp()
        if b >= a:
            moments = sums
        else:
            m0, m1, m2 = sums
            moments = [m0, m0-m1, m0-2*m1+m2]
    else:
        decay = _IV(-v).exp()
        j0 = (_IV(1)-decay)/_IV(v)
        j1 = (j0-decay)/_IV(v)
        j2 = (2*j1-decay)/_IV(v)
        factor = _IV(max(a, b)).exp()
        moments = [j0, j1, j2] if a >= b else [j0, j0-j1, j0-2*j1+j2]
    return tuple(factor * moment for moment in moments)


class _Mass:
    """Mass and unnormalized derivatives with respect to raw logits."""

    def __init__(self, value=0, gradient=None):
        self.value = _IV(value)
        self.gradient = gradient or {}

    def __add__(self, other):
        keys = self.gradient.keys() | other.gradient.keys()
        return _Mass(self.value+other.value,
                     {k: self.gradient.get(k, _IV(0))+other.gradient.get(k, _IV(0)) for k in keys})

    def __mul__(self, other):
        keys = self.gradient.keys() | other.gradient.keys()
        return _Mass(self.value*other.value,
                     {k: self.gradient.get(k, _IV(0))*other.value
                         + other.gradient.get(k, _IV(0))*self.value for k in keys})


def _rational_score(z, coordinate, anchor, stride, sign):
    q = min(max(sign*(coordinate-anchor)/stride, Fraction(0)), CAP)
    k = q.numerator // q.denominator
    weights = {k: 1-(q-k), k+1: q-k}
    return sum(z[i]*weight for i, weight in weights.items()), weights


def _certified_interval(z, side, lo, hi, anchor, stride, sign, weight=None):
    if lo >= hi:
        return _Mass()
    cuts = sorted({lo, hi, *(anchor + sign*stride*k for k in KNOTS
                             if lo < anchor + sign*stride*k < hi)})
    total = _Mass()
    for left, right in zip(cuts[:-1], cuts[1:]):
        a, qa = _rational_score(z[side], left, anchor, stride, sign)
        b, qb = _rational_score(z[side], right, anchor, stride, sign)
        m0, m1, m2 = _decimal_moments(a, b)
        w0, w1 = ((Fraction(1), Fraction(1)) if weight is None else
                  (weight[0]+weight[1]*left, weight[0]+weight[1]*right))
        h, dw = right-left, w1-w0
        mass = _IV(h)*(_IV(w0)*m0+_IV(dw)*m1)
        gradient = {}
        for k in qa.keys() | qb.keys():
            q0, dq = qa.get(k, Fraction(0)), qb.get(k, Fraction(0))-qa.get(k, Fraction(0))
            gradient[side, k] = _IV(h)*(_IV(q0*w0)*m0 + _IV(q0*dw+dq*w0)*m1 + _IV(dq*dw)*m2)
        total = total + _Mass(mass, gradient)
    return total


def _certified_mass(logits, box, anchors, stride, flags, canvas):
    z = tuple(tuple(Fraction.from_float(x) for x in row) for row in logits.detach().cpu().tolist())
    total = _Mass(1)
    for axis, (a, width) in enumerate(zip(anchors, canvas)):
        low, high = axis, axis+2
        if not flags[low] and not flags[high]:
            continue
        if flags[low] and flags[high]:
            split = min(max(a, Fraction(0)), width)
            center = (_certified_interval(z, low, Fraction(0), split, a, stride, -1)
                      * _certified_interval(z, high, split, width, a, stride, 1))
            low_constant = _IV(z[low][0]).exp()
            high_constant = _IV(z[high][0]).exp()
            lower = (_certified_interval(z, low, Fraction(0), split, a, stride, -1,
                                         (split, Fraction(-1)))
                     * _Mass(high_constant, {(high, 0): high_constant}))
            upper = (_certified_interval(z, high, split, width, a, stride, 1,
                                         (-split, Fraction(1)))
                     * _Mass(low_constant, {(low, 0): low_constant}))
            total = total * (center+lower+upper)
        else:
            side = low if flags[low] else high
            lo, hi = ((Fraction(0), min(width, box[high])) if flags[low]
                      else (max(Fraction(0), box[low]), width))
            total = total * _certified_interval(z, side, lo, hi, a, stride, -1 if side == low else 1)
    return total


def _error_bound(actual, enclosure):
    # actual is a binary64 number; Decimal.from_float embeds it exactly.
    value = Decimal.from_float(float(actual))
    return max(_DOWN.subtract(value, enclosure.hi).copy_abs(),
               _UP.subtract(value, enclosure.lo).copy_abs())


def _outward_float(value):
    return math.nextafter(float(value), math.inf)


def _qualification_metadata(diagnostics):
    diagnostics.update(
        arithmetic_scope="component float64 U value/gradient only; not C, E, AMP accumulation, model or training",
        arithmetic_envelope={"max_abs_logit": MAX_ABS_LOGIT, "max_abs_coordinate": MAX_ABS_COORDINATE,
                             "min_stride": MIN_STRIDE, "max_stride": MAX_ABS_COORDINATE,
                             "min_positive_segment_width": "2^-900", "max_evaluated_segments": 128})


def normalizer(logits, nominal_box, anchor, stride, mask, canvas, *, audit=False):
    """Return NormalizerResult(log_integral, diagnostics), tuple-unpackable.

    The differentiable output is a raw-score log integral over canvas D, in
    float64 on logits.device. With audit=True, runtime Decimal enclosures
    independently bound its value and every float64 logit derivative; both
    errors must be <=1e-8. Default fast mode uses a conditional engineering
    budget of 1e-7, not a formal outward certificate, under the assumptions
    below. Both modes share the same analytic kernel and input domain.

    Supported arithmetic envelope: finite |logits|<=65536, all coordinates
    <=2**30 in absolute value, and 2**-20<=stride<=2**30. Canvas must be positive;
    each positive segment/weight >=2**-900; at most128 evaluated segments.
    Exact zero/negative D volume is rejected. Other profiles fail explicitly.
    Native/AMP casting after this float64 U component remains caller-owned;
    the bound concerns derivatives w.r.t. promoted float64 logits, not an
    optimizer, final float16 accumulation, model parameters or the C integral.

    Fast engineering budget rationale (not a formal CPU/CUDA library proof):
    assume IEEE binary64 basic operations, correctly rounded conversions, and
    <=8 ulp errors for exp/expm1/log and their standard autograd derivatives.
    Geometry, knots and coefficients are rational until float conversion.
    q coefficient error is <=2**-53; |z_i-z_j|<=131072. Endpoint evaluation
    therefore contributes <2e-10 per active side including centering. Positive
    log-sums do not amplify score perturbations. Outside |delta|<.1, the
    negative-exponential formula's smallest barycentric mean exceeds 7e-6;
    its worst cancellation is at .1 (first derivatives subtract quantities
    of order100). Inside .1 the explicit Taylor truncation in value/first
    derivative is <1e-17. Normal interval/weight floats and <=128 evaluations
    keep accumulated log/moment/softmax rounding below a conservative 1e-8
    allocation; use another factor10 (1e-7) as the engineering budget. Runtime
    audit checks the actual library output and gradient without relying on
    this allocation. Synthetic high-precision/CPU/CUDA comparisons validate
    implementation, but are not substituted for the stated assumptions.
    """
    if (not isinstance(logits, torch.Tensor) or logits.shape != (4, 16)
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError("finite floating logits[4,16] required")
    if logits.device.type not in {"cpu", "cuda"}:
        raise UnsupportedObservation("normalizer numerical profile covers CPU/CUDA only")
    box = _numbers(nominal_box, 4, "nominal_box")
    anchors = _numbers(anchor, 2, "anchor")
    extents = _numbers(canvas, 2, "canvas")
    scale = _scalar(stride, "stride")
    if scale <= 0 or min(extents) <= 0:
        raise ValueError("positive stride and canvas required")
    if (float(logits.detach().abs().max()) > MAX_ABS_LOGIT
            or max(abs(x) for x in (*box, *anchors, *extents)) > MAX_ABS_COORDINATE
            or not MIN_STRIDE <= scale <= MAX_ABS_COORDINATE):
        raise UnsupportedObservation("outside declared normalizer arithmetic envelope")
    if not audit:
        result = _analytic_normalizer(logits, nominal_box, anchor, stride, mask, canvas)
        diagnostics = result.diagnostics
        if diagnostics["analytic_segments"] > 128:
            raise UnsupportedObservation("normalizer exceeds declared segment count")
        _qualification_metadata(diagnostics)
        diagnostics.update(qualified=True, arithmetic_mode="fast_conditional_budget", runtime_checked=False,
            qualification_reason="declared numerical domain and conditional float64 engineering budget",
            arithmetic_method="analytic log/expm1 moments; small-slope Taylor; no quadrature",
            arithmetic_assumptions=["IEEE binary64 basic operations and correctly rounded conversions",
                                    "CPU/CUDA exp, expm1, log and standard derivatives within 8 ulp",
                                    "engineering budget, not formal outward certification"])
        diagnostics["error_budget"].update(relative_arithmetic_mass_bound=1.000001e-7,
            absolute_log_error_bound=FAST_ARITHMETIC_TOLERANCE,
            logit_gradient_error_bound=FAST_ARITHMETIC_TOLERANCE,
            runtime_absolute_tolerance=None)
        return result
    original_grad = torch.is_grad_enabled() and logits.requires_grad
    with torch.enable_grad():
        checked_logits = logits.to(torch.float64)
        if not original_grad:
            checked_logits = checked_logits.detach().requires_grad_(True)
        result = _analytic_normalizer(checked_logits, nominal_box, anchor, stride, mask, canvas)
        if result.diagnostics["analytic_segments"] > 128:
            raise UnsupportedObservation("normalizer exceeds declared segment count")
        gradient = torch.autograd.grad(result.log_integral, checked_logits, retain_graph=True)[0]
    if not torch.isfinite(gradient).all():
        raise UnsupportedObservation("nonfinite normalizer float64 logit gradient")
    flags = torch.as_tensor(mask).detach().cpu().tolist()
    enclosure = _certified_mass(checked_logits, box, anchors, scale, flags, extents)
    log_enclosure = enclosure.value.ln()
    log_error = _error_bound(result.log_integral.detach().cpu(), log_enclosure)
    gradient_error = _ZERO
    grad_values = gradient.detach().cpu().tolist()
    for side in range(4):
        for k in range(16):
            derivative = enclosure.gradient.get((side, k), _IV(0)) / enclosure.value
            gradient_error = max(gradient_error, _error_bound(grad_values[side][k], derivative))
    tolerance = Decimal.from_float(ARITHMETIC_TOLERANCE)
    if max(log_error, gradient_error) > tolerance:
        raise UnsupportedObservation("normalizer runtime value/gradient enclosure exceeds 1e-8 arithmetic budget")
    relative_error = (_IV(log_error).exp()-1).hi
    diagnostics = result.diagnostics
    _qualification_metadata(diagnostics)
    diagnostics.update(qualified=True, qualification_reason="runtime Decimal value and float64 logit-gradient enclosure passed",
                       arithmetic_mode="decimal_runtime_audit", runtime_checked=True,
                       arithmetic_method="80-digit outward Decimal; analytic moments; small-slope positive series with tail bound",
                       decimal_log_integral_interval=[str(log_enclosure.lo), str(log_enclosure.hi)])
    diagnostics["error_budget"].update(
        relative_arithmetic_mass_bound=_outward_float(relative_error),
        absolute_log_error_bound=_outward_float(log_error),
        logit_gradient_error_bound=_outward_float(gradient_error),
        runtime_absolute_tolerance=ARITHMETIC_TOLERANCE)
    return NormalizerResult(result.log_integral if original_grad else result.log_integral.detach(), diagnostics)


def _union_fraction(value, name):
    """Keep rational endpoints exact, including separations below a float ULP."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool")
    try:
        if isinstance(value, Fraction):
            return value
        if hasattr(value, "as_integer_ratio"):
            numerator, denominator = value.as_integer_ratio()
            return Fraction(int(numerator), int(denominator))
        return Fraction(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be a finite rational or numeric scalar") from exc


def _interval_union(intervals):
    """Exact Lebesgue union; endpoint closure and isolated points have no mass."""
    positive, count, zero_count = [], 0, 0
    for item in intervals:
        if hasattr(item, "lo") and hasattr(item, "hi"):
            lo, hi = item.lo, item.hi
        else:
            try:
                lo, hi = item
            except (TypeError, ValueError) as exc:
                raise ValueError("interval must be a (lo,hi) pair or expose .lo/.hi") from exc
        lo, hi = _union_fraction(lo, "interval lo"), _union_fraction(hi, "interval hi")
        count += 1
        if lo > hi:
            raise ValueError("reversed interval endpoints")
        if max(abs(lo), abs(hi)) > MAX_ABS_COORDINATE:
            raise UnsupportedObservation("interval union outside declared coordinate envelope")
        if lo == hi:
            zero_count += 1
        else:
            positive.append((lo, hi))
    union = []
    for lo, hi in sorted(positive):
        if union and lo <= union[-1][1]:
            union[-1] = (union[-1][0], max(hi, union[-1][1]))
        else:
            union.append((lo, hi))
    if not union:
        raise UnsupportedObservation("interval union has zero Lebesgue volume")
    return tuple(union), count, zero_count


def _analytic_interval_union(logits, intervals, anchor, stride, sign, diagnostics):
    z = logits.to(torch.float64)
    shift = z.detach().max()
    terms = [_interval(z-shift, lo, hi, anchor, stride, sign, diagnostics) for lo, hi in intervals]
    value = torch.logsumexp(torch.stack(terms), 0)+shift
    if not torch.isfinite(value):
        raise UnsupportedObservation("nonfinite analytic interval-union integral")
    return value


def integrate_interval_union(logits16, intervals, anchor, stride, sign, *, audit=False):
    """V5 fast 1U C integral; does not change V4 normalizer or sample proposals.

    Return NormalizerResult(raw_log_integral, diagnostics), also tuple-unpackable.
    logits16 is [16]; anchor/stride are scalars; sign=-1 for left/top and +1
    for right/bottom, so distance=sign*(u-anchor)/stride. Intervals are (lo,hi)
    pairs or objects exposing .lo/.hi. Fraction endpoints/anchor/stride stay
    rational through exact union and DFL segmentation; never round endpoints
    before subtracting. Touching/overlapping intervals are merged; singleton
    components have zero Lebesgue mass, and zero total mass fails explicitly.

    This integrates the supplied union, not a source oracle: the caller must
    establish source legality, completeness and C subset D. No geometric or
    quadrature residual remains for that union. Inputs are not mutated.

    Same CPU/CUDA arithmetic profile and conditional 1e-7 value/gradient budget
    as normalizer(): |logits|<=65536, |coordinates|<=2**30, stride in
    [2**-20,2**30], positive DFL pieces>=2**-900, <=128 total pieces.
    Fast mode performs no Decimal or internal autograd audit. audit=True uses
    independent outward Decimal moments to check the value and ALL16 logit
    derivatives against 1e-8, retaining the graph for the caller's backward.
    The derivative budget concerns promoted float64 logits, not later casts.
    """
    if (not isinstance(logits16, torch.Tensor) or logits16.shape != (16,)
            or not logits16.is_floating_point() or not torch.isfinite(logits16).all()):
        raise ValueError("finite floating logits16[16] required")
    if logits16.device.type not in {"cpu", "cuda"}:
        raise UnsupportedObservation("interval union numerical profile covers CPU/CUDA only")
    if not isinstance(audit, bool):
        raise ValueError("audit must be bool")
    a, scale = _union_fraction(anchor, "anchor"), _union_fraction(stride, "stride")
    direction = _union_fraction(sign, "sign")
    if direction not in (-1, 1) or scale <= 0:
        raise ValueError("sign must be -1/+1 and stride must be positive")
    if (float(logits16.detach().abs().max()) > MAX_ABS_LOGIT or abs(a) > MAX_ABS_COORDINATE
            or not MIN_STRIDE <= scale <= MAX_ABS_COORDINATE):
        raise UnsupportedObservation("interval union outside declared arithmetic envelope")
    union, input_count, zero_count = _interval_union(intervals)
    # Preflight all exact widths/counts before allocating any differentiable
    # kernel. Many union pieces do not silently inherit a small-union budget.
    segment_count = 0
    for lo, hi in union:
        cuts = sorted({lo, hi, *(a+direction*scale*k for k in KNOTS if lo < a+direction*scale*k < hi)})
        for left, right in zip(cuts[:-1], cuts[1:]):
            _positive_log(right-left)
        segment_count += len(cuts)-1
    if segment_count > 128:
        raise UnsupportedObservation("interval union exceeds declared segment count")
    diagnostics = {
        "convention": "raw_interpolated_logits_log_integral", "measure": "final_U_coordinate_Lebesgue",
        "component": "V5_1U_interval_union", "unknown_count": 1, "dtype": "float64",
        "analytic_segments": 0, "input_intervals": input_count, "union_intervals": len(union),
        "zero_measure_input_intervals": zero_count,
        "exact_union": [[str(lo), str(hi)] for lo, hi in union],
        "exact_union_length": str(sum((hi-lo for lo, hi in union), Fraction(0))),
        "qualified": False, "error_budget": {"geometric_residual_mass": 0.0, "quadrature_residual_mass": 0.0},
    }
    _qualification_metadata(diagnostics)
    diagnostics["arithmetic_scope"] = "supplied 1U union float64 value/gradient only; source legality, D, E, AMP and training excluded"
    if not audit:
        value = _analytic_interval_union(logits16, union, a, scale, direction, diagnostics)
        diagnostics.update(qualified=True, arithmetic_mode="fast_conditional_budget", runtime_checked=False,
            qualification_reason="declared numerical domain and conditional float64 engineering budget",
            arithmetic_method="exact rational union; analytic log/expm1 moments; small-slope Taylor",
            arithmetic_assumptions=["IEEE binary64 basic operations and correctly rounded conversions",
                                    "CPU/CUDA exp, expm1, log and standard derivatives within 8 ulp",
                                    "engineering budget, not formal outward certification"])
        diagnostics["error_budget"].update(relative_arithmetic_mass_bound=1.000001e-7,
            absolute_log_error_bound=FAST_ARITHMETIC_TOLERANCE,
            logit_gradient_error_bound=FAST_ARITHMETIC_TOLERANCE, runtime_absolute_tolerance=None)
        return NormalizerResult(value, diagnostics)
    original_grad = torch.is_grad_enabled() and logits16.requires_grad
    with torch.enable_grad():
        checked = logits16.to(torch.float64)
        if not original_grad:
            checked = checked.detach().requires_grad_(True)
        value = _analytic_interval_union(checked, union, a, scale, direction, diagnostics)
        gradient = torch.autograd.grad(value, checked, retain_graph=True)[0]
    if not torch.isfinite(gradient).all():
        raise UnsupportedObservation("nonfinite interval-union float64 gradient")
    rational_logits = (tuple(Fraction.from_float(x) for x in checked.detach().cpu().tolist()),)
    enclosure = _Mass()
    for lo, hi in union:
        enclosure = enclosure + _certified_interval(rational_logits, 0, lo, hi, a, scale, direction)
    log_enclosure = enclosure.value.ln()
    log_error = _error_bound(value.detach().cpu(), log_enclosure)
    gradient_error = max(_error_bound(actual, enclosure.gradient.get((0, k), _IV(0))/enclosure.value)
                         for k, actual in enumerate(gradient.detach().cpu().tolist()))
    if max(log_error, gradient_error) > Decimal.from_float(ARITHMETIC_TOLERANCE):
        raise UnsupportedObservation("interval-union runtime value/gradient enclosure exceeds 1e-8 arithmetic budget")
    diagnostics.update(qualified=True, arithmetic_mode="decimal_runtime_audit", runtime_checked=True,
        qualification_reason="runtime Decimal value and all16 float64 logit-gradient enclosures passed",
        arithmetic_method="80-digit outward Decimal; analytic moments; positive-series tail bound",
        decimal_log_integral_interval=[str(log_enclosure.lo), str(log_enclosure.hi)])
    diagnostics["error_budget"].update(
        relative_arithmetic_mass_bound=_outward_float((_IV(log_error).exp()-1).hi),
        absolute_log_error_bound=_outward_float(log_error),
        logit_gradient_error_bound=_outward_float(gradient_error),
        runtime_absolute_tolerance=ARITHMETIC_TOLERANCE)
    return NormalizerResult(value if original_grad else value.detach(), diagnostics)
