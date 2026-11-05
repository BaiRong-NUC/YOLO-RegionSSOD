"""Continuous DFL raw-score integration and certified geometric search.

Coordinates and logits are embedded as exact Fractions for piecewise-affine
score bounds. Public log_integral is ALWAYS log integral exp(sum q*z), with
the binary64 cap 14.99; softmax constants are not included. A returned numeric
value is not a certificate: consumers must check diagnostics['qualified'].
No dataset or model is constructed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
from fractions import Fraction
from functools import lru_cache
import heapq
import itertools
import math
from typing import Any

import numpy as np
import torch

Q = Fraction
CAP = Q.from_float(14.99)
SCORE_CONVENTION = "raw_sum_q_logits_binary64_cap_14.99"
DECIMAL_PRECISION = 80


def _decimal_op(fn, rounding):
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
        ctx.rounding = rounding
        ctx.Emax, ctx.Emin = 999999, -999999
        return fn()


def _decimal_fraction(q, rounding):
    q = as_fraction(q)
    return _decimal_op(lambda: Decimal(q.numerator) / Decimal(q.denominator), rounding)


@dataclass(frozen=True)
class DecimalBounds:
    """Outward decimal interval, with exact input binary values via Fraction.

    Basic arithmetic uses decimal ROUND_FLOOR/CEILING at 80 digits. Python
    Decimal.exp/ln are correctly rounded (ROUND_HALF_EVEN); stepping one
    decimal neighbor outwards encloses the exact transcendental value. This
    is the arithmetic budget, not agreement with a high-precision reference.
    No global decimal context is changed. Endpoints must be finite.
    """
    lo: Decimal
    hi: Decimal

    def __post_init__(self):
        if not self.lo.is_finite() or not self.hi.is_finite() or self.lo > self.hi:
            raise NumericalCertificationError("invalid/nonfinite decimal enclosure")

    @classmethod
    def exact(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, Decimal):
            return cls(value, value)
        return cls(_decimal_fraction(value, ROUND_FLOOR), _decimal_fraction(value, ROUND_CEILING))

    @classmethod
    def fractions(cls, lo, hi):
        return cls(_decimal_fraction(lo, ROUND_FLOOR), _decimal_fraction(hi, ROUND_CEILING))

    def __add__(self, other):
        other = self.exact(other)
        return DecimalBounds(_decimal_op(lambda: self.lo + other.lo, ROUND_FLOOR),
                             _decimal_op(lambda: self.hi + other.hi, ROUND_CEILING))
    __radd__ = __add__

    def __neg__(self):
        return DecimalBounds(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + (-self.exact(other))

    def __rsub__(self, other):
        return self.exact(other) - self

    def __mul__(self, other):
        other = self.exact(other)
        pairs = tuple(itertools.product((self.lo, self.hi), (other.lo, other.hi)))
        return DecimalBounds(min(_decimal_op(lambda a=a, b=b: a * b, ROUND_FLOOR) for a, b in pairs),
                             max(_decimal_op(lambda a=a, b=b: a * b, ROUND_CEILING) for a, b in pairs))
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.exact(other)
        if other.lo <= 0 <= other.hi:
            raise NumericalCertificationError("interval divisor contains zero")
        pairs = tuple(itertools.product((self.lo, self.hi), (other.lo, other.hi)))
        return DecimalBounds(min(_decimal_op(lambda a=a, b=b: a / b, ROUND_FLOOR) for a, b in pairs),
                             max(_decimal_op(lambda a=a, b=b: a / b, ROUND_CEILING) for a, b in pairs))

    def exp(self):
        if max(self.lo.copy_abs(), self.hi.copy_abs()) > Decimal(1000000):
            raise NumericalCertificationError("exponent exceeds declared arithmetic resource cap",
                {"status": "ARITHMETIC_RESOURCE_LIMIT", "exponent_bounds": (str(self.lo), str(self.hi))})
        if self.lo == self.hi == 0:
            return self.exact(1)
        with localcontext() as ctx:
            ctx.prec = DECIMAL_PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            ctx.Emax, ctx.Emin = 999999, -999999
            lo, hi = self.lo.exp(), self.hi.exp()
            return DecimalBounds(lo.next_minus(), hi.next_plus())

    def log(self):
        if self.lo <= 0:
            raise NumericalCertificationError("log interval is not strictly positive")
        with localcontext() as ctx:
            ctx.prec = DECIMAL_PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            ctx.Emax, ctx.Emin = 999999, -999999
            return DecimalBounds(self.lo.ln().next_minus(), self.hi.ln().next_plus())

    def float_bounds(self):
        def convert(x, lower):
            f = float(x)
            if math.isinf(f):
                if (f > 0 and lower) or (f < 0 and not lower):
                    f = float(np.nextafter(f, 0.))
            elif (Q.from_float(f) > Q(x) if lower else Q.from_float(f) < Q(x)):
                f = float(np.nextafter(f, -math.inf if lower else math.inf))
            return f
        return convert(self.lo, True), convert(self.hi, False)


@dataclass(frozen=True)
class ExpMoments:
    mass: DecimalBounds
    left: DecimalBounds
    right: DecimalBounds


@lru_cache(maxsize=8192)
def _positive_series_moments(delta):
    """Exact rational series plus geometric remainder for 0<=delta<=1/2."""
    # Integral (1-t)*exp(delta*t) and t*exp(delta*t), respectively.
    left = right = Q(0)
    power_over_factorial = Q(1)
    threshold = Q(1, 10 ** (DECIMAL_PRECISION + 5))
    for n in range(512):
        left += power_over_factorial / ((n + 1) * (n + 2))
        right += power_over_factorial / (n + 2)
        following = power_over_factorial * delta / (n + 1)
        # Every later term ratio is <= delta/(n+2); ignoring denominator
        # factors only enlarges this positive remainder bound.
        remainder = following / (1 - delta / (n + 2))
        if remainder <= threshold:
            return (DecimalBounds.fractions(left, left + remainder),
                    DecimalBounds.fractions(right, right + remainder))
        power_over_factorial = following
    raise NumericalCertificationError("positive exponential series resource cap")


def certified_linear_exp(value_left, value_right, width):
    """Certified exp-affine integral and endpoint-basis first moments.

    For t in [0,1], a(t)=(1-t)*value_left+t*value_right; return width times
    integrals of exp(a), (1-t)*exp(a), t*exp(a). Inputs are exact Fractions or
    binary floats. For a nonnegative affine geometric weight w, its integral
    is w_left*result.left + w_right*result.right. Useful for D triangles too.
    """
    va, vb, width = map(as_fraction, (value_left, value_right, width))
    if width < 0:
        raise ValueError("negative integration width")
    if width == 0:
        zero = DecimalBounds.exact(0)
        return ExpMoments(zero, zero, zero)
    delta = abs(vb - va)
    if delta <= Q(1, 2):
        left, right = _positive_series_moments(delta)
        if va > vb:
            left, right = right, left
        factor = DecimalBounds.exact(min(va, vb)).exp() * width
    else:
        d = DecimalBounds.exact(delta)
        e = (-d).exp()
        # With the maximum endpoint factored out, B is the smaller-score
        # endpoint's basis moment and A is the larger-score endpoint's.
        b = (1 - (1 + d) * e) / (d * d)
        a = (d - 1 + e) / (d * d)
        left, right = (a, b) if va > vb else (b, a)
        factor = DecimalBounds.exact(max(va, vb)).exp() * width
    left, right = factor * left, factor * right
    return ExpMoments(left + right, left, right)


@dataclass(frozen=True)
class RectangleCertificate:
    mass: DecimalBounds
    gradient_numerators: tuple[tuple[DecimalBounds, ...], ...]
    shift: Q


def certify_rectangle(score, bounds, *, gradients=True):
    """Certify shifted mass and all exact raw-logit derivative numerators.

    Integral exp(raw score - shift) is returned. The shift is detached and
    purely for stable arithmetic; use mass.log()+shift for the raw log mass.
    """
    bounds = _bounds(bounds)
    if len(bounds) != score.dimension:
        raise ValueError("rectangle/score dimension mismatch")
    axes, axis_gradients = [], []
    zero = DecimalBounds.exact(0)
    for axis, (lo, hi) in enumerate(bounds):
        mass, grad = zero, [zero for _ in range(16)]
        knots = score.breakpoints(axis, lo, hi)
        for a, b in zip(knots, knots[1:]):
            moments = certified_linear_exp(score.axis_value(axis, a) - score.shifts[axis],
                score.axis_value(axis, b) - score.shifts[axis], b - a)
            mass = mass + moments.mass
            for x, basis in (((a, moments.left), (b, moments.right)) if gradients else ()):
                k, w = score.weights(axis, x)
                if w != 1:
                    grad[k] = grad[k] + basis * (1 - w)
                if w:
                    grad[k + 1] = grad[k + 1] + basis * w
        axes.append(mass); axis_gradients.append(grad)
    total = DecimalBounds.exact(1)
    for a in axes:
        total = total * a
    result = []
    for axis, grad in enumerate(axis_gradients if gradients else ()):
        factor = DecimalBounds.exact(1)
        for other, mass in enumerate(axes):
            if other != axis:
                factor = factor * mass
        result.append(tuple(g * factor for g in grad))
    return RectangleCertificate(total, tuple(result), sum(score.shifts, Q(0)))


class NumericalCertificationError(ValueError):
    """An explicit failed numerical attempt, carrying its last available bounds."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})
        self.diagnostics.setdefault("qualified", False)


def as_fraction(value):
    if isinstance(value, Q):
        return value
    if isinstance(value, (int, np.integer)):
        return Q(int(value))
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("scalar coordinate required")
        value = value.detach().cpu().item()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("finite scalar required")
    return Q.from_float(value)


def _bounds(bounds):
    result = tuple((as_fraction(lo), as_fraction(hi)) for lo, hi in bounds)
    if any(lo > hi for lo, hi in result):
        raise ValueError("reversed cell bounds")
    return result


def _interval_ends(interval):
    if hasattr(interval, "lo"):
        return as_fraction(interval.lo), as_fraction(interval.hi)
    return tuple(map(as_fraction, interval))


def _union(intervals):
    """Union for Lebesgue integration; endpoint closure is handled by point()."""
    merged = []
    for lo, hi in sorted(_interval_ends(item) for item in intervals):
        if hi < lo:
            raise ValueError("reversed exact interval")
        if hi == lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return tuple(merged)


@dataclass
class IntegralResult:
    log_integral: torch.Tensor
    relative_residual_bound: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PointResult:
    point: tuple[Q, ...]
    gap: Q
    diagnostics: dict[str, Any] = field(default_factory=dict)


class DFLScore:
    """a(u)=sum_j q(sign_j*(u_j-offset_j)/stride) dot logits_j.

    logits_U is [m,16]; offsets/signs use active xyxy order. Internally the
    tensor is promoted to float64 without severing its gradient. There is no
    external shift or softmax normalization. Final coordinates are not clipped.
    """

    convention = SCORE_CONVENTION

    def __init__(self, logits_U, offsets, signs, stride):
        if not isinstance(logits_U, torch.Tensor) or logits_U.ndim != 2 or logits_U.shape[1] != 16:
            raise ValueError("logits_U must be a torch tensor [m,16]")
        if not logits_U.is_floating_point() or not torch.isfinite(logits_U).all():
            raise ValueError("finite floating logits required")
        self.logits = logits_U.to(dtype=torch.float64).clone()
        self.logits_U = self.logits
        self.offsets = tuple(map(as_fraction, offsets))
        exact_signs = tuple(map(as_fraction, signs))
        if any(s not in (-1, 1) for s in exact_signs):
            raise ValueError("signs must be exactly +/-1")
        self.signs = tuple(map(int, exact_signs))
        self.stride = as_fraction(stride)
        self.dimension = self.logits.shape[0]
        if not self.dimension or len(self.offsets) != self.dimension or len(self.signs) != self.dimension:
            raise ValueError("one offset/sign per active coordinate required")
        if self.stride <= 0 or any(s not in (-1, 1) for s in self.signs):
            raise ValueError("positive stride and signs +/-1 required")
        self.exact_logits = tuple(tuple(as_fraction(v) for v in row)
                                  for row in self.logits.detach().cpu().tolist())
        self.shifts = tuple(max(row) for row in self.exact_logits)

    def distance(self, axis, coordinate):
        return self.signs[axis] * (as_fraction(coordinate) - self.offsets[axis]) / self.stride

    def weights(self, axis, coordinate):
        d = min(CAP, max(Q(0), self.distance(axis, coordinate)))
        k = d.numerator // d.denominator
        return k, d - k

    def axis_value(self, axis, coordinate):
        k, weight = self.weights(axis, coordinate)
        z = self.exact_logits[axis]
        return (1 - weight) * z[k] + weight * z[k + 1]

    def value_at(self, point):
        if len(point) != self.dimension:
            raise ValueError("point dimension mismatch")
        return sum((self.axis_value(j, u) for j, u in enumerate(point)), Q(0))

    def tensor_axis_value(self, axis, coordinate, *, shifted=False):
        k, weight = self.weights(axis, coordinate)
        w = float(weight)
        shift = float(self.shifts[axis]) if shifted else 0.
        return (self.logits[axis, k] - shift) * (1 - w) + (self.logits[axis, k + 1] - shift) * w

    def breakpoints(self, axis, lo, hi):
        lo, hi = as_fraction(lo), as_fraction(hi)
        if lo > hi:
            raise ValueError("reversed interval")
        values = {lo, hi}
        for distance in (*range(15), CAP):
            u = self.offsets[axis] + self.signs[axis] * self.stride * distance
            if lo < u < hi:
                values.add(u)
        return tuple(sorted(values))

    def cell_score_bounds(self, bounds):
        lows, highs = [], []
        for axis, (lo, hi) in enumerate(_bounds(bounds)):
            values = [self.axis_value(axis, x) for x in self.breakpoints(axis, lo, hi)]
            lows.append(min(values)); highs.append(max(values))
        return sum(lows, Q(0)), sum(highs, Q(0))

    def cell_maximizer(self, bounds):
        result = []
        for axis, (lo, hi) in enumerate(bounds):
            candidates = self.breakpoints(axis, lo, hi)
            result.append(min(candidates, key=lambda x: (-self.axis_value(axis, x), x)))
        return tuple(result)


def _log_exprel(t):
    # Nominal differentiable evaluation. This series is NOT a rounding proof;
    # certified arithmetic is reported independently in diagnostics.
    small = t.abs() < 1e-3
    a = torch.where(small, torch.ones_like(t), t.abs())
    regular = torch.maximum(t, torch.zeros_like(t)) + torch.log(-torch.expm1(-a)) - torch.log(a)
    series = t / 2 + t.square() / 24 - t.pow(4) / 2880 + t.pow(6) / 181440
    return torch.where(small, series, regular)


def _axis_log_integral(score, axis, lo, hi, *, shifted=False):
    points = score.breakpoints(axis, lo, hi)
    terms = []
    for a, b in zip(points, points[1:]):
        width = float(b - a)
        if width <= 0 or not math.isfinite(width):
            raise NumericalCertificationError("interval width not representable in binary64")
        va = score.tensor_axis_value(axis, a, shifted=True)
        vb = score.tensor_axis_value(axis, b, shifted=True)
        terms.append(math.log(width) + va + _log_exprel(vb - va))
    if not terms:
        raise NumericalCertificationError("zero Lebesgue volume", {"status": "ZERO_VOLUME"})
    value = torch.logsumexp(torch.stack(terms), 0)
    return value if shifted else value + float(score.shifts[axis])


def _batch_rectangle_logs(score, cells):
    """One vectorized autograd construction AFTER detached partitioning."""
    n, device = len(cells), score.logits.device
    total = torch.zeros(n, dtype=torch.float64, device=device)
    for axis in range(score.dimension):
        ids, ka, kb, wa, wb, widths = [], [], [], [], [], []
        for index, cell in enumerate(cells):
            knots = score.breakpoints(axis, *cell[axis])
            for a, b in zip(knots, knots[1:]):
                k0, w0 = score.weights(axis, a)
                k1, w1 = score.weights(axis, b)
                width = b - a
                ids.append(index); ka.append(k0); kb.append(k1)
                wa.append(float(w0)); wb.append(float(w1))
                widths.append(math.log(width.numerator) - math.log(width.denominator))
        idx = torch.tensor(ids, dtype=torch.long, device=device)
        ia, ib = (torch.tensor(x, dtype=torch.long, device=device) for x in (ka, kb))
        fa, fb = (torch.tensor(x, dtype=torch.float64, device=device) for x in (wa, wb))
        z = score.logits[axis] - float(score.shifts[axis])
        va, vb = z[ia] * (1 - fa) + z[ia + 1] * fa, z[ib] * (1 - fb) + z[ib + 1] * fb
        terms = torch.tensor(widths, dtype=torch.float64, device=device) + va + _log_exprel(vb - va)
        maxima = torch.full((n,), -torch.inf, dtype=torch.float64, device=device)
        maxima = maxima.scatter_reduce(0, idx, terms.detach(), reduce="amax", include_self=True)
        sums = torch.zeros_like(maxima).scatter_add(0, idx, torch.exp(terms - maxima[idx]))
        total = total + maxima + sums.log()
    return total


def _upper_float(value):
    return DecimalBounds.exact(value).float_bounds()[1]


def _sum_certificates(certificates, dimension):
    mass = DecimalBounds.exact(0)
    grad = [[DecimalBounds.exact(0) for _ in range(16)] for _ in range(dimension)]
    for cert in certificates:
        mass = mass + cert.mass
        if cert.gradient_numerators:
            for j in range(dimension):
                for k in range(16):
                    grad[j][k] = grad[j][k] + cert.gradient_numerators[j][k]
    return mass, grad


def _finish_integral(score, cells, mixed_upper, tolerance, diagnostics):
    certs = [certify_rectangle(score, cell) for cell in cells]
    mass, grad = _sum_certificates(certs, score.dimension)
    if mass.lo <= 0:
        raise NumericalCertificationError("no certified positive interior mass", diagnostics)
    remainder = DecimalBounds.exact(mixed_upper)
    total_bounds = DecimalBounds(mass.lo, (mass + remainder).hi)
    shift = sum(score.shifts, Q(0))
    log_bounds = total_bounds.log() + shift
    log_inside = torch.logsumexp(_batch_rectangle_logs(score, cells), 0)
    result = log_inside + float(shift)
    if not torch.isfinite(result):
        raise NumericalCertificationError("nonfinite Torch log integral", diagnostics)
    numeric_value = DecimalBounds.exact(float(result.detach()))
    difference = numeric_value - log_bounds
    log_error = max(difference.lo.copy_abs(), difference.hi.copy_abs())
    residual = (remainder + DecimalBounds.exact(mass.hi) - mass.lo) / DecimalBounds.exact(mass.lo)
    arithmetic_width = (DecimalBounds.exact(mass.hi) - mass.lo) / DecimalBounds.exact(mass.lo)
    r = (remainder / DecimalBounds.exact(mass.lo)).hi
    bounds = []
    for row in grad:
        out = []
        for numerator in row:
            inside_gradient = numerator / mass
            # True complete gradient is a convex combination of the inside
            # q-expectation and an unknown [0,1] residual q-expectation.
            expanded = inside_gradient + DecimalBounds(r.copy_negate(), r)
            out.append(DecimalBounds(max(Decimal(0), expanded.lo), min(Decimal(1), expanded.hi)))
        bounds.append(out)
    gradient_error = Decimal(0)
    if score.logits.requires_grad:
        actual = torch.autograd.grad(result, score.logits, retain_graph=True)[0].detach().cpu().numpy()
        if not np.isfinite(actual).all():
            raise NumericalCertificationError("nonfinite integral gradient", diagnostics)
        for j, row in enumerate(bounds):
            for k, enclosure in enumerate(row):
                delta = DecimalBounds.exact(float(actual[j, k])) - enclosure
                gradient_error = max(gradient_error, delta.lo.copy_abs(), delta.hi.copy_abs())
    else:
        # No computational gradient is returned for a detached input. Still
        # report the complete analytic derivative enclosure and its width.
        for row in bounds:
            for interval in row:
                gradient_error = max(gradient_error, (interval - interval.lo).hi)
    diagnostics = dict(diagnostics, qualified=True, status="CERTIFIED",
        score_convention=SCORE_CONVENTION, runtime_checked=True,
        arithmetic_mode="exact Fraction pieces + outward Decimal80 + Torch value/gradient audit",
        arithmetic_relative_width=_upper_float(arithmetic_width.hi),
        relative_residual_bound=_upper_float(residual.hi),
        log_integral_bounds=log_bounds.float_bounds(),
        log_value_error_bound=_upper_float(log_error), gradient_error_bound=_upper_float(gradient_error),
        gradient_bounds=[[interval.float_bounds() for interval in row] for row in bounds],
        gradient_runtime_checked=bool(score.logits.requires_grad),
        gradient_coordinates="promoted float64 logits; excludes downstream FP16/FP32 cast rounding",
        shifted_mass_bounds=(str(total_bounds.lo), str(total_bounds.hi)),
        raw_score_shift=str(shift), decimal_precision=DECIMAL_PRECISION)
    if max(_upper_float(residual.hi), _upper_float(log_error), _upper_float(gradient_error)) > tolerance:
        diagnostics.update(qualified=False, status="ARITHMETIC_OR_RESIDUAL_LIMIT")
        raise NumericalCertificationError("integral/gradient certificate exceeds tolerance", diagnostics)
    return IntegralResult(result, _upper_float(residual.hi), diagnostics)


def integrate_interval(score, axis, lo, hi, *, relative_tolerance=1e-4):
    """One-axis raw log integral and full value/gradient arithmetic audit."""
    one = DFLScore(score.logits[axis:axis + 1], (score.offsets[axis],), (score.signs[axis],), score.stride)
    return integrate_rectangle(one, ((lo, hi),), relative_tolerance=relative_tolerance)


def integrate_rectangle(score, bounds, *, relative_tolerance=1e-4):
    if not 0 < relative_tolerance < 1:
        raise ValueError("integration tolerance must be finite and between zero and one")
    bounds = _bounds(bounds)
    if len(bounds) != score.dimension or any(lo == hi for lo, hi in bounds):
        raise NumericalCertificationError("invalid/zero-volume rectangle", {"status": "ZERO_VOLUME"})
    return _finish_integral(score, [bounds], Decimal(0), relative_tolerance,
                            {"method": "exact_rectangle", "geometric_residual": 0., "cells": 1})


def _classify(domain, cell):
    kind = domain.classify(cell)
    if kind not in ("inside", "outside", "mixed"):
        raise ValueError("classify must return inside/outside/mixed")
    return kind


def _split(cell, score=None):
    widths = [hi - lo for lo, hi in cell]
    axis = max(range(len(cell)), key=lambda j: (widths[j], -j))
    lo, hi = cell[axis]
    if lo == hi:
        raise NumericalCertificationError("cannot bisect zero-volume cell")
    middle = (lo + hi) / 2
    if score is not None:
        knots = score.breakpoints(axis, lo, hi)[1:-1]
        if knots:
            middle = min(knots, key=lambda x: (abs(x - middle), x))
    left, right = list(cell), list(cell)
    left[axis], right[axis] = (lo, middle), (middle, hi)
    return tuple(left), tuple(right)


def _checked_point(domain, point, cell=None):
    witness = domain.point(tuple(map(as_fraction, point)))
    if witness is None:
        return None
    p = tuple(map(as_fraction, witness.point))
    if len(p) != len(point) or any(p[j] != as_fraction(point[j]) for j in range(len(p))):
        raise ValueError("domain.point returned a different point")
    if cell is not None and any(not lo <= p[j] <= hi for j, (lo, hi) in enumerate(cell)):
        raise ValueError("witness outside requested cell")
    return witness


def _optimize_exact_intervals(domain, score, intervals, root, tolerance):
    """Optimize the certified 1D union without source queries at every knot.

    Endpoint closure metadata is required; otherwise use the generic oracle
    search. Overlapping components need not be merged for maximization, and
    retaining every component preserves punctures and isolated closed points.
    The cell cap applies to B&B, not this finite analytic enumeration.
    """
    if any(not hasattr(i, "left_closed") or not hasattr(i, "right_closed") for i in intervals):
        return None
    best, best_value, upper = None, None, None
    candidates, nonempty = 0, 0

    def consider(x, value):
        nonlocal best, best_value, candidates
        candidates += 1
        if best_value is None or value > best_value or (value == best_value and x < best):
            best, best_value = x, value

    for item in intervals:
        lo, hi = _interval_ends(item)
        lc, rc = item.left_closed, item.right_closed
        if not isinstance(lc, (bool, np.bool_)) or not isinstance(rc, (bool, np.bool_)):
            raise ValueError("exact interval endpoint closure must be boolean")
        if lo > hi:
            raise ValueError("reversed exact interval")
        if lo == hi and not (lc and rc):
            continue
        if not root[0][0] <= lo <= hi <= root[0][1]:
            raise ValueError("exact interval outside domain bounds")
        nonempty += 1
        knots = score.breakpoints(0, lo, hi)
        values = tuple(score.axis_value(0, x) for x in knots)
        component_upper = max(values)
        upper = component_upper if upper is None else max(upper, component_upper)
        for j, (x, value) in enumerate(zip(knots, values)):
            if (x != lo or lc) and (x != hi or rc):
                consider(x, value)
                continue
            # An excluded endpoint is approached inside its adjacent affine
            # segment. Exact slope gives loss <= tolerance/2, even when the
            # supremum is unattained. Zero slope uses an interior midpoint.
            neighbor = 1 if j == 0 else j - 1
            span = abs(knots[neighbor] - x)
            slope = abs(values[neighbor] - value) / span
            step = span / 2 if slope == 0 else min(span / 2, tolerance / (2 * slope))
            interior = x + step if j == 0 else x - step
            consider(interior, score.axis_value(0, interior))
    if best is None:
        raise NumericalCertificationError("exact interval union is empty",
            {"status": "NO_WITNESS", "method": "exact_1d_point", "cells": 0, "point_calls": 0})
    gap = upper - best_value
    witness = _checked_point(domain, (best,), root)
    diagnostics = {"qualified": True, "status": "CERTIFIED", "method": "exact_1d_point",
        "cells": 0, "interval_count": len(intervals), "nonempty_components": nonempty,
        "candidate_count": candidates, "point_calls": 1,
        "score_lower": best_value, "score_upper": upper,
        "arithmetic": "exact Fraction interval closure extrema and feasible-point gap",
        "tie_rule": "highest score, then smallest active coordinate among enumerated feasible candidates; fixed E"}
    if witness is None or not 0 <= gap <= tolerance:
        diagnostics.update(qualified=False, status="INCONSISTENT_INTERVAL_ORACLE", point=(best,), gap=gap)
        raise NumericalCertificationError("certified interval union disagrees with final point witness", diagnostics)
    return PointResult((best,), gap, diagnostics)


def optimize_point(domain, score, *, tolerance=1e-5, max_cells=20000):
    """Maximize exact raw score; the gap is also the native DFL minimum gap.

    Every returned point is certified by domain.point. Open endpoints are
    approached with feasible interior points; no nonexistent lexicographic
    minimum is requested. For an exact 1D interval union with closure metadata,
    only the final selected point requires a domain.point call. With fixed E,
    active-coordinate order equals final xyxy order for tie breaking.
    """
    tol = as_fraction(tolerance)
    if tol <= 0 or not isinstance(max_cells, (int, np.integer)) or isinstance(max_cells, bool) or max_cells < 1:
        raise ValueError("positive tolerance/resource cap required")
    root = _bounds(domain.bounds)
    if len(root) != score.dimension:
        raise ValueError("domain/score dimension mismatch")
    intervals = getattr(domain, "intervals", None)
    if score.dimension == 1 and intervals is not None:
        intervals = tuple(intervals)
        result = _optimize_exact_intervals(domain, score, intervals, root, tol)
        if result is not None:
            return result
    best, best_value, best_key = None, None, None
    queue, serial, calls = [], itertools.count(), 0
    def consider(witness):
        nonlocal best, best_value, best_key
        if witness is None:
            return
        p = tuple(map(as_fraction, witness.point))
        checked = _checked_point(domain, p)
        if checked is None:
            raise ValueError("feasible_witness returned an infeasible point")
        value = score.value_at(p)
        key = tuple(map(as_fraction, getattr(witness, "xyxy", p)))
        if best_value is None or value > best_value or (value == best_value and key < best_key):
            best, best_value, best_key = p, value, key
    def add(cell):
        nonlocal calls
        calls += 1
        kind = _classify(domain, cell)
        if kind == "outside":
            return
        _, upper = score.cell_score_bounds(cell)
        if best_value is not None and upper < best_value:
            return
        maximizer = score.cell_maximizer(cell)
        consider(_checked_point(domain, maximizer, cell))
        consider(_checked_point(domain, tuple((lo + hi) / 2 for lo, hi in cell), cell))
        witness = domain.feasible_witness(cell)
        if witness is not None:
            if any(not lo <= as_fraction(witness.point[j]) <= hi for j, (lo, hi) in enumerate(cell)):
                raise ValueError("feasible witness outside cell")
            consider(witness)
        if best_value is None or upper >= best_value:
            heapq.heappush(queue, (-upper, cell, next(serial)))
    if score.dimension == 1 and intervals is not None:
        # Exact projection endpoints/knots give a global supremum, even if an
        # endpoint itself is excluded. Domain.point enforces endpoint closure.
        for item in intervals:
            lo, hi = _interval_ends(item)
            knots = score.breakpoints(0, lo, hi)
            for x in knots:
                consider(_checked_point(domain, (x,)))
            add(((lo, hi),))
    else:
        add(root)
    while queue:
        upper = max(best_value if best_value is not None else -queue[0][0], -queue[0][0])
        if best_value is not None and upper - best_value <= tol:
            return PointResult(best, upper - best_value, {"qualified": True, "status": "CERTIFIED",
                "cells": calls, "score_lower": best_value, "score_upper": upper,
                "arithmetic": "exact Fraction score/gap; logits are exact input binary values",
                "tie_rule": "highest certified score, then lexicographic final xyxy among visited witnesses"})
        if calls + 2 > max_cells:
            raise NumericalCertificationError("global point resource cap", {"status": "RESOURCE_LIMIT",
                "cells": calls, "point": best, "score_lower": best_value, "score_upper": upper,
                "gap": None if best_value is None else upper - best_value})
        neg_upper, cell, _ = heapq.heappop(queue)
        if best_value is not None and -neg_upper < best_value:
            continue
        for child in _split(cell, score):
            add(child)
    if best is None:
        raise NumericalCertificationError("no certified feasible point", {"status": "NO_WITNESS", "cells": calls})
    return PointResult(best, Q(0), {"qualified": True, "status": "CERTIFIED", "cells": calls,
        "score_lower": best_value, "score_upper": best_value})


def integrate_domain(domain, score, *, relative_tolerance=1e-4, max_cells=20000):
    """Certified inside-mass lower bound plus mixed-rectangle upper residual.

    Partitioning uses only detached rational geometry and scalar Decimal mass
    bounds. There are no per-cell GPU calls or repeatedly stacked autograd
    graphs. The accepted inside rectangles are evaluated in one final batched
    differentiable pass, audited against interval mass/gradient numerators.
    A resource failure reports bounds; it is not a proof that C is empty.
    """
    if (not 0 < relative_tolerance < 1 or not isinstance(max_cells, (int, np.integer))
            or isinstance(max_cells, bool) or max_cells < 1):
        raise ValueError("invalid integration tolerance/resource cap")
    root = _bounds(domain.bounds)
    if len(root) != score.dimension:
        raise ValueError("domain/score dimension mismatch")
    intervals = getattr(domain, "intervals", None)
    if score.dimension == 1 and intervals is not None:
        union = _union(intervals)
        if not union:
            raise NumericalCertificationError("C has zero Lebesgue volume", {"status": "ZERO_VOLUME"})
        return _finish_integral(score, [((lo, hi),) for lo, hi in union], Decimal(0),
            relative_tolerance, {"method": "exact_1d_union", "geometric_residual": 0.,
                                  "interval_count": len(union), "cells": len(union)})
    if any(lo == hi for lo, hi in root):
        raise NumericalCertificationError("C bounding box has zero Lebesgue volume", {"status": "ZERO_VOLUME"})
    inside, queue, serial, calls = [], [], itertools.count(), 0
    inside_mass = DecimalBounds.exact(0)
    mixed_mass = DecimalBounds.exact(0)
    def add(cell):
        nonlocal calls, inside_mass, mixed_mass
        calls += 1
        kind = _classify(domain, cell)
        if kind == "outside":
            return
        mass = certify_rectangle(score, cell, gradients=False).mass
        if kind == "inside":
            inside.append(cell)
            inside_mass = inside_mass + mass
        else:
            heapq.heappush(queue, (mass.hi.copy_negate(), cell, next(serial), mass))
            mixed_mass = mixed_mass + mass
    add(root)
    while True:
        if not queue:
            mixed_mass = DecimalBounds.exact(0)
        if inside_mass.lo > 0:
            residual = (DecimalBounds.exact(mixed_mass.hi) / DecimalBounds.exact(inside_mass.lo)).hi
            # Leave room for arithmetic error and for a detached-input
            # derivative interval's two-sided width, not just its radius.
            if _upper_float(residual) <= relative_tolerance / 3:
                return _finish_integral(score, inside, mixed_mass.hi, relative_tolerance,
                    {"method": "inside_mixed_bnb", "cells": calls, "inside_cells": len(inside),
                     "mixed_cells": len(queue), "geometric_residual": _upper_float(residual)})
        if not queue:
            raise NumericalCertificationError("C has no certified positive volume", {"status": "ZERO_VOLUME", "cells": calls})
        if calls + 2 > max_cells:
            raise NumericalCertificationError("integral resource cap", {"status": "RESOURCE_LIMIT", "cells": calls,
                "shifted_mass_lower": str(inside_mass.lo),
                "shifted_mass_upper": str((inside_mass + DecimalBounds.exact(mixed_mass.hi)).hi),
                "mixed_mass_upper": str(mixed_mass.hi), "raw_score_shift": str(sum(score.shifts, Q(0))),
                "relative_residual_bound": math.inf if inside_mass.lo <= 0 else _upper_float(residual),
                "arithmetic_certified": True, "qualified": False})
        _, cell, _, mass = heapq.heappop(queue)
        mixed_mass = mixed_mass - mass
        hint = getattr(domain, "split_hint", None)
        split = hint(cell) if hint is not None else None
        if split is None:
            children = _split(cell)
        else:
            axis, coordinate = int(split[0]), as_fraction(split[1])
            if not 0 <= axis < len(cell) or not cell[axis][0] < coordinate < cell[axis][1]:
                raise ValueError("split_hint must be strictly inside its cell axis")
            left, right = list(cell), list(cell)
            left[axis], right[axis] = (cell[axis][0], coordinate), (coordinate, cell[axis][1])
            children = tuple(left), tuple(right)
        for child in children:
            add(child)
