"""Constant-count exact axis maximization for the continuous SOURCE_OUTER.

This independent implementation has the same feasible halfplanes and objective
as continuous_domain._axis. It does not replace that reference or modify the
source observer. All finite inputs are interpreted as exact rational numbers.
Returned witnesses maximize visible_width / input_width within ONE clip branch;
equal-ratio witnesses need not equal the reference's sorted-vertex tie choice.

Proof, with input endpoints a,b, cap M>0 and positive visible width:

* Both clipped: visible endpoints are 0,1. Check center/size admit these.
  Maximizing a and minimizing b minimizes the denominator. These choices also
  minimize b-a, so failure of the cap proves the entire branch infeasible.
* Left clipped: b lies in [0,1] intersect size intersect twice-center intersect
  the right query; a lies in [-M,0] intersect the left query. For a<=0<b,
  b/(b-a) is nondecreasing in both a and b. Increasing a also relaxes b<=a+M.
  Thus choose the largest a, then the largest b<=a+M; check all lower bounds.
* Right clipped: a lies in [0,1] intersect (1-size) intersect (2*center-1)
  intersect the left query, and b in [1,1+M] intersect the right query.
  (1-a)/(b-a) is nonincreasing in both variables for a<1<=b. Choose the
  smallest b, then the smallest a>=b-M; check all upper bounds.
* Neither clipped: ratio is one. Let Wlo=max(0,size.lo), Whi=min(M,size.hi).
  For a in its [0,1] query interval, existence of b is exactly
    max(Blo,a+Wlo,2*cl-a) <= min(Bhi,a+Whi,2*ch-a).
  All nine lower/upper pairs are linear inequalities in a. Their intersection
  is a closed rational interval. At its smallest a, the largest admissible
  b-a equals min(Bhi-a,Whi,2*ch-2*a), a nonincreasing function of a. Therefore
  if this width is not positive, no positive witness exists anywhere. Otherwise
  use a positive endpoint of that b interval. No epsilon or strict-boundary
  approximation is used.

Work is bounded independently of query values (nine inequality eliminations at
most), excluding arbitrary-precision integer bit complexity. This certifies
the declared outer set, not an exact historical export inverse. It supplies no
new Lebesgue-measure, branch-union, floating arithmetic, or training certificate.
"""
from __future__ import annotations

from fractions import Fraction as Q
from functools import lru_cache
import math

from .bounded_retention_observation import AxisWitness, Interval

ZERO, ONE = Q(0), Q(1)
FREE = (-math.inf, math.inf)


def _query(pair):
    if len(pair) != 2:
        raise ValueError("query must have two endpoints")
    result = []
    for value in pair:
        if value in FREE:
            result.append(value)
        else:
            # Fraction rejects NaN and non-finite values other than our explicit
            # unbounded endpoints; binary floats are embedded exactly.
            result.append(Q(value))
    return tuple(result)


def _witness(a, b, left, right):
    if b <= a or right <= left:
        return None
    return AxisWitness(a, b, left, right, (right-left)/(b-a))


@lru_cache(maxsize=16384)
def axis_max(center: Interval, size: Interval, unknown: tuple[bool, bool],
             m: Q, branch: tuple[bool, bool], left=FREE, right=FREE) -> AxisWitness | None:
    """Exact maximum ratio and a full witness, or None for an empty branch.

    Queries are closed, may be unbounded, and may be reversed (an empty query).
    Clip flags must be permitted by original U. M is a finite positive cap.
    The lru_cache stores the complete witness. ``axis_max.__wrapped__`` allows
    uncached differential testing/benchmarking without changing this function.
    """
    if len(unknown) != 2 or len(branch) != 2:
        raise ValueError("two unknown flags and two clip flags required")
    lc, rc = branch
    if (lc and not unknown[0]) or (rc and not unknown[1]):
        raise ValueError("clip branch contradicts original U")
    m = Q(m)
    if m <= 0:
        raise ValueError("positive finite source span cap required")
    cl, ch, wl, wh = map(Q, (center.lo, center.hi, size.lo, size.hi))
    al, ah = _query(left)
    bl, bh = _query(right)
    if cl > ch or wl > wh or al > ah or bl > bh:
        return None

    if lc and rc:
        if not cl <= Q(1, 2) <= ch or not wl <= ONE <= wh:
            return None
        alo, a = max(al, -m), min(ah, ZERO)
        b, bhi = max(bl, ONE), min(bh, ONE+m)
        if alo > a or b > bhi or b-a > m:
            return None
        return _witness(a, b, ZERO, ONE)

    if lc:
        alo, a = max(al, -m), min(ah, ZERO)
        blo = max(bl, ZERO, wl, 2*cl)
        bhi = min(bh, ONE, wh, 2*ch)
        if alo > a:
            return None
        b = min(bhi, a+m)
        if b < blo:
            return None
        return _witness(a, b, ZERO, b)

    if rc:
        alo = max(al, ZERO, ONE-wh, 2*cl-ONE)
        ahi = min(ah, ONE, ONE-wl, 2*ch-ONE)
        b, bhi = max(bl, ONE), min(bh, ONE+m)
        if b > bhi:
            return None
        a = max(alo, b-m)
        if a > ahi:
            return None
        return _witness(a, b, a, ONE)

    alo, ahi = max(al, ZERO), min(ah, ONE)
    blo, bhi = max(bl, ZERO), min(bh, ONE)
    wl, wh = max(ZERO, wl), min(m, wh)
    if alo > ahi or blo > bhi or wl > wh:
        return None
    lower = ((0, blo), (1, wl), (-1, 2*cl))
    upper = ((0, bhi), (1, wh), (-1, 2*ch))
    for slope_lo, intercept_lo in lower:
        for slope_hi, intercept_hi in upper:
            slope = slope_lo-slope_hi
            rhs = intercept_hi-intercept_lo
            if slope > 0:
                ahi = min(ahi, rhs/slope)
            elif slope < 0:
                alo = max(alo, rhs/slope)
            elif rhs < 0:
                return None
            if alo > ahi:
                return None
    a = alo
    bmin = max(blo, a+wl, 2*cl-a)
    bmax = min(bhi, a+wh, 2*ch-a)
    # bmin can be a degenerate zero-width endpoint. Do not mistake it for an
    # empty feasible set when bmax>a. Prefer the smaller positive endpoint.
    b = bmin if bmin > a else bmax
    return _witness(a, b, a, b)
