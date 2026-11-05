"""Read-only numerical observer for the initial human tile export.

No data access, model, torch, or training dependency. This is an OUTER-set
certificate, NOT an inverse of the historical exporter. Physical G means the
ideal continuous box made from the exact values of the input binary64 xywh:
((x +/- w/2)*W - tile_origin)/tile_size, evaluated in real arithmetic.
It is not the box formed by subtracting the exporter's rounded endpoints.

The certificate is conditional on the declared source domain and export path.
It does not authenticate historical execution. Nominal float32 E targets are
owned by the loss; they never replace the physical witness in this module.

Proof of the binary64 outer envelope (RN, gradual underflow, no fast-math):
write u=2**-53, eta=2**-1075. Every finite operation obeys
|fl(z)-z| <= u*|z|+eta. Source/tile ratios are <= M, tile sizes >= S,
and finite integer source sizes <= M*T < 2**53. The public default T=548
is the prior legal geometry report's maximum, NOT inferred from token values.
Users of another population must supply its own tile_size_upper.

_arithmetic_budget propagates these bounds, in tile-normalized units, through
half-width, endpoint addition, multiplication by image size, nonexpansive
max/min, span subtraction, center calculation, division, and output clamp.
The clamp is nonexpansive relative to the ideal visible coordinate in [0,1].
The denominator budget separately propagates fl(fl(w*W)*fl(h*H)); it is NOT
computed from the rounded endpoint difference. Integer dimensions are exact.

For retained rows, let Nhat,Dhat be computed pixel areas divided by tile area.
The two rounded operations fl(fl(Nhat/Dhat)+c) >= tau imply Nhat >= k*Dhat,
where k=(((tau-eta)/(1+u)-c)-eta)/(1+u). Hence
V + dn + k*dd >= k*G. For ideal visible area V >= vmin > 0 and G >= V,
V/G >= k-(dn+k*dd)/vmin. This last weaker necessary condition defines our
outer retention rule. Each membership witness uses its OWN visible area;
there is no independent maximum-visible-area/minimum-input-area splice.
The builder requires this effective threshold to be positive. Consequently
vmin > dd >= 3*u*M**2; each ideal input span >= vmin (visible spans <=1).
With S>=1, this keeps source width/height products and their quotient far from
underflow/overflow. The finite-dimension bound prevents endpoint/area overflow.

Exact Fraction polygon vertices maximize each axis' linear-fractional visible
ratio in each clip region. Their product is the maximum over this declared
separable outer observation. It proves existence in that outer set only.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction as Q
from functools import lru_cache
from itertools import combinations
import math
from typing import Sequence

import numpy as np


# The public minimum itself was serialized from a rounded division. Widen that
# denominator downward before taking the reciprocal and widening upward.
DEFAULT_SOURCE_RATIO_UPPER = math.nextafter(
    1 / math.nextafter(.0993113425925926, 0.0), math.inf)
ZERO, ONE = Q(0), Q(1)


class UnsupportedObservation(ValueError):
    """A missing/invalid premise, not a certified empty physical event."""

    def __init__(self, status: str):
        self.status = status
        super().__init__(status)


def _q(value) -> Q:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError("finite number required")
        return Q.from_float(float(value))
    return Q(value)


@dataclass(frozen=True)
class Interval:
    lo: Q
    hi: Q

    def contains(self, value) -> bool:
        return self.lo <= _q(value) <= self.hi


def _outward(value: Q, *, upper: bool) -> float:
    f = float(value)
    if (Q.from_float(f) < value if upper else Q.from_float(f) > value):
        f = math.nextafter(f, math.inf if upper else -math.inf)
    return f


def _bounds(intervals) -> np.ndarray:
    return np.asarray([[_outward(i.lo, upper=False), _outward(i.hi, upper=True)]
                       for i in intervals], dtype=np.float64)


def serialized_interval(token: str) -> Interval:
    """Closed OUTER envelope of .10g; decimal-decade transitions included.

    Endpoints/tie behavior are deliberately covered, not asserted attainable.
    Ten significant digits, not number of printed trailing digits, set radius.
    """
    if not isinstance(token, str) or len(token) > 80:
        raise UnsupportedObservation("INVALID_TOKEN")
    try:
        d = Decimal(token)
        if not d.is_finite() or not 0 <= d <= 1 or d.adjusted() < -324:
            raise UnsupportedObservation("INVALID_NORMALIZED_TOKEN")
        value = Q(d)
        if value == 0:
            return Interval(ZERO, ZERO)
        # Guard against accepting a token which is not even a ten-digit value.
        digits = list(d.as_tuple().digits)
        while digits and digits[-1] == 0:
            digits.pop()
        if len(digits) > 10:
            raise UnsupportedObservation("NOT_10G_TOKEN")
        radius = Q(5) * Q(10) ** (d.adjusted() - 10)
        return Interval(max(ZERO, value - radius), min(ONE, value + radius))
    except (InvalidOperation, ValueError, OverflowError) as exc:
        if isinstance(exc, UnsupportedObservation):
            raise
        raise UnsupportedObservation("INVALID_TOKEN") from exc


@lru_cache(maxsize=128)
def _arithmetic_budget(m: Q, s: int, t: int, tau: Q, tolerance: Q) -> dict:
    if not (m >= 1 and 1 <= s <= t and m*t < 2**53):
        raise UnsupportedObservation("UNPROVEN_SOURCE_NUMERIC_DOMAIN")
    u, eta = Q(1, 2**53), Q(1, 2**1075)
    # Half-width underflow, x +/- half-width, then multiplication by W.
    add_error = eta + u*(Q(3, 2) + eta) + eta
    corner = m*((1+u)*add_error + Q(3, 2)*u) + eta/s
    # On a returned row the computed clipped endpoints lie within the tile.
    span = 2*corner + u + eta/s
    size_report = span + u*(1+span) + eta
    mean = corner + u*m + Q(3, 2)*eta/s
    centered = mean + u*(1+mean) + eta/s
    center_report = centered + u*(1+centered) + eta
    # These area budgets use different source expressions, intentionally.
    original_span = u*m + eta/s
    denominator = 2*m*original_span + original_span**2 + u*(m+original_span)**2 + eta/s**2
    numerator = 2*span + span**2 + u*(1+span)**2 + eta/s**2
    k = (((tau-eta)/(1+u)-tolerance)-eta)/(1+u)
    if k <= 0:
        raise UnsupportedObservation("NONPOSITIVE_RETENTION_THRESHOLD")
    # Round final positive error budgets UP, k DOWN, retaining those doubles as
    # exact rationals. This is proven outward rounding, not an empirical epsilon;
    # it also avoids 1075-bit denominators in every polygon operation.
    up = lambda q: Q.from_float(_outward(q, upper=True))
    return dict(corner=up(corner), span=up(span), center=up(center_report), size=up(size_report),
                numerator=up(numerator), denominator=up(denominator),
                comparison_k=Q.from_float(_outward(k,upper=False)),
                unit_roundoff=u, underflow_half_ulp=eta)


# Halfplanes are A*a+B*b <= C, all exact rationals.
Plane = tuple[Q, Q, Q]


def _vertices(planes: Sequence[Plane]) -> list[tuple[Q, Q]]:
    planes = tuple(dict.fromkeys(planes))
    result = set()
    for (a,b,c), (d,e,f) in combinations(planes, 2):
        det = a*e-b*d
        if not det:
            continue
        x, y = (c*e-b*f)/det, (a*f-c*d)/det
        if all(p*x+q*y <= r for p,q,r in planes):
            result.add((x,y))
    return sorted(result)


@lru_cache(maxsize=8192)
def _visible_vertices(center: Interval, size: Interval) -> tuple:
    return tuple(_vertices(_visible_planes(center,size)))


def _visible_planes(center: Interval, size: Interval) -> tuple[Plane, ...]:
    return ((Q(-1),ZERO,ZERO), (ZERO,ONE,ONE),
            (Q(-1),ONE,size.hi), (ONE,Q(-1),-size.lo),
            (ONE,ONE,2*center.hi), (Q(-1),Q(-1),-2*center.lo))


@dataclass(frozen=True)
class AxisWitness:
    input_left: Q
    input_right: Q
    visible_left: Q
    visible_right: Q
    ratio: Q

    @property
    def visible_width(self) -> Q:
        return self.visible_right-self.visible_left

    @property
    def input_width(self) -> Q:
        return self.input_right-self.input_left


@lru_cache(maxsize=16384)
def _axis_max(center: Interval, size: Interval, unknown: tuple[bool,bool],
              left_range: tuple, right_range: tuple, m: Q) -> AxisWitness | None:
    best = None
    # A known side is certified strictly inside the initial tile, not cut.
    for lc in ((False,True) if unknown[0] else (False,)):
        for rc in ((False,True) if unknown[1] else (False,)):
            # Visible endpoints are l=0 or a, r=1 or b.
            la, lb, l0 = (ZERO,ZERO,ZERO) if lc else (ONE,ZERO,ZERO)
            ra, rb, r0 = (ZERO,ZERO,ONE) if rc else (ZERO,ONE,ZERO)
            planes = []
            for p,q,r in _visible_planes(center,size):
                planes.append((p*la+q*ra, p*lb+q*rb, r-p*l0-q*r0))
            planes.extend([(ONE,ZERO,ZERO) if lc else (Q(-1),ZERO,ZERO),
                           (ZERO,Q(-1),Q(-1)) if rc else (ZERO,ONE,ONE),
                           (Q(-1),ONE,m),  # input width <= source/tile upper
                           (Q(-1),ZERO,m), (ZERO,ONE,1+m)])
            for index,(lo,hi) in enumerate((left_range,right_range)):
                if lo is not None:
                    planes.append((Q(-1) if index==0 else ZERO,
                                   Q(-1) if index==1 else ZERO,-lo))
                if hi is not None:
                    planes.append((ONE if index==0 else ZERO,
                                   ONE if index==1 else ZERO,hi))
            for a,b in _vertices(planes):
                l, r = la*a+lb*b+l0, ra*a+rb*b+r0
                if b <= a or r <= l:
                    continue
                witness = AxisWitness(a,b,l,r,(r-l)/(b-a))
                if best is None or witness.ratio > best.ratio:
                    best = witness
    return best


@dataclass(frozen=True)
class JointWitness:
    x: AxisWitness
    y: AxisWitness
    ratio: Q
    area_cap: Q

    @property
    def visible_area(self) -> Q:
        return self.x.visible_width*self.y.visible_width

    @property
    def input_area(self) -> Q:
        return self.x.input_width*self.y.input_width


@dataclass(frozen=True)
class MembershipCertificate:
    mask: np.ndarray
    statuses: tuple[str, ...]
    witnesses: tuple[JointWitness | None, ...]
    certificate: dict


@dataclass(frozen=True)
class PhysicalObservation:
    tokens: tuple[str, ...]
    serialized_intervals: tuple[Interval, ...]
    intervals: tuple[Interval, ...]  # physical visible x,y,w,h, NOT nominal E
    edge_intervals: tuple[Interval, ...]
    _unknown: tuple[bool, ...]
    source_ratio_upper: Q
    effective_retention: Q
    certificate: dict

    @property
    def unknown(self) -> np.ndarray:
        return np.asarray(self._unknown, dtype=bool)

    @property
    def original_unknown(self) -> np.ndarray:
        return self.unknown

    @property
    def visible_edge_bounds(self) -> np.ndarray:
        """OUTER ideal-visible xyxy bounds, normalized to the ORIGINAL tile.

        A left/top input edge <= its visible edge's upper bound; a right/bottom
        input edge >= its visible edge's lower bound. These are not cut flags.
        No affine/canvas arithmetic is performed or certified by this API.
        """
        return _bounds(self.edge_intervals)

    def compatible(self, unknown_xyxy) -> np.ndarray:
        points = np.asarray(unknown_xyxy,dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("expected [N,4] original-normalized physical xyxy")
        ranges = np.repeat(points[...,None],2,axis=2)
        ranges[:,~self.unknown,:] = (-np.inf,np.inf)
        return self.compatible_ranges(ranges)

    def compatible_ranges(self, ranges) -> np.ndarray:
        """Existence mask in the certified OUTER set, never exact-source inverse.

        Known E positions must have [-inf,+inf]; no prediction can modify them.
        Original-U may be a point, bounded interval, or final-canvas half-line.
        Negative PHYSICAL coordinates are legal; negative distances are a loss
        concern. Reversed/infinite points/nonfinite queries are rejected.
        """
        return self.certify_ranges(ranges).mask

    def certify_ranges(self, ranges) -> MembershipCertificate:
        values = np.asarray(ranges,dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (4,2):
            raise ValueError("expected ranges [N,4,2]")
        mask, states, witnesses = [], [], []
        for row in values:
            status, joint = self._certify_row(row)
            mask.append(status == "OUTER_COMPATIBLE")
            states.append(status)
            witnesses.append(joint)
        result = np.asarray(mask,dtype=bool)
        result.setflags(write=False)
        return MembershipCertificate(result,tuple(states),tuple(witnesses),dict(self.certificate))

    def _certify_row(self, row):
        if np.isnan(row).any() or (row[:,0] > row[:,1]).any() or np.isposinf(row[:,0]).any() or np.isneginf(row[:,1]).any():
            return "INVALID_RANGE", None
        if any(not (np.isneginf(row[i,0]) and np.isposinf(row[i,1]))
               for i in range(4) if not self._unknown[i]):
            return "UNSUPPORTED_KNOWN_EDGE_QUERY", None
        parsed = tuple((None if np.isneginf(lo) else _q(lo),
                        None if np.isposinf(hi) else _q(hi)) for lo,hi in row)
        x,y,w,h = self.intervals
        wx = _axis_max(x,w,(self._unknown[0],self._unknown[2]),
                       parsed[0],parsed[2],self.source_ratio_upper)
        wy = _axis_max(y,h,(self._unknown[1],self._unknown[3]),
                       parsed[1],parsed[3],self.source_ratio_upper)
        if wx is None or wy is None:
            return "EMPTY_JOINT_GEOMETRY", None
        ratio = wx.ratio*wy.ratio
        joint = JointWitness(wx,wy,ratio,wx.visible_width*wy.visible_width/self.effective_retention)
        if ratio < self.effective_retention:
            return "RETENTION_EXCLUDED", joint
        return "OUTER_COMPATIBLE", joint


def _assemble(tokens, serialized, intervals, m, threshold, certificate):
    x,y,w,h = intervals
    vertices = [_visible_vertices(x,w), _visible_vertices(y,h)]
    if any(not points for points in vertices):
        raise UnsupportedObservation("EMPTY_VISIBLE_JOINT_INTERVALS")
    edges = tuple(Interval(min(p[side] for p in vertices[axis]),
                           max(p[side] for p in vertices[axis]))
                  for axis,side in ((0,0),(1,0),(0,1),(1,1)))
    unknown = (edges[0].lo==0,edges[1].lo==0,edges[2].hi==1,edges[3].hi==1)
    return PhysicalObservation(tuple(tokens),tuple(serialized),tuple(intervals),edges,unknown,
                               m,threshold,certificate)


def build_observation(xywh_tokens: Sequence[str], *,
                      source_ratio_upper=DEFAULT_SOURCE_RATIO_UPPER,
                      min_visibility: float=.35, retention_tolerance: float=1e-12,
                      tile_size_lower: int=64, tile_size_upper: int=548) -> PhysicalObservation:
    """Build a conditional source-OUTER certificate from four raw .10g tokens.

    Public premises: each source dimension / corresponding tile dimension <= M;
    source dimensions and tile origins/sizes are integers; source xywh binary64
    components are finite in [0,1], w,h>0; initial tile lies in source image;
    tile sizes in [tile_size_lower,tile_size_upper]; the reviewed Python exporter
    was used with these tau/tolerance values. No hidden data verifies premises.
    """
    if len(xywh_tokens) != 4:
        raise UnsupportedObservation("EXPECTED_FOUR_TOKENS")
    if not isinstance(tile_size_lower,int) or not isinstance(tile_size_upper,int):
        raise UnsupportedObservation("INTEGER_TILE_BOUNDS_REQUIRED")
    try:
        m, tau, tol = _q(source_ratio_upper), _q(float(min_visibility)), _q(float(retention_tolerance))
    except (ValueError,OverflowError) as exc:
        raise UnsupportedObservation("INVALID_PUBLIC_PARAMETERS") from exc
    if not (0 < tau <= 1 and 0 <= tol < tau):
        raise UnsupportedObservation("INVALID_RETENTION_PARAMETERS")
    serial = tuple(serialized_interval(t) for t in xywh_tokens)
    budget = dict(_arithmetic_budget(m,tile_size_lower,tile_size_upper,tau,tol))
    # Guarantee ideal visible spans >0 before invoking clip nonexpansiveness.
    # span/report bounds remain valid for the un-clamped positive size result;
    # width is positive and <=1 along a returned path (integer tile width).
    errors = (budget["center"],budget["center"],budget["size"],budget["size"])
    intervals = tuple(Interval(max(ZERO,i.lo-e),min(ONE,i.hi+e)) for i,e in zip(serial,errors))
    if intervals[2].lo <= 0 or intervals[3].lo <= 0:
        raise UnsupportedObservation("UNPROVEN_POSITIVE_PHYSICAL_VISIBLE_AREA")
    vmin = intervals[2].lo*intervals[3].lo
    k = budget["comparison_k"]
    slack = budget["numerator"]+k*budget["denominator"]
    effective = k-slack/vmin
    if effective <= 0:
        raise UnsupportedObservation("UNINFORMATIVE_RETENTION_ENVELOPE")
    cert = dict(kind="SOURCE_OUTER", exact_source_inverse=False,
                historical_execution_verified=False,
                physical_definition="ideal real box from exact input binary64 xywh",
                scope="initial tile; no transform/DFL-support certificate",
                source_ratio_upper=m, tile_size_lower=tile_size_lower,tile_size_upper=tile_size_upper,
                source_dimension_upper=m*tile_size_upper,
                source_tau=tau,source_additive_tolerance=tol,
                arithmetic=budget, minimum_visible_area=vmin,additive_area_slack=slack,
                effective_retention=effective,
                coverage="retained ideal G is contained in this outer set under declared source premises",
                witness_semantics="one joint visible/input box in the outer set; not a source replay",
                nominal_e="not consumed or modified; caller retains baseline targets")
    return _assemble(xywh_tokens,serial,intervals,m,effective,cert)


def build_ideal_observation(xywh: Sequence, *, source_ratio_upper=DEFAULT_SOURCE_RATIO_UPPER,
                          retention=Q(35,100)) -> PhysicalObservation:
    """Zero-envelope algebra oracle only; NEVER a historical-source certificate."""
    if len(xywh)!=4:
        raise UnsupportedObservation("EXPECTED_FOUR_COORDINATES")
    coords = tuple(_q(v) for v in xywh)
    m, tau = _q(source_ratio_upper), _q(retention)
    if not (m>=1 and 0<tau<=1 and all(0<=c<=1 for c in coords) and coords[2]>0 and coords[3]>0):
        raise UnsupportedObservation("INVALID_IDEAL_OBSERVATION")
    intervals = tuple(Interval(c,c) for c in coords)
    return _assemble(tuple(str(c) for c in coords),intervals,intervals,m,tau,
                     dict(kind="IDEAL_ALGEBRA_ONLY",exact_source_inverse=False,
                          historical_execution_verified=False,effective_retention=tau))
