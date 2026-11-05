"""Exact-real continuous SOURCE_OUTER adapter; no model, data access or measure.

Coordinates are final canvas pixels, ordered by ``active_sides``. Bounds are
finite Fraction pairs. ``intervals`` is the exact one-dimensional union, with
open endpoints where positive output area requires them. Multi-U classification
is deliberately three-valued. A missing feasible witness is NOT an emptiness
certificate. All certificates refer to the declared independent-parameter
transport outer, never to an exact historical exporter/augmentation inverse.

Nominal E is fixed in final xyxy/D, as in F_nom. Source-known endpoints remain
existential physical variables constrained by the complete xywh observation;
only active output U is constrained by transport. A paired source witness
certifies physical retention, NOT that its four transported endpoints equal
the nominal final box, nor that the nominal box obeys the source area ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from itertools import product
import math

from .bounded_retention_observation import (
    AxisWitness, JointWitness, UnsupportedObservation, _vertices, _visible_planes,
)
from .bounded_retention_continuous_axis import axis_max as _axis

NEG, POS = -math.inf, math.inf
FREE = (NEG, POS)


def _q(x):
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        raise ValueError("finite coordinate required")
    return Q(x)


def _ext(x):
    if hasattr(x, "item"):
        x = x.item()
    if x == NEG or x == POS:
        return x
    return _q(x)


def _pair(value, *, extended=False):
    if len(value) != 2:
        raise ValueError("interval must have two endpoints")
    a, b = map(_ext if extended else _q, value)
    if a > b:
        raise ValueError("reversed interval")
    return a, b


@dataclass(frozen=True)
class Interval1D:
    lo: Q
    hi: Q
    left_closed: bool = True
    right_closed: bool = True

    def __iter__(self):
        return iter((self.lo, self.hi))

    @property
    def empty(self):
        return self.lo > self.hi or (self.lo == self.hi and not
                                     (self.left_closed and self.right_closed))

    def contains(self, x):
        x = _q(x)
        return ((x > self.lo or (x == self.lo and self.left_closed)) and
                (x < self.hi or (x == self.hi and self.right_closed)))

    def intersection(self, other):
        lo, hi = max(self.lo, other.lo), min(self.hi, other.hi)
        lc = (self.left_closed if lo == self.lo else True) and (
            other.left_closed if lo == other.lo else True)
        rc = (self.right_closed if hi == self.hi else True) and (
            other.right_closed if hi == other.hi else True)
        return Interval1D(lo, hi, lc, rc)


def _merge(intervals):
    out = []
    for x in sorted((i for i in intervals if not i.empty),
                    key=lambda i: (i.lo, not i.left_closed, i.hi)):
        if not out or x.lo > out[-1].hi or (
                x.lo == out[-1].hi and not (x.left_closed or out[-1].right_closed)):
            out.append(x)
            continue
        a = out[-1]
        hi = max(a.hi, x.hi)
        rc = (a.right_closed if hi == a.hi else False) or (
            x.right_closed if hi == x.hi else False)
        out[-1] = Interval1D(a.lo, hi, a.left_closed, rc)
    return tuple(out)


@dataclass(frozen=True)
class DomainWitness:
    """Fixed nominal-E output paired with an existential physical source box.

    Only active U is transported from source; xyxy is not a full-box inverse.
    """
    point: tuple[Q, ...]
    xyxy: tuple[Q, Q, Q, Q]
    source: JointWitness


@dataclass(frozen=True)
class AxisCellProof:
    branch: tuple[bool, bool]
    corners: tuple  # (full active-coordinate corner, AxisWitness)
    ratio_lower: Q


@dataclass(frozen=True)
class CellCertificate:
    status: str
    reason: str
    bounds: tuple
    axis_proofs: tuple[AxisCellProof, ...] = ()


def _branches(unknown):
    return tuple(product(*[(False, True) if u else (False,) for u in unknown]))


@lru_cache(maxsize=4096)
def _planes(center, size, unknown, m, branch):
    lc, rc = branch
    if (lc and not unknown[0]) or (rc and not unknown[1]):
        raise ValueError("clip branch contradicts original U")
    la, lb, l0 = (Q(0), Q(0), Q(0)) if lc else (Q(1), Q(0), Q(0))
    ra, rb, r0 = (Q(0), Q(0), Q(1)) if rc else (Q(0), Q(1), Q(0))
    planes = [(p*la+q*ra, p*lb+q*rb, r-p*l0-q*r0)
              for p, q, r in _visible_planes(center, size)]
    planes += [(Q(1), Q(0), Q(0)) if lc else (Q(-1), Q(0), Q(0)),
               (Q(0), Q(-1), Q(-1)) if rc else (Q(0), Q(1), Q(1)),
               (Q(-1), Q(1), m), (Q(-1), Q(0), m), (Q(0), Q(1), 1+m)]
    return tuple(planes), (la, lb, l0), (ra, rb, r0)


@lru_cache(maxsize=16384)
def _axis_reference(center, size, unknown, m, branch, left=FREE, right=FREE):
    """Preserved vertex oracle for differential validation, not production queries."""
    base, lv, rv = _planes(center, size, unknown, m, branch)
    planes = list(base)
    for j, (lo, hi) in enumerate((left, right)):
        if lo != NEG:
            planes.append((Q(-1) if j == 0 else Q(0), Q(-1) if j == 1 else Q(0), -lo))
        if hi != POS:
            planes.append((Q(1) if j == 0 else Q(0), Q(1) if j == 1 else Q(0), hi))
    best = None
    for a, b in _vertices(planes):
        l, r = lv[0]*a+lv[1]*b+lv[2], rv[0]*a+rv[1]*b+rv[2]
        if b <= a or r <= l:
            continue
        w = AxisWitness(a, b, l, r, (r-l)/(b-a))
        if best is None or w.ratio > best.ratio:
            best = w
    return best


def _axis_options(key, axis, ranges):
    intervals, unknown, m, _ = key
    u = (unknown[axis], unknown[axis+2])
    return tuple((branch, w) for branch in _branches(u)
                 if (w := _axis(intervals[axis], intervals[axis+2], u, m, branch,
                                ranges[axis], ranges[axis+2])) is not None)


@lru_cache(maxsize=32768)
def _source_projection(key, side):
    """Exact projection of the full retained outer set onto one original U."""
    intervals, unknown, m, tau = key
    axis = side % 2
    others = _axis_options(key, 1-axis, (FREE,)*4)
    if not others:
        return ()
    needed = tau / max(w.ratio for _, w in others)
    u = (unknown[axis], unknown[axis+2])
    projected = []
    for branch in _branches(u):
        base, lv, rv = _planes(intervals[axis], intervals[axis+2], u, m, branch)
        na, nb, nc = (rv[i]-lv[i] for i in range(3))
        # needed * (b-a) <= visible_right-visible_left.
        vertices = _vertices(base + ((-needed-na, needed-nb, nc),))
        if vertices:
            coordinates = [v[0 if side < 2 else 1] for v in vertices]
            projected.append(Interval1D(min(coordinates), max(coordinates)))
    return _merge(projected)


@lru_cache(maxsize=32768)
def _source_split_coordinates(key, side):
    """Keep all branch polygon vertex coordinates before projection merging.

    These are subdivision hints only. The other axis is freely maximized just
    as in _source_projection; these marginal coordinates do not establish a
    feasible Cartesian product or license deletion of any branch/cell.
    """
    intervals, unknown, m, tau = key
    axis = side % 2
    points = {Q(0), Q(1)}
    others = _axis_options(key, 1-axis, (FREE,)*4)
    if others:
        needed = tau / max(w.ratio for _, w in others)
        u = (unknown[axis], unknown[axis+2])
        for branch in _branches(u):
            base, lv, rv = _planes(intervals[axis], intervals[axis+2], u, m, branch)
            na, nb, nc = (rv[i]-lv[i] for i in range(3))
            vertices = _vertices(base + ((-needed-na, needed-nb, nc),))
            points.update(v[0 if side < 2 else 1] for v in vertices)
    return tuple(sorted(points))


def _forward(source, scale, offset, lower, upper):
    """Exact interval image of independent s,t,l,h with l<=h."""
    if lower[0] > upper[1]:
        return None
    xs = [g*s for g in source for s in scale]
    a, b = min(xs)+offset[0], max(xs)+offset[1]
    return max(lower[0], min(a, upper[0])), min(upper[1], max(b, lower[1]))


def _inverse(output, scale, offset, lower, upper):
    """Exact existential preimage of an output interval, including clip faces."""
    lo, hi = max(output[0], lower[0]), min(output[1], upper[1])
    if lo > hi:
        return None
    prelo = NEG if max(lo, lower[0]) <= min(hi, lower[1]) else lo
    prehi = POS if max(lo, upper[0]) <= min(hi, upper[1]) else hi
    values = [(u-t)/s for u in (prelo, prehi) for t in offset for s in scale]
    return min(values), max(values)


def _active_mapping(record):
    physical = record["physical"]
    affine = tuple(_pair(x) for x in record["affine_bounds"])
    clip = tuple(_pair(x, extended=True) for x in record["augmentation_clip_window_bounds"])
    if len(affine) != 4 or len(clip) != 4:
        raise ValueError("expected four transport intervals")
    for scale in affine[:2]:
        if scale[0] <= 0 <= scale[1]:
            raise UnsupportedObservation("AFFINE_SCALE_CROSSES_ZERO")
    for axis in (0, 1):
        if clip[axis][0] > clip[axis+2][1]:
            raise ValueError("empty clip parameter domain")
        if clip[axis][1] == POS or clip[axis+2][0] == NEG:
            raise ValueError("invalid infinite clip endpoint")
    original = tuple(bool(x) for x in record["original_unknown"])
    if original != tuple(bool(x) for x in physical.unknown):
        raise ValueError("original U must match source observation")
    mask, mapping = [False]*4, {}
    for side, unknown in enumerate(original):
        if not unknown:
            continue
        axis = side % 2
        scale, offset = affine[axis], affine[axis+2]
        mapped = side if scale[0] > 0 else (side+2) % 4
        edge = physical.edge_intervals[side]
        extreme = edge.hi if side < 2 else edge.lo
        vals = [s*extreme+t for s in scale for t in offset]
        resolved = (max(vals) <= clip[mapped][0] if mapped < 2 else
                    min(vals) >= clip[mapped][1])
        if not resolved:
            mask[mapped] = True
            mapping[mapped] = side
    return tuple(mask), mapping, affine, clip


def active_mask(record):
    """Exact Fraction resolve decision, independent of nominal box and canvas.

    Use this same helper in the consumer's early all-known branch. In particular,
    do not compare against V3's additionally rounded floating-point mask.
    """
    return _active_mapping(record)[0]


class ContinuousDomain:
    """Immutable-by-convention observation snapshot with exact Fraction geometry.

    ``d_bounds`` encloses D. ``bounds``/``search_bounds`` enclose the FULL C via
    exact one-side projections with all other original U freely eliminated.
    ``classify`` concerns closed cells; faces
    excluded by strict positive output area remain mixed or outside. Neither a
    zero-length interval nor a nonempty face implies positive Lebesgue measure.
    """

    def __init__(self, record, boxxyxy, canvas, *, point_cache_size=1024):
        if not isinstance(point_cache_size, int) or point_cache_size < 0:
            raise ValueError("point_cache_size must be a nonnegative integer")
        self.physical = record["physical"]
        self.box = tuple(map(_q, boxxyxy))
        self.canvas = tuple(map(_q, canvas))
        if len(self.box) != 4 or len(self.canvas) != 2 or min(self.canvas) <= 0:
            raise ValueError("expected xyxy and positive (width,height)")
        self.mask, self._mapping, self.affine_bounds, self.clip_bounds = _active_mapping(record)
        original = tuple(bool(x) for x in record["original_unknown"])
        self._key = (tuple(self.physical.intervals), original,
                     self.physical.source_ratio_upper, self.physical.effective_retention)
        mask = self.mask
        self.active_sides = tuple(i for i, flag in enumerate(mask) if flag)
        self.dimension = len(self.active_sides)
        bounds, d_intervals = [], []
        for side in self.active_sides:
            lo, hi, lc, rc = Q(0), self.canvas[side % 2], True, True
            other = (side+2) % 4
            if not mask[other]:
                if side < 2 and self.box[other] <= hi:
                    hi, rc = self.box[other], False
                if side >= 2 and self.box[other] >= lo:
                    lo, lc = self.box[other], False
            d_intervals.append(Interval1D(lo, hi, lc, rc))
            bounds.append((lo, hi) if lo <= hi else (Q(0), Q(0)))
        self.d_bounds = tuple(bounds)
        self._d_intervals = tuple(d_intervals)
        self._empty_D = (any(i.empty for i in d_intervals) or any(
            not 0 <= self.box[j] <= self.canvas[j % 2] for j in range(4) if not mask[j]) or
            any(not mask[a] and not mask[a+2] and self.box[a] >= self.box[a+2] for a in (0, 1)))
        marginals = tuple(self._marginal(j) for j in range(self.dimension))
        self._empty_C = self._empty_D or any(not intervals for intervals in marginals)
        self.bounds = tuple((xs[0].lo, xs[-1].hi) if xs else (Q(0), Q(0)) for xs in marginals)
        self.search_bounds = self.bounds
        self.intervals = marginals[0] if self.dimension == 1 else None
        self.certificate = {"kind": "CONTINUOUS_SOURCE_TRANSPORT_OUTER", "exact_source_inverse": False,
                            "arithmetic": "Fraction; recorded binary64 bounds interpreted exactly",
                            "transport": "independent per-side scale/offset/ordered clip endpoints",
                            "transported_output_sides": self.active_sides,
                            "full_box_transport_inverse": False,
                            "known_edges": "fixed nominal supervision; source physical E remains existential in xywh intervals",
                            "measure": "not assigned by domain; lower-dimensional sets retained"}
        # Instance-local and bounded: cache the complete joint certificate, not
        # an approximate geometry decision. The observation snapshot is fixed.
        self._point_cached = lru_cache(maxsize=point_cache_size)(self._point_exact)
        self._split_cuts = None  # Build lazily; 1U analytic paths need no cuts.

    def _transport(self, side):
        axis = side % 2
        return (self.affine_bounds[axis], self.affine_bounds[axis+2],
                self.clip_bounds[axis], self.clip_bounds[axis+2])

    def split_hint(self, bounds):
        """Return (active-coordinate index, Fraction cut), or None.

        A hint only bisects one cell axis at a STRICT interior coordinate.
        Both children must be retained and normally certified; their union is
        exactly the parent (their shared face has zero Lebesgue volume). This
        method changes neither C nor point/classify, and makes no convergence
        or positive-mass claim. In particular a face-only intersection is not
        declared outside. Callers choose their ordinary fallback if no hint
        remains. Negative scales and interval clip parameters use the same
        exact forward interval image as source projection.
        """
        cell = self._cell(bounds)
        if self._split_cuts is None:
            cuts = []
            for side in self.active_sides:
                original = self._mapping[side]
                points = set(_source_split_coordinates(self._key, original))
                edge = self.physical.edge_intervals[original]
                points.update((edge.lo, edge.hi))
                transport = self._transport(side)
                output = set()
                for point in points:
                    image = _forward((point, point), *transport)
                    if image is not None:
                        output.update(image)
                # Offset is an inverse-affine sign transition; clip interval
                # endpoints separate finite from half-infinite inverse queries.
                output.update(x for pair in transport[1:] for x in pair)
                cuts.append(tuple(sorted(x for x in output if x not in FREE)))
            self._split_cuts = tuple(cuts)
        choices = [(abs(x-(lo+hi)/2)/(hi-lo), j, x)
                   for j, (lo, hi) in enumerate(cell) if hi > lo
                   for x in self._split_cuts[j] if lo < x < hi]
        if choices:
            _, axis, coordinate = min(choices)
            return axis, coordinate
        return None

    def _marginal(self, j):
        if self._empty_D:
            return ()
        side = self.active_sides[j]
        output = []
        for interval in _source_projection(self._key, self._mapping[side]):
            image = _forward(tuple(interval), *self._transport(side))
            if image is not None:
                output.append(Interval1D(*image).intersection(self._d_intervals[j]))
        return _merge(output)

    def _cell(self, bounds):
        cell = tuple(_pair(x) for x in bounds)
        if len(cell) != self.dimension:
            raise ValueError("cell dimension must match active_sides")
        return cell

    def _xyxy(self, point):
        point = tuple(map(_q, point))
        if len(point) != self.dimension:
            raise ValueError("point dimension must match active_sides")
        box = list(self.box)
        for side, value in zip(self.active_sides, point):
            box[side] = value
        return point, tuple(box)

    def _in_D(self, box):
        return (not self._empty_D and all(0 <= v <= self.canvas[i % 2] for i, v in enumerate(box))
                and box[0] < box[2] and box[1] < box[3])

    def _ranges(self, cell):
        ranges = [FREE]*4
        for side, interval in zip(self.active_sides, cell):
            inverse = _inverse(interval, *self._transport(side))
            if inverse is None:
                return None
            ranges[self._mapping[side]] = inverse
        return tuple(ranges)

    def _joints(self, ranges):
        if ranges is None:
            return ()
        xs, ys = (_axis_options(self._key, axis, ranges) for axis in (0, 1))
        tau = self._key[3]
        return tuple(JointWitness(x, y, x.ratio*y.ratio, x.visible_width*y.visible_width/tau)
                     for _, x in xs for _, y in ys if x.ratio*y.ratio >= tau)

    def point(self, point):
        point = tuple(map(_q, point))
        if len(point) != self.dimension:
            raise ValueError("point dimension must match active_sides")
        return self._point_cached(point)

    def _point_exact(self, point):
        point, box = self._xyxy(point)
        if not self._in_D(box):
            return None
        joints = self._joints(self._ranges(tuple((v, v) for v in point)))
        return DomainWitness(point, box, joints[0]) if joints else None

    def classify_D(self, bounds):
        cell = self._cell(bounds)
        if self._empty_D:
            return "outside"
        box = [(x, x) for x in self.box]
        for side, interval in zip(self.active_sides, cell):
            box[side] = interval
        if any(b < 0 or a > self.canvas[j % 2] for j, (a, b) in enumerate(box)):
            return "outside"
        clipped = [(max(Q(0), a), min(self.canvas[j % 2], b)) for j, (a, b) in enumerate(box)]
        if any(clipped[a][0] >= clipped[a+2][1] for a in (0, 1)):
            return "outside"
        inside = (all(0 <= a <= b <= self.canvas[j % 2] for j, (a, b) in enumerate(box)) and
                  all(box[a][1] < box[a+2][0] for a in (0, 1)))
        return "inside" if inside else "mixed"

    def _transport_piece(self, cell):
        for side, (lo, hi) in zip(self.active_sides, cell):
            scale, offset, lower, upper = self._transport(side)
            if any(lo < v < hi for v in offset+lower+upper):
                return False
            ends = [_inverse((v, v), scale, offset, lower, upper) for v in (lo, hi)]
            if any(x is None for x in ends):
                return False
            if any((ends[0][i] in FREE) != (ends[1][i] in FREE) for i in (0, 1)):
                return False
        return True

    def certify_cell(self, bounds):
        cell = self._cell(bounds)
        if self._empty_C:
            return CellCertificate("outside", "empty source projection or output domain", cell)
        ds = self.classify_D(cell)
        if ds == "outside":
            return CellCertificate("outside", "no positive output box in cell", cell)
        if self.dimension == 1:
            segment = Interval1D(*cell[0])
            hits = [segment.intersection(i) for i in self.intervals]
            if not any(not i.empty for i in hits):
                return CellCertificate("outside", "exact interval union disjoint", cell)
            if any(i.contains(segment.lo) and i.contains(segment.hi) for i in self.intervals):
                return CellCertificate("inside", "exact interval union contains cell", cell)
            return CellCertificate("mixed", "interval or strict-area boundary", cell)
        ranges = self._ranges(cell)
        if not self._joints(ranges):
            return CellCertificate("outside", "source excludes enclosing inverse ranges", cell)
        if ds != "inside" or not self._transport_piece(cell):
            return CellCertificate("mixed", "output geometry or transport branch boundary", cell)
        corners = tuple(product(*(tuple(dict.fromkeys(pair)) for pair in cell)))
        proofs = []
        intervals, unknown, m, tau = self._key
        for axis in (0, 1):
            candidates = []
            u = (unknown[axis], unknown[axis+2])
            for branch in _branches(u):
                witnesses = []
                for corner in corners:
                    query = self._ranges(tuple((v, v) for v in corner))
                    w = _axis(intervals[axis], intervals[axis+2], u, m, branch,
                              query[axis], query[axis+2])
                    if w is None:
                        break
                    witnesses.append((corner, w))
                else:
                    candidates.append(AxisCellProof(branch, tuple(witnesses),
                                                    min(w.ratio for _, w in witnesses)))
            if not candidates:
                return CellCertificate("mixed", "no single source branch covers corners", cell)
            proofs.append(max(candidates, key=lambda p: p.ratio_lower))
        if proofs[0].ratio_lower*proofs[1].ratio_lower < tau:
            return CellCertificate("mixed", "branchwise ratio lower bound insufficient", cell)
        return CellCertificate("inside", "fixed-branch convex witness interpolation", cell, tuple(proofs))

    def classify(self, bounds):
        return self.certify_cell(bounds).status

    def feasible_witness(self, bounds):
        cell = self._cell(bounds)
        if self.classify_D(cell) == "outside":
            return None
        if self.dimension == 1:
            for interval in self.intervals:
                hit = interval.intersection(Interval1D(*cell[0]))
                if not hit.empty:
                    witness = self.point(((hit.lo+hit.hi)/2,))
                    if witness is None:
                        raise RuntimeError("exact projection/point certificate disagreement")
                    return witness
            return None
        for joint in self._joints(self._ranges(cell)):
            source = (joint.x.input_left, joint.y.input_left, joint.x.input_right, joint.y.input_right)
            reachable = []
            for side, interval in zip(self.active_sides, cell):
                g = source[self._mapping[side]]
                image = _forward((g, g), *self._transport(side))
                reachable.append((max(interval[0], image[0], Q(0)),
                                  min(interval[1], image[1], self.canvas[side % 2])))
            if any(a > b for a, b in reachable):
                continue
            point = [(a+b)/2 for a, b in reachable]
            _, box = self._xyxy(point)
            for axis in (0, 1):
                if box[axis] >= box[axis+2]:
                    for side, end in ((axis, 0), (axis+2, 1)):
                        if side in self.active_sides:
                            j = self.active_sides.index(side)
                            point[j] = reachable[j][end]
            witness = self.point(point)
            if witness is not None:
                return witness
        # These are candidate witnesses only, never an inside/outside test.
        for point in ((tuple((a+b)/2 for a, b in cell)), *product(*cell)):
            witness = self.point(point)
            if witness is not None:
                return witness
        return None
