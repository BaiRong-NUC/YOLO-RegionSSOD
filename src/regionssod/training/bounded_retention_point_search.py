"""Independent exact point-search prototype; never installed in a train entry.

Same feasible set, raw DFL score and tolerance as V6. Conditional source
projections only REMOVE regions proved infeasible. Every incumbent still goes
through the original domain.point oracle. The old traversal-dependent tie
selection is not reproduced: callers must disclose geometry/assignment changes.
"""
from fractions import Fraction as Q
import heapq
import itertools

from .bounded_retention_continuous_domain import (
    FREE, _axis_options, _branches, _forward, _planes,
)
from .bounded_retention_continuous_numerics import (
    NumericalCertificationError, PointResult, _bounds, _checked_point,
    _split, as_fraction, optimize_point as reference_optimize_point,
)
from .bounded_retention_observation import _vertices
from .bounded_retention_domain_v6 import NativeEDomain


def contract_bounds(domain, bounds):
    """One exact sweep of necessary conditional marginal projections.

    In each source axis, R_other is the exact maximum retention ratio on the
    other axis's enclosing inverse ranges. Every feasible source must satisfy
    r_axis >= tau/R_other. On each source clip branch this is a linear
    halfplane in (left,right). Project its bounded polygon, map it through the
    SAME interval transport, and intersect the output cell. Taking a hull over
    all branches may retain infeasible points, but cannot delete a feasible
    one. The witness oracle, NOT this outer contraction, accepts points.
    """
    if not isinstance(domain, NativeEDomain):
        raise TypeError("conditional projection requires the frozen native-E domain")
    cell = list(_bounds(bounds))
    if domain.classify_D(cell) == "outside":
        return None
    intervals, unknown, span_cap, tau = domain._key
    for j, side in enumerate(domain.active_sides):
        ranges = domain._ranges(cell)
        if ranges is None:
            return None
        source_side = domain._mapping[side]
        axis = source_side % 2
        other = _axis_options(domain._key, 1-axis, ranges)
        if not other:
            return None
        needed = tau / max(w.ratio for _, w in other)
        u = (unknown[axis], unknown[axis+2])
        images = []
        for branch in _branches(u):
            planes, lv, rv = _planes(intervals[axis], intervals[axis+2], u, span_cap, branch)
            planes = list(planes)
            for k, (lo, hi) in enumerate((ranges[axis], ranges[axis+2])):
                if lo != FREE[0]:
                    planes.append((-Q(1) if k == 0 else Q(0), -Q(1) if k == 1 else Q(0), -lo))
                if hi != FREE[1]:
                    planes.append((Q(1) if k == 0 else Q(0), Q(1) if k == 1 else Q(0), hi))
            na, nb, nc = (rv[k]-lv[k] for k in range(3))
            planes.append((-needed-na, needed-nb, nc))
            vertices = _vertices(planes)
            if not vertices:
                continue
            coords = [v[0 if source_side < 2 else 1] for v in vertices]
            image = _forward((min(coords), max(coords)), *domain._transport(side))
            if image is not None:
                images.append(image)
        if not images:
            return None
        lo = max(cell[j][0], min(x[0] for x in images))
        hi = min(cell[j][1], max(x[1] for x in images))
        if lo > hi:
            return None
        cell[j] = (lo, hi)
    return tuple(cell)


def optimize_point(domain, score, *, tolerance=1e-5, max_cells=20000):
    """Exact B&B with necessary-domain contraction and no inside-cell proofs.

    Point optimization does not need an integral's all-corners inside proof.
    Budget counts every visited B&B node; contraction is a bounded sweep per
    node, not an increased cell limit. Its arithmetic cost is measured separately.
    Analytic 1U delegates unchanged to the old solver.
    """
    tol = as_fraction(tolerance)
    if tol <= 0 or not isinstance(max_cells, int) or isinstance(max_cells, bool) or max_cells < 1:
        raise ValueError("positive tolerance/resource cap required")
    if not isinstance(domain, NativeEDomain):
        raise TypeError("point-search prototype requires native-E domain")
    if score.dimension != domain.dimension:
        raise ValueError("domain/score dimension mismatch")
    if domain.dimension == 1:
        return reference_optimize_point(domain, score, tolerance=tolerance, max_cells=max_cells)
    queue, serial = [], itertools.count()
    best, value, best_key = None, None, None
    calls, contractions, point_calls = 0, 0, 0

    def consider(point):
        nonlocal best, value, best_key, point_calls
        point_calls += 1
        witness = _checked_point(domain, point)
        if witness is None:
            return
        v = score.value_at(point)
        key = tuple(witness.xyxy)
        if value is None or v > value or (v == value and key < best_key):
            best, value, best_key = tuple(point), v, key

    def add(cell):
        nonlocal calls, contractions
        calls += 1
        # A box maximum is a valid upper bound before any geometric work.
        _, upper = score.cell_score_bounds(cell)
        if value is not None and upper <= value:
            return
        consider(score.cell_maximizer(cell))
        if value is not None and upper <= value:
            return
        contractions += 1
        cell = contract_bounds(domain, cell)
        if cell is None:
            return
        _, upper = score.cell_score_bounds(cell)
        if value is not None and upper <= value:
            return
        consider(score.cell_maximizer(cell))
        if value is not None and upper <= value:
            return
        witness = domain.feasible_witness(cell)
        if witness is not None:
            consider(witness.point)
        if value is None or upper > value:
            heapq.heappush(queue, (-upper, cell, next(serial)))

    def diagnostics(upper, qualified):
        return dict(qualified=qualified, status="CERTIFIED" if qualified else "RESOURCE_LIMIT",
                    method="conditional_source_projection", cells=calls,
                    contractions=contractions, point_calls=point_calls,
                    score_lower=value, score_upper=upper,
                    arithmetic="exact Fraction necessary projections and raw DFL bounds",
                    tie_rule="highest visited feasible score then xyxy; certified early exit; not old traversal parity")

    add(_bounds(domain.bounds))
    while queue:
        upper = max(value if value is not None else -queue[0][0], -queue[0][0])
        if value is not None and upper-value <= tol:
            return PointResult(best, upper-value, diagnostics(upper, True))
        if calls+2 > max_cells:
            d = diagnostics(upper, False)
            d.update(point=best, gap=None if value is None else upper-value)
            raise NumericalCertificationError("global point resource cap", d)
        neg_upper, cell, _ = heapq.heappop(queue)
        if value is not None and -neg_upper <= value:
            continue
        for child in _split(cell, score):
            add(child)
    if best is None:
        raise NumericalCertificationError("no certified feasible point", diagnostics(None, False))
    return PointResult(best, Q(0), diagnostics(value, True))
