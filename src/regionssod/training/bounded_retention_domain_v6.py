"""V6 output domain: only active U is restricted to the finite canvas.

Native E stays fixed, including finite values just outside the canvas. Every
output box must still have strictly positive width and height. Source outer
constraints, transport, active-side resolution, and joint certificates are
inherited unchanged from the frozen continuous domain.
"""
from fractions import Fraction as Q

from .bounded_retention_continuous_domain import ContinuousDomain


class NativeEDomain(ContinuousDomain):
    """ContinuousDomain with the native_e_revision_contract D definition."""

    def __init__(self, record, boxxyxy, canvas, *, point_cache_size=1024):
        super().__init__(record, boxxyxy, canvas, point_cache_size=point_cache_size)
        # The inherited U intervals already intersect canvas with the strict
        # opposite-E inequality correctly. Only the blanket E-canvas gate was
        # additional. Rebuild its downstream projections before any query.
        self._empty_D = any(i.empty for i in self._d_intervals) or any(
            not self.mask[a] and not self.mask[a + 2]
            and self.box[a] >= self.box[a + 2] for a in (0, 1)
        )
        marginals = tuple(self._marginal(j) for j in range(self.dimension))
        self._empty_C = self._empty_D or any(not xs for xs in marginals)
        self.bounds = tuple((xs[0].lo, xs[-1].hi) if xs else (Q(0), Q(0))
                            for xs in marginals)
        self.search_bounds = self.bounds
        self.intervals = marginals[0] if self.dimension == 1 else None

    def _in_D(self, box):
        return (not self._empty_D
                and all(0 <= box[j] <= self.canvas[j % 2] for j in self.active_sides)
                and box[0] < box[2] and box[1] < box[3])

    def classify_D(self, bounds):
        """Certify a closed U cell without clipping or replacing fixed E."""
        cell = self._cell(bounds)
        if self._empty_D:
            return "outside"
        box = [(x, x) for x in self.box]
        all_within = True
        for side, (lo, hi) in zip(self.active_sides, cell):
            width = self.canvas[side % 2]
            if hi < 0 or lo > width:
                return "outside"
            all_within &= 0 <= lo <= hi <= width
            box[side] = (max(Q(0), lo), min(width, hi))
        if any(box[a][0] >= box[a + 2][1] for a in (0, 1)):
            return "outside"
        positive_everywhere = all(box[a][1] < box[a + 2][0] for a in (0, 1))
        return "inside" if all_within and positive_everywhere else "mixed"
