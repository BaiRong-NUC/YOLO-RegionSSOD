"""Independent zero-training primitives for retention-constrained observations.

This module does not authorize training. Decimal intervals describe serialized
measurements, not recovered hidden boxes. A fixed-coordinate loss requires its
caller to certify that coordinate semantics match the observation contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import math

import torch
from torch.nn import functional as F
from ultralytics.utils.loss import DFLoss


@dataclass(frozen=True)
class Interval:
    lo: Decimal
    hi: Decimal

    def contains(self, value):
        return self.lo <= Decimal(value) <= self.hi


def serialized_interval(token: str) -> Interval:
    """Outward .10g decimal rounding envelope, BEFORE float arithmetic error.

    At decimal powers the adjacent lower decade has a smaller quantum; taking
    the upper decade quantum on both sides is conservative. Trailing zeros
    omitted by format do not change the ten-significant-digit quantum.
    This does not invert source arithmetic, clip, or later float32 conversion.
    """
    value = Decimal(token)
    if not value.is_finite() or not 0 <= value <= 1:
        raise ValueError("normalized finite decimal required")
    if value == 0:
        # Python's general format does not round positive floats to literal 0.
        return Interval(value, value)
    with localcontext() as ctx:
        ctx.prec = 80
        radius = Decimal(5).scaleb(value.adjusted() - 10)
        return Interval(max(Decimal(0), value - radius), min(Decimal(1), value + radius))


def edge_intervals(tokens: list[str]) -> tuple[Interval, ...]:
    if len(tokens) != 4:
        raise ValueError("xywh requires four tokens")
    x, y, w, h = [serialized_interval(t) for t in tokens]
    if w.lo <= 0 or h.lo <= 0:
        raise ValueError("positive sizes required")
    with localcontext() as ctx:
        ctx.prec = 80
        return (Interval(x.lo - w.hi / 2, x.hi - w.lo / 2),
                Interval(y.lo - h.hi / 2, y.hi - h.lo / 2),
                Interval(x.lo + w.lo / 2, x.hi + w.hi / 2),
                Interval(y.lo + h.lo / 2, y.hi + h.hi / 2))


class UnsupportedObservation(ValueError):
    """No legal finite-support conditioning event; never silently clamp it."""


def feasible_atoms(distances, unknown, area_cap, bins=16):
    """Enumerate unknown distances, fixing exact continuous known distances.

    Distances are l,t,r,b in stride units. Unknown coordinates are lower bounds;
    no claim is made that an input bound was actually truncated. This function
    assumes certified unrounded coordinates; serialized-interval certification
    is deliberately separate. Returns all unknown atoms with their geometry.
    """
    if distances.shape != (4,) or unknown.shape != (4,) or unknown.dtype != torch.bool:
        raise ValueError("expected four distances and boolean unknown mask")
    if bins < 2 or not torch.isfinite(distances).all() or not math.isfinite(float(area_cap)) or area_cap <= 0:
        raise UnsupportedObservation("nonfinite or invalid observation")
    if (distances < 0).any():
        raise UnsupportedObservation("negative distance: anchor outside observed box")
    # Original DFL uses max 14.99 for K=16. Do not turn an exact target at 15
    # into 14.99 in a partially unknown event.
    if (distances[~unknown] > bins - 1 - .01).any():
        raise UnsupportedObservation("exact distance outside unclamped DFL support")
    indices = unknown.nonzero().flatten()
    if not len(indices):
        geom = distances[None]
        if (geom[0, 0] + geom[0, 2]) * (geom[0, 1] + geom[0, 3]) > area_cap:
            raise UnsupportedObservation("inconsistent area bound")
        return torch.empty((1, 0), dtype=torch.long, device=distances.device), geom
    support = torch.arange(bins, device=distances.device)
    values = torch.cartesian_prod(*([support] * len(indices))).reshape(-1, len(indices))
    geom = distances.repeat(len(values), 1)
    geom[:, indices] = values.to(distances.dtype)
    valid = (geom[:, indices] >= distances[indices]).all(1)
    area = (geom[:, 0] + geom[:, 2]) * (geom[:, 1] + geom[:, 3])
    valid &= (area > 0) & (area <= area_cap)
    if not valid.any():
        raise UnsupportedObservation("empty finite-support observation")
    return values[valid], geom[valid]


def bounded_dfl(logits, distances, unknown, area_cap, *, mode="mass"):
    """Per-anchor four-side mean, WITHOUT downstream TAL weight/normalization.

    mass = sum of compatible atoms; map = strongest compatible atom.
    Both receive the identical set. The all-known branch calls native DFL on a
    clone (including its original saturation convention) for exact parity.
    Native saturation is not a certificate for a partially unknown event.
    """
    if mode not in {"mass", "map"}:
        raise ValueError("mode must be mass or map")
    if logits.ndim != 2 or logits.shape[0] != 4 or not torch.isfinite(logits).all():
        raise ValueError("finite [4,K] logits required")
    if unknown.shape != (4,) or unknown.dtype != torch.bool or distances.shape != (4,):
        raise ValueError("expected four distances and boolean unknown mask")
    if not torch.isfinite(distances).all():
        raise UnsupportedObservation("nonfinite distances")
    if not unknown.any():
        return DFLoss(logits.shape[1])(logits, distances[None].clone()).squeeze()
    atoms, _ = feasible_atoms(distances, unknown, area_cap, logits.shape[1])
    logp = F.log_softmax(logits, dim=-1)
    known = ~unknown
    target = distances[known]
    lower = target.long()
    fraction = target - lower
    rows = known.nonzero().flatten()
    exact_ce = -(logp[rows, lower] * (1 - fraction) + logp[rows, lower + 1] * fraction).sum()
    atom_logp = logp[unknown.nonzero().flatten()[None], atoms].sum(1)
    logz = torch.logsumexp(atom_logp, dim=0) if mode == "mass" else atom_logp.max()
    return (exact_ce - logz) / 4
