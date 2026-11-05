"""Certified envelope rejection for the continuous C posterior (V5).

The ideal algorithm uses independent infinite uniform bit strings. This
implementation uses a private PCG64 stream, assumes its words behave as those
independent uniform bits, and FAILS if 256-bit prefixes cannot decide. PCG64 is
not a mathematical source of perfect randomness. No midpoint membership test
is used. Only q is approximated after a complete inside/acceptance certificate.
Finite-K sampling variance is not a deterministic integral error certificate.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction as Q
import hashlib
import heapq
import json
import math

import numpy as np
import torch

from .bounded_retention_continuous_numerics import (
    DecimalBounds as DB, DFLScore, NumericalCertificationError, as_fraction,
    certify_rectangle,
)

K = 256
Q_ERROR = Q(1, 10**10)


@dataclass
class PosteriorResult:
    q_mean: torch.Tensor
    log_mass_lower: float
    log_mass_upper: float
    diagnostics: dict


class SamplingCertificationError(NumericalCertificationError):
    pass


def derive_sampling_seed(*, source_object_id, occurrence_id, anchor, stride,
                         call_index, seed0=0):
    """Stable private-stream key; never Python hash() or a global RNG draw."""
    for value in (seed0, call_index):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
            raise ValueError("seed0 and call_index must be nonnegative integers")
    if isinstance(source_object_id, np.generic):
        source_object_id = source_object_id.item()
    if isinstance(occurrence_id, np.generic):
        occurrence_id = occurrence_id.item()
    payload = dict(version="continuous-envelope-v1", seed0=int(seed0),
        source_object_id=source_object_id, occurrence_id=occurrence_id,
        anchor=[str(as_fraction(x)) for x in anchor],
        stride=str(as_fraction(stride)), call_index=int(call_index))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=16).digest(), "big")


class _Prefix:
    """Closed rational enclosure of a latent uniform with a retained prefix."""
    def __init__(self):
        self.word, self.bits = 0, 0

    def extend(self, rng):
        self.word = (self.word << 64) | int(rng.random_raw())
        self.bits += 64

    @property
    def bounds(self):
        return Q(self.word, 1 << self.bits), Q(self.word + 1, 1 << self.bits)


@dataclass
class _Leaf:
    bounds: tuple
    kind: str
    maximum: Q
    envelope: DB
    interior: DB


def _kind(domain, cell):
    kind = domain.classify(cell)
    if kind not in ("inside", "outside", "mixed"):
        raise ValueError("classify must return inside/outside/mixed")
    return kind


def _failure(message, diagnostics, status):
    raise SamplingCertificationError(message, dict(diagnostics, qualified=False, status=status))


def _mass_diagnostics(inside, envelope, shift):
    result = {"shifted_mass_lower": str(inside.lo), "shifted_mass_upper": str(envelope.hi),
              "raw_score_shift": str(shift), "arithmetic_certified": True}
    if inside.lo > 0 and envelope.hi > 0:
        result["acceptance_lower_bound"] = DB.exact(Q(inside.lo) / Q(envelope.hi)).float_bounds()[0]
        logs = DB(inside.lo, envelope.hi).log() + shift
        result["log_mass_lower"], result["log_mass_upper"] = logs.float_bounds()
    else:
        result["acceptance_lower_bound"] = 0.
    return result


def _partition(domain, score, max_cells):
    root = tuple((as_fraction(lo), as_fraction(hi)) for lo, hi in domain.bounds)
    if len(root) != score.dimension or any(lo > hi for lo, hi in root):
        raise ValueError("invalid domain bounds/dimension")
    if any(lo == hi for lo, hi in root):
        _failure("zero Lebesgue volume cannot be thickened", {}, "ZERO_VOLUME")
    shift = sum(score.shifts, Q(0))
    leaves, queue, calls, serial = {}, [], 0, 0
    inside = envelope = DB.exact(0)

    def add(cell):
        nonlocal calls, serial, inside, envelope
        calls += 1
        kind = _kind(domain, cell)
        if kind == "outside":
            return
        maximum = score.cell_score_bounds(cell)[1]
        volume = math.prod(hi - lo for lo, hi in cell)
        env = DB.exact(maximum - shift).exp() * volume
        mass = certify_rectangle(score, cell, gradients=False).mass if kind == "inside" else DB.exact(0)
        leaf = _Leaf(cell, kind, maximum, env, mass)
        leaves[serial] = leaf
        # Refine non-flat inside cells too: envelope excess wastes proposals.
        waste = (env - mass).hi
        heapq.heappush(queue, (waste.copy_negate(), serial))
        serial += 1
        inside, envelope = inside + mass, envelope + env

    add(root)
    while inside.lo <= 0 or Q(inside.lo) * 8 < Q(envelope.hi):
        diag = dict(cells=calls, cover_cells=len(leaves),
                    **_mass_diagnostics(inside, envelope, shift))
        if not queue:
            _failure("no positive-volume certified cover", diag, "ZERO_VOLUME")
        if calls + 2 > max_cells:
            _failure("envelope partition resource cap", diag, "PARTITION_RESOURCE_LIMIT")
        _, key = heapq.heappop(queue)
        leaf = leaves.pop(key)
        inside, envelope = inside - leaf.interior, envelope - leaf.envelope
        cell = leaf.bounds
        hint = getattr(domain, "split_hint", None)
        split = hint(cell) if hint is not None else None
        if split is None:
            widths = [(hi - lo) / (root[j][1] - root[j][0]) for j, (lo, hi) in enumerate(cell)]
            if leaf.kind == "inside":
                variation = []
                for j, (lo, hi) in enumerate(cell):
                    values = [score.axis_value(j, x) for x in score.breakpoints(j, lo, hi)]
                    variation.append(max(values) - min(values))
                axis = max(range(score.dimension), key=lambda j: (variation[j], widths[j], -j))
            else:
                axis = max(range(score.dimension), key=lambda j: (widths[j], -j))
            lo, hi = cell[axis]
            cut = (lo + hi) / 2
            knots = score.breakpoints(axis, lo, hi)[1:-1]
            if knots:
                cut = min(knots, key=lambda x: (abs(x - cut), x))
        else:
            axis, cut = int(split[0]), as_fraction(split[1])
            if not 0 <= axis < score.dimension or not cell[axis][0] < cut < cell[axis][1]:
                raise ValueError("invalid domain split_hint")
        left, right = list(cell), list(cell)
        left[axis], right[axis] = (cell[axis][0], cut), (cut, cell[axis][1])
        add(tuple(left)); add(tuple(right))

    # Re-sum the frozen leaves to remove subtraction dependency overestimation.
    frozen = tuple(leaves[key] for key in sorted(leaves))
    inside, envelope = DB.exact(0), DB.exact(0)
    for leaf in frozen:
        inside, envelope = inside + leaf.interior, envelope + leaf.envelope
    diag = dict(cells=calls, cover_cells=len(frozen),
                inside_cells=sum(x.kind == "inside" for x in frozen),
                mixed_cells=sum(x.kind == "mixed" for x in frozen),
                **_mass_diagnostics(inside, envelope, shift))
    if inside.lo <= 0 or Q(inside.lo) * 8 < Q(envelope.hi):
        _failure("frozen envelope failed acceptance certificate", diag, "ARITHMETIC_LIMIT")
    return frozen, envelope, diag


def _cdf(leaves, total):
    # The true categorical weights are Vol*exp(exact maximum), not rounded
    # midpoints or upper bounds. Intervals certify their normalized CDF.
    cumulative = DB.exact(0)
    bounds = [(Q(0), Q(0))]
    for leaf in leaves[:-1]:
        cumulative = cumulative + leaf.envelope
        value = cumulative / total
        # Intersect with the analytically known [0,1] CDF range, also keeping
        # the final boundary monotone when interval upper rounding exceeds 1.
        bounds.append((max(Q(0), Q(value.lo)), min(Q(1), Q(value.hi))))
    bounds.append((Q(1), Q(1)))
    return bounds


def _select(cdf, rng, max_bits, diagnostics):
    prefix = _Prefix()
    uppers = [hi for lo, hi in cdf]
    # Cumulative positive interval upper bounds are monotone by construction.
    for _ in range(max_bits // 64):
        prefix.extend(rng)
        diagnostics["maxbits"] = max(diagnostics["maxbits"], prefix.bits)
        lo, hi = prefix.bounds
        index = bisect_right(uppers, lo) - 1
        if 0 <= index < len(cdf) - 1 and hi <= cdf[index + 1][0]:
            return index
    _failure("categorical CDF undecided at prefix cap", diagnostics, "BITS_RESOURCE_LIMIT")


def _q_at_midpoint(score, cell):
    q = [[Q(0)] * 16 for _ in range(score.dimension)]
    error = Q(0)
    for j, (lo, hi) in enumerate(cell):
        k, weight = score.weights(j, (lo + hi) / 2)
        q[j][k], q[j][k + 1] = 1 - weight, weight
        # Each clipped DFL basis function is globally 1-Lipschitz in distance,
        # including intervals that cross knots or saturation boundaries.
        error = max(error, (hi - lo) / (2 * score.stride))
    return q, error


def sample_posterior(domain, score: DFLScore, *, seed, samples=256,
                     max_proposals=None, max_cells=20000, max_bits=256):
    """Return fixed-K posterior q mean with certified decisions and log bounds.

    Defaults: K=256, 64*K=16384 proposals, 20000 oracle partition cells,
    256 bits per latent uniform. Smaller caps may be used for failure tests;
    larger caps and changing K are rejected. Intended for multi-U only.
    q_mean is detached float64 on the score device. Its approximation error
    bound excludes Monte Carlo variance and downstream lower-precision casts.
    """
    if not isinstance(samples, (int, np.integer)) or samples != K or isinstance(samples, bool):
        raise ValueError("the frozen estimator requires samples=256")
    if not 2 <= score.dimension <= 4:
        raise ValueError("posterior sampler is multi-U; 1U uses analytic integration")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("nonnegative explicit integer seed required")
    max_proposals = 64 * K if max_proposals is None else max_proposals
    for name, value, limit in (("max_proposals", max_proposals, 64*K), ("max_cells", max_cells, 20000)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or not 1 <= value <= limit:
            raise ValueError(f"{name} must be a positive integer <= {limit}")
    if not isinstance(max_bits, (int, np.integer)) or isinstance(max_bits, bool) or max_bits not in (64, 128, 192, 256):
        raise ValueError("max_bits must be 64, 128, 192, or 256")
    leaves, total, diagnostics = _partition(domain, score, max_cells)
    if not all(math.isfinite(diagnostics[key]) for key in ("log_mass_lower", "log_mass_upper")):
        _failure("raw log-mass bounds cannot be represented as finite binary64", diagnostics, "ARITHMETIC_RESOURCE_LIMIT")
    diagnostics.update(qualified=False, status="SAMPLING", accepted=0, proposals=0, maxbits=0,
        seed=int(seed), samples=K, max_proposals=max_proposals, max_cells=max_cells, max_bits=max_bits,
        arithmetic="Fraction geometry/score + outward Decimal80 CDF/exp decisions",
        rng="private PCG64 random_raw uint64 prefixes",
        rng_assumption="ideal proof assumes independent perfect bits; PCG64 pseudorandomness is an explicit implementation assumption",
        full_cover=True, midpoint_membership=False)
    retention = getattr(getattr(domain, "physical", None), "effective_retention", None)
    if retention is not None:
        diagnostics["retention_tau"] = str(as_fraction(retention))
    cdf = _cdf(leaves, total)
    rng = np.random.PCG64(int(seed))
    accepted, errors = [], []
    while len(accepted) < K:
        if diagnostics["proposals"] >= max_proposals:
            _failure("fixed K not reached before proposal cap", diagnostics, "PROPOSAL_RESOURCE_LIMIT")
        diagnostics["proposals"] += 1
        leaf = leaves[_select(cdf, rng, max_bits, diagnostics)]
        coordinates = [_Prefix() for _ in range(score.dimension)]
        accept = _Prefix()
        decided = False
        for _ in range(max_bits // 64):
            for prefix in (*coordinates, accept):
                prefix.extend(rng)
            diagnostics["maxbits"] = max(diagnostics["maxbits"], accept.bits)
            cell = tuple((lo + (hi-lo)*p.bounds[0], lo + (hi-lo)*p.bounds[1])
                         for (lo, hi), p in zip(leaf.bounds, coordinates))
            low, high = score.cell_score_bounds(cell)
            threshold = DB.fractions(low - leaf.maximum, high - leaf.maximum).exp()
            rlo, rhi = accept.bounds
            if rlo >= Q(threshold.hi):
                decided = True
                break
            kind = "inside" if leaf.kind == "inside" else _kind(domain, cell)
            if kind == "outside":
                decided = True
                break
            if kind == "inside" and rhi <= Q(threshold.lo):
                q, error = _q_at_midpoint(score, cell)
                # Reserve half the error budget for final binary64 conversion.
                if error <= Q_ERROR / 2:
                    accepted.append(q); errors.append(error)
                    diagnostics["accepted"] = len(accepted)
                    decided = True
                    break
        if not decided:
            _failure("membership/acceptance/q undecided at prefix cap", diagnostics, "BITS_RESOURCE_LIMIT")
    mean = [[sum((q[j][k] for q in accepted), Q(0)) / K for k in range(16)]
            for j in range(score.dimension)]
    values = [[float(x) for x in row] for row in mean]
    conversion = max(abs(Q.from_float(values[j][k]) - mean[j][k])
                     for j in range(score.dimension) for k in range(16))
    error_bound = sum(errors, Q(0)) / K + conversion
    if error_bound > Q_ERROR:
        _failure("q mean conversion exceeds certified budget", diagnostics, "ARITHMETIC_LIMIT")
    qs = np.array([[[float(x) for x in row] for row in q] for q in accepted], dtype=np.float64)
    diagnostics.update(qualified=True, status="CERTIFIED", accepted=K,
        acceptance_observed=K / diagnostics["proposals"],
        q_midpoint_error_bound=DB.exact(max(errors)).float_bounds()[1],
        q_mean_error_bound=DB.exact(error_bound).float_bounds()[1],
        # Uniform bound for the q approximation, not this realization's
        # smaller error bound promoted into an unconditional bias claim.
        q_approximation_bias_bound=DB.exact(Q_ERROR).float_bounds()[1],
        source_membership="entire small cell inside; inherited when envelope leaf is certified inside",
        variance_of_mean=(qs.var(axis=0, ddof=1) / K).tolist(),
        variance_bound_per_component=1 / (4*K),
        variance_semantics="empirical variance/K; universal bound assumes ideal independent posterior samples",
        gradient_semantics="posterior first-gradient estimator, not deterministic integration or Hessian")
    return PosteriorResult(torch.tensor(values, dtype=torch.float64, device=score.logits.device).detach(),
                           diagnostics["log_mass_lower"], diagnostics["log_mass_upper"], diagnostics)


sample_domain = sample_posterior
