"""Same continuous density target, analytic 1U and posterior-gradient multi-U.

V4's failed deterministic whole-domain quadrature remains available unchanged.
Displayed multi-U loss is the detached midpoint of an NLL enclosure, not its
exact value. Only the first derivative is the declared stochastic estimator.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction

import torch
from ultralytics.utils.loss import DFLoss

from .bounded_retention import UnsupportedObservation
from .bounded_retention_objective_v4 import (
    ContinuousDensityLoss, continuous_support, current_unknown, score_for,
    partial_dfl as point_or_deterministic_dfl,
)


def private_seed(identity, anchor, stride, call_number, seed=0):
    if len(identity) != 2 or any(x is None for x in identity):
        raise ValueError("source and augmentation occurrence are required for private RNG")
    key = ["continuous-posterior-v5", int(seed), *map(int, identity),
           *[float(x).hex() for x in anchor], float(stride).hex(), int(call_number)]
    return int.from_bytes(hashlib.sha256(json.dumps(key).encode("ascii")).digest()[:16], "big")


def nll_enclosure(log_d, log_mass_lower, log_mass_upper, arithmetic_budget=1e-7):
    """Combine already-outward mass endpoints without rounding bounds inward."""
    if not all(math.isfinite(x) for x in (log_d, log_mass_lower, log_mass_upper, arithmetic_budget)):
        raise UnsupportedObservation("nonfinite posterior NLL enclosure input")
    if arithmetic_budget < 0 or log_mass_upper < log_mass_lower:
        raise UnsupportedObservation("invalid posterior mass/arithmetic enclosure")
    d, e, lo, hi = map(Fraction, (log_d, arithmetic_budget, log_mass_lower, log_mass_upper))
    lower, upper = max(Fraction(0), d-e-hi), d+e-lo
    if upper < lower:
        raise UnsupportedObservation("invalid posterior NLL enclosure")
    a, b = float(lower), float(upper)
    if Fraction(a) > lower:
        a = math.nextafter(a, -math.inf)
    if Fraction(b) < upper:
        b = math.nextafter(b, math.inf)
    return a, b


def partial_dfl(logits, support, anchor_row, mode, *, rng_seed=0, audit=False):
    if mode == "control":
        return point_or_deterministic_dfl(logits, support, anchor_row, mode)
    if mode != "candidate":
        raise ValueError("partial loss is only defined for candidate/control")
    # Imports stay local so reviewing this integration shell cannot silently
    # qualify missing numerical components or construct a model.
    from .bounded_retention_continuous_normalizer import normalizer, integrate_interval_union
    mask, target = support["mask"], support["target"][anchor_row]
    known = (~mask).nonzero().flatten()
    exact = (DFLoss(logits.shape[-1])(logits[known], target[known][None].clone()).squeeze()
             * len(known)) if len(known) else logits.sum()*0
    anchor, stride = support["anchors"][anchor_row], support["strides"][anchor_row].item()
    score = score_for(logits, mask, anchor, stride)
    denominator = normalizer(logits, support["nominal_box"], anchor, stride, mask,
                             support["canvas"], audit=audit)
    log_d, d_diag = denominator
    if not d_diag.get("qualified", False):
        raise UnsupportedObservation("continuous normalizer arithmetic is not qualified")
    domain = support["domain"]
    if domain.dimension == 1:
        result = integrate_interval_union(score.logits[0], domain.intervals,
            score.offsets[0], score.stride, score.signs[0], audit=audit)
        log_c, c_diag = result
        if not c_diag.get("qualified", False):
            raise UnsupportedObservation("analytic C interval arithmetic is not qualified")
        unknown = log_d-log_c
        if not torch.isfinite(unknown) or float(unknown.detach()) < -2e-7:
            raise UnsupportedObservation("invalid analytic continuous event mass")
        diag = {"method": "analytic_1u", "qualified": True, "numerator": c_diag,
                "denominator": d_diag}
    else:
        from .bounded_retention_continuous_sampling import sample_posterior
        sample = sample_posterior(domain, score, seed=rng_seed, samples=256)
        diag = sample.diagnostics
        if not diag.get("qualified", False):
            raise UnsupportedObservation("posterior sampler is not qualified")
        q = sample.q_mean.to(device=logits.device, dtype=torch.float64).detach()
        if q.shape != score.logits.shape or not torch.isfinite(q).all():
            raise UnsupportedObservation("invalid posterior interpolation moments")
        if ((q < 0) | (q > 1)).any() or not torch.allclose(q.sum(-1), torch.ones_like(q[:, 0]), atol=1e-12, rtol=0):
            raise UnsupportedObservation("posterior moments are not interpolation weights")
        # The true I_C is at most I_D. This tightens an event-mass bound; it does
        # not clamp coordinates, turn an empty C nonempty, or alter gradients.
        logd = float(log_d.detach())
        d_error = 1e-7
        lower, upper = nll_enclosure(logd, float(sample.log_mass_lower),
                                    float(sample.log_mass_upper), d_error)
        surrogate = log_d - (q * score.logits).sum()
        unknown = surrogate - surrogate.detach() + (lower+upper)/2
        diag = {**diag, "method": "posterior_gradient", "nll_interval": [lower, upper],
                "reported_value": "detached_interval_midpoint_not_exact_NLL",
                "denominator": d_diag}
    support.setdefault("density_diagnostics", {})[int(anchor_row)] = diag
    return (exact + unknown)/4


class ContinuousSamplingLoss(ContinuousDensityLoss):
    def __init__(self, model):
        super().__init__(model)
        self.posterior_call_number = 0

    def partial_dfl(self, logits, support, anchor_row, mode):
        seed = private_seed(support["identity"], support["anchors"][anchor_row].tolist(),
            support["strides"][anchor_row].item(), self.posterior_call_number)
        return partial_dfl(logits, support, anchor_row, mode, rng_seed=seed)

    def get_assigned_targets_and_loss(self, preds, batch):
        self.posterior_call_number += 1
        result = super().get_assigned_targets_and_loss(preds, batch)
        events = [d for s in self.supports.values() for d in s.get("density_diagnostics", {}).values()]
        stochastic = [d for d in events if d["method"] == "posterior_gradient"]
        self.diagnostics.update(analytic_density_anchors=len(events)-len(stochastic),
            posterior_density_anchors=len(stochastic), posterior_samples_per_anchor=256,
            posterior_event_details=[{"identity": s["identity"], "anchor_row": row,
                "nll_interval": d["nll_interval"], "reported_value": d["reported_value"]}
                for s in self.supports.values() for row, d in s.get("density_diagnostics", {}).items()
                if d["method"] == "posterior_gradient"],
            posterior_nll_interval_scope="unknown_event_before_quarter_normalization_TAL_weight_and_gain",
            posterior_nll_max_interval_width=max((d["nll_interval"][1]-d["nll_interval"][0]
                                                    for d in stochastic), default=0.),
            posterior_proposals=sum(d.get("proposals", 0) for d in stochastic),
            posterior_partition_cells=sum(d.get("cells", 0) for d in stochastic),
            posterior_min_acceptance_bound=min((d.get("acceptance_lower_bound", 0.) for d in stochastic), default=1.),
            posterior_max_q_error=max((d.get("q_mean_error_bound", math.inf) for d in stochastic), default=0.),
            posterior_variance_bound_per_component=1/1024,
            posterior_call_number=self.posterior_call_number)
        return result
