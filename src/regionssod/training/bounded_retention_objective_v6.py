"""V6: original nominal E, with canvas constraints applying only to latent U.

Frozen V4/V5 files and their failed integration snapshots remain untouched.
The objective, posterior sampler and global point control are unchanged.
"""
from fractions import Fraction
import numpy as np
import torch
from ultralytics.utils.loss import DFLoss
from ultralytics.utils.metrics import bbox_iou
from .bounded_retention import UnsupportedObservation
from .bounded_retention_domain_v6 import NativeEDomain as ContinuousDomain
from .bounded_retention_continuous_numerics import optimize_point
from .bounded_retention_objective_v4 import (
    current_unknown, score_for, partial_dfl as point_or_deterministic_dfl,
    ContinuousObservationAssigner as FrozenAssigner)
from .bounded_retention_objective_v5 import ContinuousSamplingLoss, private_seed, nll_enclosure


def continuous_support(record, box, anchors, strides, canvas, logits):
    domain = ContinuousDomain(record, box.detach().cpu().tolist(), canvas)
    expected, _ = current_unknown(record, *canvas)
    mask = torch.as_tensor(expected, dtype=torch.bool, device=box.device)
    if not mask.any():
        raise UnsupportedObservation("continuous support called without active U")
    if not np.array_equal(np.asarray(domain.mask, dtype=bool), expected):
        raise UnsupportedObservation("continuous domain changed original active-edge semantics")
    target = torch.cat((anchors - box[:2], box[2:] - anchors), 1) / strides
    geometry = box.double()[None, None].expand(len(anchors), 1, 4).clone()
    certificates, rounding = [], []
    for row in range(len(anchors)):
        score = score_for(logits[row].detach(), mask, anchors[row], strides[row].item())
        try:
            certificate = optimize_point(domain, score)
        except Exception as exc:
            exc.diagnostics = {**(getattr(exc, "diagnostics", None) or {}),
                "source_object_id": record.get("source_object_id"),
                "occurrence_id": record.get("occurrence_id"),
                "anchor": anchors[row].detach().cpu().tolist(),
                "stride": strides[row].item(), "nominal_box": box.detach().cpu().tolist()}
            raise
        if not certificate.diagnostics.get("qualified", False) or certificate.gap > 1e-5:
            raise UnsupportedObservation("global continuous point DFL gap exceeds contract")
        geometry[row, 0, mask] = torch.tensor([float(v) for v in certificate.point],
                                             device=box.device, dtype=torch.float64)
        rounding.append(tuple(abs(Fraction(float(v)) - v) for v in certificate.point))
        certificates.append(certificate)
    # E comes directly from the nominal float32 target, never a distance roundtrip.
    geometry[..., ~mask] = box[~mask].double()
    if not ((geometry[..., 2:] - geometry[..., :2]) > 0).all():
        raise UnsupportedObservation("continuous witness has nonpositive nominal geometry")
    return {"mask": mask, "target": target, "domain": domain, "geometry": geometry,
        "valid": torch.ones(len(anchors), 1, dtype=torch.bool, device=box.device),
        "anchors": anchors, "strides": strides, "canvas": canvas, "nominal_box": box.clone(),
        "point_certificates": certificates, "geometry_rounding_error": rounding,
        "identity": (record.get("source_object_id"), record.get("occurrence_id"))}


class NativeEAssigner(FrozenAssigner):
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        owner = self.owner_ref()
        candidates = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        candidates = candidates * mask_gt
        metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, candidates)
        self.legal_assignment = torch.ones_like(candidates, dtype=torch.bool)
        for (batch_i, gt_i), record in owner.observations.items():
            box = gt_bboxes[batch_i, gt_i]
            effective, _ = current_unknown(record, *owner.canvas)
            if not effective.any():
                continue
            # Use only the original P1 pool, including its tiny-box expansion.
            # Encoding saturation does not remove continuous physical coordinates.
            indices = candidates[batch_i, gt_i].bool().nonzero().flatten()
            candidates[batch_i, gt_i] = False
            metric[batch_i, gt_i] = 0
            overlaps[batch_i, gt_i] = 0
            if not len(indices):
                raise UnsupportedObservation("partially unknown object has no native TAL candidate anchor")
            support = continuous_support(record, box, anc_points[indices], owner.strides[indices], owner.canvas, owner.detached_dist[batch_i, indices].view(-1, 4, owner.reg_max))
            known = support["target"][:, ~support["mask"]]
            owner.diagnostics["negative_known_distances"] += int((known < 0).sum())
            owner.diagnostics["overflow_known_distances"] += int((known > owner.reg_max - 1 - .01).sum())
            has_atoms = support["valid"].any(1)
            owner.diagnostics["empty_anchor_events"] += int((~has_atoms).sum())
            if not has_atoms.any():
                raise UnsupportedObservation("partially unknown object has no certified continuous anchor event")
            geometric_iou = bbox_iou(pd_bboxes[batch_i, indices, None, :], support["geometry"],
                                      xywh=False, CIoU=True).squeeze(-1)
            geometric_iou = geometric_iou.masked_fill(~support["valid"], -torch.inf)
            quality, chosen = geometric_iou.max(1)
            quality = torch.where(has_atoms, quality.clamp(min=0), torch.zeros_like(quality)).to(overlaps.dtype)
            cls = gt_labels[batch_i, gt_i, 0].long()
            overlaps[batch_i, gt_i, indices] = quality
            metric[batch_i, gt_i, indices] = pd_scores[batch_i, indices, cls].pow(self.alpha) * quality.pow(self.beta)
            candidates[batch_i, gt_i, indices] = has_atoms.to(candidates.dtype)
            support["indices"] = indices
            support["chosen"] = chosen
            self.legal_assignment[batch_i, gt_i] = False
            self.legal_assignment[batch_i, gt_i, indices] = has_atoms
            owner.supports[batch_i, gt_i] = support
            owner.diagnostics["supported_objects"] += 1
        topk = self.select_topk_candidates(metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        return topk * candidates * mask_gt.bool(), metric, overlaps


def partial_dfl(logits, support, anchor_row, mode, *, rng_seed=0, audit=False):
    if mode == "control":
        return point_or_deterministic_dfl(logits, support, anchor_row, mode)
    if mode != "candidate":
        raise ValueError("partial loss is only defined for candidate/control")
    # Imports stay local so reviewing this integration shell cannot silently
    # qualify missing numerical components or construct a model.
    from .bounded_retention_normalizer_v6 import normalizer, integrate_interval_union
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


class ContinuousNativeELoss(ContinuousSamplingLoss):
    def __init__(self, model):
        super().__init__(model)
        self.assigner = NativeEAssigner(self.assigner, self)

    def partial_dfl(self, logits, support, anchor_row, mode):
        seed = private_seed(support["identity"], support["anchors"][anchor_row].tolist(),
            support["strides"][anchor_row].item(), self.posterior_call_number)
        return partial_dfl(logits, support, anchor_row, mode, rng_seed=seed)
