"""Conservative TAL candidate-pool adapter, with an image-level coverage guard."""
from __future__ import annotations

import torch
from ultralytics.utils.tal import TaskAlignedAssigner


class SupportedPointAssigner(TaskAlignedAssigner):
    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        expanded = super().select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt, eps)
        distances = torch.cat((xy_centers[None, None] - gt_bboxes[:, :, None, :2],
                               gt_bboxes[:, :, None, 2:] - xy_centers[None, None]), -1)
        inside = (distances.amin(-1) > eps) & mask_gt.bool()
        return torch.where(inside.any(-1, keepdim=True), inside, expanded)


def preserve_target_coverage(baseline, preferred):
    """Fallback the whole image if any baseline effective GT disappears."""
    base_scores, base_fg, base_ids = baseline[2:]
    new_scores, new_fg, new_ids = preferred[2:]
    fallback = []
    for image in range(base_fg.shape[0]):
        old = base_ids[image, base_fg[image] & (base_scores[image].sum(-1) > 0)]
        new = new_ids[image, new_fg[image] & (new_scores[image].sum(-1) > 0)]
        fallback.append(~torch.isin(old, new).all())
    mask = torch.stack(fallback)
    result = tuple(torch.where(mask.reshape(-1, *([1] * (old.ndim - 1))), old, new)
                   for old, new in zip(baseline, preferred))
    return result, mask


class CoverageGuardedAssigner(TaskAlignedAssigner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preferred = SupportedPointAssigner(*args, **kwargs)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        baseline = super().forward(*args, **kwargs)
        preferred = self.preferred(*args, **kwargs)
        result, _ = preserve_target_coverage(baseline, preferred)
        return result
