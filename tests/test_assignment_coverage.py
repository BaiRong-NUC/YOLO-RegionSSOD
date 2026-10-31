import torch

from regionssod.training.assignment import preserve_target_coverage


def outputs(ids, scores):
    idx = torch.tensor(ids, dtype=torch.long)
    quality = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
    shape = (*idx.shape, 4)
    return torch.zeros_like(idx), torch.zeros(shape), quality, quality[..., 0] > 0, idx


def test_fallback_preserves_entire_image_if_any_effective_target_disappears():
    baseline = outputs([[0, 1, 1], [0, 1, 1]], [[.5, .2, .3], [.4, .2, .1]])
    preferred = outputs([[0, 0, 0], [1, 0, 0]], [[.8, .1, 0], [.5, .7, .1]])
    merged, mask = preserve_target_coverage(baseline, preferred)
    assert mask.tolist() == [True, False]
    assert all(torch.equal(part[0], old[0]) for part, old in zip(merged, baseline))
    assert all(torch.equal(part[1], new[1]) for part, new in zip(merged, preferred))


def test_zero_score_foreground_does_not_falsely_satisfy_coverage():
    baseline = outputs([[0, 1]], [[.5, .3]])
    preferred = outputs([[0, 1]], [[.5, 0.]])
    preferred[3][:] = True
    merged, mask = preserve_target_coverage(baseline, preferred)
    assert mask.tolist() == [True]
    assert torch.equal(merged[2], baseline[2])


def test_empty_baseline_needs_no_fallback_and_no_gt_is_created():
    baseline = outputs([[0, 0]], [[0., 0.]])
    preferred = outputs([[0, 0]], [[0., 0.]])
    merged, mask = preserve_target_coverage(baseline, preferred)
    assert mask.tolist() == [False]
    assert not merged[3].any()
