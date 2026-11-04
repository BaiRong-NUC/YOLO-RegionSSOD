"""Counterexamples for the diagnostic; these do not certify a student loss."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("pqt", Path(__file__).parents[1] / "scripts/diagnose_positive_quality_transport.py")
pqt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pqt)


def test_rearrangement_preserves_mass_zeros_and_changes_inverse_quality():
    target = torch.tensor([.9, 0., .1, .4])
    quality = torch.tensor([.2, .99, .8, .5])
    changed = pqt.permute_targets(target, quality)
    assert torch.equal(changed, torch.tensor([.1, 0., .9, .4]))
    assert torch.equal(changed.sort().values, target.sort().values)
    assert (changed * quality).sum() > (target * quality).sum()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_equal_geometry_is_identity_and_singletons_are_protected(dtype):
    target = torch.tensor([.6, 0., .2, .8], dtype=dtype)
    assert torch.equal(pqt.permute_targets(target, torch.ones(4)), target)
    single = torch.tensor([0., .3, 0.], dtype=dtype)
    assert torch.equal(pqt.permute_targets(single, torch.tensor([1., 0., 1.])), single)


def test_pairs_distinguish_score_error_from_target_error_and_gap():
    target = torch.tensor([.8, .2])
    scores = torch.tensor([.01, .9])
    iou = torch.tensor([.7, .8])
    row = pqt.object_row(target, scores, iou, iou, torch.tensor([[2., 2.], [3., 3.]]), torch.tensor([0., 0., 10., 10.]))
    assert row["inverse_pairs"] == row["large_gap_inverse_pairs"] == 1
    assert row["score_inverse_pairs"] == 0
    assert row["weighted_iou_after"] > row["weighted_iou_before"]


@pytest.mark.parametrize("target,quality", [([float("nan")], [1.]), ([1.1], [1.]), ([.1], [float("inf")])])
def test_corrupt_cache_rejected(target, quality):
    with pytest.raises(ValueError):
        pqt.permute_targets(torch.tensor(target), torch.tensor(quality))
