"""Ablating one supervision mechanism must retain the other mechanism."""
import pytest
import torch

from regionssod.training.p1 import classification_factors


@pytest.mark.parametrize(
    "soft_weights, uncertain_ignore, expected_gradient",
    [(True, True, [.2, 0., .5]), (False, True, [.5, 0., .5]), (True, False, [.2, .5, .5])],
)
def test_independent_ablation_preserves_remaining_supervision(soft_weights, uncertain_ignore, expected_gradient):
    anchors = torch.tensor([[10., 10.], [30., 30.], [90., 90.]])
    foreground = torch.tensor([[True, False, False]])
    target_indices = torch.zeros((1, 3), dtype=torch.long)
    batch = {"pseudo_weights": torch.tensor([.4]), "batch_idx": torch.tensor([0]),
             "ignore_bboxes": torch.tensor([[.3, .3, .6, .6]]), "ignore_batch_idx": torch.tensor([0])}
    factors = classification_factors(anchors, foreground, target_indices, batch, 1, "cpu",
                                     torch.tensor([100, 100]), soft_weights=soft_weights,
                                     uncertain_ignore=uncertain_ignore)
    logits = torch.zeros((1, 3, 2), requires_grad=True)
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction="none") * factors).sum()
    loss.backward()
    assert torch.allclose(logits.grad[0], torch.tensor(expected_gradient).unsqueeze(-1).expand(-1, 2))
