"""Source transport and full P1 criterion invariants, using generated fixtures."""
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from test_p1 import make_p1_dataset, loss_fixture
from regionssod.training.p1 import P1DetectionLoss
from regionssod.training.reliable_competitor_negative import (
    SourceDataset, ReliableCompetitorLoss, ReliableCompetitorModel, validate_batch_provenance,
)


def source_batch(batch):
    n = len(batch["cls"])
    batch["pseudo_weights"] = torch.ones(n)
    batch["object_is_human"] = torch.tensor([True, False])[:n]
    batch["object_id"] = torch.arange(2, 2+n)
    batch["object_class"] = batch["cls"].reshape(-1).long()
    batch["object_table"] = torch.stack((batch["object_class"].float(), torch.ones(n),
        torch.zeros(n), batch["object_is_human"].float(), batch["object_id"].float()), 1)
    batch["ignore_bboxes"] = torch.tensor([[.1, .1, .15, .15]])
    batch["ignore_batch_idx"] = torch.tensor([0])
    return batch


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


@pytest.mark.parametrize("closed", [False, True])
def test_augmented_source_geometry_matches_frozen_p1(tmp_path, closed):
    original, hyp = make_p1_dataset(tmp_path, augment=True)
    newer = SourceDataset(img_path=str(tmp_path / "images.txt"), metadata_path=tmp_path / "metadata.jsonl",
        imgsz=128, batch_size=2, augment=True, hyp=deepcopy(hyp), data=original.data)
    if closed:
        original.close_mosaic(deepcopy(hyp))
        newer.close_mosaic(deepcopy(hyp))
    # Separate identically seeded instances avoid mosaic buffer-state differences.
    for step in range(15):
        seed(step)
        left = original[step % len(original)]
        seed(step)
        right = newer[step % len(newer)]
        for key in ("img", "cls", "bboxes", "pseudo_weights", "ignore_bboxes", "ignore_batch_idx", "batch_idx"):
            assert torch.equal(left[key], right[key]), key
        combined = newer.collate_fn([right])
        validate_batch_provenance(combined, 4)
        table = combined["object_table"]
        assert torch.equal(table, newer.p1_lookup[combined["object_id"]])
        assert (table[combined["object_is_human"], 1] == 1).all()


@pytest.mark.parametrize("empty", [False, True])
def test_full_loss_assignment_regression_and_untouched_gradient(empty):
    model, preds, batch = loss_fixture()
    if empty:
        batch = {k: v[:0] for k, v in batch.items()}
    batch = source_batch(batch)
    baseline = P1DetectionLoss(model).get_assigned_targets_and_loss(preds, batch)
    baseline_grad = torch.autograd.grad(baseline[1].sum(), (preds["scores"], preds["boxes"]), retain_graph=True)
    outputs, grads, criteria = {}, {}, {}
    for mode in ("reference", "candidate", "control"):
        model.reliable_competitor_mode = mode
        model.reliable_competitor_diagnostics = True
        criterion = ReliableCompetitorLoss(model)
        outputs[mode] = criterion.get_assigned_targets_and_loss(preds, batch)
        grads[mode] = torch.autograd.grad(outputs[mode][1].sum(), (preds["scores"], preds["boxes"]), retain_graph=True)
        criteria[mode] = criterion
        assert all(torch.equal(x, y) for x, y in zip(baseline[0], outputs[mode][0]))
        assert torch.equal(baseline[1][[0, 2]], outputs[mode][1][[0, 2]])
        assert torch.equal(grads[mode][1], baseline_grad[1])
        assert criterion.last_diagnostics["changed_positive_target_weights"] == 0
        assert torch.equal(criterion.last_targets, criteria["reference"].last_targets)
        labels = criterion.last_assignment[0]
        foreground = criterion.last_assignment[3]
        idx = torch.arange(labels.shape[0])[:, None].expand_as(labels)
        anchor = torch.arange(labels.shape[1])[None].expand_as(labels)
        assert (criterion.last_weights[idx[foreground], anchor[foreground], labels[foreground].long()] == 1).all()
    assert torch.equal(baseline[1], outputs["reference"][1])
    assert all(torch.equal(x, y) for x, y in zip(baseline_grad, grads["reference"]))
    candidate = criteria["candidate"]
    unchanged = candidate.last_weights == 1
    assert torch.equal(grads["candidate"][0].permute(0, 2, 1)[unchanged], baseline_grad[0].permute(0, 2, 1)[unchanged])
    added = [(grads[m][0] - baseline_grad[0]).abs().sum() for m in ("candidate", "control")]
    torch.testing.assert_close(added[0], added[1], rtol=3e-5, atol=2e-7)


@pytest.mark.parametrize("field", ["object_id", "object_class", "object_is_human", "object_table", "cls", "pseudo_weights", "batch_idx"])
def test_corrupt_source_rejected_before_zero_quality(field):
    _, _, batch = loss_fixture()
    batch = source_batch(batch)
    if field == "object_is_human":
        batch[field][0] = False
    elif field == "object_table":
        batch[field][0, 3] = 0
    else:
        batch[field].view(-1)[0] += 0.5 if batch[field].is_floating_point() else 1
    with pytest.raises(ValueError):
        validate_batch_provenance(batch, 2)


def test_standard_validation_uses_original_loss_even_with_cached_training_criterion():
    model, preds, batch = loss_fixture()
    model.reliable_competitor_mode = "candidate"
    model.criterion = ReliableCompetitorLoss(model)
    model.eval()
    expected = P1DetectionLoss(model)(preds, batch)
    actual = ReliableCompetitorModel.loss(model, batch, preds)
    assert torch.equal(actual[0], expected[0])
    model.train()
    with pytest.raises(ValueError, match="provenance"):
        model.criterion(preds, batch)
