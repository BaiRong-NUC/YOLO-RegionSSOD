import pytest
import torch
import sys
from pathlib import Path
from ultralytics.cfg import get_cfg

from regionssod.training.assignment import CoverageGuardedAssigner
from regionssod.training.coverage_guard import CoverageGuardLoss, CoverageGuardModel
from regionssod.training.p1 import P1DetectionLoss
from test_p1 import loss_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_p1_coverage_guard import qualifies


@pytest.mark.parametrize("ap,recall,expected", [
    (.1, 0., True), (.099, 0., False), (0., .5, True),
    (-.001, 1., False), (.2, -.001, False), (float("nan"), 1., False),
    (1., float("inf"), False)])
def test_prospective_exploratory_handoff_threshold(ap, recall, expected):
    assert qualifies({"mAP50_95": ap, "small_recall": recall}) is expected


@pytest.mark.parametrize("empty", [False, True])
def test_integrated_loss_uses_guarded_assignment_and_finite_gradients(empty):
    model, preds, batch = loss_fixture()
    if empty:
        batch = {key: value[:0] for key, value in batch.items()}
    criterion = CoverageGuardLoss(model)
    assert isinstance(criterion.assigner, CoverageGuardedAssigner)
    result = criterion.get_assigned_targets_and_loss(preds, batch)
    base = P1DetectionLoss(model).get_assigned_targets_and_loss(preds, batch)
    for image in range(2):
        old = base[0][1][image, base[0][0][image]]
        new = result[0][1][image, result[0][0][image]]
        assert torch.isin(old, new).all()
    assert torch.isfinite(result[1]).all()
    result[1].sum().backward()
    assert all(torch.isfinite(preds[key].grad).all() for key in ("boxes", "scores"))


def test_validation_bypasses_cached_guard_criterion_and_training_still_requires_metadata():
    config = {"nc": 2,
              "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [32, 3, 2]]],
              "head": [[[0, 1], 1, "Detect", [2]]]}
    model = CoverageGuardModel(config, nc=2, ch=3, verbose=False)
    model.args = get_cfg()
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    batch = {"img": torch.rand(2, 3, 64, 64),
             "batch_idx": torch.tensor([0., 1.]), "cls": torch.tensor([[0.], [1.]]),
             "bboxes": torch.tensor([[.5, .5, .25, .25], [.5, .5, .25, .25]])}
    weighted = batch | {"pseudo_weights": torch.tensor([.4, 1.])}
    model.train()
    loss, _ = model.loss(weighted)
    assert torch.isfinite(loss).all()
    cached = model.criterion
    assert isinstance(cached, CoverageGuardLoss)
    model.eval()
    with torch.no_grad():
        preds = model(batch["img"])
        expected, _ = P1DetectionLoss(model)(preds, batch)
        observed, _ = model.loss(batch, preds)
    assert torch.equal(observed, expected)
    assert model.criterion is cached
    model.train()
    with pytest.raises(ValueError, match="provenance"):
        model.loss(batch)
