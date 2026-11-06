"""Cached production wiring: synthetic heads only, no data/weights/optimizer.

These checks complement (do not replace) the real factory/setup_model/model.loss
and all-parameter backward audit owned by the production-entry integration gate.
"""
from contextlib import nullcontext
from copy import deepcopy
import io
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ultralytics.utils.torch_utils import ModelEMA

from regionssod.training import bounded_retention_objective_cached as cached
from regionssod.training import bounded_retention_objective_v6 as v6
from regionssod.training import bounded_retention_training_cached as training
from regionssod.training.bounded_retention_observation import build_ideal_observation
from regionssod.training.bounded_retention_point_geometry_cache import _ATTRIBUTE as CACHE_ATTRIBUTE, cache_info, optimize_point
from regionssod.training.bounded_retention_training import BoundedRetentionModel, BoundedRetentionTrainer
from regionssod.training.p1 import P1DetectionLoss
from test_bounded_retention_continuous_loss import partial_fixture
from test_bounded_retention_p1_encoding import known_head_fixture


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA AMP requires the installed GPU"))]


def bytes_equal(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.detach().contiguous().reshape(-1).view(torch.uint8),
        b.detach().contiguous().reshape(-1).view(torch.uint8)))


def model_shell(source, mode):
    """Exercise real model.loss dispatch with synthetic head metadata, no forward."""
    model = training.CachedNativeEModel.__new__(training.CachedNativeEModel)
    torch.nn.Module.__init__(model)
    model.register_parameter("dummy", torch.nn.Parameter(source.dummy.detach().clone()))
    model.model = deepcopy(source.model)
    model.args = deepcopy(source.args)
    model.p1_options = dict(source.p1_options)
    model.bounded_retention_mode = mode
    return model


def two_u_fixture():
    source, preds, batch = partial_fixture()
    physical = build_ideal_observation(["9/10", "9/10", "1/5", "1/5"])
    batch["retention_observations"][0].update(physical=physical, original_unknown=physical.unknown)
    batch["bboxes"][0] = batch["bboxes"].new_tensor([.45, .45, .1, .1])
    return source, preds, batch


def plain_batch(batch):
    return {key: value for key, value in batch.items()
            if key in {"batch_idx", "cls", "bboxes"}}


def test_only_two_private_global_dependencies_change():
    pairs = [(cached.continuous_support, v6.continuous_support, "optimize_point", optimize_point),
             (cached.CachedNativeEAssigner.get_pos_mask, v6.NativeEAssigner.get_pos_mask,
              "continuous_support", cached.continuous_support)]
    for actual, original, key, replacement in pairs:
        assert actual.__code__ is original.__code__
        assert actual.__defaults__ == original.__defaults__
        assert actual.__kwdefaults__ == original.__kwdefaults__
        assert actual.__closure__ is original.__closure__
        assert actual.__globals__ is not original.__globals__
        assert actual.__globals__.keys() == original.__globals__.keys()
        assert actual.__globals__[key] is replacement
        assert original.__globals__[key] is not replacement
        assert all(actual.__globals__[name] is value for name, value in original.__globals__.items()
                   if name != key)
    assert cached.CachedNativeELoss.partial_dfl is v6.ContinuousNativeELoss.partial_dfl
    assert (cached.CachedNativeELoss.get_assigned_targets_and_loss
            is v6.ContinuousNativeELoss.get_assigned_targets_and_loss)
    assert training.CachedNativeEModel.loss is BoundedRetentionModel.loss
    assert training.CachedNativeETrainer.__init__ is BoundedRetentionTrainer.__init__
    assert training.CachedNativeETrainer.build_dataset is BoundedRetentionTrainer.build_dataset


@pytest.mark.parametrize("mode", ["candidate", "control"])
@pytest.mark.parametrize("weights", [None, "initial-weights"])
def test_trainer_names_options_mode_precede_weight_loading(monkeypatch, mode, weights):
    events = []
    options = {"soft_weights": True, "uncertain_ignore": True}

    class Model:
        def __init__(self, cfg, nc, ch, verbose):
            assert not verbose
            events.append(("create", cfg, nc, ch))

        def load(self, value):
            assert self.names == {0: "a", 1: "b"}
            assert self.p1_options is options and self.bounded_retention_mode == mode
            assert not hasattr(self, "criterion")
            events.append(("load", value))

    def names(model):
        assert not hasattr(model, "p1_options") and not hasattr(model, "bounded_retention_mode")
        model.names = {0: "a", 1: "b"}
        events.append(("names",))
        return model

    monkeypatch.setattr(training, "CachedNativeEModel", Model)
    trainer = object.__new__(training.CachedNativeETrainer)
    trainer.data = {"nc": 2, "channels": 3}
    trainer.p1_options, trainer.mode = options, mode
    trainer.set_model_names_for_load = names
    model = trainer.get_model("synthetic", weights, verbose=False)
    assert model.p1_options is options and model.bounded_retention_mode == mode
    assert not hasattr(model, "criterion")
    assert events == [("create", "synthetic", 2, 3), ("names",)] + (
        [("load", weights)] if weights else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["candidate", "control"])
@pytest.mark.parametrize("case", ["negative", "overflow"])
def test_model_loss_all_known_exact_p1_values_and_head_gradients(case, mode, device):
    source, preds, batch = known_head_fixture(case, device)
    batch["retention_observations"][0]["physical"] = build_ideal_observation(
        ["1/2", "1/2", "1/5", "1/5"])
    model = model_shell(source, mode)
    # Unlike the real audit's preinstalled hook, exercise native lazy creation:
    # never assign model.criterion or call model.init_criterion() in this test.
    assert not hasattr(model, "criterion")
    with torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
        actual = model.loss(batch, preds)[0]
        expected = P1DetectionLoss(model)(preds, batch)[0]
    assert type(model.criterion) is cached.CachedNativeELoss
    assert type(model.criterion.assigner) is cached.CachedNativeEAssigner
    assert model.criterion.mode == mode
    assert model.criterion.supports == {}
    first_criterion = model.criterion
    assert first_criterion.posterior_call_number == 1
    assert bytes_equal(actual, expected)
    inputs = (preds["boxes"], preds["scores"])
    ga = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)
    gb = torch.autograd.grad(expected.sum(), inputs)
    assert all(bytes_equal(a, b) and torch.isfinite(a).all() for a, b in zip(ga, gb))
    with torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
        repeated = model.loss(batch, preds)[0]
    assert model.criterion is first_criterion
    assert first_criterion.posterior_call_number == 2
    assert bytes_equal(repeated, expected)


@pytest.mark.parametrize("mode", ["candidate", "control"])
def test_validation_without_metadata_uses_p1_without_creating_cached_criterion(mode):
    source, preds, batch = known_head_fixture("negative")
    model = model_shell(source, mode).eval()
    batch = plain_batch(batch)
    actual, expected = model.loss(batch, preds)[0], P1DetectionLoss(model)(preds, batch)[0]
    assert bytes_equal(actual, expected)
    assert getattr(model, "criterion", None) is None
    model.train()
    with pytest.raises(ValueError, match="explicit aligned retention observations"):
        model.loss(batch, preds)


def test_partial_model_loss_shares_cached_geometry_but_not_posterior_gradient():
    source, preds, batch = two_u_fixture()
    original_globals = dict(vars(v6))
    candidate, control = model_shell(source, "candidate"), model_shell(source, "control")
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    a, b = candidate.loss(batch, preds)[0], control.loss(batch, preds)[0]
    ca, cb = candidate.criterion, control.criterion
    assert ca.diagnostics["effective_anchors"] > 0 and ca.diagnostics["posterior_density_anchors"] > 0
    assert bytes_equal(a[:2], b[:2])
    assert ca.supports.keys() == cb.supports.keys()
    for key, support in ca.supports.items():
        other = cb.supports[key]
        for field in ("geometry", "indices", "chosen", "valid"):
            assert bytes_equal(support[field], other[field])
        assert support["point_certificates"] == other["point_certificates"]
        assert support["domain"] is not other["domain"]
        # This flat-score fixture may certify a feasible box maximizer before
        # any contraction. The instance-local solver/cache must still be wired.
        # Actual repeated contraction hits are required by the real 3U audit.
        assert hasattr(support["domain"], CACHE_ATTRIBUTE)
        assert hasattr(other["domain"], CACHE_ATTRIBUTE)
    ga = torch.autograd.grad(a.sum(), preds["boxes"], retain_graph=True)[0]
    gb = torch.autograd.grad(b.sum(), preds["boxes"], retain_graph=True)[0]
    assert torch.isfinite(ga).all() and torch.isfinite(gb).all() and not bytes_equal(ga, gb)
    assert ca.posterior_call_number == cb.posterior_call_number == 1
    candidate.loss(batch, preds)
    assert candidate.criterion is ca and ca.posterior_call_number == 2
    assert control.criterion is cb and cb.posterior_call_number == 1
    assert vars(v6).keys() == original_globals.keys()
    assert all(vars(v6)[key] is value for key, value in original_globals.items())
    assert random.getstate() == state[0]
    assert np.random.get_state()[0] == state[1][0]
    np.testing.assert_array_equal(np.random.get_state()[1], state[1][1])
    assert np.random.get_state()[2:] == state[1][2:]
    assert bytes_equal(torch.get_rng_state(), state[2])


def test_native_ema_before_lazy_loss_stays_cache_free_through_copy_and_serialization():
    source, preds, batch = two_u_fixture()
    student = model_shell(source, "candidate")
    assert getattr(student, "criterion", None) is None
    ema = ModelEMA(student)
    assert type(ema.ema) is training.CachedNativeEModel and not ema.ema.training
    student.loss(batch, preds)
    # Even an empty geometry lookup owns a closure over the student's domain;
    # that closure must not be transplanted into the pre-loss EMA checkpoint.
    assert any(hasattr(s["domain"], CACHE_ATTRIBUTE) for s in student.criterion.supports.values())
    ema.update(student)
    # Use the installed trainer's restricted attribute transfer, not a blanket
    # copy that would transplant the student's training-only criterion/cache.
    ema.update_attr(student, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
    assert getattr(ema.ema, "criterion", None) is None
    validation = plain_batch(batch)
    assert bytes_equal(ema.ema.loss(validation, preds)[0], P1DetectionLoss(ema.ema)(preds, validation)[0])
    assert getattr(ema.ema, "criterion", None) is None
    snapshot = deepcopy(ema.ema).half()
    buffer = io.BytesIO()
    torch.save({"ema": snapshot}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)["ema"]
    assert type(restored) is training.CachedNativeEModel
    assert getattr(restored, "criterion", None) is None
    assert not restored.training
    assert restored.bounded_retention_mode == "candidate"
    assert all(bytes_equal(v, restored.state_dict()[key]) for key, v in snapshot.state_dict().items())
