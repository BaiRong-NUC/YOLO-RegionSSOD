"""Read-only callback checks: synthetic tensors/stubs, no detector/data/training."""
from copy import deepcopy
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from regionssod.training.bounded_retention_run_diagnostics import (
    BoundedRetentionRunDiagnostics, DiagnosticsError, METRICS, NLL_SCOPE,
    build_callbacks, make_callbacks,
)


def support(dimension=1, indices=(1,), identity=(10, 20), stride=1., nominal=(-1., 0., 9., 10.), point_scale=2., device="cpu"):
    n = len(indices)
    box = torch.tensor(nominal, dtype=torch.float64, device=device)
    first = box.expand(n, 4).clone()
    second = first.clone()
    second[:, 2] = second[:, 0]+point_scale*(box[2]-box[0])
    return {"mask": torch.tensor([j < dimension for j in range(4)], device=device),
        "identity": identity, "indices": torch.tensor(indices, device=device),
        "nominal_box": box, "strides": torch.full((n, 1), stride, device=device),
        "anchors": torch.zeros(n, 2, device=device),
        "geometry": torch.stack((first, second), dim=1),
        "chosen": torch.ones(n, dtype=torch.long, device=device),
        "valid": torch.ones(n, 2, dtype=torch.bool, device=device)}


def posterior(dimension=2, row=0, lower=.2, upper=.9):
    return {row: {"method": "posterior_gradient", "qualified": True, "samples": 256,
        "accepted": 256, "proposals": 400, "cells": 17, "seed": 345,
        "nll_interval": [lower, upper], "reported_value": "detached_interval_midpoint_not_exact_NLL",
        "variance_of_mean": [[.0001]*16 for _ in range(dimension)],
        "variance_bound_per_component": 1/1024,
        "variance_semantics": "empirical variance/K; universal bound assumes ideal independent posterior samples",
        "gradient_semantics": "posterior first-gradient estimator, not deterministic integration or Hessian",
        "q_approximation_bias_bound": 1e-7, "acceptance_lower_bound": .125,
        "q_mean_error_bound": 1e-12}}


def trainer(tmp_path, supports=None, *, mode="candidate", device="cpu"):
    if supports is None:
        supports = {(0, 0): support(device=device)}
    d = {"supported_objects": len(supports), "effective_anchors": 0,
        "negative_known_distances": 2, "overflow_known_distances": 3, "empty_anchor_events": 0,
        "posterior_call_number": 1, "posterior_density_anchors": 0,
        "posterior_event_details": [], "posterior_proposals": 0, "posterior_partition_cells": 0,
        "posterior_samples_per_anchor": 256, "posterior_variance_bound_per_component": 1/1024,
        "posterior_nll_interval_scope": NLL_SCOPE}
    for s in supports.values():
        for row, event in s.get("density_diagnostics", {}).items():
            if event["method"] == "posterior_gradient":
                d["posterior_density_anchors"] += 1
                d["posterior_proposals"] += event["proposals"]
                d["posterior_partition_cells"] += event["cells"]
                d["posterior_event_details"].append({"identity": s["identity"], "anchor_row": row,
                    "nll_interval": event["nll_interval"], "reported_value": event["reported_value"]})
    criterion = SimpleNamespace(mode=mode, supports=supports, diagnostics=d,
        detached_dist=torch.zeros(2, 8, 64, device=device, requires_grad=True))
    return SimpleNamespace(model=SimpleNamespace(criterion=criterion), epoch=0, rank=-1, save_dir=tmp_path)


def run_epoch(t, callbacks):
    for event in ("on_train_epoch_start", "on_train_batch_end", "on_train_epoch_end"):
        callbacks[event](t)


def load_report(directory, epoch=1, rank=-1):
    return json.loads((Path(directory)/f"epoch_{epoch:04d}.rank_{rank}.json").read_text("utf-8"))


def test_all_native_pools_equal_object_means_and_three_dimensions(tmp_path):
    supports = {(0, 0): support(1, (1,), stride=1., point_scale=2.),
        (0, 1): support(1, (2, 4, 6), stride=2., point_scale=4.),
        (1, 0): support(2, (7, 2), identity=(11, 21)),
        (1, 1): support(3, (1, 3), identity=(12, 22))}
    t = trainer(tmp_path, supports)
    # Pools are intentionally noncontiguous; non-candidate row must not leak in.
    with torch.no_grad():
        t.model.criterion.detached_dist[:, 0] = torch.nan
    run_epoch(t, make_callbacks(tmp_path, "candidate"))
    r = load_report(tmp_path)
    one = r["groups_by_active_u"]["1"]
    assert one["object_occurrences"] == 2
    assert one["native_candidate_object_pairs"] == 4
    assert one["mean_over_objects"][METRICS[0]] == pytest.approx(math.log(16))
    # Each uniform side has mean bin7.5: native area225*stride**2.
    assert one["mean_over_objects"][METRICS[1]] == pytest.approx((2.25+9.)/2)
    assert one["mean_over_objects"][METRICS[1]] != pytest.approx((2.25+3*9.)/4)
    assert one["mean_over_objects"][METRICS[2]] == 3.
    for dim in (2, 3):
        assert r["groups_by_active_u"][str(dim)]["object_occurrences"] == 1
        assert r["groups_by_active_u"][str(dim)]["native_candidate_object_pairs"] == 2
    assert r["criterion_counter_totals"]["effective_anchors"] == 0
    assert r["criterion_counter_totals"]["negative_known_distances"] == 2
    assert r["criterion_counter_totals"]["overflow_known_distances"] == 3
    assert len(r["batches"][0]["objects"]) == 4  # no filtering for zero foreground
    assert "not FG" in r["population"]
    assert r["diagnostic_only"] and not r["participates_in_loss_tuning_or_stopping"]


def test_selected_geometry_and_active_edge_entropy_not_all_edges(tmp_path):
    s = support(2, (1, 3, 5))
    s["mask"] = torch.tensor([True, False, True, False])
    s["chosen"] = torch.tensor([0, 1, 1])
    t = trainer(tmp_path, {(1, 2): s})
    with torch.no_grad():
        logits = t.model.criterion.detached_dist.view(2, 8, 4, 16)
        logits[1, :, 1::2, :] = -1000
        logits[1, :, 1::2, 0] = 0
    run_epoch(t, make_callbacks(tmp_path, "candidate"))
    obj = load_report(tmp_path)["batches"][0]["objects"][0]
    assert obj[METRICS[0]] == pytest.approx(math.log(16))
    assert obj[METRICS[1]] == 0.  # Legitimate decoded zero height, not a dropped object.
    assert obj[METRICS[2]] == pytest.approx(5/3)
    assert obj["image_index"] == 1 and obj["gt_row"] == 2


def test_batch_accumulation_reset_exclusive_output_and_rank_local(tmp_path):
    t = trainer(tmp_path)
    t.rank = 2
    c = make_callbacks(tmp_path, "candidate")
    c["on_train_epoch_start"](t)
    c["on_train_batch_end"](t)
    t.model.criterion.diagnostics["posterior_call_number"] += 1
    c["on_train_batch_end"](t)
    c["on_train_epoch_end"](t)
    r = load_report(tmp_path, rank=2)
    assert r["batch_count"] == 2
    assert r["groups_by_active_u"]["1"]["object_occurrences"] == 2
    assert r["criterion_counter_totals"]["negative_known_distances"] == 4
    first = (tmp_path/"epoch_0001.rank_2.json").read_bytes()
    t.epoch = 1
    run_epoch(t, c)
    assert load_report(tmp_path, epoch=2, rank=2)["batch_count"] == 1
    t.epoch = 0
    with pytest.raises(FileExistsError):
        run_epoch(t, c)
    assert (tmp_path/"epoch_0001.rank_2.json").read_bytes() == first


@pytest.mark.parametrize("explicit", [True, False])
def test_genuine_original_p1_reference_has_no_fabricated_zero_signal(tmp_path, explicit):
    t = SimpleNamespace(model=SimpleNamespace(criterion=SimpleNamespace()), epoch=0, rank=-1, save_dir=tmp_path)
    callbacks = make_callbacks(tmp_path, "reference") if explicit else build_callbacks(tmp_path)
    run_epoch(t, callbacks)
    r = load_report(tmp_path)
    assert r["availability"] == "not_available"
    for key in ("groups_by_active_u", "criterion_counter_totals", "posterior_summary"):
        assert r[key] is None
    assert r["batches"][0]["objects"] is None


@pytest.mark.parametrize("mode", ["candidate", "control"])
def test_empty_supports_are_available_without_reading_stale_logits(tmp_path, mode):
    t = trainer(tmp_path, {}, mode=mode)
    del t.model.criterion.detached_dist
    run_epoch(t, make_callbacks(tmp_path, mode))
    r = load_report(tmp_path)
    assert r["availability"] == "available"
    assert r["batches"][0]["objects"] == []
    assert r["posterior_summary"]["event_count"] == 0
    assert r["posterior_summary"]["variance_bound_max"] is None
    for g in r["groups_by_active_u"].values():
        assert g["object_occurrences"] == 0
        assert all(v is None for v in g["mean_over_objects"].values())


def test_every_posterior_event_endpoints_semantics_variance_and_counts(tmp_path):
    a, b = support(2, (1, 4)), support(3, (2,), identity=(30, 40))
    a["density_diagnostics"] = posterior(2, row=1)
    b["density_diagnostics"] = posterior(3, lower=.01, upper=1.5)
    t = trainer(tmp_path, {(0, 0): a, (0, 1): b})
    t.model.criterion.diagnostics["effective_anchors"] = 2
    run_epoch(t, make_callbacks(tmp_path, "candidate"))
    r = load_report(tmp_path)
    p = r["posterior_summary"]
    assert p["K_histogram"] == {"256": 2}
    assert p["proposals_total"] == 800 and p["partition_cells_total"] == 34
    assert p["accepted_total"] == 512
    assert p["nll_interval_extrema"] == [.01, 1.5]
    assert p["variance_bound_max"] == 1/1024
    assert p["empirical_variance_of_mean_max"] == .0001
    assert p["nll_scope"] == NLL_SCOPE
    assert p["nll_extrema_are_not_an_epoch_loss_interval"]
    assert p["reported_value_semantics"] == ["detached_interval_midpoint_not_exact_NLL"]
    event = r["batches"][0]["objects"][0]["posterior_events"][0]
    assert event["anchor_row"] == 1 and event["nll_interval"] == [.2, .9]
    assert event["variance_of_mean"] == [[.0001]*16]*2
    assert "ideal independent" in event["variance_semantics"]
    assert "not parameter-gradient variance" in event["variance_scope"]
    assert "not deterministic integration" in event["gradient_semantics"]
    assert event["sampling_arithmetic"]["q_approximation_bias_bound"] == 1e-7
    assert r["batches"][0]["criterion_diagnostics"]["posterior_event_details"][1]["identity"] == [30, 40]


def test_analytic_events_not_misrepresented_as_posterior(tmp_path):
    s = support()
    s["density_diagnostics"] = {0: {"method": "analytic_1u", "qualified": True}}
    t = trainer(tmp_path, {(0, 0): s})
    run_epoch(t, make_callbacks(tmp_path, "candidate"))
    assert load_report(tmp_path)["posterior_summary"]["event_count"] == 0


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s.pop("identity"), "identity"),
    (lambda s: s.update(indices=torch.tensor([8])), "pool indices"),
    (lambda s: s.update(indices=torch.tensor([1, 1])), "duplicated"),
    (lambda s: s.update(mask=torch.ones(4, dtype=torch.bool)), "1..3"),
    (lambda s: s.update(nominal_box=torch.tensor([0., 0., 0., 2.])), "nominal"),
    (lambda s: s.update(nominal_box=torch.tensor([0., 0., float("nan"), 2.])), "nonfinite"),
    (lambda s: s.update(strides=torch.tensor([[0.]])), "stride"),
    (lambda s: s.update(chosen=torch.tensor([2])), "witness"),
    (lambda s: s.update(valid=torch.zeros(1, 2, dtype=torch.bool)), "not valid"),
    (lambda s: s.update(geometry=torch.zeros(1, 2, 4)), "point area"),
    (lambda s: s.update(density_diagnostics={0: {"method": "unknown"}}), "unrecognized"),
])
def test_malformed_second_object_raises_without_partial_batch_or_silent_skip(tmp_path, mutation, match):
    bad = support()
    mutation(bad)
    t = trainer(tmp_path, {(0, 0): support(), (0, 1): bad})
    observer = BoundedRetentionRunDiagnostics(tmp_path, "candidate")
    observer.on_train_epoch_start(t)
    with pytest.raises(DiagnosticsError, match=match):
        observer.on_train_batch_end(t)
    assert observer._state["batches"] == []
    with pytest.raises(DiagnosticsError, match="earlier diagnostic"):
        observer.on_train_epoch_end(t)
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("mutation", [
    lambda c: delattr(c, "supports"), lambda c: delattr(c, "detached_dist"),
    lambda c: c.diagnostics.update(supported_objects=2),
    lambda c: c.diagnostics.update(posterior_density_anchors=1),
    lambda c: c.diagnostics.update(posterior_proposals=1),
    lambda c: c.diagnostics.update(posterior_partition_cells=1),
    lambda c: c.diagnostics.update(posterior_event_details=[{"identity": [3, 4]}]),
    lambda c: c.diagnostics.update(posterior_nll_interval_scope="whole_loss"),
    lambda c: c.diagnostics.update(effective_anchors=-1),
    lambda c: c.diagnostics.update(extra=float("inf")),
])
def test_missing_or_inconsistent_criterion_signals_fail(tmp_path, mutation):
    t = trainer(tmp_path)
    mutation(t.model.criterion)
    with pytest.raises(DiagnosticsError):
        run_epoch(t, make_callbacks(tmp_path, "candidate"))
    assert not list(tmp_path.glob("*.json"))


def test_wrapper_cycle_wrong_mode_stale_sidecar_and_lifecycle_fail(tmp_path):
    t = trainer(tmp_path)
    c = make_callbacks(tmp_path, "candidate")
    with pytest.raises(DiagnosticsError, match="outside"):
        c["on_train_batch_end"](t)
    c["on_train_epoch_start"](t)
    with pytest.raises(DiagnosticsError, match="not been saved"):
        c["on_train_epoch_start"](t)
    c["on_train_batch_end"](t)
    with pytest.raises(DiagnosticsError, match="stale"):
        c["on_train_batch_end"](t)
    with pytest.raises(DiagnosticsError, match="mode"):
        run_epoch(t, make_callbacks(tmp_path, "control"))
    t.model.module = t.model
    with pytest.raises(DiagnosticsError, match="cyclic"):
        run_epoch(t, make_callbacks(tmp_path, "candidate"))


def test_output_default_and_repo_unignored_rejection(tmp_path):
    t = trainer(tmp_path)
    run_epoch(t, build_callbacks(mode="candidate"))
    assert load_report(tmp_path/"bounded_retention_diagnostics")["batch_count"] == 1
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(DiagnosticsError, match="ignored artifact"):
        make_callbacks(root/"src"/"forbidden_diagnostics", "candidate")["on_train_epoch_start"](t)


def assert_nested_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for k in actual:
            assert_nested_equal(actual[k], expected[k])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_inputs_parameters_gradients_rng_unchanged_no_forward_or_optimizer(tmp_path, device, dtype):
    t = trainer(tmp_path, device=device)
    criterion = t.model.criterion
    criterion.detached_dist = torch.arange(1024, device=device, dtype=torch.float64).reshape(2, 8, 64).sin().to(dtype).requires_grad_()
    parameter = torch.nn.Parameter(torch.tensor([3., 4.], device=device))
    parameter.grad = torch.tensor([5., 6.], device=device)
    original_logits = criterion.detached_dist.detach().clone()
    original_supports, original_diagnostics = deepcopy(criterion.supports), deepcopy(criterion.diagnostics)
    original_parameter, original_grad = parameter.detach().clone(), parameter.grad.clone()
    class Poison:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected oracle/optimizer access: {name}")
        def __call__(self, *args, **kwargs):
            raise AssertionError("unexpected forward")
    criterion.supports[0, 0]["domain"] = Poison()
    class Model:
        def __init__(self):
            self.criterion, self.parameter = criterion, parameter
        def __call__(self, *args, **kwargs):
            raise AssertionError("unexpected model forward")
    t.model = SimpleNamespace(module=SimpleNamespace(_orig_mod=Model()))
    t.optimizer = Poison()
    t.stop = False
    py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else None
    with torch.autocast(device_type=device, enabled=True, dtype=torch.float16 if device == "cuda" else torch.bfloat16):
        run_epoch(t, make_callbacks(tmp_path, "candidate"))
    assert random.getstate() == py_rng
    np_now = np.random.get_state()
    assert np_now[0] == np_rng[0] and np.array_equal(np_now[1], np_rng[1]) and np_now[2:] == np_rng[2:]
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng))
    assert torch.equal(criterion.detached_dist, original_logits)
    assert criterion.detached_dist.grad is None
    for key, value in original_supports.items():
        for field, original in value.items():
            assert_nested_equal(criterion.supports[key][field], original)
    assert_nested_equal(criterion.diagnostics, original_diagnostics)
    assert torch.equal(parameter, original_parameter) and torch.equal(parameter.grad, original_grad)
    assert not t.stop
    # Independently recompute the selected pool's native categorical metrics.
    logits = original_logits[0, 1].cpu().double().reshape(4, 16)
    p, logp = logits.softmax(-1), logits.log_softmax(-1)
    means = p @ torch.arange(16, dtype=torch.float64)
    obj = load_report(tmp_path)["batches"][0]["objects"][0]
    assert obj[METRICS[0]] == pytest.approx(float(-(p[0]*logp[0]).sum()), abs=2e-12)
    assert obj[METRICS[1]] == pytest.approx(float((means[0]+means[2])*(means[1]+means[3])/100), abs=2e-12)


def test_saved_batch_snapshots_do_not_alias_mutable_criterion(tmp_path):
    t = trainer(tmp_path)
    c = make_callbacks(tmp_path, "candidate")
    c["on_train_epoch_start"](t)
    c["on_train_batch_end"](t)
    t.model.criterion.diagnostics["negative_known_distances"] = 1000
    t.model.criterion.supports[0, 0]["nominal_box"].fill_(0)
    c["on_train_epoch_end"](t)
    r = load_report(tmp_path)
    assert r["criterion_counter_totals"]["negative_known_distances"] == 2
    assert r["batches"][0]["objects"][0]["nominal_visible_area"] == 100.


def test_factory_mode_validation(tmp_path):
    with pytest.raises(ValueError):
        make_callbacks(tmp_path, "other")
