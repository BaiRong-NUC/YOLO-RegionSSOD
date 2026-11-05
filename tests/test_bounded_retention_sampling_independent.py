"""Independent V5 consumer and curved-moment review; pure tensors/mocks only.

No detector, source labels, datasets, checkpoints, optimizer or validation.
Curved reference geometry is derived explicitly, never obtained by querying
domain membership. SciPy quadrature is a numerical cross-check, not an outward
certificate. The original capacity case uses IDEAL EXACT tau=35/100; the public
binary64 effective threshold is a separate parameter case, not its replay.
"""
from contextlib import nullcontext
from fractions import Fraction as Q
from types import SimpleNamespace
import importlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import pytest
from scipy.integrate import quad_vec
import torch
from ultralytics.utils.loss import DFLoss

from regionssod.training import bounded_retention_objective_v4 as v4
from regionssod.training import bounded_retention_objective_v5 as v5
from regionssod.training import bounded_retention_continuous_normalizer as normalizers
from regionssod.training import bounded_retention_continuous_sampling as sampling
from regionssod.training.bounded_retention import UnsupportedObservation
from regionssod.training.bounded_retention_continuous_domain import ContinuousDomain
from regionssod.training.bounded_retention_continuous_numerics import DFLScore
from regionssod.training.bounded_retention_observation import build_ideal_observation
from regionssod.training.p1 import P1DetectionLoss


IDEAL_TAU = Q(35, 100)
PUBLIC_EFFECTIVE_TAU = Q.from_float(.35)-Q.from_float(1e-12)


def physical_three_u(tau):
    physical = build_ideal_observation([Q(1, 2), Q(9, 10), Q(1), Q(1, 5)], retention=tau)
    affine = np.array([64., 64., 64., 64.])
    record = {"physical": physical, "original_unknown": physical.unknown,
              "affine": affine, "affine_bounds": np.repeat(affine[:, None], 2, axis=1),
              "augmentation_clip_window_bounds": np.repeat(np.array([0., 0., 256., 256.])[:, None], 2, axis=1)}
    domain = ContinuousDomain(record, [64., 115.2, 128., 128.], (256, 256))
    assert domain.active_sides == (0, 2, 3) and domain._key[-1] == tau
    return domain


def reference_q(x, anchor, sign, stride=8.):
    """Independent native two-bin basis; no production DFL/score helpers."""
    distance = min(14.99, max(0., sign*(x-anchor)/stride))
    k = math.floor(distance)
    weights = np.zeros(16)
    weights[k], weights[k+1] = 1-(distance-k), distance-k
    return weights


def integrate_q_times_linear(lo, hi, intercept, slope, anchor, sign):
    """Exact polynomial antiderivative per DFL piece (evaluated in float64).

    Both q and the marginal weight are affine in the normalized coordinate t;
    integrate their quadratic product using coefficients 1,1/2,1/3. This is
    not a fixed-node numerical integration of a nonpolynomial inner function.
    """
    total = np.zeros(16)
    if lo >= hi:
        return total
    cuts = sorted({lo, hi, *(anchor+sign*8*k for k in [*range(15), 14.99]
                             if lo < anchor+sign*8*k < hi)})
    for left, right in zip(cuts[:-1], cuts[1:]):
        q0 = reference_q(left, anchor, sign)
        dq = reference_q(right, anchor, sign)-q0
        w0, dw = intercept+slope*left, slope*(right-left)
        total += (right-left)*(q0*w0+(q0*dw+dq*w0)/2+dq*dw/3)
    return total


def independent_three_u_reference(tau):
    """49-vector: volume and L/R/B moment numerators, integrated only over h.

    L in [max(0,128-A/h),64], R in [128,L+A/h], B=576/5+h.
    The only external parameter is tau. No domain/point/classify is called.
    """
    area = float(Q(4096, 5)/tau)
    h0, transition, h2 = 64/5, area/128, area/64
    def cross_section(h):
        width = area/h
        lmin = max(0., 128-width)
        cross = .5*(64-lmin)*(64+lmin+2*width-256)
        left = integrate_q_times_linear(lmin, 64., width-128, 1., 96., -1)
        rmax, turn = 64+width, width+lmin
        right = integrate_q_times_linear(128., min(turn, rmax), 64-lmin, 0., 96., 1)
        right += integrate_q_times_linear(max(128., turn), rmax, 64+width, -1., 96., 1)
        bottom = reference_q(576/5+h, 120., 1)*cross
        return np.concatenate(([cross], left, right, bottom))
    points = {transition}
    for k in [*range(15), 14.99]:
        left_knot, right_knot = 96-8*k, 96+8*k
        if 128-left_knot > 0:
            points.add(area/(128-left_knot))
        if right_knot-64 > 0:
            points.add(area/(right_knot-64))
        if right_knot > 0:
            points.add(area/right_knot)
        points.add(120+8*k-576/5)
    integral, numerical_error = quad_vec(cross_section, h0, h2, epsabs=1e-7, epsrel=1e-11,
                                         points=sorted(p for p in points if h0 < p < h2), norm="max")
    # Independent elementary volume antiderivatives on the two h ranges.
    f = lambda h: 64*area*math.log(h)-6144*h
    g = lambda h: (-area*area/h-128*area*math.log(h)+4096*h)/2
    volume = f(transition)-f(h0)+g(h2)-g(transition)
    assert abs(integral[0]-volume) < 1e-6 and numerical_error < 1e-5
    means = integral[1:].reshape(3, 16)/integral[0]
    np.testing.assert_allclose(means.sum(1), np.ones(3), rtol=0, atol=1e-10)
    assert means.min() > -1e-12
    return volume, means, numerical_error


@pytest.mark.parametrize("tau", [pytest.param(IDEAL_TAU, id="ideal_exact_tau_35_100"),
                                pytest.param(PUBLIC_EFFECTIVE_TAU, id="public_effective_tau_shape_mass")])
def test_physical_three_u_curve_against_independent_volume_and_moments(tau):
    domain = physical_three_u(tau)
    volume, means, error = independent_three_u_reference(domain._key[-1])
    if tau == IDEAL_TAU:
        assert volume == pytest.approx(28240.812573, abs=1e-5)
    score = DFLScore(torch.zeros(3, 16, dtype=torch.float64), (96, 96, 120), (-1, 1, 1), 8)
    # Frozen seeds, no selection/retry of favorable runs. Public-effective tau
    # is a separate shape/mass check; only the ideal case uses the 2048-sample
    # simultaneous 48-component moment threshold requested by the protocol.
    seeds = range(8) if tau == IDEAL_TAU else range(1)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    results = [sampling.sample_posterior(domain, score, seed=seed, samples=256) for seed in seeds]
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)
    assert np.random.get_state()[0] == numpy_state[0] and np.random.get_state()[2:] == numpy_state[2:]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    for result in results:
        assert result.diagnostics["qualified"] and result.diagnostics["accepted"] == 256
        assert result.log_mass_lower <= math.log(volume) <= result.log_mass_upper
        assert result.q_mean.shape == (3, 16)
    maximum_error = float(np.abs(torch.stack([r.q_mean for r in results]).mean(0).numpy()-means).max())
    if tau == IDEAL_TAU:
        assert 96*math.exp(-2*2048*.07**2) < 2e-7
        assert maximum_error <= .07, "fixed 8x256 posterior means violate the predeclared simultaneous threshold"
    print({"tau": str(tau), "independent_volume": volume, "quadrature_error_estimate": error,
           "seeds": list(seeds), "accepted_total": 256*len(results), "max_component_error": maximum_error,
           "proposals": [r.diagnostics["proposals"] for r in results]})


def consumer_support(dimensions=2, device="cpu"):
    mask = torch.tensor([False, False, True, dimensions == 2], device=device)
    return {"mask": mask, "target": torch.tensor([[-2., -4., 10., 12.]], device=device),
        "anchors": torch.tensor([[0., 0.]], device=device), "strides": torch.tensor([[.5]], device=device),
        "nominal_box": torch.tensor([1., 2., 5., 6.], device=device), "canvas": (8., 8.),
        "domain": SimpleNamespace(dimension=dimensions, intervals=[(Q(3), Q(3)+Q(1, 2**80))]),
        "identity": (17, 23), "indices": torch.tensor([0], device=device)}


def install_posterior_mock(monkeypatch, *, qualified=True):
    calls = []
    def fake(domain, score, *, seed, samples):
        calls.append((seed, samples))
        # Deliberately differentiable: consumer MUST detach this estimator.
        q = score.logits.softmax(-1)
        return SimpleNamespace(q_mean=q, log_mass_lower=0., log_mass_upper=1.,
            diagnostics={"qualified": qualified, "seed": seed, "samples": samples,
                "accepted": samples, "proposals": 320, "q_mean_error_bound": 1e-10,
                "variance_of_mean": [[.0001]*16 for _ in range(domain.dimension)]})
    monkeypatch.setattr(sampling, "sample_posterior", fake)
    return calls


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="synthetic CUDA AMP consumer"))])
def test_consumer_surrogate_gradient_detaches_q_and_midpoint_is_only_display(monkeypatch, device):
    install_posterior_mock(monkeypatch)
    dtype = torch.float64 if device == "cpu" else torch.float16
    z = torch.linspace(-1., 1., 64, device=device, dtype=dtype).reshape(4, 16).requires_grad_()
    support = consumer_support(device=device)
    with torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
        result = v5.partial_dfl(z, support, 0, "candidate", rng_seed=123)
        exact = DFLoss(16)(z[~support["mask"]], support["target"][:, ~support["mask"]].clone()).sum()*2
    log_d = normalizers.normalizer(z, support["nominal_box"], support["anchors"][0], .5,
                                  support["mask"], support["canvas"]).log_integral
    q = z[support["mask"]].double().softmax(-1).detach()
    expected_surrogate = (exact+log_d-(q*z[support["mask"]].double()).sum())/4
    actual_gradient = torch.autograd.grad(result, z, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected_surrogate, z, retain_graph=True)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert result.item() == pytest.approx((exact.detach().item()+log_d.detach().item()-.5)/4, abs=1e-12)
    assert abs(result.item()-expected_surrogate.item()) > .01
    diag = support["density_diagnostics"][0]
    assert diag["reported_value"] == "detached_interval_midpoint_not_exact_NLL"
    assert diag["nll_interval"][0] < diag["nll_interval"][1]


@pytest.mark.parametrize("audit", [False, True])
def test_consumer_really_calls_fraction_union_api_and_no_sampler_for_one_u(monkeypatch, audit):
    support = consumer_support(1)
    seen = []
    original = normalizers.integrate_interval_union
    def spy(logits, intervals, anchor, stride, sign, **kwargs):
        seen.append((intervals, anchor, stride, sign, kwargs))
        return original(logits, intervals, anchor, stride, sign, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("1U called the sampler")
    monkeypatch.setattr(normalizers, "integrate_interval_union", spy)
    monkeypatch.setattr(sampling, "sample_posterior", forbidden)
    z = torch.linspace(-1., 1., 64, dtype=torch.float64).reshape(4, 16).requires_grad_()
    result = v5.partial_dfl(z, support, 0, "candidate", audit=audit)
    assert len(seen) == 1 and seen[0][0] is support["domain"].intervals
    assert seen[0][1:4] == (Q(0), Q(1, 2), 1) and seen[0][4] == {"audit": audit}
    assert support["density_diagnostics"][0]["numerator"]["exact_union_length"] == str(Q(1, 2**80))
    assert torch.isfinite(torch.autograd.grad(result, z)[0]).all()


@pytest.mark.parametrize("component", ["D", "C", "posterior"])
def test_consumer_rejects_unqualified_dependency(monkeypatch, component):
    install_posterior_mock(monkeypatch, qualified=component != "posterior")
    if component in {"D", "C"}:
        name = "normalizer" if component == "D" else "integrate_interval_union"
        monkeypatch.setattr(normalizers, name, lambda *a, **k: (torch.tensor(0.), {"qualified": False}))
    z = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True)
    with pytest.raises(UnsupportedObservation, match="not qualified"):
        v5.partial_dfl(z, consumer_support(1 if component == "C" else 2), 0, "candidate")


def synthetic_head_metadata(case="negative", device="cpu"):
    # Metadata namespace only: no torch model/detector is constructed or run.
    metadata = SimpleNamespace(parameters=lambda: iter([torch.zeros(1, device=device)]),
        model=[SimpleNamespace(stride=torch.tensor([8.], device=device), nc=2, reg_max=16)],
        args=SimpleNamespace(box=7.5, cls=.5, dfl=1.5),
        p1_options={"soft_weights": True, "uncertain_ignore": True}, bounded_retention_mode="candidate")
    size = 4 if case == "negative" else 32
    dtype = torch.float16 if device == "cuda" else torch.float32
    raw = torch.linspace(-1., 1., 64, device=device, dtype=dtype).reshape(1, 64, 1).expand(1, 64, size**2).clone()
    if case == "negative":
        distributions = raw.view(1, 4, 16, -1)
        distributions.fill_(-10)
        distributions[:, :, :2] = 0
        distributions[:, 1, 0] = 10
    preds = {"boxes": raw.requires_grad_(),
             "scores": torch.zeros(1, 2, size**2, device=device, dtype=dtype, requires_grad=True),
             "feats": [torch.empty(1, 1, size, size, device=device, dtype=dtype)]}
    box = torch.tensor([8., 28.8, 16., 32.] if case == "negative" else [0., 0., 256., 256.], device=device)
    xywh = torch.cat(((box[:2]+box[2:])/2, box[2:]-box[:2]))/(8*size)
    batch = {"batch_idx": torch.tensor([0.], device=device), "cls": torch.tensor([[0.]], device=device),
        "bboxes": xywh[None], "pseudo_weights": torch.ones(1, device=device),
        "ignore_bboxes": torch.empty(0, 4, device=device), "ignore_batch_idx": torch.empty(0, device=device),
        "object_is_human": torch.tensor([True], device=device), "object_id": torch.tensor([17], device=device),
        "obs": {"occurrence_id": np.array([23])},
        "retention_observations": [{"source_object_id": 17, "occurrence_id": 23}]}
    return metadata, preds, batch


@pytest.mark.parametrize("case", ["negative", "overflow"])
@pytest.mark.parametrize("mode", ["candidate", "control", "reference"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="synthetic native AMP parity"))])
def test_all_known_full_native_parity_and_no_sampling_rng(monkeypatch, case, mode, device):
    model, preds, batch = synthetic_head_metadata(case, device)
    model.bounded_retention_mode = mode
    monkeypatch.setattr(v4, "current_unknown", lambda *a: (np.zeros(4, dtype=bool), None))
    def forbidden(*args, **kwargs):
        raise AssertionError("all-known path touched partial sampling")
    monkeypatch.setattr(sampling, "sample_posterior", forbidden)
    revised, native = v5.ContinuousSamplingLoss(model), P1DetectionLoss(model)
    py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
    with torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
        assignment, result, _ = revised.get_assigned_targets_and_loss(preds, batch)
        baseline = native.get_assigned_targets_and_loss(preds, batch)[1]
    assert assignment[0].any(), "positive native regression must actually be exercised"
    fg, _, target_boxes, anchors, strides = assignment
    distance = torch.cat((anchors-target_boxes[..., :2]/strides,
                          target_boxes[..., 2:]/strides-anchors), dim=-1)[fg]
    assert bool((distance < 0).any()) if case == "negative" else bool((distance > 14.99).any())
    assert torch.equal(result, baseline)
    ga = torch.autograd.grad(result.sum(), (preds["boxes"], preds["scores"]), retain_graph=True)
    gb = torch.autograd.grad(baseline.sum(), (preds["boxes"], preds["scores"]))
    assert all(torch.equal(a, b) for a, b in zip(ga, gb))
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), cpu_state)
    assert np.random.get_state()[0] == np_state[0] and np.random.get_state()[2:] == np_state[2:]
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    if device == "cuda":
        assert torch.equal(torch.cuda.get_rng_state(), cuda_state)
    assert revised.posterior_call_number == 1


@pytest.mark.parametrize("field", ["human", "object", "occurrence"])
def test_identity_mismatch_stops_before_geometry(monkeypatch, field):
    model, preds, batch = synthetic_head_metadata()
    if field == "human":
        batch["object_is_human"][0] = False
    elif field == "object":
        batch["object_id"][0] += 1
    else:
        batch["obs"]["occurrence_id"][0] += 1
    def forbidden(*args, **kwargs):
        raise AssertionError("bad identity reached geometry")
    monkeypatch.setattr(v4, "current_unknown", forbidden)
    with pytest.raises(ValueError, match="human evidence|identity mismatch"):
        v5.ContinuousSamplingLoss(model)(preds, batch)


def mock_parent_call(monkeypatch):
    calls = install_posterior_mock(monkeypatch)
    z = torch.linspace(-1., 1., 64, dtype=torch.float64).reshape(4, 16).requires_grad_()
    support = consumer_support()
    def parent(self, preds, batch):
        self.supports, self.diagnostics = {(0, 0): support}, {}
        return self.partial_dfl(z, support, 0, "candidate")
    monkeypatch.setattr(v4.ContinuousDensityLoss, "get_assigned_targets_and_loss", parent)
    owner = object.__new__(v5.ContinuousSamplingLoss)
    owner.posterior_call_number = 0
    return owner, calls


def test_call_counter_changes_private_seed_and_reconstruction_replays(monkeypatch):
    owner, calls = mock_parent_call(monkeypatch)
    state = torch.get_rng_state()
    owner.get_assigned_targets_and_loss(None, None)
    owner.get_assigned_targets_and_loss(None, None)
    assert [n for _, n in calls] == [256, 256] and calls[0][0] != calls[1][0]
    assert calls[0][0] == v5.private_seed((17, 23), (0., 0.), .5, 1)
    assert owner.posterior_call_number == 2 and torch.equal(torch.get_rng_state(), state)
    replay = object.__new__(v5.ContinuousSamplingLoss)
    replay.posterior_call_number = 0
    replay.get_assigned_targets_and_loss(None, None)
    assert calls[2] == calls[0]


def test_actual_training_diagnostics_retain_interval_proposals_and_variance(monkeypatch):
    """Consumer contract: the exported per-call report cannot lose MC evidence."""
    owner, _ = mock_parent_call(monkeypatch)
    owner.get_assigned_targets_and_loss(None, None)
    serialized = json.dumps(owner.diagnostics)
    for required in ("nll_interval", "proposals", "reported_value"):
        assert required in serialized, f"actual criterion report loses {required}; supports are overwritten next call"
    assert "variance_of_mean" in serialized or "variance_bound_per_component" in serialized


def test_consumer_rejects_nan_mass_enclosure_even_if_sampler_claims_qualified(monkeypatch):
    install_posterior_mock(monkeypatch)
    original = sampling.sample_posterior
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result.log_mass_upper = float("nan")
        return result
    monkeypatch.setattr(sampling, "sample_posterior", corrupt)
    with pytest.raises(UnsupportedObservation):
        v5.partial_dfl(torch.zeros(4, 16, dtype=torch.float64, requires_grad=True), consumer_support(), 0, "candidate")


def test_eval_model_dispatch_uses_native_p1_without_training_metadata(monkeypatch):
    from regionssod.training import bounded_retention_training as base
    from regionssod.training.bounded_retention_training_v5 import ContinuousSamplingModel
    calls = []
    sentinel = object()
    dummy = SimpleNamespace(training=False)
    monkeypatch.setattr(base, "P1DetectionLoss", lambda owner: lambda preds, batch: calls.append((owner, preds, batch)) or sentinel)
    preds, batch = object(), {"img": object()}
    assert ContinuousSamplingModel.loss(dummy, batch, preds) is sentinel
    assert calls == [(dummy, preds, batch)]


def test_trainer_model_options_and_names_are_set_before_loading_mock_weights(monkeypatch):
    from regionssod.training import bounded_retention_training_v5 as training
    order = []
    def constructor(cfg, **kwargs):
        assert cfg == "synthetic-config" and kwargs["nc"] == 2 and kwargs["ch"] == 3
        result = SimpleNamespace()
        def load(weights):
            assert result.p1_options == {"soft_weights": True} and result.bounded_retention_mode == "candidate"
            order.append(("load", weights))
        result.load = load
        return result
    monkeypatch.setattr(training, "ContinuousSamplingModel", constructor)
    def names(model):
        order.append(("names", True))
        return model
    trainer = SimpleNamespace(data={"nc": 2, "channels": 3}, p1_options={"soft_weights": True},
                              mode="candidate", set_model_names_for_load=names)
    training.ContinuousSamplingTrainer.get_model(trainer, "synthetic-config", "mock-weights")
    assert order == [("names", True), ("load", "mock-weights")]


def test_val_dataset_lexical_super_follows_legal_mro_without_real_constructor(monkeypatch):
    from regionssod.training.bounded_retention_training_v5 import ContinuousSamplingTrainer
    from regionssod.training.p1 import P1Trainer
    sentinel = object()
    monkeypatch.setattr(P1Trainer, "build_dataset", lambda self, path, mode, batch: sentinel)
    trainer = object.__new__(ContinuousSamplingTrainer)
    assert trainer.build_dataset("synthetic:val", mode="val", batch=8) is sentinel


def load_engineering(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    return importlib.import_module("check_bounded_retention_continuous")


def test_actual_head_audit_does_not_overwrite_training_sample_diagnostics(monkeypatch):
    load_engineering(monkeypatch)
    checker = importlib.import_module("check_bounded_retention_sampling")
    support = consumer_support()
    original = {"seed": 987, "reported_value": "training-sample"}
    support["density_diagnostics"] = {0: original.copy()}
    def audit_loss(z, local_support, *args, **kwargs):
        local_support.setdefault("density_diagnostics", {})[0] = {"seed": 0, "reported_value": "audit-sample"}
        return z.square().sum()
    monkeypatch.setattr(checker, "partial_dfl", audit_loss)
    criterion = SimpleNamespace(supports={(0, 0): support})
    report = {}
    checker.audit_actual_logits(criterion, {"boxes": torch.zeros(1, 64, 1)}, report)
    assert report["actual_head_numeric_audits"]
    assert support["density_diagnostics"][0] == original, "audit replaced the actual forward's sample evidence"


def mocked_engineering_attempt(monkeypatch, tmp_path, *, train_images=11310, optimizer=False,
        geometry_failure=False, scan_all=False, close_mosaic=0, stream_batches=1, hash_drift=False,
        dimension_schedule=None):
    """Run only script control flow against tensor stubs; no real Trainer/model.

    ROOT/config/factory/fingerprints/snapshots are redirected or replaced before
    main() is invoked. Generated receipts go exclusively to pytest's temp dir.
    """
    checker = load_engineering(monkeypatch)
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    config_file = tmp_path/"configs/candidates/positive_quality_transport_v2/reference30.yaml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"p1": {}, "train": {"close_mosaic": close_mosaic}, "model": "synthetic-checkpoint",
                                      "data": "synthetic-data"}), encoding="utf-8")
    events = {"forwards": 0, "batches": 0, "optimizer_steps": 0}
    trace = []
    forwarded_batch_ids = []
    monkeypatch.setattr(checker, "fingerprint", lambda path: (
        "changed-synthetic-sha" if hash_drift and events["forwards"] else "synthetic-sha-no-file-read"))
    monkeypatch.setattr(checker, "snapshot_inputs", lambda *args: None)
    factory = SimpleNamespace(counts={"objects": 105057}, reads={}, premises={"synthetic": True})
    monkeypatch.setattr(checker, "BoundSourceFactory", lambda: factory)
    monkeypatch.setattr(sys, "argv", ["synthetic-review", "--name", "mock_attempt"]+
                        (["--scan-all-batches"] if scan_all else []))
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *a, **k: SimpleNamespace(get_scale=lambda: 1., scale=lambda x: x))
    def preflight(*args):
        phase = args[2].get("geometry_phase")
        assert events["forwards"] == 0, "geometry scan occurred after a detector forward"
        trace.append(("geometry", phase, args[2].get("geometry_batch_index")))
        if geometry_failure is True or geometry_failure == phase:
            raise UnsupportedObservation("synthetic geometry STOP")
        args[2]["geometry_preflight"] = {"synthetic": True,
            "active_dimensions": dict(args[0]["_mock_active_dimensions"])}
    monkeypatch.setattr(checker, "preflight", preflight)

    class Population:
        def __init__(self, count):
            self.count = count
            self.mosaic = True
            self.human_labels = SimpleNamespace(counts={"raw_rows": 105058, "retained_rows": 105057,
                "native_duplicate_rows": 1}, duplicate_aliases=[])
        def __len__(self):
            return self.count
        def close_mosaic(self, hyp):
            assert not self.mosaic and hyp.close_mosaic == close_mosaic
            trace.append(("dataset_close_mosaic",))

    class Loader:
        def __init__(self, count):
            self.dataset = Population(count)
        def __iter__(self):
            phase = "mosaic_open" if self.dataset.mosaic else "mosaic_closed"
            for index in range(stream_batches):
                events["batches"] += 1
                dimensions = {} if dimension_schedule is None else dimension_schedule[phase][index]
                yield {"img": torch.tensor([float(events["batches"])]), "cls": torch.zeros(1, 1),
                       "_mock_active_dimensions": dimensions}
        def reset(self):
            assert not self.dataset.mosaic
            trace.append(("loader_reset",))

    class TensorProducer:
        """A single scalar graph; this is not an nn.Module or detection model."""
        def __init__(self):
            self.p = torch.tensor([1.], requires_grad=True)
        def to(self, device):
            return self
        def train(self):
            return self
        def named_parameters(self):
            return [("synthetic_scalar", self.p)]
        def parameters(self):
            return [self.p]
        def __call__(self, image):
            trace.append(("forward",))
            forwarded_batch_ids.append(int(image.item()))
            events["forwards"] += 1
            return {"boxes": self.p*torch.ones(1, 64, 1)}

    def forbidden_step():
        events["optimizer_steps"] += 1
        raise AssertionError("optimizer step forbidden")
    trainer = SimpleNamespace(model=TensorProducer(), device="cpu", data={"train": "synthetic-train", "val": "synthetic-val"},
        args=SimpleNamespace(close_mosaic=close_mosaic),
        setup_model=lambda: None, set_model_attributes=lambda: None, preprocess_batch=lambda batch: batch,
        get_dataloader=lambda path, **kw: Loader(train_images if kw["mode"] == "train" else 548))
    # Exercise the installed native transition method on a namespace; still
    # no Trainer construction, actual data loader, detector or source access.
    from ultralytics.engine.trainer import BaseTrainer
    trainer._close_dataloader_mosaic = lambda: BaseTrainer._close_dataloader_mosaic(trainer)
    if optimizer:
        trainer.optimizer = SimpleNamespace(step=forbidden_step)

    class TensorLoss:
        def __init__(self, model):
            self.mode, self.diagnostics = "candidate", {"synthetic": True}
        def __call__(self, predictions, batch):
            s = predictions["boxes"].sum()
            factor = {"candidate": 1., "control": 2., "reference": 3.}[self.mode]
            result = torch.stack((s, s, factor*s))
            return result, result.detach()

    code = checker.main(Trainer=lambda **kw: trainer, Loss=TensorLoss,
                        objective="SYNTHETIC_MOCK_CONSUMER_REVIEW_NO_DETECTOR")
    report = json.loads((tmp_path/"outputs/bounded_retention_continuous_20261105/mock_attempt/result.json").read_text(encoding="utf-8"))
    report["_synthetic_trace"] = trace
    report["_synthetic_forwarded_batch_ids"] = forwarded_batch_ids
    return code, report, events


def test_engineering_rejects_wrong_train_population_before_batch_forward(monkeypatch, tmp_path):
    code, report, events = mocked_engineering_attempt(monkeypatch, tmp_path, train_images=7)
    assert code == 2 and report["status"] == "STOP_BEFORE_STUDENT_TRAINING", "train11310 contract is currently only reported, not gated"
    assert events["forwards"] == 0 and events["batches"] == 0


def test_engineering_zero_optimizer_component_receipt_with_only_tensor_stubs(monkeypatch, tmp_path):
    code, report, events = mocked_engineering_attempt(monkeypatch, tmp_path)
    assert code == 0 and report["status"] == "INTEGRATION_COMPONENT_PASSED_NOT_STUDENT_PERMIT"
    assert report["parameter_hash_unchanged"] and report["optimizer_steps"] == 0
    assert not report["training_started"] and report["validation_inference_calls"] == 0
    assert report["dataloaders"]["train_images"] == 11310
    assert events == {"forwards": 1, "batches": 1, "optimizer_steps": 0}


@pytest.mark.parametrize("cause", ["optimizer", "geometry"])
def test_engineering_stop_paths_cannot_step_or_override_failure(monkeypatch, tmp_path, cause):
    code, report, events = mocked_engineering_attempt(monkeypatch, tmp_path,
        optimizer=cause == "optimizer", geometry_failure=cause == "geometry")
    assert code == 2 and report["status"] == "STOP_BEFORE_STUDENT_TRAINING"
    assert not report["training_started"] and events["optimizer_steps"] == 0
    if cause == "geometry":
        assert events["forwards"] == report["model_forward_attempts"] == 0
    else:
        assert "optimizer" in report["error"]["message"]


def test_engineering_native_close_reset_two_complete_streams_before_only_forward(monkeypatch, tmp_path):
    code, report, events = mocked_engineering_attempt(monkeypatch, tmp_path,
        scan_all=True, close_mosaic=10, stream_batches=1414)
    assert code == 0 and report["all_stream_geometry_checked"]
    checks = report["geometry_batch_checks"]
    assert [x["phase"] for x in checks] == ["mosaic_open"]*1414+["mosaic_closed"]*1414
    assert events == {"forwards": 1, "batches": 2828, "optimizer_steps": 0}
    trace = report["_synthetic_trace"]
    assert trace[1413:1417] == [("geometry", "mosaic_open", 1413),
        ("dataset_close_mosaic",), ("loader_reset",), ("geometry", "mosaic_closed", 0)]
    assert trace[-2:] == [("geometry", "mosaic_closed", 1413), ("forward",)]
    started = json.loads((tmp_path/"outputs/bounded_retention_continuous_20261105/mock_attempt/started.json").read_text(encoding="utf-8"))
    assert started["status"] == "RUNNING" and started["model_forward_calls"] == 0
    assert started["pid"] == report["pid"] > 0 and started["command"] == report["command"]
    assert started["objective"] == "SYNTHETIC_MOCK_CONSUMER_REVIEW_NO_DETECTOR"


def test_engineering_closed_mosaic_failure_stops_before_any_forward(monkeypatch, tmp_path):
    code, report, events = mocked_engineering_attempt(monkeypatch, tmp_path,
        scan_all=True, close_mosaic=10, stream_batches=1414, geometry_failure="mosaic_closed")
    assert code == 2 and report["status"] == "STOP_BEFORE_STUDENT_TRAINING"
    assert report["geometry_phase"] == "mosaic_closed" and report["geometry_batch_index"] == 0
    assert len(report["geometry_batch_checks"]) == 1414
    assert events == {"forwards": 0, "batches": 1415, "optimizer_steps": 0}
    assert not report.get("all_stream_geometry_checked", False)


def test_engineering_terminal_hash_drift_cannot_receive_pass(monkeypatch, tmp_path):
    code, report, events = mocked_engineering_attempt(monkeypatch, tmp_path, hash_drift=True)
    assert code == 2 and report["status"] == "STOP_BEFORE_STUDENT_TRAINING"
    assert "input changed during integration audit" in report["error"]["message"]
    assert events["optimizer_steps"] == 0 and not report["training_started"]


def test_engineering_first_maximum_dimension_across_phases_selects_actual_batch(monkeypatch, tmp_path):
    # Public synthetic metadata only. Object counts deliberately disagree with
    # dimension ordering, and the later 3U tie must not replace its first batch.
    schedule = {
        "mosaic_open": [{"1": 99}, {"1": 5, "2": 1}, {}, {"2": 40}],
        "mosaic_closed": [{"2": 30}, {"1": 100, "3": 1}, {"3": 9}, {"1": 500}],
    }
    code, report, events = mocked_engineering_attempt(monkeypatch, tmp_path,
        scan_all=True, close_mosaic=10, stream_batches=4, dimension_schedule=schedule)
    assert code == 0 and report["all_stream_geometry_checked"]
    assert report["forward_batch_selection"] == {
        "rule": "first_maximum_active_dimension", "phase": "mosaic_closed",
        "batch_index": 1, "active_dimension": 3,
    }
    assert report["_synthetic_forwarded_batch_ids"] == [6], "selection metadata must match the actual forwarded tensor"
    assert events == {"forwards": 1, "batches": 8, "optimizer_steps": 0}
    assert report["_synthetic_trace"][-2:] == [("geometry", "mosaic_closed", 3), ("forward",)]


@pytest.mark.parametrize("d,lo,hi,error", [
    (1., 0., 0., 2.**-54),  # Both nearest-rounded endpoints are inward.
    (196608., 196608., 196608., 1e-7),  # Cancellation after raw-score shifts.
    (-65536., -65537., -65536., 1e-7),
    (1., .3, .7, 1e-7),
    (0., -1., 1., 1e-7),  # Probability constraint tightens the lower end to 0.
    (2.**-1022, 0., 0., 2.**-1074),
    (1., 1., 1., 0.),
])
def test_nll_enclosure_outward_against_exact_fraction_arithmetic(d, lo, hi, error):
    a, b = v5.nll_enclosure(d, lo, hi, error)
    exact_lower, exact_upper = max(Q(0), Q(d)-Q(error)-Q(hi)), Q(d)+Q(error)-Q(lo)
    assert 0 <= Q(a) <= exact_lower <= exact_upper <= Q(b)
    # Tight directed rounding: not a gratuitously widened interval that merely
    # passes containment. Compare the neighboring floats in exact arithmetic.
    assert Q(math.nextafter(a, math.inf)) > exact_lower
    assert Q(math.nextafter(b, -math.inf)) < exact_upper


@pytest.mark.parametrize("values", [
    (float("nan"), 0., 1., 1e-7), (1., float("nan"), 1., 1e-7),
    (1., 0., float("nan"), 1e-7), (1., 0., 1., float("nan")),
    (1., -math.inf, 1., 1e-7), (1., 0., math.inf, 1e-7),
    (1., 2., 1., 1e-7), (1., 0., 1., -1e-7), (0., 1., 1., 0.),
])
def test_nll_enclosure_rejects_bad_mass_and_budget_inputs(values):
    with pytest.raises(UnsupportedObservation):
        v5.nll_enclosure(*values)


def test_partial_consumer_dispatches_to_outward_nll_enclosure(monkeypatch):
    install_posterior_mock(monkeypatch)
    seen = []
    def spy(*args):
        seen.append(args)
        return .125, .75
    monkeypatch.setattr(v5, "nll_enclosure", spy)
    support = consumer_support()
    z = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True)
    result = v5.partial_dfl(z, support, 0, "candidate")
    assert len(seen) == 1 and seen[0][1:] == (0., 1., 1e-7)
    assert support["density_diagnostics"][0]["nll_interval"] == [.125, .75]
    exact = DFLoss(16)(z[:2], support["target"][:, :2].clone()).sum()*2
    assert result.item() == pytest.approx((exact.item()+(.125+.75)/2)/4, abs=1e-14)


@pytest.mark.parametrize("audit", [False, True])
def test_one_u_full_canvas_event_cancels_analytically_without_full_domain_flag(monkeypatch, audit):
    support = consumer_support(1)
    # D right coordinate is (fixed left=1, canvas right=8); endpoint inclusion
    # has no effect on this Lebesgue integral. C is explicitly this same union.
    support["domain"].intervals = [(Q(1), Q(8))]
    def forbidden(*args, **kwargs):
        raise AssertionError("analytic full-domain event called sampling")
    monkeypatch.setattr(sampling, "sample_posterior", forbidden)
    z = torch.linspace(-1., 1., 64, dtype=torch.float64).reshape(4, 16).requires_grad_()
    result = v5.partial_dfl(z, support, 0, "candidate", audit=audit)
    exact = DFLoss(16)(z[~support["mask"]], support["target"][:, ~support["mask"]].clone()).sum()*3/4
    assert abs((result-exact).item()) <= 1e-14
    residual = torch.autograd.grad(result-exact, z)[0]
    torch.testing.assert_close(residual, torch.zeros_like(z), rtol=0, atol=1e-14)
    assert support["density_diagnostics"][0]["method"] == "analytic_1u"


def test_multi_u_full_canvas_event_has_zero_mean_gradient_not_forced_zero_samples():
    class CanvasRectangle:
        """Explicit geometry, up to zero-measure faces; no production full-D flag."""
        dimension = 2
        bounds = ((Q(1), Q(8)), (Q(2), Q(8)))
        def classify(self, cell):
            assert all(a <= lo <= hi <= b for (a, b), (lo, hi) in zip(self.bounds, cell))
            return "inside"
    support = consumer_support()
    support["domain"] = CanvasRectangle()
    z = torch.stack([torch.arange(16, dtype=torch.float64)*s for s in (.1, -.1, .2, -.15)]).requires_grad_()
    exact = DFLoss(16)(z[:2], support["target"][:, :2].clone()).sum()*2/4
    residuals, display = [], []
    for seed in range(8):
        value = v5.partial_dfl(z, support, 0, "candidate", rng_seed=seed)
        residuals.append(torch.autograd.grad(value-exact, z, retain_graph=True)[0])
        display.append(float((value-exact).detach()))
        assert support["density_diagnostics"][0]["method"] == "posterior_gradient"
    mean = torch.stack(residuals).mean(0)
    assert torch.equal(mean[:2], torch.zeros_like(mean[:2]))
    assert float(mean[2:].abs().max()) <= .07/4
    assert any(bool(r[2:].abs().max() > 0) for r in residuals), "must not pretend finite-K noise vanishes"
    assert max(map(abs, display)) <= 1e-7


def test_nonzero_logits_physical_two_u_curved_posterior_independent_quadrature():
    """Numerical cross-check of a real synthetic curved C, not a certificate.

    Fixed E left/top=576/5; widths w,h >=64/5 and w*h <= (1/25)/tau*4096.
    Reference integration uses these equations and independent q only; no
    membership/classify/source oracle supplies reference values or moments.
    """
    physical = build_ideal_observation([Q(9, 10), Q(9, 10), Q(1, 5), Q(1, 5)], retention=IDEAL_TAU)
    affine = np.array([64., 64., 64., 64.])
    record = {"physical": physical, "original_unknown": physical.unknown, "affine": affine,
        "affine_bounds": np.repeat(affine[:, None], 2, axis=1),
        "augmentation_clip_window_bounds": np.repeat(np.array([0., 0., 256., 256.])[:, None], 2, axis=1)}
    domain = ContinuousDomain(record, [115.2, 115.2, 128., 128.], (256, 256))
    assert domain.active_sides == (2, 3) and domain._key[-1] == IDEAL_TAU
    area, minimum, origin = float(Q(4096, 25)/IDEAL_TAU), 64/5, 576/5
    k = np.arange(16)
    raw = np.stack((.4*np.sin(.8*k), .5*np.cos(.7*k)))
    def section(height):
        upper = origin+area/height
        def inner(right):
            qr = reference_q(right, 120., 1)
            return np.concatenate(([1.], qr))*math.exp(qr@raw[0])
        inner_value, inner_error = quad_vec(inner, 128., upper, epsabs=1e-10, epsrel=1e-11,
            points=[120+8*x for x in range(15) if 128 < 120+8*x < upper], norm="max")
        assert inner_error < 1e-7
        qb = reference_q(origin+height, 120., 1)
        return np.concatenate((inner_value, qb*inner_value[0]))*math.exp(qb@raw[1])
    upper_height = area/minimum
    points = {120+8*x-origin for x in range(15)}
    points.update(area/(120+8*x-origin) for x in range(15) if 120+8*x > origin)
    integral, error = quad_vec(section, minimum, upper_height, epsabs=1e-8, epsrel=1e-11,
        points=sorted(x for x in points if minimum < x < upper_height), norm="max")
    assert error < 1e-6
    means = integral[1:].reshape(2, 16)/integral[0]
    np.testing.assert_allclose(means.sum(1), np.ones(2), rtol=0, atol=1e-11)
    score = DFLScore(torch.tensor(raw, dtype=torch.float64), (120, 120), (1, 1), 8)
    results = [sampling.sample_posterior(domain, score, seed=seed, samples=256) for seed in range(8)]
    for result in results:
        assert result.diagnostics["qualified"] and result.diagnostics["accepted"] == 256
        assert result.log_mass_lower <= math.log(integral[0]) <= result.log_mass_upper
    max_error = float(np.abs(torch.stack([r.q_mean for r in results]).mean(0).numpy()-means).max())
    assert 64*math.exp(-2*2048*.07**2) < 2e-7
    assert max_error <= .07
    print({"case": "nonzero_logits_curved_2U", "tau": str(IDEAL_TAU), "raw_mass": integral[0],
           "quadrature_error_estimate": error, "max_component_error": max_error, "accepted": 2048})
