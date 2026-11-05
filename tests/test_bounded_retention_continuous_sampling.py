"""V5 synthetic posterior tests: no real labels, detector, or trainer."""
from fractions import Fraction as Q
from decimal import Decimal, localcontext
import math
import random
import time

import numpy as np
import pytest
import torch

from regionssod.training.bounded_retention_continuous_numerics import DFLScore
from regionssod.training.bounded_retention_continuous_sampling import (
    SamplingCertificationError, derive_sampling_seed, sample_posterior, _partition, _cdf, _select,
)


class Box:
    def __init__(self, dim=2):
        self.bounds = tuple((Q(0), Q(1)) for _ in range(dim))

    def classify(self, cell):
        assert all(0 <= lo <= hi <= 1 for lo, hi in cell)
        return "inside"

    def point(self, point):
        raise AssertionError("midpoint membership is forbidden")


class Simplex(Box):
    def classify(self, cell):
        if sum(hi for lo, hi in cell) <= 1:
            return "inside"
        if sum(lo for lo, hi in cell) > 1:
            return "outside"
        return "mixed"


def make_score(dim=2, slopes=None, shift=0., device="cpu"):
    z = torch.zeros(dim, 16, dtype=torch.float64, device=device)
    if slopes is not None:
        for j, slope in enumerate(slopes):
            z[j] = torch.arange(16, dtype=torch.float64, device=device) * slope
    return DFLScore((z + shift).requires_grad_(), (0,)*dim, (1,)*dim, 1)


def check(result):
    diag = result.diagnostics
    assert diag["qualified"] and diag["arithmetic_certified"]
    assert diag["accepted"] == 256 and 256 <= diag["proposals"] <= 64*256
    assert diag["acceptance_lower_bound"] >= 1/8
    assert diag["q_midpoint_error_bound"] <= 1e-10 and diag["q_mean_error_bound"] <= 1e-10
    assert diag["maxbits"] <= 256 and diag["full_cover"] and not diag["midpoint_membership"]
    assert "PCG64" in diag["rng_assumption"]
    assert result.q_mean.dtype == torch.float64 and not result.q_mean.requires_grad
    assert torch.allclose(result.q_mean.sum(1), torch.ones(result.q_mean.shape[0], dtype=torch.float64), atol=1e-15, rtol=0)
    assert torch.all((result.q_mean >= 0) & (result.q_mean <= 1))
    assert math.isfinite(result.log_mass_lower) and result.log_mass_lower <= result.log_mass_upper


def test_uniform_rectangle_raw_mass_fixed_k_and_global_rng_isolation():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    score = make_score(2, shift=3.)
    result = sample_posterior(Box(), score, seed=0)
    check(result)
    assert result.log_mass_lower <= 6 <= result.log_mass_upper
    assert result.diagnostics["proposals"] == 256 and result.diagnostics["cells"] == 1
    assert torch.max(torch.abs(result.q_mean[:, 1] - .5)) < .12
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1]) and after[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    again = sample_posterior(Box(), score, seed=0)
    assert torch.equal(again.q_mean, result.q_mean)
    assert again.diagnostics == result.diagnostics


def test_nonflat_inside_cells_refine_envelope_without_changing_target():
    result = sample_posterior(Box(), make_score(2, (12., -9.)), seed=11)
    check(result)
    exact = math.log(math.expm1(12)/12) + math.log(-math.expm1(-9)/9)
    assert result.log_mass_lower <= exact <= result.log_mass_upper
    assert result.diagnostics["cells"] > 1 and result.diagnostics["mixed_cells"] == 0
    expected = torch.tensor([1/(-math.expm1(-12))-1/12, 1/(-math.expm1(9))+1/9], dtype=torch.float64)
    assert torch.max(torch.abs(result.q_mean[:, 1] - expected)) < .1


def test_nonzero_score_mean_and_c_equals_d_gradient_over_fixed_seed_schedule():
    score = make_score(2, (1.2, -.8))
    expected = torch.tensor([1/(-math.expm1(-s))-1/s for s in (1.2, -.8)], dtype=torch.float64)
    results = [sample_posterior(Box(), score, seed=i) for i in range(8)]
    for result in results:
        check(result)
    mean = torch.stack([r.q_mean[:, 1] for r in results]).mean(0)
    # Predeclared simultaneous Hoeffding check for 2048 samples, 2 components:
    # P(max error>.05) <= 4 exp(-2*2048*.05**2) < 1.5e-4.
    assert torch.max(torch.abs(mean - expected)) < .05
    assert all(r.diagnostics["variance_bound_per_component"] == 1/1024 for r in results)


@pytest.mark.parametrize("dim", [2, 3])
def test_joint_simplex_not_independent_marginal_allowlists(dim):
    result = sample_posterior(Simplex(dim), make_score(dim), seed=31+dim)
    check(result)
    exact_mass = 1/math.factorial(dim)
    assert result.log_mass_lower <= math.log(exact_mass) <= result.log_mass_upper
    assert result.diagnostics["mixed_cells"] > 0
    assert torch.max(torch.abs(result.q_mean[:, 1] - 1/(dim+1))) < .12
    assert float(result.q_mean[:, 1].sum()) < 1


@pytest.mark.parametrize("dim", [2, 3])
def test_nonzero_score_coupled_domain_posterior_and_surrogate_gradient(dim):
    from scipy.integrate import quad
    slope = 1.2
    # Independent radial simplex integral: volume density s**(m-1)/(m-1)!.
    radial = quad(lambda s: s**(dim-1)*math.exp(slope*s), 0, 1, epsabs=1e-12)[0]
    moment = quad(lambda s: s**dim*math.exp(slope*s), 0, 1, epsabs=1e-12)[0]
    expected_c = moment/radial/dim
    exact_log = math.log(radial/math.factorial(dim-1))
    score = make_score(dim, (slope,)*dim)
    results = [sample_posterior(Simplex(dim), score, seed=i) for i in range(8)]
    for r in results:
        check(r)
        assert r.log_mass_lower <= exact_log <= r.log_mass_upper
    qmean = torch.stack([r.q_mean for r in results]).mean(0)
    # Fixed 8 seeds, no best-seed selection. Joint Hoeffding failure probability
    # <= 2*m*exp(-2*2048*.05**2), at most .000215 for m=3.
    assert torch.max(torch.abs(qmean[:, 1] - expected_c)) < .05
    delta = score.logits[:, 1] - score.logits[:, 0]
    log_d = (score.logits[:, 0] + torch.log(torch.expm1(delta)/delta)).sum()
    gradient, = torch.autograd.grad(log_d - (qmean*score.logits).sum(), score.logits)
    expected_d = 1/(-math.expm1(-slope)) - 1/slope
    assert torch.max(torch.abs(gradient[:, 1] - (expected_d-expected_c))) < .05
    assert torch.max(torch.abs(gradient[:, 0] + (expected_d-expected_c))) < .05
    assert torch.count_nonzero(gradient[:, 2:]) == 0


def test_seed_key_is_stable_and_distinguishes_occurrence_anchor_and_call():
    args = dict(source_object_id=17, occurrence_id=3, anchor=(Q(1, 3), 2), stride=8, call_index=0)
    seed = derive_sampling_seed(**args)
    assert seed == derive_sampling_seed(**args)
    for key, value in (("occurrence_id", 4), ("anchor", (Q(1, 3), 3)), ("call_index", 1)):
        assert seed != derive_sampling_seed(**dict(args, **{key: value}))


def test_partition_and_proposal_caps_are_explicit_failures():
    with pytest.raises(SamplingCertificationError) as failure:
        sample_posterior(Simplex(2), make_score(), seed=0, max_cells=1)
    assert failure.value.diagnostics["status"] == "PARTITION_RESOURCE_LIMIT"
    assert not failure.value.diagnostics["qualified"]
    with pytest.raises(SamplingCertificationError) as failure:
        sample_posterior(Box(), make_score(), seed=0, max_proposals=255)
    assert failure.value.diagnostics["status"] == "PROPOSAL_RESOURCE_LIMIT"
    assert failure.value.diagnostics["accepted"] == 255
    assert not failure.value.diagnostics["qualified"]


def test_midpoint_cannot_rescue_undecided_membership(monkeypatch):
    class Opaque(Box):
        def classify(self, cell):
            return "inside" if cell[0][1] <= Q(1, 2) else "mixed"
    class Words:
        def random_raw(self):
            return 3*(1 << 62)  # Select the mixed right half; never resolve it.
    monkeypatch.setattr(np.random, "PCG64", lambda seed: Words())
    with pytest.raises(SamplingCertificationError) as failure:
        sample_posterior(Opaque(), make_score(), seed=0)
    diag = failure.value.diagnostics
    assert diag["status"] == "BITS_RESOURCE_LIMIT" and diag["maxbits"] == 256
    assert diag["accepted"] == 0 and not diag["qualified"]


def test_categorical_prefix_refines_instead_of_rounding_cdf():
    class Words:
        def __init__(self, words):
            self.words = iter(words)
        def random_raw(self):
            return next(self.words)
    diag = {"maxbits": 0}
    cdf = [(Q(0), Q(0)), (Q(1, 3), Q(1, 3)), (Q(1), Q(1))]
    assert _select(cdf, Words([(1 << 64)//3, 0]), 128, diag) == 0
    assert diag["maxbits"] == 128
    with pytest.raises(SamplingCertificationError) as failure:
        _select([(Q(0), Q(0)), (Q(2, 5), Q(3, 5)), (Q(1), Q(1))],
                Words([1 << 63, 0, 0, 0]), 256, {"maxbits": 0})
    assert failure.value.diagnostics["status"] == "BITS_RESOURCE_LIMIT"


def test_exponential_acceptance_refines_retained_prefix(monkeypatch):
    with localcontext() as ctx:
        ctx.prec = 100
        near_threshold = int(Decimal("-0.5").exp() * (1 << 64))
    class Words:
        def __init__(self):
            self.i = 0
        def random_raw(self):
            words = (0, 1 << 63, 1 << 63, near_threshold, 0, 0, 0)
            out = words[self.i % len(words)]
            self.i += 1
            return out
    monkeypatch.setattr(np.random, "PCG64", lambda seed: Words())
    result = sample_posterior(Box(), make_score(2, (1., 0.)), seed=0)
    check(result)
    assert result.diagnostics["maxbits"] == 128
    assert result.diagnostics["proposals"] == 256


def test_thin_disjoint_component_is_retained_and_can_be_certifiably_sampled(monkeypatch):
    class Thin(Box):
        def classify(self, cell):
            lo, hi = cell[0]
            if hi <= Q(1, 4) or lo >= 1-Q(1, 2**80):
                return "inside"
            if lo > Q(1, 4) and hi < 1-Q(1, 2**80):
                return "outside"
            return "mixed"
    domain, score = Thin(), make_score()
    leaves, total, _ = _partition(domain, score, 20000)
    cdf = _cdf(leaves, total)
    index = next(j for j, leaf in enumerate(leaves) if leaf.bounds[0][1] == 1)
    selector = int(((cdf[index][1]+cdf[index+1][0])/2) * (1 << 64))
    class Words:
        def __init__(self):
            self.i = 0
        def random_raw(self):
            words = (selector, (1 << 64)-1, 1 << 63, 1 << 63,
                     (1 << 64)-1, 1 << 63, 1 << 63)
            out = words[self.i % len(words)]
            self.i += 1
            return out
    monkeypatch.setattr(np.random, "PCG64", lambda seed: Words())
    result = sample_posterior(domain, score, seed=0)
    check(result)
    assert result.diagnostics["maxbits"] == 128
    assert result.q_mean[0, 1] == 1
    # Tiny positive q0 is retained even though q1 rounds to binary64 one.
    assert 0 < result.q_mean[0, 0] < 2**-80


def test_zero_volume_and_nonfrozen_sample_count_rejected():
    d = Box(); d.bounds = ((Q(0), Q(0)), (Q(0), Q(1)))
    with pytest.raises(SamplingCertificationError) as failure:
        sample_posterior(d, make_score(), seed=0)
    assert failure.value.diagnostics["status"] == "ZERO_VOLUME"
    with pytest.raises(ValueError, match="256"):
        sample_posterior(Box(), make_score(), seed=0, samples=32)
    with pytest.raises(ValueError, match="1U"):
        sample_posterior(Box(1), make_score(1), seed=0)


def test_finite_logits_with_unrepresentable_raw_log_bounds_fail_explicitly():
    score = make_score(3, shift=1e308)
    with pytest.raises(SamplingCertificationError) as failure:
        sample_posterior(Box(3), score, seed=0)
    assert failure.value.diagnostics["status"] == "ARITHMETIC_RESOURCE_LIMIT"
    assert not failure.value.diagnostics["qualified"]


@pytest.mark.parametrize("retention", [Q(35, 100), Q.from_float(.35)-Q.from_float(1e-12)],
                         ids=["ideal_tau", "public_effective_tau"])
def test_complete_physical_curved_3u_capacity_case(retention):
    from regionssod.training.bounded_retention_observation import build_ideal_observation
    from regionssod.training.bounded_retention_continuous_domain import ContinuousDomain
    physical = build_ideal_observation([Q(1, 2), Q(9, 10), Q(1), Q(1, 5)], retention=retention)
    assert physical.effective_retention == retention
    record = {"physical": physical, "original_unknown": physical.original_unknown,
        "affine_bounds": tuple((Q(x), Q(x)) for x in (64, 64, 64, 64)),
        "augmentation_clip_window_bounds": tuple((Q(x), Q(x)) for x in (0, 0, 256, 256))}
    domain = ContinuousDomain(record, [64., 115.2, 128., 128.], (256, 256))
    score = DFLScore(torch.zeros(3, 16, dtype=torch.float64), (96, 96, 120), (-1, 1, 1), 8)
    start = time.perf_counter()
    result = sample_posterior(domain, score, seed=0)
    check(result)
    assert result.diagnostics["retention_tau"] == str(retention)
    print({"ideal_source_geometry": True, "tau_fraction": str(retention), "tau_float": float(retention),
        "full_physical_3U": {k: result.diagnostics[k] for k in
        ("qualified", "cells", "cover_cells", "accepted", "proposals", "maxbits", "acceptance_lower_bound", "q_mean_error_bound")},
        "seconds": time.perf_counter()-start,
        "log_mass_bounds": (result.log_mass_lower, result.log_mass_upper)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_same_private_stream_and_detached_output():
    cpu = sample_posterior(Box(), make_score(), seed=8)
    gpu = sample_posterior(Box(), make_score(device="cuda"), seed=8)
    assert gpu.q_mean.device.type == "cuda" and not gpu.q_mean.requires_grad
    assert torch.equal(cpu.q_mean, gpu.q_mean.cpu())
    assert cpu.diagnostics == gpu.diagnostics
