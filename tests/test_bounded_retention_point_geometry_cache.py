"""Cache equivalence includes ties, errors, isolation and lifecycle."""
from fractions import Fraction as Q
import gc
import weakref

import pytest
import torch

from regionssod.training import bounded_retention_point_geometry_cache as cached
from regionssod.training import bounded_retention_point_search_v2 as reference
from regionssod.training.bounded_retention_continuous_numerics import DFLScore, NumericalCertificationError
from regionssod.training.bounded_retention_domain_v6 import NativeEDomain
from regionssod.training.bounded_retention_observation import build_ideal_observation
from test_bounded_retention_point_search import fixture
from test_bounded_retention_continuous_domain import record


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('flip,uncertain', [(False, False), (True, True)])
@pytest.mark.parametrize('shape', ['flat', 'ramp'])
def test_complete_results_match_uncached_search(dimension, flip, uncertain, shape):
    a, b = fixture(dimension, flip, uncertain), fixture(dimension, flip, uncertain)
    logits = torch.zeros(dimension, 16, dtype=torch.float64) if shape == 'flat' else torch.arange(
        16, dtype=torch.float64).repeat(dimension, 1)
    s = DFLScore(logits, (25,)*dimension, (-1, 1, 1)[:dimension], 1)
    expected = reference.optimize_point(a, s)
    before = dict(reference.optimize_point.__globals__)
    for _ in range(2):
        actual = cached.optimize_point(b, s)
        assert actual == expected
        assert b.point(actual.point) is not None
    assert reference.optimize_point.__globals__ == before


def test_reuse_different_scores_preserves_full_result_and_original_code():
    d = fixture(3)
    for i in (0, 1, 2):
        logits = torch.arange(16, dtype=torch.float64).repeat(3, 1).roll(i, dims=1)
        s = DFLScore(logits, (25, 25, 25), (-1, 1, 1), 1)
        assert cached.optimize_point(d, s) == reference.optimize_point(fixture(3), s)
    assert cached.cache_info(d)['hits'] > 0
    assert getattr(d, cached._ATTRIBUTE).solve.__code__ is reference.optimize_point.__code__


def test_single_unknown_is_unchanged_without_geometry_cache():
    p = build_ideal_observation([Q(3, 4), Q(1, 2), Q(1, 2), Q(1, 5)])
    d = NativeEDomain(record(p), [32, 24, 33, 26], (64, 64))
    s = DFLScore(-torch.arange(16, dtype=torch.float64)[None], (30,), (1,), 1)
    assert cached.optimize_point(d, s) == reference.optimize_point(d, s)
    assert not hasattr(d, cached._ATTRIBUTE)


@pytest.mark.parametrize('empty', [False, True])
def test_error_and_full_diagnostics_match_without_retry(empty):
    d = fixture()
    if empty:
        d = NativeEDomain(record(d.physical), [100, 100, 101, 101], (64, 64))
    s = DFLScore(torch.arange(16, dtype=torch.float64).repeat(2, 1), (28, 28), (1, 1), 1)
    with pytest.raises(NumericalCertificationError) as old:
        reference.optimize_point(d, s, max_cells=1)
    for _ in range(2):
        with pytest.raises(NumericalCertificationError) as new:
            cached.optimize_point(d, s, max_cells=1)
        assert str(new.value) == str(old.value)
        assert new.value.diagnostics == old.value.diagnostics


@pytest.mark.parametrize('field', ['canvas', 'dimension', '_mapping', 'affine_bounds', '_key'])
def test_observation_mutation_is_rejected(field):
    d = fixture()
    s = DFLScore(torch.zeros(2, 16), (28, 28), (1, 1), 1)
    cached.optimize_point(d, s)
    replacements = dict(canvas=(Q(63), Q(64)), dimension=1,
                        _mapping={k: (v+1)%4 for k,v in d._mapping.items()},
                        affine_bounds=((Q(1), Q(2)),)+d.affine_bounds[1:],
                        _key=d._key[:-1]+(Q(1,2),))
    setattr(d, field, replacements[field])
    with pytest.raises(ValueError, match='mutated'):
        cached.optimize_point(d, s)


def test_subclass_with_custom_geometry_keeps_original_solver():
    class Derived(NativeEDomain):
        pass
    original = fixture()
    d = Derived(record(original.physical), original.box, original.canvas)
    s = DFLScore(torch.zeros(d.dimension,16), (28,)*d.dimension, (1,)*d.dimension, 1)
    assert cached.optimize_point(d,s) == reference.optimize_point(d,s)
    assert not hasattr(d, cached._ATTRIBUTE)


def test_dynamic_geometry_override_fails_explicitly():
    d = fixture()
    s = DFLScore(torch.zeros(2,16), (28,28), (1,1), 1)
    cached.optimize_point(d,s)
    d._ranges = lambda cell: None
    with pytest.raises(ValueError, match='dynamic geometry'):
        cached.optimize_point(d,s)


@pytest.mark.parametrize('kwargs', [{'max_cells':0}, {'max_cells':True}, {'tolerance':0}, {'tolerance':-1}])
def test_invalid_solver_options_keep_original_exception(kwargs):
    d = fixture()
    s = DFLScore(torch.zeros(2,16), (28,28), (1,1), 1)
    with pytest.raises(ValueError) as old:
        reference.optimize_point(d,s,**kwargs)
    with pytest.raises(ValueError) as new:
        cached.optimize_point(d,s,**kwargs)
    assert str(new.value) == str(old.value)


def test_domains_do_not_share_cache_and_are_collectable():
    a, b = fixture(), fixture()
    s = DFLScore(torch.arange(16, dtype=torch.float64).repeat(2, 1), (28, 28), (1, 1), 1)
    cached.optimize_point(a, s)
    assert cached.cache_info(b)['currsize'] == 0
    cached.optimize_point(b, s)
    assert getattr(a, cached._ATTRIBUTE) is not getattr(b, cached._ATTRIBUTE)
    cached.clear_geometry_cache(b)
    assert cached.cache_info(b)['currsize'] == 0
    ref = weakref.ref(a)
    del a
    gc.collect()
    assert ref() is None


def test_cache_is_bounded_and_keeps_exact_keys(monkeypatch):
    d = fixture()
    calls = []
    def contract(domain, cell):
        calls.append(cell)
        return cell
    monkeypatch.setattr(reference, 'contract_bounds', contract)
    cache = cached._GeometryCache(d)
    for i in range(cached.CACHE_SIZE + 1):
        cell = ((Q(i, 10000), Q(i+1, 10000)), (Q(0), Q(1)))
        assert cache.lookup(cell) == cell
    assert cache.lookup.cache_info().currsize == cached.CACHE_SIZE
    assert len(calls) == cached.CACHE_SIZE + 1
    assert calls[0] != calls[1]
    assert cache.lookup(calls[0]) == calls[0]
    assert cache.lookup.cache_info().misses == cached.CACHE_SIZE + 2
