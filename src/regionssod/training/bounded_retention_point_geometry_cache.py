"""Instance-local exact geometry memoization for the frozen v2 point solver.

The original code object executes with a private globals dictionary. Only its
contract_bounds dependency changes; no process-wide monkeypatch, new search
heuristic, score cache, altered node budget, or cross-observation reuse.
"""
from functools import lru_cache
from types import FunctionType

from . import bounded_retention_point_search_v2 as reference
from .bounded_retention_domain_v6 import NativeEDomain

CACHE_SIZE = 1024
_ATTRIBUTE = '_exact_point_geometry_cache_v1'


def _signature(domain):
    # All snapshot fields consumed by contract_bounds/classify_D/_ranges and
    # transport, plus original enclosing bounds. Mutable cache state is excluded.
    return (type(domain), domain._key, domain.box, domain.canvas, domain.mask,
            domain.active_sides, domain.dimension, tuple(sorted(domain._mapping.items())),
            domain.affine_bounds, domain.clip_bounds, domain.bounds, domain.d_bounds,
            domain._d_intervals, domain._empty_D, domain._empty_C)


class _GeometryCache:
    def __init__(self, domain):
        self.domain = domain
        self.signature = _signature(domain)

        @lru_cache(maxsize=CACHE_SIZE)
        def lookup(cell):
            return reference.contract_bounds(domain, cell)

        self.lookup = lookup

        def contract(observation, bounds):
            if observation is not domain:
                raise ValueError('geometry cache belongs to another observation')
            return lookup(reference._bounds(bounds))

        namespace = dict(reference.optimize_point.__globals__)
        namespace['contract_bounds'] = contract
        self.solve = FunctionType(reference.optimize_point.__code__, namespace,
                                  reference.optimize_point.__name__,
                                  reference.optimize_point.__defaults__,
                                  reference.optimize_point.__closure__)
        self.solve.__kwdefaults__ = dict(reference.optimize_point.__kwdefaults__)

    def verify(self):
        if _signature(self.domain) != self.signature:
            raise ValueError('observation geometry mutated after cache creation')


def optimize_point(domain, score, *, tolerance=1e-5, max_cells=20000):
    """Same v2 traversal and complete result; memoize only exact contraction.

    Each domain owns a bounded cache, released with that observation. Domain
    snapshots must remain immutable, as required by their original point cache.
    Detected mutation fails explicitly; it never returns a stale cached result.
    """
    cache = getattr(domain, _ATTRIBUTE, None)
    if cache is not None:
        cache.verify()
    # A subclass may add unrecorded dependencies to its geometric methods.
    # Keep its original solver behavior instead of guessing a valid cache key.
    if type(domain) is not NativeEDomain or domain.dimension == 1:
        return reference.optimize_point(domain, score, tolerance=tolerance, max_cells=max_cells)
    if any(name in vars(domain) for name in ('classify_D', '_ranges', '_transport', '_cell')):
        raise ValueError('dynamic geometry methods are unsupported by this cache')
    if cache is None:
        cache = _GeometryCache(domain)
        setattr(domain, _ATTRIBUTE, cache)
    return cache.solve(domain, score, tolerance=tolerance, max_cells=max_cells)


def cache_info(domain):
    cache = getattr(domain, _ATTRIBUTE, None)
    if cache is None:
        return dict(hits=0, misses=0, maxsize=CACHE_SIZE, currsize=0)
    return dict(cache.lookup.cache_info()._asdict())


def clear_geometry_cache(domain):
    cache = getattr(domain, _ATTRIBUTE, None)
    if cache is not None:
        cache.lookup.cache_clear()
        delattr(domain, _ATTRIBUTE)
