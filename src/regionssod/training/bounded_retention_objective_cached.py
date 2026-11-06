"""V6 loss with the certified instance-local exact point-geometry cache.

Only the support solver dependency changes. The frozen support and assigner
code objects run with private globals; their original module is never patched.
DFL, posterior sampling, call numbering and the native-E semantics are inherited.
"""
from types import FunctionType

from . import bounded_retention_objective_v6 as v6
from .bounded_retention_point_geometry_cache import optimize_point


def _bind(function, replacements, qualname):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    bound = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = (None if function.__kwdefaults__ is None
                            else dict(function.__kwdefaults__))
    bound.__annotations__ = dict(function.__annotations__)
    bound.__doc__ = function.__doc__
    bound.__module__ = __name__
    bound.__qualname__ = qualname
    return bound


continuous_support = _bind(v6.continuous_support, {"optimize_point": optimize_point},
                           "continuous_support")


class CachedNativeEAssigner(v6.NativeEAssigner):
    get_pos_mask = _bind(v6.NativeEAssigner.get_pos_mask,
                         {"continuous_support": continuous_support},
                         "CachedNativeEAssigner.get_pos_mask")


class CachedNativeELoss(v6.ContinuousNativeELoss):
    def __init__(self, model):
        super().__init__(model)
        self.assigner = CachedNativeEAssigner(self.assigner, self)
