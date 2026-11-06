"""Independent cached V6 model/trainer; original P1 remains a separate trainer.

Keep criterion creation lazy: native Trainer constructs EMA before the first
training loss. Per-observation caches belong to the student's criterion only.
The inherited eval loss uses original P1 without retention metadata. No cache,
criterion, optimizer or EMA is eagerly attached by this initialization shell.
"""
from .bounded_retention_training_v6 import ContinuousNativeEModel, ContinuousNativeETrainer


class CachedNativeEModel(ContinuousNativeEModel):
    def init_criterion(self):
        from .bounded_retention_objective_cached import CachedNativeELoss
        return CachedNativeELoss(self)


class CachedNativeETrainer(ContinuousNativeETrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(CachedNativeEModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        model.bounded_retention_mode = self.mode
        if weights:
            model.load(weights)
        return model
