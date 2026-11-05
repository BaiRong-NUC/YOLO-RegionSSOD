"""V6 native-E domain strategy with the audited real Trainer initialization chain."""
from .bounded_retention_training import BoundedRetentionModel, BoundedRetentionTrainer


class ContinuousNativeEModel(BoundedRetentionModel):
    def init_criterion(self):
        from .bounded_retention_objective_v6 import ContinuousNativeELoss
        return ContinuousNativeELoss(self)


class ContinuousNativeETrainer(BoundedRetentionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(ContinuousNativeEModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        model.bounded_retention_mode = self.mode
        if weights:
            model.load(weights)
        return model
