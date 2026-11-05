"""V5 numerical strategy with the audited real Trainer initialization chain."""
from .bounded_retention_training import BoundedRetentionModel, BoundedRetentionTrainer


class ContinuousSamplingModel(BoundedRetentionModel):
    def init_criterion(self):
        from .bounded_retention_objective_v5 import ContinuousSamplingLoss
        return ContinuousSamplingLoss(self)


class ContinuousSamplingTrainer(BoundedRetentionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(ContinuousSamplingModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        model.bounded_retention_mode = self.mode
        if weights:
            model.load(weights)
        return model
