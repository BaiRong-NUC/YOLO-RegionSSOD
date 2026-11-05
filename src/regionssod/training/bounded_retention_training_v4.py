"""Independent continuous-density model; audited Dataset/Trainer MRO retained.

This module does not launch or authorize training. The caller must supply the
bound source factory and pass the separate source/numerics/integration gates.
"""
from .bounded_retention_training import BoundedRetentionModel, BoundedRetentionTrainer


class ContinuousDensityModel(BoundedRetentionModel):
    def init_criterion(self):
        from .bounded_retention_objective_v4 import ContinuousDensityLoss
        return ContinuousDensityLoss(self)


class ContinuousDensityTrainer(BoundedRetentionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(ContinuousDensityModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        model.bounded_retention_mode = self.mode
        if weights:
            model.load(weights)
        return model
