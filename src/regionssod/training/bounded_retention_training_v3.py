"""Native-event model selection; original lexical Dataset/Trainer MRO retained."""
from .bounded_retention_training import BoundedRetentionModel, BoundedRetentionTrainer
from .bounded_retention_objective_v3 import P1EncodingLoss


class P1EncodingModel(BoundedRetentionModel):
    def init_criterion(self):
        return P1EncodingLoss(self)


class P1EncodingTrainer(BoundedRetentionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(P1EncodingModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        model.bounded_retention_mode = self.mode
        if weights:
            model.load(weights)
        return model
