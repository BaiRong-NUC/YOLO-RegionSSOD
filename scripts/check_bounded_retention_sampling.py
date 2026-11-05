"""V5 source/real Trainer/full augmentation/AMP audit; zero optimizer steps."""
from pathlib import Path
import torch
import check_bounded_retention_continuous as engineering
from regionssod.training.bounded_retention_training_v5 import ContinuousSamplingTrainer
from regionssod.training.bounded_retention_objective_v5 import ContinuousSamplingLoss, partial_dfl


def audit_actual_logits(criterion, preds, report):
    audits = []
    for (image, _), support in criterion.supports.items():
        index = support["indices"][0]
        z = preds["boxes"][image, :, index].detach().double().reshape(4, 16).requires_grad_()
        # Auditing is a separate draw/calculation, never the recorded training
        # draw. The remaining geometry/certificates are read-only here.
        audit_support = {**support, "density_diagnostics": {}}
        value = partial_dfl(z, audit_support, 0, "candidate", rng_seed=0, audit=True)
        gradient = torch.autograd.grad(value, z)[0]
        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            raise ValueError("independent audit of actual head values failed")
        audits.append({"identity": support["identity"], "dimension": support["domain"].dimension,
                       "diagnostics": audit_support["density_diagnostics"][0]})
    if not audits:
        raise ValueError("actual first batch has no partial observation for numerical audit")
    report["actual_head_numeric_audits"] = audits


if __name__ == "__main__":
    raise SystemExit(engineering.main(Trainer=ContinuousSamplingTrainer, Loss=ContinuousSamplingLoss,
        partial_loss=partial_dfl, default_name="posterior_integration_01",
        objective="V5_same_continuous_density_posterior_gradient",
        extra_sources=[Path(__file__).resolve()], numeric_audit=audit_actual_logits))
