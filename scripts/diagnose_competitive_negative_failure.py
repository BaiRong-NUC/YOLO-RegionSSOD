"""One real batch, no training loop/backward/optimizer step; never edits frozen runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.torch_utils import autocast

from regionssod.training.competitive_negative import (
    CompetitiveNegativeLoss, CompetitiveNegativeTrainer, _NegativeBCE, negative_multipliers,
)
from regionssod.training.p1 import P1DetectionLoss


ROOT = Path(__file__).resolve().parents[1]
FAILED = ROOT / "outputs/p1_competitive_negative_main30_s0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(logits, targets, teacher, flags, labels, raw):
    """Keep zero-mass and casting loss distinct from class/provenance errors."""
    mass, cls = targets.max(-1)
    raw_mass = raw.max(-1).values
    pseudo = flags.bool()
    top = teacher.argmax(-1)
    positive = mass > 0
    return {
        "foreground": len(logits), "pseudo_foreground": int(pseudo.sum()),
        "raw_dtype": str(raw.dtype), "bce_target_dtype": str(targets.dtype),
        "teacher_vs_assigned_class_mismatch": int((pseudo & (top != labels)).sum()),
        "raw_zero_mass_pseudo": int((pseudo & (raw_mass == 0)).sum()),
        "cast_only_zero_mass_pseudo": int((pseudo & (raw_mass > 0) & ~positive).sum()),
        "bce_zero_mass_pseudo": int((pseudo & ~positive).sum()),
        "positive_mass_wrong_class": int((pseudo & positive & (cls != top)).sum()),
        "multiple_nonzero_classes": int((pseudo & ((targets > 0).sum(-1) > 1)).sum()),
    }


def replay_foreground(rows):
    """Controlled replay: hold logits/provenance fixed, vary only target casting."""
    result = {}
    for mode in ("semantic", "strength"):
        result[mode] = {}
        for target_name in ("targets", "raw"):
            errors = []
            for image, row in enumerate(rows):
                try:
                    negative_multipliers(row["logits"], row[target_name], row["teacher"], row["flags"], mode=mode)
                except ValueError as exc:
                    errors.append({"image": image, "error": str(exc)})
            result[mode][target_name] = errors
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if ROOT / "outputs" not in out.parents or out.exists():
        raise ValueError("Use a NEW directory under ignored outputs/")
    out.mkdir(parents=True)
    watched = [FAILED / n for n in ("status.json", "dispatch.json", "preflight.json", "candidate_stdout.log")]
    watched += list((ROOT / "src/regionssod").rglob("*.py"))
    watched += list((ROOT / "runs/p1_maincmp_improved30_s0").glob("*.yaml"))
    watched += list((ROOT / "data/visdrone_area10").rglob("*.cache"))
    before = {str(p): sha(p) for p in watched}
    report = {"scope": "engineering_one_batch_not_performance", "optimizer_steps": 0,
              "backward_calls": 0, "historical_batch_bitwise_identity": "NOT_ESTABLISHED"}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Training/backward/optimizer step forbidden in diagnostic")

    original_bce = _NegativeBCE.forward
    captured = []

    def capture_bce(self, logits, targets):
        labels, _, scores, foreground, ids = self.capture.assigned
        batch = self.owner.current_batch
        for image in range(len(logits)):
            anchors = foreground[image].nonzero(as_tuple=True)[0]
            local = ids[image, anchors].long()
            objects = batch["batch_idx"].long() == image
            row = {"logits": logits[image, anchors], "targets": targets[image, anchors],
                   "teacher": batch["teacher_class_scores"][objects][local],
                   "flags": batch["is_pseudo"][objects][local],
                   "labels": labels[image, anchors], "raw": scores[image, anchors]}
            report.setdefault("images", []).append(summarize(**row))
            captured.append({k: v.detach().cpu() for k, v in row.items()})
        torch.save(captured, out / "foreground_tensors.pt")
        return original_bce(self, logits, targets)

    config = yaml.safe_load((FAILED / "candidate_config.yaml").read_text(encoding="utf-8"))
    overrides = yaml.safe_load((ROOT / "runs/p1_maincmp_improved30_s0/args.yaml").read_text(encoding="utf-8"))
    overrides.pop("save_dir", None)
    overrides.update(project=str(out), name="setup_only", exist_ok=False, plots=False)
    with patch.object(BaseTrainer, "train", forbidden), patch.object(torch.Tensor, "backward", forbidden), \
            patch.object(torch.optim.SGD, "step", forbidden):
        yolo = YOLO(config["model"])
        trainer = CompetitiveNegativeTrainer(overrides=overrides, _callbacks=yolo.callbacks,
            mode="semantic", p1_options=config["p1"],
            class_score_path=ROOT / config["competitive_negative"]["class_score_path"],
            enabled=True, diagnostics=False)
        trainer.model = trainer.get_model(weights=yolo.model, cfg=yolo.model.yaml)
        trainer._setup_train()
        trainer.epoch = 0
        trainer._model_train()
        parameter_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                                 for p in trainer.model.parameters())).hexdigest()
        batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
        report["image_paths"] = list(batch["im_file"])
        report["amp"] = trainer.amp
        report["train_objects"] = len(batch["cls"])
        # No backward. BatchNorm buffers may update in memory; no model is saved.
        with torch.no_grad(), autocast(trainer.amp, device=trainer.device.type):
            preds = trainer.model(batch["img"])
            criterion = CompetitiveNegativeLoss(trainer.model)
            try:
                with patch.object(_NegativeBCE, "forward", capture_bce):
                    criterion(preds, batch)
                report["original_error"] = None
            except ValueError as exc:
                report["original_error"] = str(exc)
            base = P1DetectionLoss(trainer.model)(preds, batch)
            report["original_p1_loss_finite"] = bool(torch.isfinite(base[0]).all())
        after_parameters = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                                  for p in trainer.model.parameters())).hexdigest()
        report["parameters_unchanged"] = parameter_hash == after_parameters
    after = {str(p): sha(p) for p in watched}
    report["watched_files"] = before
    report["modified_watched_files"] = [p for p in before if before[p] != after[p]]
    report["tensor_sha256"] = sha(out / "foreground_tensors.pt")
    report["controlled_cpu_replay"] = replay_foreground(captured)
    (out / "diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "watched_files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
