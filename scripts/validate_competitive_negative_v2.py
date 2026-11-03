"""Cached boundary replay + one real AMP batch; forbids training/backward/steps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

from diagnose_competitive_negative import ROOT, read, sha, verify
from regionssod.training.competitive_negative import CompetitiveNegativeLoss, negative_multipliers
from regionssod.training.competitive_negative_v2 import (
    CompetitiveNegativeLossV2, CompetitiveNegativeTrainerV2, negative_multipliers_v2,
)
from regionssod.training.p1 import P1DetectionLoss

CONFIG = ROOT / "configs/candidates/competitive_negative_v2/semantic30.yaml"
CACHE = ROOT / "outputs/p1_competitive_negative_debug_20261103_v1/foreground_tensors.pt"
CACHE_SHA = "8a1aa278e75756ac5e5315248235e0c60b33f6ab943d76cf57ba01700fb41bb6"
OUT = ROOT / "outputs/p1_competitive_negative_v2_validation_20261103"


def cached_replay():
    if sha(CACHE) != CACHE_SHA:
        raise ValueError("Diagnostic tensor identity changed")
    rows = torch.load(CACHE, map_location="cpu", weights_only=True)
    summary = {"foreground": 0, "pseudo_foreground": 0, "zero_effective_mass": 0, "modes": {}}
    for row in rows:
        summary["foreground"] += len(row["logits"])
        summary["pseudo_foreground"] += int(row["flags"].sum())
        summary["zero_effective_mass"] += int((row["flags"] & (row["targets"].sum(-1) == 0)).sum())
    for mode in ("semantic", "strength"):
        total = 0
        for row in rows:
            weights, active = negative_multipliers_v2(row["logits"], row["targets"], row["teacher"],
                row["flags"], row["labels"].long(), row["raw"], mode=mode)
            if (weights.gather(1, row["labels"].long()[:, None]) != 1).any():
                raise AssertionError("Assigned class modified")
            zero = row["targets"].sum(-1) == 0
            if (weights[zero] != 1).any() or active[zero].any():
                raise AssertionError("Zero-mass foreground modified")
            nonzero = ~zero
            old, was_active = negative_multipliers(row["logits"][nonzero], row["targets"][nonzero],
                row["teacher"][nonzero], row["flags"][nonzero], mode=mode)
            if not torch.equal(weights[nonzero], old) or not torch.equal(active[nonzero], was_active):
                raise AssertionError("Ordinary target behavior changed")
            total += int(active.sum())
        summary["modes"][mode] = {"accepted": True, "active": total, "normal_matches_v1": True}
    return summary


def parameters_sha(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def validate(out):
    out = out.resolve()
    if ROOT / "outputs" not in out.parents or out.exists():
        raise FileExistsError("Use a new ignored outputs directory")
    out.mkdir(parents=True)
    sources = [CONFIG, Path(__file__).resolve(), CACHE]
    sources += list((ROOT / "src/regionssod").rglob("*.py"))
    before = {str(p): sha(p) for p in sources}
    report = {"cached_replay": cached_replay(), "optimizer_steps": 0, "backward_calls": 0,
        "torch_version": torch.__version__, "scope": "engineering_only_not_student_performance"}
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Training/backward/optimizer step forbidden")
    from ultralytics.utils.torch_utils import autocast
    from ultralytics.utils.tal import TaskAlignedAssigner
    import inspect
    before[inspect.getfile(TaskAlignedAssigner)] = sha(inspect.getfile(TaskAlignedAssigner))
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    overrides = yaml.safe_load((ROOT / "runs/p1_maincmp_improved30_s0/args.yaml").read_text(encoding="utf-8"))
    overrides.pop("save_dir", None)
    overrides.update(project=str(out), name="setup_only", exist_ok=False)
    with patch.object(BaseTrainer, "train", forbidden), patch.object(torch.Tensor, "backward", forbidden), \
            patch.object(torch.optim.SGD, "step", forbidden):
        yolo = YOLO(config["model"])
        trainer = CompetitiveNegativeTrainerV2(overrides=overrides, _callbacks=yolo.callbacks,
            mode="semantic", p1_options=config["p1"],
            class_score_path=ROOT / config["competitive_negative"]["class_score_path"], enabled=True, diagnostics=False)
        trainer.model = trainer.get_model(weights=yolo.model, cfg=yolo.model.yaml)
        trainer._setup_train()
        trainer.epoch = 0
        trainer._model_train()
        before_parameters = parameters_sha(trainer.model)
        batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
        with torch.no_grad(), autocast(trainer.amp, device=trainer.device.type):
            preds = trainer.model(batch["img"])
            original_error = None
            try:
                CompetitiveNegativeLoss(trainer.model)(preds, batch)
            except ValueError as exc:
                original_error = str(exc)
            base = P1DetectionLoss(trainer.model).get_assigned_targets_and_loss(preds, batch)
            criterion = CompetitiveNegativeLossV2(trainer.model)
            criterion.diagnostics = True
            fixed = criterion.get_assigned_targets_and_loss(preds, batch)
        report["real_batch"] = {
            "images": len(batch["img"]), "amp": trainer.amp, "device": str(trainer.device),
            "v1_error": original_error,
            "assignment_identical": all(torch.equal(a, b) for a, b in zip(base[0], fixed[0], strict=True)),
            "box_DFL_identical": torch.equal(base[1][[0, 2]], fixed[1][[0, 2]]),
            "v2_loss_finite": bool(torch.isfinite(fixed[1]).all()),
            "parameters_unchanged": parameters_sha(trainer.model) == before_parameters,
            "diagnostics": {k: v.item() if isinstance(v, torch.Tensor) else v
                            for k, v in criterion.last_diagnostics.items()},
        }
    verify(before)
    report["source_sha256"] = {k: v for k, v in before.items() if k != str(CACHE)}
    report["artifact_sha256"] = {str(CACHE): CACHE_SHA, str(out / "setup_only/args.yaml"): sha(out / "setup_only/args.yaml")}
    r = report["real_batch"]
    d = r["diagnostics"]
    report["accepted"] = (r["assignment_identical"] and r["box_DFL_identical"] and r["v2_loss_finite"]
        and r["parameters_unchanged"] and r["amp"] and r["device"].startswith("cuda")
        and r["v1_error"] == "original hard pseudo-positive TAL target required"
        and d["changed_positive_target_weights"] == 0 and d["changed_assigned_class_weights"] == 0
        and d["target_tensor_unchanged"] and d["cast_underflow_pseudo_anchors"] >= 1)
    (out / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("sha256")}, ensure_ascii=False, indent=2))
    if not report["accepted"]:
        raise AssertionError("Engineering validation failed; inspect receipt")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    validate(parser.parse_args().output)
