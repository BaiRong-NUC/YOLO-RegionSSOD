"""Actual P1 data transport, YOLO11s AMP backward and validator; zero optimizer steps."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator

from diagnose_p1_class_ambiguity import ROOT, sha
from evaluate_paper1_small_pr import write
from run_p1_class_ambiguity import OUT, CONFIG, SCORES
from regionssod.training.runner import load_experiment_config
from regionssod.training.class_ambiguity import ClassAmbiguityDataset, ClassAmbiguityModel, ClassAmbiguityLoss
from regionssod.training.p1 import P1DetectionLoss


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().item()
    return {k: plain(v) for k, v in value.items()} if isinstance(value, dict) else value


def main():
    if (OUT / "dispatch.json").exists() or (OUT / "engineering_check.json").exists():
        raise FileExistsError("Existing engineering receipt/dispatch; inspect without overwrite")
    torch.manual_seed(0)
    config = load_experiment_config(CONFIG)
    data = check_det_dataset(config["data"])
    args = get_cfg(overrides=config["train"])
    dataset = ClassAmbiguityDataset(
        img_path=data["train"], metadata_path=Path(data["p1_metadata"]),
        class_score_path=SCORES / "class_scores.jsonl", imgsz=640, batch_size=2,
        augment=False, hyp=args, rect=False, cache=False, single_cls=False, stride=32,
        pad=0., prefix="engineering: ", task="detect", data=data, fraction=1.)
    # Deterministic engineering probe of both origins; augmentation transport has separate tests.
    human = next(i for i, p in enumerate(dataset.im_files) if str(Path(p).resolve()) not in dataset.p1_tokens)
    pseudo = next(i for i, p in enumerate(dataset.im_files) if str(Path(p).resolve()) in dataset.p1_tokens)
    batch = dataset.collate_fn([dataset[human], dataset[pseudo]])
    if not batch["is_pseudo"].any() or not (~batch["is_pseudo"]).any():
        raise ValueError("Mixed provenance not exercised")
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch["img"] = batch["img"].float() / 255
    pretrained = YOLO(config["model"]).model
    model = ClassAmbiguityModel(pretrained.yaml, nc=10, ch=3, verbose=False)
    model.load(pretrained)
    model.p1_options = config["p1"]
    model.class_ambiguity_diagnostics = True
    model.args = args
    model = model.cuda().train()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, loss_items = model(batch)
    loss.sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not torch.isfinite(loss).all() or not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError("Nonfinite actual AMP forward/backward")
    diagnostics = plain(model.criterion.last_diagnostics)
    if diagnostics["pseudo_positive"]["anchors"] <= 0 or diagnostics["human_positive"]["anchors"] <= 0:
        raise ValueError("No real pseudo/human positive anchors exercised")
    for group, values in diagnostics.items():
        if not isinstance(values, dict):
            continue
        for key in ("mass", "weighted_mass"):
            if not math.isclose(values["old_"+key], values["new_"+key], rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("Classification target mass changed")
    ema = deepcopy(model).float()
    if not isinstance(ema.criterion, ClassAmbiguityLoss):
        raise ValueError("Validation must exercise cached training criterion")
    validator = DetectionValidator(
        save_dir=OUT / "engineering_validation",
        args=get_cfg(overrides={"data": config["data"], "imgsz": 640, "batch": 2, "workers": 0,
                                "plots": False, "save_json": False, "conf": .25, "max_det": 500}))
    validator.data = data
    validator.device = torch.device("cuda:0")
    validator.stride = int(ema.stride.max())
    loader = validator.get_dataloader(data["val"], batch_size=2)
    validator.dataloader = DataLoader(Subset(loader.dataset, [0, 1]), batch_size=2,
                                    num_workers=0, collate_fn=loader.dataset.collate_fn)
    ordinary = next(iter(validator.dataloader))
    if "is_pseudo" in ordinary:
        raise ValueError("Validation must use ordinary labels")
    probe = SimpleNamespace(
        ema=SimpleNamespace(ema=ema), device=validator.device, data=data, amp=True,
        loss_items=loss_items, stopper=SimpleNamespace(possible_stop=False), epoch=0, epochs=30,
        world_size=1, label_loss_items=lambda values, prefix: {f"{prefix}/{k}": v for k, v in values.items()})
    values = validator(trainer=probe)
    val_loss = {k: v for k, v in values.items() if k.startswith("val/")}
    if validator.seen != 2 or len(val_loss) != 3 or not all(math.isfinite(v) for v in val_loss.values()):
        raise ValueError("Actual two-image validation failed")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        ordinary = validator.preprocess(ordinary)
        predictions = ema(ordinary["img"])
        observed = ema.loss(ordinary, predictions)[0]
        expected = P1DetectionLoss(ema)(predictions, ordinary)[0]
    if not torch.equal(observed, expected):
        raise ValueError("Ordinary validation changed")
    paths = [Path(__file__).resolve(), CONFIG, ROOT / "scripts/run_p1_class_ambiguity.py",
             ROOT / "src/regionssod/training/class_ambiguity.py",
             ROOT / "src/regionssod/training/p1.py", SCORES / "class_scores.jsonl",
             SCORES / "receipt.json"]
    write(OUT / "engineering_check.json", {
        "accepted": True, "optimizer_steps": 0, "epochs": 0, "amp": True, "train_imgsz": 640,
        "dataset_images": len(dataset), "pseudo_positive_objects": int(dataset.pseudo_lookup.sum()),
        "loss": loss.detach().float().cpu().tolist(), "finite_gradient_tensors": len(gradients),
        "classification_diagnostics": diagnostics, "validation_images": validator.seen,
        "validation_matches_original_p1": True, "validation_loss": val_loss,
        "source_sha256": {str(p): sha(p) for p in paths}})
    print("Accepted actual data + YOLO11s640 AMP backward + two-image validation; zero optimizer steps.")


if __name__ == "__main__":
    main()
