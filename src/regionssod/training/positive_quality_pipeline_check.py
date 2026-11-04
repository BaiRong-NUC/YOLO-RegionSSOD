"""Synthetic disk fixtures exercise the real train/val dataloader dispatch.

No real VisDrone samples, detector forward, optimizer or training loop. A stride
placeholder supplies only the shape metadata needed by dataset construction.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset

from .positive_quality_transport import PositiveQualityTrainer, PositiveQualityModel
from .positive_quality_transport_v2 import PositiveQualityTrainerV2
from .reliable_competitor_negative import SourceDataset, validate_batch_provenance


def fixture(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    paths = []
    metadata = []
    for i in range(4):
        source = "labeled" if i < 2 else "pseudo"
        p = root / source / "images/train" / f"fixture_{i}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (80+i, 120, 160)).save(p)
        label = root / source / "labels/train" / f"fixture_{i}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        paths.append(str(p))
        if source == "pseudo":
            metadata.append({"image": str(p), "positive": [{"class_id": 1, "confidence": 1.,
                             "xywhn": [.5, .5, .5, .5]}], "ignore": []})
    train = root / "train.txt"
    train.write_text("\n".join(paths) + "\n", encoding="utf-8")
    sidecar = root / "metadata.jsonl"
    sidecar.write_text("".join(json.dumps(r)+"\n" for r in metadata), encoding="utf-8")
    val = root / "validation/images/val/fixture_val.jpg"
    val.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), (120, 140, 160)).save(val)
    label = root / "validation/labels/val/fixture_val.txt"
    label.parent.mkdir(parents=True)
    label.write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    return {"train": str(train), "val": str(val.parent), "p1_metadata": str(sidecar),
        "names": {i: str(i) for i in range(10)}, "nc": 10, "channels": 3, "path": str(root)}


def trainer_stub(cls, data, mode):
    trainer = object.__new__(cls)
    trainer.args = get_cfg(overrides={"imgsz": 64, "batch": 8, "workers": 0,
        "device": "cpu", "seed": 0, "deterministic": True, "plots": False})
    trainer.data = data
    trainer.mode = mode
    trainer.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    trainer.model = torch.nn.Module()
    trainer.model.stride = torch.tensor([8., 16., 32.])
    trainer.device = torch.device("cpu")
    return trainer


def check_pipeline(root):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    data = fixture(root)
    old = trainer_stub(PositiveQualityTrainer, data, "reference")
    try:
        old.build_dataset(data["val"], "val", 8)
    except TypeError as e:
        if "super(type, obj)" not in str(e):
            raise
        old_error = str(e)
    else:
        raise AssertionError("regression fixture did not reproduce V1 failure")
    results = {}
    for mode in ("reference", "candidate", "control"):
        trainer = trainer_stub(PositiveQualityTrainerV2, data, mode)
        train = trainer.get_dataloader(data["train"], batch_size=8, rank=-1, mode="train")
        val = trainer.get_dataloader(data["val"], batch_size=8, rank=-1, mode="val")
        assert type(train.dataset) is SourceDataset
        assert type(val.dataset) is YOLODataset
        train_batch = next(iter(train))
        val_batch = next(iter(val))
        validate_batch_provenance(train_batch, 10)
        assert "object_is_human" not in val_batch and "pseudo_weights" not in val_batch
        assert len(train.dataset) == 4 and len(val.dataset) == 1
        # Test the original inherited get_model method's mode binding without
        # instantiating or forwarding a real detector.
        constructed = torch.nn.Module()
        trainer.set_model_names_for_load = lambda m: m
        with patch("regionssod.training.positive_quality_transport.PositiveQualityModel", return_value=constructed):
            actual = trainer.get_model(cfg={}, verbose=False)
        assert actual.positive_quality_mode == mode and actual.p1_options == trainer.p1_options
        results[mode] = {"train_dataset": type(train.dataset).__name__, "val_dataset": type(val.dataset).__name__,
            "synthetic_train_images": len(train.dataset), "synthetic_val_images": len(val.dataset),
            "synthetic_train_batch_shape": list(train_batch["img"].shape),
            "synthetic_val_batch_shape": list(val_batch["img"].shape), "get_model_mode_bound": True,
            "real_detector_instantiated": False}
    return {"accepted": True, "v1_val_failure_reproduced": old_error, "modes": results,
        "actual_installed_get_dataloader": True, "new_real_batches": 0, "model_forward_calls": 0,
        "optimizer_steps": 0, "student_training_iterations": 0,
        "scope": "actual dataloader path on synthetic images with stride-only placeholder; not full trainer setup/training"}
