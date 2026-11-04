from __future__ import annotations
import ast
import inspect
from types import SimpleNamespace

import pytest

from regionssod.training.p1 import P1Trainer
from regionssod.training.positive_quality_transport import PositiveQualityTrainer, PositiveQualityModel
from regionssod.training.positive_quality_transport_v2 import PositiveQualityTrainerV2
from regionssod.training.positive_quality_pipeline_check import check_pipeline, fixture, trainer_stub
from regionssod.training.reliable_competitor_negative import ReliableCompetitorTrainer


def test_real_train_val_dataloaders_reproduce_and_fix_lexical_super(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("detector forward/optimizer forbidden")
    monkeypatch.setattr(PositiveQualityModel, "forward", forbidden)
    result = check_pipeline(tmp_path / "synthetic")
    assert result["accepted"] and "super(type, obj)" in result["v1_val_failure_reproduced"]
    assert set(result["modes"]) == {"reference", "candidate", "control"}
    assert result["new_real_batches"] == result["model_forward_calls"] == result["optimizer_steps"] == 0
    assert all(v["get_model_mode_bound"] for v in result["modes"].values())


def test_v2_scientific_model_factory_and_constructor_are_inherited_unchanged():
    assert PositiveQualityTrainerV2.get_model is PositiveQualityTrainer.get_model
    assert PositiveQualityTrainerV2.__init__ is PositiveQualityTrainer.__init__


def test_train_dataset_body_matches_frozen_constructor():
    import textwrap
    def body(cls):
        f = ast.parse(textwrap.dedent(inspect.getsource(cls.build_dataset))).body[0]
        # Only the first `mode != train` branch may differ.
        return [ast.dump(n, include_attributes=False) for n in f.body[1:]]
    assert body(PositiveQualityTrainerV2) == body(ReliableCompetitorTrainer)


@pytest.mark.parametrize("mode", ["reference", "candidate", "control"])
def test_inherited_constructor_keeps_mode_and_runtime_gate(monkeypatch, mode):
    monkeypatch.setattr(P1Trainer, "__init__", lambda self, *a, **k: setattr(self, "args", SimpleNamespace(cls_pw=0)))
    trainer = PositiveQualityTrainerV2(mode=mode)
    assert trainer.mode == mode
    with pytest.raises(ValueError, match="legal mode"):
        PositiveQualityTrainerV2(mode="other")


def test_validation_has_no_training_sidecar_dependency(tmp_path):
    data = fixture(tmp_path / "synthetic")
    data["p1_metadata"] = str(tmp_path / "absent.jsonl")
    trainer = trainer_stub(PositiveQualityTrainerV2, data, "candidate")
    loader = trainer.get_dataloader(data["val"], batch_size=8, rank=-1, mode="val")
    assert len(loader.dataset) == 1
    assert "object_is_human" not in next(iter(loader))
    with pytest.raises(FileNotFoundError):
        trainer.build_dataset(data["train"], "train", 8)


@pytest.mark.parametrize("key,value", [("single_cls", True), ("classes", [0]), ("fraction", .5)])
def test_training_subset_changes_rejected(tmp_path, key, value):
    trainer = trainer_stub(PositiveQualityTrainerV2, {}, "candidate")
    setattr(trainer.args, key, value)
    with pytest.raises(ValueError, match="all classes and fraction=1"):
        trainer.build_dataset(str(tmp_path), "train", 8)
