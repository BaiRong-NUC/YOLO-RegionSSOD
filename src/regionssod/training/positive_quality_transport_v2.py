"""Integration-only repair; frozen positive-quality loss/model stay unchanged."""
from __future__ import annotations

from functools import partial
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO

from .p1 import P1Trainer
from .positive_quality_transport import PositiveQualityTrainer
from .reliable_competitor_negative import SourceDataset
from .runner import load_experiment_config, _optional_git_commit

IMPLEMENTATION = "positive_quality_transport_trainerfix_v2"


class PositiveQualityTrainerV2(PositiveQualityTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            # P1Trainer is in this instance's MRO; its lexical super() correctly
            # reaches DetectionTrainer. Do not transplant another class method.
            return P1Trainer.build_dataset(self, img_path, mode, batch)
        if self.args.single_cls or self.args.classes is not None or self.args.fraction != 1:
            raise ValueError("all classes and fraction=1 required")
        return SourceDataset(img_path=img_path, metadata_path=Path(self.data["p1_metadata"]),
            imgsz=self.args.imgsz, batch_size=batch, augment=True, hyp=self.args,
            rect=self.args.rect, cache=self.args.cache or None, single_cls=False,
            stride=max(int(self.model.stride.max()), 32), pad=0., prefix="train: ",
            task="detect", data=self.data, fraction=1.)


def run_training(config_path):
    """Explicit new worker only; never runs at import, preflight or validation."""
    import yaml
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    mode = config["positive_quality_transport"]["mode"]
    run = Path(config["project"]).resolve() / config["name"]
    if run.exists():
        raise FileExistsError(run)
    data = Path(config["data"]).resolve()
    metadata = Path(yaml.safe_load(data.read_text(encoding="utf-8"))["p1_metadata"])
    sources = [config_path, data, metadata, Path(config["model"]).resolve()]
    sources += list(Path(__file__).resolve().parents[1].rglob("*.py"))
    provenance = {"config": config, "implementation": IMPLEMENTATION, "mode": mode,
        "source_commit": _optional_git_commit(config_path.parent),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    yolo = YOLO(config["model"])
    def save_metadata(trainer):
        (Path(trainer.save_dir) / "experiment.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    yolo.add_callback("on_pretrain_routine_end", save_metadata)
    result = yolo.train(data=str(data), name=config["name"], project=str(run.parent), exist_ok=False,
        trainer=partial(PositiveQualityTrainerV2, mode=mode, p1_options=config["p1"]), **config["train"])
    return Path(result.save_dir).resolve()
