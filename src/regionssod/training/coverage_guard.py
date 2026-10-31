"""P1 with the previously diagnosed, image-level coverage-guarded TAL rule."""
from __future__ import annotations

from functools import partial
import hashlib
import json
from pathlib import Path

from ultralytics.nn.tasks import DetectionModel

from regionssod.training.assignment import CoverageGuardedAssigner
from regionssod.training.p1 import P1DetectionLoss, P1Trainer
from regionssod.training.runner import _optional_git_commit, load_experiment_config


class CoverageGuardLoss(P1DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.assigner = CoverageGuardedAssigner(
            topk=10, num_classes=self.nc, alpha=.5, beta=6.,
            stride=self.stride.tolist(), topk2=None)


class CoverageGuardModel(DetectionModel):
    def init_criterion(self):
        return CoverageGuardLoss(self)

    def loss(self, batch, preds=None):
        if self.training:
            if "pseudo_weights" not in batch:
                raise ValueError("coverage guard training requires per-object provenance")
            return super().loss(batch, preds)
        if preds is None:
            preds = self.forward(batch["img"])
        # Standard validation has no pseudo metadata and measures the original loss.
        return P1DetectionLoss(self)(preds, batch)


class CoverageGuardTrainer(P1Trainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(CoverageGuardModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        if weights:
            model.load(weights)
        return model


def run_coverage_guard_training(config_path):
    import ultralytics
    from ultralytics import YOLO
    import yaml

    if ultralytics.__version__ != "8.4.172":
        raise RuntimeError("use pinned ultralytics==8.4.172")
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    if (config["method"] != "p1" or config["assignment"] != "image_coverage_guard"
            or config["p1"] != {"soft_weights": True, "uncertain_ignore": True}):
        raise ValueError("expected the single coverage-guard extension")
    data = Path(config["data"]).resolve()
    dataset = yaml.safe_load(data.read_text(encoding="utf-8"))
    metadata_path = Path(dataset["p1_metadata"])
    project = Path(config["project"]).resolve()
    run = project / config["name"]
    if run.exists():
        raise FileExistsError(run)
    sources = [config_path, data, metadata_path, metadata_path.parent / "summary.json"]
    sources += list(Path(__file__).resolve().parents[1].rglob("*.py"))
    metadata = {
        "config_path": str(config_path), "model": config["model"], "data": str(data),
        "project": str(project), "name": config["name"], "train_args": config["train"],
        "method": "p1", "p1": config["p1"], "assignment": config["assignment"],
        "ultralytics_version": ultralytics.__version__,
        "git_commit": _optional_git_commit(config_path.parent),
        "pseudo_summary": json.loads((metadata_path.parent / "summary.json").read_text(encoding="utf-8")),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    yolo = YOLO(config["model"])

    def save_metadata(trainer):
        (Path(trainer.save_dir) / "experiment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    yolo.add_callback("on_pretrain_routine_end", save_metadata)
    yolo.train(data=str(data), name=config["name"], project=str(project), exist_ok=False,
               trainer=partial(CoverageGuardTrainer, p1_options=config["p1"]), **config["train"])
    return run
