from __future__ import annotations

import json
from functools import partial
import hashlib
from pathlib import Path
import subprocess
from typing import Any

import yaml


TRAIN_KEYS = {
    "epochs",
    "batch",
    "imgsz",
    "device",
    "workers",
    "seed",
    "deterministic",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "amp",
    "plots",
    "max_det",
    "close_mosaic",
    "patience",
    "pretrained",
    "cache",
    "rect",
    "cos_lr",
    "single_cls",
    "fraction",
    "verbose",
}


def load_experiment_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    if "name" not in config or "data" not in config:
        raise ValueError(f"{path}: both name and data are required")
    train = config.get("train", {})
    if not isinstance(train, dict):
        raise ValueError(f"{path}: train must be a mapping")
    unknown = set(train) - TRAIN_KEYS
    if unknown:
        raise ValueError(f"{path}: unsupported train keys: {sorted(unknown)}")
    if config.get("method", "baseline") not in {"baseline", "p1"}:
        raise ValueError("method must be baseline or p1")
    options = config.get("p1", {})
    if not isinstance(options, dict) or set(options) - {"soft_weights", "uncertain_ignore"}:
        raise ValueError("p1 options must be soft_weights and/or uncertain_ignore")
    if any(not isinstance(v, bool) for v in options.values()):
        raise ValueError("p1 options must be booleans")
    return config


def _optional_git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run_training(
    config_path: Path,
    *,
    model: str | None = None,
    data: Path | None = None,
    name: str | None = None,
    project: str | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    seed: int | None = None,
    exist_ok: bool = False,
) -> Path:
    from ultralytics import YOLO
    import ultralytics

    config_path = config_path.resolve()
    config = load_experiment_config(config_path)
    train_args = dict(config.get("train", {}))

    model_path = model or str(config.get("model", "yolo11s.pt"))
    data_path = (data or Path(str(config["data"]))).resolve()
    run_name = name or str(config["name"])
    run_project = str(Path(project or str(config.get("project", "runs"))).resolve())

    if not data_path.exists():
        raise FileNotFoundError(f"dataset YAML not found: {data_path}")
    if not data_path.is_file():
        raise ValueError(f"dataset path must be a YAML file: {data_path}")

    overrides = {
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "seed": seed,
    }
    train_args.update({key: value for key, value in overrides.items() if value is not None})

    method = config.get("method", "baseline")
    dataset = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if bool(dataset.get("p1_metadata")) != (method == "p1"):
        raise ValueError("P1 dataset and method: p1 must be used together")
    trainer_args = {}
    p1_options = {"soft_weights": True, "uncertain_ignore": True} | config.get("p1", {})
    if method == "p1":
        if ultralytics.__version__ != "8.4.172":
            raise RuntimeError("P1 adapter is verified only with ultralytics==8.4.172")
        from regionssod.training.p1 import P1Trainer

        trainer_args["trainer"] = partial(P1Trainer, p1_options=p1_options)
    metadata = {
        "config_path": str(config_path),
        "model": model_path,
        "data": str(data_path),
        "project": run_project,
        "name": run_name,
        "train_args": train_args,
        "ultralytics_version": ultralytics.__version__,
        "git_commit": _optional_git_commit(config_path.parent),
        "method": method,
        "p1": p1_options if method == "p1" else None,
    }
    audit_files = [config_path, data_path] + list(Path(__file__).resolve().parents[1].rglob("*.py"))
    if method == "p1":
        metadata_file = Path(dataset["p1_metadata"])
        audit_files += [metadata_file, metadata_file.parent / "summary.json"]
        metadata["pseudo_summary"] = json.loads((metadata_file.parent / "summary.json").read_text(encoding="utf-8"))
    metadata["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in audit_files}
    yolo = YOLO(model_path)

    def save_metadata(trainer):
        save_dir = Path(trainer.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "experiment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    yolo.add_callback("on_pretrain_routine_end", save_metadata)
    results = yolo.train(
        data=str(data_path), name=run_name, project=run_project, exist_ok=exist_ok,
        **trainer_args, **train_args,
    )
    save_dir = Path(getattr(results, "save_dir", Path(run_project) / run_name)).resolve()
    return save_dir
