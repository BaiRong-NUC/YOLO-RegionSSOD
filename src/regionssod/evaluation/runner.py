from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_validation(
    weights: Path,
    data: Path,
    *,
    imgsz: int = 960,
    batch: int = 8,
    device: str = "0",
    project: str = "runs/val",
    name: str | None = None,
) -> Path:
    from ultralytics import YOLO

    weights = weights.resolve()
    data = data.resolve()
    project = str(Path(project).resolve())
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data}")

    run_name = name or f"{weights.parent.parent.name}_{weights.stem}_{imgsz}"
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=run_name,
        plots=False,
        save_json=False,
        max_det=500,
    )
    save_dir = Path(
        getattr(metrics, "save_dir", Path(project) / run_name)
    ).resolve()

    box = getattr(metrics, "box", None)
    payload = {
        "weights": str(weights),
        "data": str(data),
        "imgsz": imgsz,
        "results": _to_builtin(getattr(metrics, "results_dict", {})),
        "speed_ms_per_image": _to_builtin(getattr(metrics, "speed", {})),
        "class_indices": _to_builtin(getattr(box, "ap_class_index", [])),
        "class_maps": _to_builtin(getattr(box, "maps", [])),
        "names": _to_builtin(getattr(metrics, "names", {})),
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    results = payload["results"]
    print(f"validation_dir: {save_dir}")
    for key in (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ):
        if key in results:
            print(f"{key}: {results[key]:.6f}")
    return save_dir
