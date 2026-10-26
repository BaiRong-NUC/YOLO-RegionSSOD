from pathlib import Path

from regionssod.training.runner import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_configs_are_loadable() -> None:
    for name in (
        "train_partial_05.yaml",
        "train_partial_10.yaml",
        "train_partial_20.yaml",
        "train_naive_10.yaml",
    ):
        config = load_experiment_config(ROOT / "configs" / name)
        assert config["name"]
        assert config["data"]
        assert config["train"]["epochs"] == 100
        assert config["train"]["imgsz"] == 640
