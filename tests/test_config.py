from pathlib import Path

from regionssod.training.runner import load_experiment_config
import pytest
import yaml


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


def test_p1_pilot_matches_baseline_hyperparameters():
    baseline = load_experiment_config(ROOT / "configs/train_naive_10.yaml")
    p1 = load_experiment_config(ROOT / "configs/train_p1_10.yaml")
    assert p1["method"] == "p1"
    assert p1["train"] == baseline["train"] | {"epochs": 30}
    assert p1["model"] == baseline["model"]


@pytest.mark.parametrize("p1", [{"soft_weights": "false"}, {"unknown_option": True}])
def test_p1_rejects_silently_ignored_loss_options(tmp_path, p1):
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump({"name": "bad", "data": "unused.yaml", "method": "p1", "p1": p1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(config)
