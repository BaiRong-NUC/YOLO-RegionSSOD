"""检验 VisDrone 真值处理、配对训练权重和 COCO 评测协议。"""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from scripts.reproduce.esod import evaluate_visdrone as evaluation


@pytest.mark.parametrize("masked", [False, True])
def test_author_validation_pixels_and_boxes(tmp_path, masked):
    # 纹理图可以检出忽略区涂灰后是否多做或漏做了一次 JPEG 编解码。
    path = tmp_path / "sample.jpg"
    annotation = tmp_path / "sample.txt"
    texture = np.random.default_rng(42).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), texture)
    expected = cv2.imread(str(path))
    rows = ["5,6,7,8,1,10,1,2", "4,4,4,4,1,1,2,0"]
    if masked:
        rows.extend(["0,0,12,12,0,0,0,0,", "24,8,9,8,0,11,0,0"])
        expected[:12, :12] = 85
        expected[8:16, 24:33] = 85
        disk = tmp_path / "sample_masked.jpg"
        assert cv2.imwrite(str(disk), expected)
        expected = cv2.imread(str(disk))
    annotation.write_text("\n".join(rows), encoding="utf-8")
    actual, boxes = evaluation.prepare_image(path, annotation)
    np.testing.assert_array_equal(actual, expected)
    # 遮挡等级 2 仍进入真值，截断等级 2 被丢弃，类别编号保持原始 1–10。
    assert boxes == [(10, [5, 6, 7, 8])]


def test_default_and_local_cli():
    output = ["--output", "out"]
    default = evaluation.parse_args(output)
    assert (default.head, default.long_edge, default.limit) == ("sparse", 1536, 548)
    assert default.weights is default.sha256 is None
    for head in ("sparse", "dense", "vanilla-timing"):
        assert evaluation.parse_args([*output, "--head", head, "--long-edge", "1920"]).head == head
    for head in ("vanilla", "sparse"):
        for edge in (640, 768):
            args = evaluation.parse_args([*output, "--head", head, "--weights", "best.pt",
                                          "--sha256", "AB" * 32, "--long-edge", str(edge)])
            assert (args.weights.name, args.sha256, args.long_edge) == ("best.pt", "AB" * 32, edge)


@pytest.mark.parametrize("options", [
    ["--head", "vanilla"], ["--weights", "best.pt"], ["--sha256", "0" * 64],
    ["--weights", "best.pt", "--sha256", "invalid"],
    ["--head", "vanilla-timing", "--weights", "best.pt", "--sha256", "0" * 64],
])
def test_incomplete_or_ambiguous_cli(options):
    with pytest.raises(SystemExit) as error:
        evaluation.parse_args(["--output", "out", *options])
    assert error.value.code == 2


def test_wrong_digest_precedes_deserialization(tmp_path, monkeypatch):
    weight = tmp_path / "best.pt"
    weight.write_bytes(b"invalid checkpoint")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("摘要不符时不应反序列化"))
    with pytest.raises(ValueError, match="SHA-256"):
        evaluation.load_trained_model(weight, "0" * 64, "vanilla", device="cpu")


@pytest.fixture
def author_cpu(monkeypatch):
    # 基础 yolo 环境跑 CLI 和摘要测试，作者模型使用复现卡列出的额外依赖。
    for name in ("pkg_resources", "skimage", "timm", "einops"):
        pytest.importorskip(name)
    monkeypatch.syspath_prepend(str(evaluation.UPSTREAM))
    from models.yolo import Model

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield Model
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize(("head", "family", "branch"), [
    ("vanilla", "vanilla", "model"), ("sparse", "esod", "ema"),
])
@torch.inference_mode()
def test_training_checkpoint_and_real_forward(tmp_path, author_cpu, monkeypatch, head, family, branch):
    cfg = evaluation.UPSTREAM / "models/cfg" / family / "visdrone_yolov5m.yaml"
    model = author_cpu(str(cfg)).eval().half()
    names = yaml.safe_load((evaluation.UPSTREAM / "data/visdrone.yaml").read_text(encoding="utf-8"))["names"]
    model.names = dict(reversed(list(enumerate(names))))
    detector = model.model[-1]
    with pytest.raises(ValueError, match="内嵌配置"):
        evaluation.validate_model(model, "sparse" if head == "vanilla" else "vanilla")
    with monkeypatch.context() as patch:
        patch.setattr(model, "names", names[::-1])
        with pytest.raises(ValueError, match="类别名称或顺序"):
            evaluation.validate_model(model, head)
    with monkeypatch.context() as patch:
        patch.setattr(detector, "nc", 9)
        with pytest.raises(ValueError, match="Detect"):
            evaluation.validate_model(model, head)

    # 以作者的整模型格式构造 CPU 测试夹具，分别走 model 回退和 EMA 优先路径。
    checkpoint = {"epoch": 3, "model": None, "ema": None}
    checkpoint[branch] = model
    weight = tmp_path / "best.pt"
    torch.save(checkpoint, weight)
    with weight.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    loaded, info = evaluation.load_trained_model(weight, digest.upper(), head, device="cpu")
    assert info["sha256"] == digest and info["checkpoint_branch"] == branch and info["checkpoint_epoch"] == 3
    assert info["validated_config"] == cfg.relative_to(evaluation.UPSTREAM).as_posix()
    assert loaded.model[-1].sparse == (head == "sparse")
    tensor = torch.zeros((1, 3, 64, 96))
    if head == "sparse":
        # 控制热图，让同一检测头先有切片再无切片，覆盖真实的提前返回分支。
        mask = next(layer for layer in loaded.model if type(layer).__name__ == "Segmenter").m[0]
        mask.weight.zero_()
        mask.bias.fill_(5)
    (decoded, raw), heatmap = loaded(tensor)
    assert isinstance(decoded, torch.Tensor) and decoded.is_floating_point()
    assert decoded.ndim == 3 and decoded.shape[::2] == (1, 15) and decoded.shape[1] > 0
    assert raw is not None
    if head == "sparse":
        assert isinstance(heatmap, list)
        mask.bias.fill_(-5)
        detection, heatmap = loaded(tensor)
        assert detection == (None, None) and isinstance(heatmap, list)
    else:
        assert heatmap is None


class FixedOutput(torch.nn.Module):
    """给真实 NMS 提供可手算的单图输出，以核对 COCO 类别和坐标。"""

    def __init__(self, output):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.output = output

    def forward(self, tensor):
        return (self.output, []), None


@pytest.mark.parametrize(("head", "empty"), [("vanilla", False), ("vanilla", True), ("sparse", True)])
def test_coco_pipeline_and_empty_images(tmp_path, monkeypatch, author_cpu, head, empty):
    images = tmp_path / "images/val"
    annotations = tmp_path / "original_annotations/val"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    # 模拟完整的图像清单，--limit 1 只需将第一张合成图写入临时目录。
    paths = [images / f"{image_id:03}.jpg" for image_id in range(548)]
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(paths))
    assert cv2.imwrite(str(paths[0]), np.zeros((64, 64, 3), np.uint8))
    (annotations / "000.txt").write_text("10,20,20,10,1,1,0,0\n10,20,20,10,1,10,0,0", encoding="utf-8")
    decoded = torch.zeros((1, 2, 15))
    # 原图长边 64，放大至 768 后中心坐标与框宽高均乘 12。
    decoded[0, 0, :5] = torch.tensor([240, 300, 240, 120, 1])
    decoded[0, 0, [5, 14]] = 0.8
    # 作者 NMS 的 max_nms=-1 会截掉最后一项，补一个低分候选以保留两个目标类别。
    decoded[0, 1, :6] = torch.tensor([240, 300, 240, 120, 1, 0.01])
    if empty:
        decoded = None if head == "sparse" else torch.empty((1, 0, 15))
    model = FixedOutput(decoded)
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["evaluate_visdrone.py", "--data", str(tmp_path), "--head", head,
                                      "--weights", "best.pt", "--sha256", "AB" * 32, "--long-edge", "768",
                                      "--limit", "1", "--output", str(output)])
    monkeypatch.setattr(evaluation, "load_trained_model", lambda *args: (model, {}))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU 测试夹具")
    evaluation.main()
    predictions = json.loads((output / "predictions.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    durations = json.loads((output / "durations_ms.json").read_text())
    assert summary["long_edge"] == 768
    assert summary["images"] == len(durations) == 1 and summary["ground_truth"] == 2
    assert summary["detections"] == len(predictions)
    assert summary["pipeline_ms"] >= summary["forward_ms"] > 0 and summary["pipeline_fps"] > 0
    assert summary["AP_small"] == pytest.approx(0 if empty else 100)
    assert sorted(row["category_id"] for row in predictions) == ([] if empty else [1, 10])
    for row in predictions:
        assert row["bbox"] == [10, 20, 20, 10] and row["image_id"] == 1
        assert row["score"] == pytest.approx(0.8)
