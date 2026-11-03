"""One mixed-source train batch, one AMP forward, no backward/optimizer/training.

--replay uses only the immutable saved forward tensors; never loads another batch.
It permits finishing engineering assertions after fixes without consuming a
second real batch. Outputs are engineering evidence, not student performance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.trainer import BaseTrainer

from regionssod.training.p1 import P1DetectionLoss
from regionssod.training.reliable_competitor_negative import (
    IMPLEMENTATION, SourceDataset, ReliableCompetitorModel, ReliableCompetitorLoss, negative_multipliers,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reliable_competitor_negative_engineering"
CACHE = ROOT / "outputs/p1_competitive_negative_debug_20261103_v1/foreground_tensors.pt"
CACHE_SHA = "8a1aa278e75756ac5e5315248235e0c60b33f6ab943d76cf57ba01700fb41bb6"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def parameter_sha(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def foreground_cache():
    if sha(CACHE) != CACHE_SHA:
        raise ValueError("existing training tensor identity drift")
    # Explicit source flags/assigned labels were captured by the old validated
    # class-score training dataset, not inferred from confidence values.
    receipt = json.loads((ROOT / "outputs/p1_competitive_negative_v2_validation_20261103/validation.json").read_text())
    if receipt["artifact_sha256"][str(CACHE)] != CACHE_SHA or not receipt["accepted"]:
        raise ValueError("unbound existing cache")
    rows = torch.load(CACHE, map_location="cpu", weights_only=True)
    count, active, zeros, delta = 0, 0, 0, 0.
    for row in rows:
        logits, target, human, labels, raw = row["logits"], row["targets"], ~row["flags"], row["labels"].long(), row["raw"]
        a, enabled = negative_multipliers(logits, target, human, labels, raw, mode="candidate")
        b, enabled_b = negative_multipliers(logits, target, human, labels, raw, mode="control")
        assert torch.equal(enabled, enabled_b)
        p = logits.double().sigmoid()
        first, second = [((w.double()-1) * p).sum(1) for w in (a, b)]
        torch.testing.assert_close(first, second, rtol=2e-6, atol=1e-10)
        delta = max(delta, float((first-second).abs().max()) if len(first) else 0.)
        active += int(enabled.sum())
        zeros += int((human & (target.sum(1) == 0)).sum())
        count += len(logits)
    return {"anchors": count, "eligible_human": active, "human_zero_mass": zeros,
            "max_added_negative_logit_l1_abs_error": delta, "cache_sha256": CACHE_SHA}


def capture(out):
    import yaml
    from ultralytics.utils.torch_utils import init_seeds
    config = yaml.safe_load((ROOT / "configs/train_p1_10.yaml").read_text())
    init_seeds(0, deterministic=True)
    random.seed(0)
    np.random.seed(0)
    data = check_det_dataset(config["data"])
    args = get_cfg(overrides=config["train"])
    dataset = SourceDataset(img_path=data["train"], metadata_path=Path(data["p1_metadata"]),
        imgsz=640, batch_size=8, augment=True, hyp=args, rect=False, cache=False,
        single_cls=False, stride=32, pad=0., prefix="engineering: ", task="detect", data=data, fraction=1.)
    human, pseudo = [], []
    for i, label in enumerate(dataset.labels):
        if not len(label["cls"]):
            continue
        key = str(Path(label["im_file"]).resolve())
        (pseudo if key in dataset.p1_metadata else human).append(i)
    indices = human[:4] + pseudo[:4]
    if len(indices) != 8:
        raise ValueError("mixed-source engineering batch unavailable")
    batch = dataset.collate_fn([dataset[i] for i in indices])
    if not batch["object_is_human"].any() or batch["object_is_human"].all():
        raise ValueError("augmentation must exercise both sources")
    batch["img"] = batch["img"].float().cuda() / 255.
    pretrained = YOLO(config["model"]).model
    model = ReliableCompetitorModel(pretrained.yaml, nc=10, ch=3, verbose=False)
    model.load(pretrained)
    model.p1_options = config["p1"]
    model.reliable_competitor_mode = "reference"
    model.args = args
    model = model.cuda().train()
    before = parameter_sha(model)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        predictions = model(batch["img"])
    after = parameter_sha(model)
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu(v) for v in value]
        return value
    torch.save({"predictions": cpu(predictions), "batch": cpu(batch),
        "stride": model.stride.cpu(), "reg_max": model.model[-1].reg_max,
        "args": vars(args)}, out / "real_batch_cache.pt")
    write_new(out / "capture.json", {"images": 8, "indices": indices, "amp": True,
        "real_forward_calls": 1, "device": "cuda:0", "dataset_images": len(dataset),
        "human_objects": int(batch["object_is_human"].sum()),
        "pseudo_objects": int((~batch["object_is_human"]).sum()),
        "parameter_hash_before": before, "parameter_hash_after": after,
        "parameter_hash_unchanged": before == after,
        "cache_sha256": sha(out / "real_batch_cache.pt"),
        "source_sha256": sha(ROOT / "src/regionssod/training/reliable_competitor_negative.py"),
        "note": "train-mode forward only; BN buffers may update; no backward/optimizer"})


def replay(out):
    capture_info = json.loads((out / "capture.json").read_text())
    if sha(out / "real_batch_cache.pt") != capture_info["cache_sha256"]:
        raise ValueError("real engineering cache drift")
    cache = torch.load(out / "real_batch_cache.pt", weights_only=True, map_location="cuda:0")
    model = nn_fixture(cache)
    preds, batch = cache["predictions"], cache["batch"]
    # The only real forward has already happened; all arms consume identical tensors.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        base = P1DetectionLoss(model).get_assigned_targets_and_loss(preds, batch)
        results, criteria = {}, {}
        for mode in ("reference", "candidate", "control"):
            model.reliable_competitor_mode = mode
            criterion = ReliableCompetitorLoss(model)
            criteria[mode] = criterion
            results[mode] = criterion.get_assigned_targets_and_loss(preds, batch)
            assert all(torch.equal(x, y) for x, y in zip(base[0], results[mode][0], strict=True))
            assert torch.equal(base[1][[0, 2]], results[mode][1][[0, 2]])
            assert torch.isfinite(results[mode][1]).all()
            assert criterion.last_diagnostics["changed_positive_target_weights"] == 0
        assert torch.equal(base[1], results["reference"][1])
    a, b = criteria["candidate"], criteria["control"]
    assert torch.equal(a.last_targets, b.last_targets)
    assert torch.equal(a.last_raw_targets, b.last_raw_targets)
    label, _, raw, foreground, local = a.last_assignment
    p = preds["scores"].permute(0, 2, 1).double().sigmoid()
    added = [((c.last_weights.double()-1) * p).sum(2) for c in (a, b)]
    torch.testing.assert_close(*added, rtol=2e-6, atol=1e-10)
    nonzero_actions = ((a.last_weights != 1).any(2))
    human_anchors = torch.zeros_like(foreground)
    for image in range(len(foreground)):
        selected = foreground[image]
        objects = batch["batch_idx"] == image
        human_anchors[image, selected] = batch["object_is_human"][objects][local[image, selected].long()]
    mass = a.last_targets.gather(2, label.clamp(min=0).long().unsqueeze(2)).squeeze(2)
    eligible = foreground & human_anchors & (mass > 0)
    assert torch.equal(nonzero_actions, eligible)
    assert eligible.any() and (foreground & ~human_anchors).any()
    # Standard validation fallback with cached training criterion, same predictions,
    # no val image/forward. Training missing metadata must still refuse.
    ordinary = {k: v for k, v in batch.items() if not k.startswith("object_") and k != "pseudo_weights"}
    model.criterion = a
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        actual = ReliableCompetitorModel.loss(model, ordinary, preds)
        expected = P1DetectionLoss(model)(preds, ordinary)
    assert torch.equal(actual[0], expected[0])
    try:
        a.get_assigned_targets_and_loss(preds, ordinary)
    except ValueError:
        pass
    else:
        raise AssertionError("missing training provenance accepted")
    return capture_info | {"assignment_identical": True, "box_DFL_identical": True,
        "reference_loss_exact": True, "positive_targets_unchanged": True,
        "action_support_exact": True, "validation_fallback_same_cached_predictions": True,
        "eligible_human_anchors": int(eligible.sum()), "pseudo_anchors_protected": int((foreground & ~human_anchors).sum()),
        "zero_effective_human_anchors": int((foreground & human_anchors & (mass == 0)).sum()),
        "added_negative_logit_l1_max_abs_error": float((added[0]-added[1]).abs().max()),
        "loss_components": {m: r[1].float().cpu().tolist() for m, r in results.items()}}


def nn_fixture(cache):
    model = torch.nn.Module().cuda()
    model.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1, device="cuda")))
    model.model = [SimpleNamespace(stride=cache["stride"], nc=10, reg_max=cache["reg_max"])]
    model.args = SimpleNamespace(**cache["args"])
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    model.reliable_competitor_diagnostics = True
    return model


def validate(out, replay_only=False):
    out = out.resolve()
    if ROOT / "outputs" not in out.parents:
        raise ValueError("ignored outputs location required")
    if not replay_only:
        out.mkdir(parents=True, exist_ok=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("training/backward/optimizer step forbidden")
    start = time.monotonic()
    with ExitStack() as stack:
        for cls, name in ((BaseTrainer, "train"), (torch.Tensor, "backward"),
                          (torch.optim.SGD, "step"), (torch.optim.Adam, "step"), (torch.optim.AdamW, "step")):
            stack.enter_context(patch.object(cls, name, forbidden))
        old = foreground_cache()
        if not replay_only:
            capture(out)
        real = replay(out)
    import ultralytics
    import ultralytics.data.augment
    import ultralytics.utils.loss
    import ultralytics.utils.tal
    sources = [Path(__file__).resolve(), ROOT / "configs/train_p1_10.yaml",
        ROOT / "docs/paper/candidates/reliable_competitor_negative/engineering_protocol.md"]
    sources += list((ROOT / "src/regionssod").rglob("*.py"))
    sources += [Path(inspect.getfile(m)) for m in
        (ultralytics.data.augment, ultralytics.utils.loss, ultralytics.utils.tal)]
    report = {"accepted": True, "implementation": IMPLEMENTATION, "optimizer_steps": 0,
        "backward_calls": 0, "training_loop_calls": 0, "validation_inference_calls": 0,
        "parameter_hash_unchanged": real["parameter_hash_unchanged"],
        "torch_version": torch.__version__, "ultralytics_version": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(), "cached_replay": old, "real_batch": real,
        "seconds": time.monotonic()-start, "student_performance": "NOT_TESTED", "mechanism": "NOT_TESTED",
        "source_sha256": {str(p): sha(p) for p in sources},
        "artifact_sha256": {str(p): sha(p) for p in (CACHE, out / "real_batch_cache.pt", out / "capture.json")}}
    assert report["parameter_hash_unchanged"]
    write_new(out / ("validation_final.json" if replay_only else "validation.json"), report)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("sha256")}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--replay", action="store_true", help="cached tensors only, no new real batch")
    args = parser.parse_args()
    validate(args.output, args.replay)
