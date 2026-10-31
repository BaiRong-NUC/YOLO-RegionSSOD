"""Fresh base-tile confirmation for coverage-guarded assignment, without training."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import torch
import yaml
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors

from audit_assignment_geometry import ROOT, WEIGHTS, digest, read, replay_batch, summarize, write
from regionssod.training.assignment import CoverageGuardedAssigner

OUT = ROOT / "outputs/assignment_coverage_20261031"
PROTOCOL = ROOT / "docs/paper/assignment_coverage_protocol.md"
OLD = ROOT / "outputs/assignment_geometry_20261031"


def guarded_rows(cache):
    rows = replay_batch(cache)
    arrays = [torch.from_numpy(cache[key]) for key in
              ("scores", "boxes", "points", "labels", "targets", "valid")]
    guarded = CoverageGuardedAssigner(topk=10, num_classes=10, alpha=.5, beta=6.,
                                      stride=[8, 16, 32])(*arrays)
    actual = {int(i): k for k, i in enumerate(cache["sample_ids"])}
    image_fallback = {}
    for image in actual:
        image_rows = [r for r in rows if r["image"] == image]
        image_fallback[image] = any(r["baseline"]["anchors"] > 0 and
                                   r["candidate"]["anchors"] == 0 for r in image_rows)
    for row in rows:
        image, gt = actual[row["image"]], row["gt"]
        selected = guarded[3][image] & (guarded[4][image] == gt)
        mass = guarded[2][image].sum(-1)
        selected &= mass > 0
        points = arrays[2]
        target = arrays[4][image, gt]
        inside = ((points > target[:2]) & (points < target[2:])).all(-1)
        outside = selected & ~inside
        measured = {"anchors": int(selected.sum()), "score_mass": float(mass[selected].sum()),
                    "outside_anchors": int(outside.sum()), "outside_mass": float(mass[outside].sum())}
        expected = row["baseline"] if image_fallback[row["image"]] else row["candidate"]
        if measured != expected:
            raise ValueError("coverage-guard implementation differs from frozen whole-image rule")
        row["candidate"] = measured
        row["fallback"] = image_fallback[row["image"]]
    return rows


def analysis(rows):
    # Reuse the same thresholds without presenting reused discovery inputs as new evidence.
    fixed = summarize(rows)["groups"]["confirmation"]
    return {"fresh_confirmation": fixed, "worth_30_epoch_confirmation": fixed["passed"],
            "fallback_images": len({r["image"] for r in rows if r["fallback"]}),
            "scope": "128 new base tiles, full-pool mosaic may overlap; frozen P1 checkpoint; "
                     "no hidden/val GT, no training or student performance"}


def verify(check):
    receipt = read(OUT / "receipt.json")
    for path, checksum in receipt["sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"changed source: {path}")
    for path, checksum in read(ROOT / "outputs/p1_small_nwd_pilot30_s0/preflight.json")["sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"changed frozen input: {path}")
    rows = []
    for path, checksum in read(OUT / "cache_manifest.json").items():
        if digest(path) != checksum:
            raise ValueError(f"changed raw predictions: {path}")
        with np.load(path, allow_pickle=False) as cache:
            rows.extend(guarded_rows(cache))
    result = analysis(rows)
    if check:
        if rows != read(OUT / "targets.json") or result != read(OUT / "analysis.json"):
            raise ValueError("fresh confirmation replay differs")
    else:
        write(OUT / "targets.json", rows)
        write(OUT / "analysis.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def collect():
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from regionssod.training.p1 import P1Dataset, P1DetectionLoss

    if Path.cwd() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("use repository root and pinned yolo environment")
    if OUT.exists():
        raise FileExistsError("confirmation already exists; use --check")
    config = yaml.safe_load((ROOT / "configs/train_p1_10.yaml").read_text(encoding="utf-8"))
    data_cfg = yaml.safe_load((ROOT / config["data"]).read_text(encoding="utf-8"))
    paths = sorted(Path(data_cfg["train"]).read_text(encoding="utf-8").splitlines())
    old_receipt = read(OLD / "receipt.json")
    old = set(old_receipt["selected_images"])
    remaining = [p for p in paths if p not in old]
    if len(paths) != 11310 or len(remaining) != 11054:
        raise ValueError("expected full data excluding exactly 256 old base tiles")
    indices = np.random.default_rng(20261101).choice(len(remaining), 128, replace=False)
    selected = [remaining[int(i)] for i in indices]
    source_paths = [Path(__file__), PROTOCOL, Path(inspect.getfile(CoverageGuardedAssigner)),
                    OLD / "receipt.json", OLD / "analysis.json"]
    hashes = {**old_receipt["sha256"], **{str(p): digest(p) for p in source_paths}}
    for path, checksum in hashes.items():
        if digest(path) != checksum:
            raise ValueError(f"changed source before dispatch: {path}")
    relative = [p.relative_to(ROOT).as_posix() for p in source_paths[:3]]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                   capture_output=True, check=True)
    subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *relative], check=True)
    OUT.mkdir()
    (OUT / "raw").mkdir()
    write(OUT / "receipt.json", {"selected_images": selected, "sample_seed": 20261101,
        "new_forward_inputs": 128, "optimizer_steps": 0, "sha256": hashes,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    dataset = P1Dataset(img_path=data_cfg["train"], metadata_path=Path(data_cfg["p1_metadata"]),
                        imgsz=640, batch_size=8, augment=True, hyp=get_cfg(overrides=config["train"]),
                        rect=False, cache=False, single_cls=False, stride=32, pad=0.,
                        task="detect", data=data_cfg, fraction=1.)
    image_indices = {str(Path(p).resolve()): i for i, p in enumerate(dataset.im_files)}
    model: Any = YOLO(str(WEIGHTS)).model
    if not isinstance(model, torch.nn.Module):
        raise ValueError("expected loaded detection model")
    model = model.float().cuda().eval()
    model.args = get_cfg(overrides=config["train"])
    model.p1_options = config["p1"]
    model.model[-1].training = True
    criterion = P1DetectionLoss(model)
    manifest, image_hashes = {}, []
    for start in range(0, 128, 8):
        samples = []
        for order in range(start, start + 8):
            random.seed(20261101 + order)
            np.random.seed(20261101 + order)
            torch.manual_seed(20261101 + order)
            sample = dataset[image_indices[str(Path(selected[order]).resolve())]]
            image_hashes.append(hashlib.sha256(sample["img"].numpy().tobytes()).hexdigest())
            samples.append(sample)
        batch = dataset.collate_fn(samples)
        with torch.no_grad():
            predictions = model(batch["img"].cuda().float() / 255.)
            points, stride = make_anchors(predictions["feats"], model.model[-1].stride, .5)
            boxes = criterion.bbox_decode(points, predictions["boxes"].permute(0, 2, 1).contiguous()) * stride
            scores = predictions["scores"].permute(0, 2, 1).contiguous().sigmoid()
        target_list = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1),
                                 batch["bboxes"]), 1)
        targets = criterion.preprocess(target_list.cuda(), 8, scale_tensor=torch.tensor(
            [640., 640., 640., 640.], device="cuda")).cpu()
        confidence = torch.zeros(targets.shape[:2])
        for image in range(8):
            values = batch["pseudo_weights"][batch["batch_idx"] == image]
            confidence[image, :len(values)] = values
        path = OUT / "raw" / f"batch_{start // 8:03d}.npz"
        np.savez_compressed(path, scores=scores.cpu().numpy(), boxes=boxes.cpu().numpy(),
                            points=(points * stride).cpu().numpy(), labels=targets[..., :1].numpy(),
                            targets=targets[..., 1:].numpy(),
                            valid=(targets[..., 1:].sum(-1, keepdim=True) > 0).numpy(),
                            confidence=confidence.numpy(), sample_ids=np.arange(128+start, 136+start))
        manifest[str(path)] = digest(path)
        print(f"Cached {start+8}/128 new augmented inputs; zero training steps", flush=True)
    write(OUT / "cache_manifest.json", manifest)
    write(OUT / "augmented_image_sha256.json", image_hashes)
    verify(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    torch.set_num_threads(4)
    if options.check:
        verify(True)
    else:
        collect()
