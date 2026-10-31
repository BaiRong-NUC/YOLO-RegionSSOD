"""Frozen-checkpoint assignment geometry diagnostic; no optimizer or hidden GT."""
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/assignment_geometry_20261031"
WEIGHTS = ROOT / "runs/partial10_p1_pilot30_s0/weights/best.pt"
PROTOCOL = ROOT / "docs/paper/assignment_geometry_protocol.md"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def strict_inside(points, boxes, eps=1e-9):
    delta = torch.cat((points[None, None] - boxes[:, :, None, :2],
                       boxes[:, :, None, 2:] - points[None, None]), dim=-1)
    return delta.amin(-1) > eps


class SupportedPointAssigner(TaskAlignedAssigner):
    """Prefer representable points, retaining the original pool for point-free GT."""

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        expanded = super().select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt, eps)
        inside = strict_inside(xy_centers, gt_bboxes, eps) & mask_gt.bool()
        return torch.where(inside.any(-1, keepdim=True), inside, expanded)


def replay_batch(cache):
    scores, boxes, points, labels, targets, valid = [
        torch.from_numpy(cache[name]) for name in
        ("scores", "boxes", "points", "labels", "targets", "valid")]
    strict = strict_inside(points, targets)
    outputs = [cls(topk=10, num_classes=10, alpha=.5, beta=6., stride=[8, 16, 32])(
        scores, boxes, points, labels, targets, valid)
               for cls in (TaskAlignedAssigner, SupportedPointAssigner)]
    rows = []
    for image in range(scores.shape[0]):
        for gt in range(targets.shape[1]):
            if not valid[image, gt, 0]:
                continue
            sides = targets[image, gt, 2:] - targets[image, gt, :2]
            row = {"image": int(cache["sample_ids"][image]), "gt": gt,
                   "small": bool(sides.prod().item() < 32**2),
                   "strict_points": int(strict[image, gt].sum()),
                   "box": targets[image, gt].tolist(),
                   "confidence": float(cache["confidence"][image, gt])}
            for name, output in zip(("baseline", "candidate"), outputs):
                _, _, target_scores, fg, indices = output
                weight = target_scores[image].sum(-1)
                selected = fg[image] & (indices[image] == gt) & (weight > 0)
                outside = selected & ~strict[image, gt]
                row[name] = {"anchors": int(selected.sum()), "score_mass": float(weight[selected].sum()),
                             "outside_anchors": int(outside.sum()),
                             "outside_mass": float(weight[outside].sum())}
            rows.append(row)
    return rows


def summarize(rows):
    result = {}
    for group, subset in (("discovery", [r for r in rows if r["image"] < 128]),
                          ("confirmation", [r for r in rows if r["image"] >= 128])):
        small = [r for r in subset if r["small"]]
        base_active = [r for r in subset if r["baseline"]["anchors"] > 0]
        small_active = [r for r in small if r["baseline"]["anchors"] > 0]
        before = sum(r["baseline"]["score_mass"] for r in small)
        after = sum(r["candidate"]["score_mass"] for r in small)
        outside = sum(r["baseline"]["outside_mass"] for r in small)
        changed_outside = sum(r["candidate"]["outside_mass"] for r in small)
        retained = lambda active: (sum(r["candidate"]["anchors"] > 0 for r in active)
                                   / len(active) if active else 0.)
        retention = {"all": retained(base_active), "small": retained(small_active)}
        outside_objects = sum(r["baseline"]["outside_anchors"] > 0 for r in small)
        checks = {
            "small_active_at_least_100": len(small_active) >= 100,
            "outside_small_at_least_50": outside_objects >= 50,
            "outside_mass_at_least_one_percent": before > 0 and outside / before >= .01,
            "outside_mass_halved": outside > 0 and changed_outside <= .5 * outside,
            "small_coverage_retained": retention["small"] >= .99,
            "all_coverage_retained": retention["all"] >= .99,
            "small_mass_retained": before > 0 and after >= .9 * before,
        }
        result[group] = {"targets": len(subset), "small_targets": len(small),
                         "baseline_small_active": len(small_active),
                         "baseline_small_outside_objects": outside_objects,
                         "baseline_small_mass": before, "candidate_small_mass": after,
                         "baseline_small_outside_mass": outside,
                         "candidate_small_outside_mass": changed_outside,
                         "baseline_small_outside_fraction": outside / before if before else 0.,
                         "retention": retention, "checks": checks, "passed": all(checks.values())}
    return {"groups": result, "worth_30_epoch_confirmation": all(r["passed"] for r in result.values()),
            "scope": "single frozen checkpoint, augmented train positives, no hidden/val GT; "
                     "mosaic source overlap; assignment diagnostic only, not student performance"}


def verify(check):
    receipt = read(OUT / "receipt.json")
    for path, checksum in receipt["sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"changed source: {path}")
    frozen = read(ROOT / "outputs/p1_small_nwd_pilot30_s0/preflight.json")
    for path, checksum in frozen["sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"changed frozen input: {path}")
    manifest = read(OUT / "cache_manifest.json")
    rows = []
    for path, checksum in manifest.items():
        if digest(path) != checksum:
            raise ValueError(f"changed cache: {path}")
        with np.load(path, allow_pickle=False) as cache:
            rows.extend(replay_batch(cache))
    analysis = summarize(rows)
    if check:
        if analysis != read(OUT / "analysis.json") or rows != read(OUT / "targets.json"):
            raise ValueError("assignment replay differs")
    else:
        write(OUT / "targets.json", rows)
        write(OUT / "analysis.json", analysis)
    print(json.dumps(analysis, indent=2), flush=True)
    return analysis


def collect():
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    import ultralytics.data.augment
    import ultralytics.data.dataset
    import ultralytics.nn.modules.head
    import ultralytics.utils.loss
    from regionssod.training.p1 import P1Dataset, P1DetectionLoss

    if Path.cwd() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("use repository root and pinned yolo environment")
    if OUT.exists():
        raise FileExistsError("existing diagnostic; use --check, never overwrite")
    config = yaml.safe_load((ROOT / "configs/train_p1_10.yaml").read_text(encoding="utf-8"))
    dataset_config = yaml.safe_load((ROOT / config["data"]).read_text(encoding="utf-8"))
    paths = sorted(Path(dataset_config["train"]).read_text(encoding="utf-8").splitlines())
    if len(paths) != 11310:
        raise ValueError("expected full P1 dataset")
    selected = np.random.default_rng(20261031).choice(len(paths), 256, replace=False).tolist()
    frozen_path = ROOT / "outputs/p1_small_nwd_pilot30_s0/preflight.json"
    frozen = read(frozen_path)
    for path, checksum in frozen["sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"changed frozen input: {path}")
    tracked = [Path(__file__), PROTOCOL]
    relative = [p.relative_to(ROOT).as_posix() for p in tracked]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                   capture_output=True, check=True)
    subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *relative], check=True)
    source_paths = [*tracked, WEIGHTS, frozen_path, Path(inspect.getfile(TaskAlignedAssigner)),
                    Path(inspect.getfile(P1DetectionLoss))]
    for module in (
        ultralytics.data.augment, ultralytics.data.dataset, ultralytics.nn.modules.head,
        ultralytics.utils.loss):
        if module.__file__ is None:
            raise ValueError("runtime module has no inspectable source")
        source_paths.append(Path(module.__file__))
    OUT.mkdir()
    (OUT / "raw").mkdir()
    write(OUT / "receipt.json", {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sample_seed": 20261031, "selected_indices_sorted_paths": selected,
        "selected_images": [paths[i] for i in selected], "forward_inputs": 256,
        "optimizer_steps": 0, "sha256": {str(p): digest(p) for p in source_paths}})
    hyp = get_cfg(overrides=config["train"])
    data = P1Dataset(img_path=dataset_config["train"],
                     metadata_path=Path(dataset_config["p1_metadata"]), imgsz=640, batch_size=8,
                     augment=True, hyp=hyp, rect=False, cache=False, single_cls=False, stride=32,
                     pad=0., task="detect", data=dataset_config, fraction=1.)
    indices = {str(Path(p).resolve()): i for i, p in enumerate(data.im_files)}
    checkpoint: Any = YOLO(str(WEIGHTS)).model
    if not isinstance(checkpoint, torch.nn.Module):
        raise ValueError("expected a loaded PyTorch detection checkpoint")
    checkpoint = checkpoint.float().cuda().eval()
    checkpoint.args = get_cfg(overrides=config["train"])
    checkpoint.p1_options = config["p1"]
    checkpoint.model[-1].training = True  # raw head outputs; BN remains in eval mode
    criterion = P1DetectionLoss(checkpoint)
    cache_manifest, image_hashes = {}, []
    for start in range(0, 256, 8):
        samples = []
        for order in range(start, start + 8):
            seed = 20261031 + order
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sample = data[indices[str(Path(paths[selected[order]]).resolve())]]
            image_hashes.append(hashlib.sha256(sample["img"].numpy().tobytes()).hexdigest())
            samples.append(sample)
        batch = data.collate_fn(samples)
        with torch.no_grad():
            predictions = checkpoint(batch["img"].cuda().float() / 255.)
            distributions = predictions["boxes"].permute(0, 2, 1).contiguous()
            scores = predictions["scores"].permute(0, 2, 1).contiguous().sigmoid()
            points, stride = make_anchors(predictions["feats"], checkpoint.model[-1].stride, .5)
            boxes = criterion.bbox_decode(points, distributions) * stride
        labels = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1),
                            batch["bboxes"]), 1)
        targets = criterion.preprocess(labels.cuda(), 8, scale_tensor=torch.tensor(
            [640., 640., 640., 640.], device="cuda")).cpu()
        confidence = torch.zeros(targets.shape[:2])
        for image in range(8):
            weights = batch["pseudo_weights"][batch["batch_idx"] == image]
            confidence[image, :len(weights)] = weights
        cache_path = OUT / "raw" / f"batch_{start // 8:03d}.npz"
        np.savez_compressed(cache_path, scores=scores.cpu().numpy(), boxes=boxes.cpu().numpy(),
                            points=(points * stride).cpu().numpy(), labels=targets[..., :1].numpy(),
                            targets=targets[..., 1:].numpy(),
                            valid=(targets[..., 1:].sum(-1, keepdim=True) > 0).numpy(),
                            confidence=confidence.numpy(), sample_ids=np.arange(start, start + 8))
        cache_manifest[str(cache_path)] = digest(cache_path)
        print(f"Cached {start + 8}/256 augmented inputs; zero training steps", flush=True)
    write(OUT / "cache_manifest.json", cache_manifest)
    write(OUT / "augmented_image_sha256.json", image_hashes)
    verify(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.check:
        verify(True)
    else:
        collect()


if __name__ == "__main__":
    main()
