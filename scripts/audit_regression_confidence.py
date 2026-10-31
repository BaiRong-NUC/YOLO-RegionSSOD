"""Check fixed confidence regression weighting using cached diagnostic quality only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/positive_localization_20261030"
OUT = ROOT / "outputs/regression_confidence_20261031"
PROTOCOL = ROOT / "docs/paper/regression_confidence_protocol.md"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(rows):
    if not rows:
        return {"boxes": 0, "candidate": False}
    weights = np.array([row["confidence"] for row in rows], dtype=float)
    iou = np.array([row["gt_same_iou"] for row in rows], dtype=float)
    unique = np.array([row["unique_match_050"] for row in rows], dtype=float)
    if (not np.isfinite(weights).all() or not np.isfinite(iou).all()
            or (weights <= 0).any() or (weights > 1).any()
            or (iou < 0).any() or (iou > 1).any()):
        raise ValueError("invalid cached confidence or geometric quality")
    baseline = {"mean_iou": float(iou.mean()), "quality_fraction": float((iou >= .75).mean()),
                "unique_fraction": float(unique.mean())}
    weighted = {"mean_iou": float(np.average(iou, weights=weights)),
                "quality_fraction": float(np.average(iou >= .75, weights=weights)),
                "unique_fraction": float(np.average(unique, weights=weights))}
    delta = {key: weighted[key] - baseline[key] for key in baseline}
    classes = []
    for class_id in range(10):
        selected = np.array([row["class_id"] == class_id for row in rows])
        if selected.sum() >= 30:
            gap = float(np.average(iou[selected], weights=weights[selected]) - iou[selected].mean())
            classes.append({"class_id": class_id, "boxes": int(selected.sum()), "mean_iou_gap": gap})
    controlled = (sum(r["boxes"] * r["mean_iou_gap"] for r in classes)
                  / sum(r["boxes"] for r in classes)) if classes else None
    effective = float(weights.sum()**2 / (weights**2).sum())
    checks = {"at_least_300": len(rows) >= 300, "effective_fraction": effective >= .75*len(rows),
              "mean_iou_gain": delta["mean_iou"] >= .02,
              "quality_fraction_gain": delta["quality_fraction"] >= .02,
              "unique_match_not_lower": delta["unique_fraction"] >= 0,
              "enough_classes": len(classes) >= 3,
              "class_controlled_gain": controlled is not None and controlled >= .01}
    return {"boxes": len(rows), "mean_confidence": float(weights.mean()),
            "effective_samples": effective, "baseline": baseline, "weighted": weighted,
            "delta": delta, "class_control": classes, "class_controlled_iou_gap": controlled,
            "checks": checks, "candidate": all(checks.values())}


def compute(rows):
    samples = {}
    seen = {}
    for name in ("discovery", "confirmation"):
        group = [row for row in rows if row["sample"] == name]
        seen[name] = {row["tile_index"] for row in group}
        samples[name] = {size: describe(group if size == "all" else
                                      [row for row in group if row["scale"] == size])
                         for size in ("all", "small", "medium", "large")}
    if seen["discovery"] & seen["confirmation"]:
        raise ValueError("cached base-tile groups overlap")
    return {"samples": samples, "worth_30_epoch_confirmation": all(
        samples[name]["small"]["candidate"] for name in samples),
        "new_inference_images": 0, "new_training_epochs": 0,
        "scope": "reused non-blind cached pseudo-box quality; hidden GT diagnostic only; "
                 "weight equals original confidence without fitting or label export"}


def main(check=False):
    hashes = read(SOURCE / "manifest.json")["input_sha256"].copy()
    for path in (SOURCE / "manifest.json", SOURCE / "diagnostics.jsonl", PROTOCOL, Path(__file__)):
        hashes[str(path)] = digest(path)
    for path, checksum in hashes.items():
        if digest(path) != checksum:
            raise ValueError(f"changed source: {path}")
    if check:
        if read(OUT / "source_sha256.json") != hashes:
            raise ValueError("diagnostic sources differ")
    else:
        for path in (PROTOCOL, Path(__file__)):
            relative = path.relative_to(ROOT).as_posix()
            subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative],
                           capture_output=True, check=True)
            subprocess.run(["git", "diff", "HEAD", "--quiet", "--", relative], check=True)
        OUT.mkdir()
    rows = [json.loads(line) for line in (SOURCE / "diagnostics.jsonl").read_text(
        encoding="utf-8").splitlines()]
    result = compute(rows)
    if check:
        if result != read(OUT / "analysis.json"):
            raise ValueError("confidence quality replay differs")
    else:
        (OUT / "analysis.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        (OUT / "source_sha256.json").write_text(json.dumps(hashes, indent=2)+"\n", encoding="utf-8")
    for name, group in result["samples"].items():
        print(name, json.dumps(group["small"], indent=2), flush=True)
    print("worth_30_epoch_confirmation:", result["worth_30_epoch_confirmation"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args().check)
