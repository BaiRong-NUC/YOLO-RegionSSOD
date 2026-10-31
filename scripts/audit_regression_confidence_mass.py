"""Describe fixed pseudo-box confidence mass without GT, inference or training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/regression_confidence_mass_20261031"
DATA = ROOT / "data/visdrone_area10/dataset_p1.yaml"
PROTOCOL = ROOT / "docs/paper/regression_confidence_mass_protocol.md"
PREFLIGHT = ROOT / "outputs/p1_small_nwd_pilot30_s0/preflight.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(rows, total_count, total_mass):
    confidence = np.array([row["confidence"] for row in rows], dtype=np.float64)
    count = len(rows)
    mass = float(confidence.sum())
    count_share = count / total_count if total_count else 0.
    mass_share = mass / total_mass if total_mass else 0.
    return {
        "boxes": count, "sum_confidence": mass,
        "mean_confidence": float(confidence.mean()) if count else None,
        "confidence_five_numbers": np.quantile(confidence, [0, .25, .5, .75, 1]).tolist() if count else None,
        "effective_samples": float(mass**2 / np.square(confidence).sum()) if count else 0.,
        "count_share": count_share, "confidence_mass_share": mass_share,
        "share_delta_pp": 100 * (mass_share - count_share),
    }


def compute(rows):
    if not rows:
        raise ValueError("expected accepted pseudo boxes")
    for row in rows:
        if (isinstance(row["class_id"], bool) or not isinstance(row["class_id"], int)
                or row["class_id"] not in range(10) or row["scale"] not in ("small", "medium", "large")
                or not np.isfinite(row["confidence"]) or not 0 < row["confidence"] <= 1):
            raise ValueError("invalid fixed pseudo metadata")
    count = len(rows)
    mass = float(np.array([row["confidence"] for row in rows], dtype=np.float64).sum())
    return {
        "all": describe(rows, count, mass),
        "per_scale": {scale: describe([row for row in rows if row["scale"] == scale], count, mass)
                      for scale in ("small", "medium", "large")},
        "per_class": {str(cls): describe([row for row in rows if row["class_id"] == cls], count, mass)
                      for cls in range(10)},
        "per_class_scale": {str(cls): {
            scale: describe([row for row in rows if row["class_id"] == cls and row["scale"] == scale],
                            count, mass) for scale in ("small", "medium", "large")}
            for cls in range(10)},
        "new_inference_images": 0, "new_training_epochs": 0,
        "uses_hidden_or_validation_gt": False,
        "scope": "accepted pseudo boxes in the fixed train list; native-tile scale; "
                 "box-level proxy only, not actual TAL-weighted loss/gradient or student performance",
    }


def main(check=False):
    if Path.cwd() != ROOT:
        raise ValueError("use repository root")
    if not check:
        for path in (PROTOCOL, Path(__file__)):
            relative = path.relative_to(ROOT).as_posix()
            subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative],
                           check=True, capture_output=True)
            subprocess.run(["git", "diff", "HEAD", "--quiet", "--", relative], check=True)
        if OUT.exists():
            raise FileExistsError(OUT)
    data = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    train = Path(data["train"])
    metadata = Path(data["p1_metadata"])
    summary_path = metadata.parent / "summary.json"
    frozen = json.loads(PREFLIGHT.read_text(encoding="utf-8"))["sha256"]
    paths = [DATA, train, metadata, summary_path]
    for path in paths:
        if digest(path) != frozen[str(path)]:
            raise ValueError(f"changed frozen training input: {path}")
    images = train.read_text(encoding="utf-8").splitlines()
    if len(images) != len(set(images)) or len(images) != 11310:
        raise ValueError("expected the unique fixed 11310 training tiles")
    image_set = {str(Path(image).resolve()) for image in images}
    seen = set()
    rows = []
    for line in metadata.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        image = str(Path(row["image"]).resolve())
        if image not in image_set or image in seen or not row["positive"]:
            raise ValueError("positive metadata missing from train, duplicated, or empty")
        seen.add(image)
        rows.extend(row["positive"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(rows) != summary["accepted_boxes"] or len(seen) != summary["accepted_images"]:
        raise ValueError("counts differ from fixed pseudo generation summary")
    result = compute(rows)
    result.update(pseudo_images=len(seen), train_tiles=len(images))
    paths += [PREFLIGHT, PROTOCOL, Path(__file__)]
    hashes = {str(path): digest(path) for path in paths}
    if check:
        if json.loads((OUT / "source_sha256.json").read_text(encoding="utf-8")) != hashes:
            raise ValueError("mass diagnostic sources changed")
        if json.loads((OUT / "analysis.json").read_text(encoding="utf-8")) != result:
            raise ValueError("mass diagnostic replay differs")
    else:
        OUT.mkdir()
        for name, content in (("analysis.json", result), ("source_sha256.json", hashes)):
            (OUT / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all": result["all"], "per_scale": result["per_scale"]}, indent=2), flush=True)
    print("Replayed" if check else "Written", OUT, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args().check)
