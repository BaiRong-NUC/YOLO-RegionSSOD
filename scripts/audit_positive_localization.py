"""Replay cached views to diagnose existing pseudo-box localization; no inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from audit_paper1_gaps import DATA, PSEUDO, match_iou, overlaps, size
from regionssod.data.annotations import (
    Tile, YoloBox, clip_box_to_tile, label_path_for_image, read_label_file,
    replace_images_component,
)
from regionssod.pseudo.localization import localization_stability
from run_p1_scale_seed_pair import ROOT, digest, read, write

CONFIG = ROOT / "configs/positive_localization_diagnostic.yaml"


def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def describe(rows, quality_iou):
    if not rows:
        return {"boxes": 0}
    return {
        "boxes": len(rows),
        "mean_same_class_gt_iou": float(np.mean([r["gt_same_iou"] for r in rows])),
        "same_class_iou_ge_050_fraction": float(np.mean([r["gt_same_iou"] >= .5 for r in rows])),
        "same_class_iou_ge_quality_fraction": float(np.mean(
            [r["gt_same_iou"] >= quality_iou for r in rows])),
        "unique_match_050_fraction": float(np.mean([r["unique_match_050"] for r in rows])),
        "other_class_overlap_fraction": float(np.mean([r["other_class_overlap"] for r in rows])),
        "duplicate_competition_fraction": float(np.mean([r["duplicate_competition"] for r in rows])),
        "ambiguous_view_fraction": float(np.mean([r["ambiguous"] for r in rows])),
    }


def controlled_gap(rows, config, *, by_class=False):
    rule = config["class_confidence_control" if by_class else "confidence_control"]
    strata = defaultdict(lambda: {"stable": [], "unstable": []})
    edges = config["confidence_edges"]
    for row in rows:
        if row["group"] not in ("stable", "unstable"):
            continue
        band = int(np.searchsorted(edges, row["confidence"], side="right")-1)
        if not 0 <= band < len(edges)-1:
            raise ValueError("confidence outside fixed bins")
        key = f"{row['class_id']}:{band}" if by_class else str(band)
        strata[key][row["group"]].append(row)
    details, weights, gaps = {}, [], []
    comparable = Counter(stable=0, unstable=0)
    for key, groups in sorted(strata.items()):
        counts = {g: len(v) for g, v in groups.items()}
        accepted = min(counts.values()) >= rule["min_per_group_per_stratum"]
        entry = {"counts": counts, "comparable": accepted}
        if accepted:
            fraction = {g: float(np.mean([r["gt_same_iou"] >= config["quality_iou"]
                                          for r in v])) for g, v in groups.items()}
            gap = fraction["stable"]-fraction["unstable"]
            weights.append(min(counts.values()))
            gaps.append(gap)
            comparable.update(counts)
            entry.update(quality_fractions=fraction, stable_minus_unstable=gap)
        details[key] = entry
    delta = float(np.average(gaps, weights=weights)) if weights else None
    fraction = sum(comparable.values())/len(rows) if rows else 0
    checks = {
        "enough_strata": len(weights) >= rule["min_strata"],
        "enough_each_group": min(comparable.values()) >= rule["min_comparable_per_group"],
        "enough_coverage": fraction >= rule["min_comparable_fraction"],
        "quality_gap": delta is not None and delta >= rule["min_quality_fraction_gap"],
    }
    return {"strata": details, "comparable_counts": dict(comparable),
            "comparable_fraction": fraction, "weighted_quality_gap": delta,
            "checks": checks, "passed": all(checks.values())}


def summarize(rows, config):
    groups = {}
    for scale in ("all", "small", "medium", "large"):
        part = rows if scale == "all" else [r for r in rows if r["scale"] == scale]
        groups[scale] = {
            g: describe(part if g == "all" else [r for r in part if r["group"] == g],
                        config["quality_iou"])
            for g in ("all", "stable", "middle", "unstable", "unsupported")}
    primary = [r for r in rows if r["scale"] == config["primary_size"]]
    control = controlled_gap(primary, config)
    class_control = controlled_gap(primary, config, by_class=True)
    return {"groups": groups, "confidence_control": control,
            "class_confidence_control": class_control,
            "diagnostic_candidate": control["passed"] and class_control["passed"]}


def analyse(config):
    inputs = {}

    def freeze(path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        checksum = inputs.get(key)
        if checksum is None:
            checksum = digest(path)
            inputs[key] = checksum
        if expected is not None and checksum != expected:
            raise ValueError(f"changed input: {path}")

    for path in (CONFIG, Path(__file__), ROOT / "src/regionssod/pseudo/localization.py",
                 ROOT / "src/regionssod/pseudo/consistency.py",
                 ROOT / "scripts/audit_paper1_gaps.py",
                 ROOT / "src/regionssod/data/annotations.py"):
        freeze(path)
    records = jsonlines(DATA / "manifest.jsonl")
    original = {r["source_image"]: r for r in jsonlines(PSEUDO / "candidates.jsonl")}
    metadata = {r["source_image"]: r for r in jsonlines(PSEUDO / "metadata.jsonl")}
    source, split = read(DATA / "source.json"), read(DATA / "summary.json")
    samples, indices = {}, {}
    all_rows = []
    for name, key in (("discovery", "discovery_cache"), ("confirmation", "confirmation_cache")):
        cache = ROOT / config[key]
        receipt = read(cache / "manifest.json")
        freeze(cache / "manifest.json")
        for path, checksum in receipt["input_sha256"].items():
            freeze(path, checksum)
        for filename, checksum in receipt["output_sha256"].items():
            freeze(cache / filename, checksum)
        protocol = read(cache / "protocol.json")
        if (protocol["original_imgsz"] != config["original_imgsz"]
                or protocol["view_imgsz"] != config["view_imgsz"]
                or protocol["conf"] != config["min_view_conf"]
                or protocol["iou"] != .7 or protocol["max_det"] != 500):
            raise ValueError("cached view protocol differs")
        indices[name] = protocol["sample_indices"]
        if len(indices[name]) != 256 or len(set(indices[name])) != 256:
            raise ValueError("unexpected sample")
        view_records = jsonlines(cache / "view.jsonl")
        expected_images = [records[i]["unlabeled_tiles"][0]["image"] for i in indices[name]]
        if [r["source_image"] for r in view_records] != expected_images:
            raise ValueError("cached view ordering differs")
        view = {r["source_image"]: r["candidates"] for r in view_records}
        rows, excluded = [], 0
        for index in indices[name]:
            record = records[index]
            entry = record["unlabeled_tiles"][0]
            image = entry["image"]
            if image not in metadata:
                excluded += 1
                continue
            tile = Tile(*entry["tile"])
            if original[image]["shape"] != [tile.height, tile.width]:
                raise ValueError("tile shape differs")
            positive = [p for p in original[image]["candidates"] if p["state"] == "positive"]
            if positive != metadata[image]["positive"]:
                raise ValueError("positive metadata differs")
            label_path = replace_images_component(Path(metadata[image]["image"])).with_suffix(".txt")
            freeze(label_path)
            boxes = read_label_file(label_path)
            if len(boxes) != len(positive):
                raise ValueError("positive count differs")
            for item, box in zip(positive, boxes, strict=True):
                if item["class_id"] != box.class_id or not np.allclose(
                        item["xywhn"], [box.x, box.y, box.w, box.h], atol=1e-9, rtol=0):
                    raise ValueError("saved training box differs")
            features = localization_stability(
                positive, view[image], **{k: config[k] for k in (
                    "min_view_conf", "min_pair_iou", "unstable_below", "stable_at")})
            gt_path = label_path_for_image(Path(record["source_image"]),
                                          Path(source["images_train"]), Path(source["labels_train"]))
            freeze(gt_path)
            gt = [g for b in read_label_file(gt_path)
                  if (g := clip_box_to_tile(b, tile, *record["source_size"],
                                           split["min_box_visibility"])) is not None]
            if len(gt) != entry["visible_gt_boxes_hidden_from_training"]:
                raise ValueError("hidden GT reconstruction differs")
            overlap = overlaps(boxes, gt)
            same = (np.asarray([b.class_id for b in boxes])[:, None]
                    == np.asarray([g.class_id for g in gt])[None, :])
            same_overlap = np.where(same, overlap, 0)
            best = same_overlap.argmax(axis=1) if gt else np.full(len(boxes), -1)
            same_iou = same_overlap.max(axis=1, initial=0)
            any_iou = overlap.max(axis=1, initial=0)
            duplicates = Counter(int(best[p]) for p in range(len(boxes)) if same_iou[p] >= .5)
            hits = {p for p, _ in match_iou(boxes, gt)}
            for p, (item, box, feature) in enumerate(zip(positive, boxes, features, strict=True)):
                rows.append({
                    "sample": name, "tile_index": index, "positive_index": p,
                    "class_id": item["class_id"], "confidence": item["confidence"],
                    "scale": size(box, tile), **feature, "gt_same_iou": float(same_iou[p]),
                    "unique_match_050": p in hits,
                    "other_class_overlap": bool(any_iou[p] >= .5 and same_iou[p] < .5),
                    "duplicate_competition": bool(same_iou[p] >= .5 and duplicates[int(best[p])] > 1),
                })
        samples[name] = {"sample_indices": indices[name], "tiles": 256,
                         "accepted_tiles": 256-excluded, "excluded_tiles": excluded,
                         **summarize(rows, config)}
        all_rows.extend(rows)
    if set(indices["discovery"]) & set(indices["confirmation"]):
        raise ValueError("discovery and confirmation overlap")
    return {"samples": samples,
            "diagnostic_candidate": all(s["diagnostic_candidate"] for s in samples.values()),
            "scope": "cached non-blind disjoint samples; hidden GT diagnostic only; no training labels exported",
            "new_inference_images": 0, "new_training_epochs": 0}, all_rows, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    if output.exists() and not args.check:
        raise FileExistsError(output)
    result, rows, inputs = analyse(config)
    if args.check:
        receipt = read(output / "manifest.json")
        if inputs != receipt["input_sha256"]:
            raise ValueError("diagnostic inputs changed")
        for filename, checksum in receipt["output_sha256"].items():
            if digest(output / filename) != checksum:
                raise ValueError(f"changed output: {filename}")
        if result != read(output / "analysis.json") or rows != jsonlines(output / "diagnostics.jsonl"):
            raise ValueError("replayed statistics differ")
        print(f"Verified {len(inputs)} inputs and {len(rows)} diagnostic positives")
    else:
        output.mkdir(parents=True)
        write(output / "protocol.json", config)
        write(output / "analysis.json", result)
        (output / "diagnostics.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False)+"\n" for r in rows), encoding="utf-8")
        if any(digest(path) != checksum for path, checksum in inputs.items()):
            raise ValueError("input changed during diagnostic")
        write(output / "manifest.json", {
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "input_sha256": inputs,
            "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        })
    for name, sample in result["samples"].items():
        print(name, "accepted tiles:", sample["accepted_tiles"],
              "small groups:", {g: v["boxes"] for g, v in sample["groups"]["small"].items()})
        for key in ("confidence_control", "class_confidence_control"):
            control = sample[key]
            print(key, control["weighted_quality_gap"], control["checks"])
    print("Diagnostic candidate:", result["diagnostic_candidate"])


if __name__ == "__main__":
    main()
