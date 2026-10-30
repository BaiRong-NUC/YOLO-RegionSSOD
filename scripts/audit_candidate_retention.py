"""Bounded NMS/max-det audit, with replay and no training-label exports."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np

from audit_paper1_gaps import DATA, PSEUDO, match_iou, overlaps, size
from regionssod.data.annotations import (
    Tile, YoloBox, clip_box_to_tile, label_path_for_image, read_label_file,
    replace_images_component,
)
from regionssod.pseudo.scale import ScalePolicy
from run_p1_scale_seed_pair import ROOT, digest, read

OUT = ROOT / "outputs/candidate_retention_128_20261030_v2"
PRIOR = ROOT / "outputs/candidate_retention_128_20261030"
FLIP = ROOT / "outputs/cross_view_flip_20261030/analysis.json"
SEED, SAMPLE = 20261030, 128
ARMS = {
    "baseline": {"iou": .7, "max_det": 500},
    "nms090": {"iou": .9, "max_det": 500},
    "maxdet1000": {"iou": .7, "max_det": 1000},
}
# An exploratory continuation rule, not a statistical or paper acceptance gate.
GATE = {
    "min_added_boxes": 20,
    "min_added_tp_fraction": .5,
    "min_added_small_tp": 10,
    "min_small_coverage_gain": .01,
    "max_locked_precision_drop": .01,
}


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def boxes(items: list[dict]) -> list[YoloBox]:
    return [YoloBox(item["class_id"], *item["xywhn"]) for item in items]


def novel_indices(original: list[YoloBox], proposed: list[YoloBox],
                  threshold: float = .5) -> list[int]:
    """Exclude same-class overlap with any retained baseline box, without GT."""
    iou = overlaps(proposed, original)
    same = (np.asarray([p.class_id for p in proposed])[:, None]
            == np.asarray([p.class_id for p in original])[None, :])
    return np.flatnonzero(~(same & (iou >= threshold)).any(axis=1)).tolist()


def newly_retained(original: list[YoloBox], proposed: list[YoloBox]) -> list[int]:
    """Compare prediction identities, preserving nearby distinct proposals."""
    old = np.asarray([(b.x, b.y, b.w, b.h) for b in original]).reshape(-1, 4)
    new = np.asarray([(b.x, b.y, b.w, b.h) for b in proposed]).reshape(-1, 4)
    same = (np.asarray([p.class_id for p in proposed])[:, None]
            == np.asarray([p.class_id for p in original])[None, :])
    identical = (np.abs(new[:, None, :] - old[None, :, :]) <= 1e-5).all(axis=2)
    return np.flatnonzero(~(same & identical).any(axis=1)).tolist()


def quality(predictions: list[YoloBox], gt: list[YoloBox], tile: Tile) -> Counter:
    pairs = match_iou(predictions, gt)
    result = Counter(tiles=1, predictions=len(predictions), gt=len(gt), tp=len(pairs))
    for scale in ("small", "medium", "large"):
        result[f"{scale}_gt"] = sum(size(g, tile) == scale for g in gt)
        result[f"{scale}_tp"] = sum(size(gt[g], tile) == scale for _, g in pairs)
    return result


def qualifies(counts: dict) -> bool:
    added, gt, positives = counts["added"], counts["small_gt"], counts["positives"]
    if not added or not gt or not positives:
        return False
    return (
        added >= GATE["min_added_boxes"]
        and counts["added_tp"] / added >= GATE["min_added_tp_fraction"]
        and counts["added_small_tp"] >= GATE["min_added_small_tp"]
        and counts["added_small_tp"] / gt >= GATE["min_small_coverage_gain"]
        and ((counts["baseline_tp"] + counts["added_tp"]) / (positives + added)
             >= counts["baseline_tp"] / positives - GATE["max_locked_precision_drop"])
    )


def selection() -> tuple[list[dict], dict, dict[str, str]]:
    settings = read(PSEUDO / "summary.json")
    records = [json.loads(line) for line in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    excluded = set(read(FLIP)["sample_indices"])
    selected = sorted(random.Random(SEED).sample(
        [i for i in range(len(records)) if i not in excluded], SAMPLE))
    records = [records[i] for i in selected]
    paths = [DATA / "manifest.jsonl", DATA / "source.json", DATA / "summary.json",
             PSEUDO / "candidates.jsonl", PSEUDO / "metadata.jsonl",
             PSEUDO / "summary.json", Path(settings["weights"]), FLIP,
             Path(__file__).resolve(), ROOT / "scripts/audit_paper1_gaps.py",
             ROOT / "src/regionssod/data/annotations.py", ROOT / "src/regionssod/pseudo/scale.py"]
    source = read(DATA / "source.json")
    metadata = {r["source_image"]: r for r in map(
        json.loads, (PSEUDO / "metadata.jsonl").read_text(encoding="utf-8").splitlines())}
    for record in records:
        if len(record["unlabeled_tiles"]) != 1:
            raise ValueError("This audit requires one unlabeled tile per source")
        image = record["unlabeled_tiles"][0]["image"]
        paths.append(Path(image))
        paths.append(label_path_for_image(Path(record["source_image"]),
                     Path(source["images_train"]), Path(source["labels_train"])))
        if image in metadata:
            paths.append(replace_images_component(Path(metadata[image]["image"])).with_suffix(".txt"))
    hashes = {str(p.resolve()): digest(p) for p in paths}
    if digest(settings["weights"]) != settings["teacher_sha256"]:
        raise ValueError("Frozen teacher changed")
    return records, {
        "sample_seed": SEED, "sample_size": SAMPLE, "sample_indices": selected,
        "excluded_flip_indices": sorted(excluded), "arms": ARMS, "gate": GATE,
        "teacher_sha256": settings["teacher_sha256"],
        "imgsz": 960, "conf": .05, "batch": 8, "device": "0",
        "policy": settings["policy"], "ultralytics_version": settings["ultralytics_version"],
        "candidate_identity": "same class and maximum normalized-coordinate difference <=1e-5",
        "increment_rule": "newly retained P1-positive boxes; reject same-class IoU>=0.70 against original positives",
    }, hashes


def infer(records: list[dict], protocol: dict) -> dict:
    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != protocol["ultralytics_version"]:
        raise ValueError("Ultralytics version differs")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for this bounded audit")
    images = [r["unlabeled_tiles"][0]["image"] for r in records]
    image_list = OUT / "prediction_images.txt"
    image_list.write_text("\n".join(images) + "\n", encoding="utf-8")
    model = YOLO(read(PSEUDO / "summary.json")["weights"])
    policy = ScalePolicy(**protocol["policy"])
    warmup = time.perf_counter()
    model.predict(source=images[0], imgsz=960, conf=.05, device="0", verbose=False)
    torch.cuda.synchronize()
    costs = {"warmup_seconds": time.perf_counter() - warmup}
    for name, params in ARMS.items():
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        rows = []
        for index, result in enumerate(model.predict(
                source=str(image_list), stream=True, imgsz=960, conf=.05,
                batch=8, device="0", verbose=False, **params)):
            if index >= len(images) or Path(result.path).resolve() != Path(images[index]).resolve():
                raise ValueError("Unexpected prediction ordering")
            selected = policy.select(result.boxes.cls.cpu().numpy(),
                                     result.boxes.conf.cpu().numpy(),
                                     result.boxes.xywhn.cpu().numpy(), result.orig_shape)
            rows.append({"source_image": images[index], "shape": list(result.orig_shape),
                         "candidates": selected})
        torch.cuda.synchronize()
        if len(rows) != SAMPLE:
            raise ValueError("Incomplete inference")
        costs[name] = {"seconds": time.perf_counter() - start,
                       "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}
        (OUT / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"{name}: {len(rows)} tiles, {costs[name]['seconds']:.3f} seconds", flush=True)
    return costs


def analyse(records: list[dict]) -> dict:
    saved = {r["source_image"]: r for r in map(json.loads, (PSEUDO / "candidates.jsonl").read_text(encoding="utf-8").splitlines())}
    metadata = {r["source_image"]: r for r in map(json.loads, (PSEUDO / "metadata.jsonl").read_text(encoding="utf-8").splitlines())}
    predictions = {name: {r["source_image"]: r for r in map(
        json.loads, (OUT / f"{name}.jsonl").read_text(encoding="utf-8").splitlines())} for name in ARMS}
    expected = {r["unlabeled_tiles"][0]["image"] for r in records}
    if any(set(rows) != expected for rows in predictions.values()):
        raise ValueError("Saved prediction scope differs")
    source, split = read(DATA / "source.json"), read(DATA / "summary.json")
    totals: dict[str, Counter] = {}
    reproduction = Counter(tiles=0, equivalent_tiles=0)
    for record in records:
        entry = record["unlabeled_tiles"][0]
        image, tile = entry["image"], Tile(*entry["tile"])
        label = label_path_for_image(Path(record["source_image"]),
                                    Path(source["images_train"]), Path(source["labels_train"]))
        gt = [g for b in read_label_file(label)
              if (g := clip_box_to_tile(b, tile, *record["source_size"],
                                       split["min_box_visibility"])) is not None]
        if len(gt) != entry["visible_gt_boxes_hidden_from_training"]:
            raise ValueError("Hidden GT scope differs")
        positive = [] if image not in metadata else read_label_file(
            replace_images_component(Path(metadata[image]["image"])).with_suffix(".txt"))
        locked = {g for _, g in match_iou(positive, gt)}
        residual = [g for i, g in enumerate(gt) if i not in locked]
        fresh = predictions["baseline"][image]["candidates"]
        old = saved[image]["candidates"]
        equivalent = len(fresh) == len(old) and all(
            a["class_id"] == b["class_id"] and a["state"] == b["state"]
            and abs(a["confidence"] - b["confidence"]) <= 1e-4
            and np.allclose(a["xywhn"], b["xywhn"], atol=1e-4, rtol=0)
            for a, b in zip(fresh, old))
        reproduction.update(tiles=1, equivalent_tiles=int(equivalent))
        scope = "kept" if image in metadata else "excluded"
        for name, rows in predictions.items():
            row = rows[image]
            if row["shape"] != [tile.height, tile.width]:
                raise ValueError("Native tile shape differs")
            items = row["candidates"]
            total = totals.setdefault(f"{scope}/{name}", Counter())
            total.update(quality(boxes(items), gt, tile))
            total["at_max_det"] += int(len(items) == ARMS[name]["max_det"])
            total["positive_boxes"] += sum(c["state"] == "positive" for c in items)
            total["positive_tp"] += len(match_iou(
                boxes([c for c in items if c["state"] == "positive"]), gt))
            total["max_candidates_in_tile"] = max(total["max_candidates_in_tile"], len(items))
            if name == "baseline":
                continue
            retained = boxes(fresh)
            novel = novel_indices(retained, boxes(items))
            total["novel_candidates"] += len(novel)
            new_indices = newly_retained(retained, boxes(items))
            total["newly_retained"] += len(new_indices)
            positive_novel = [boxes(items)[i] for i in new_indices if items[i]["state"] == "positive"]
            # Fixed GT-free screening: only new P1-positive boxes with no overlap
            # against any original saved positive. No hidden labels select boxes.
            added = [positive_novel[i] for i in novel_indices(positive, positive_novel, .7)]
            pairs = match_iou(added, residual)
            raw_pairs = match_iou(positive_novel, residual)
            increment = totals.setdefault(f"{scope}/{name}/positive_increment", Counter())
            increment.update(tiles=1, positives=len(positive), baseline_tp=len(locked),
                             small_gt=sum(size(g, tile) == "small" for g in gt),
                             raw_added=len(positive_novel), raw_added_tp=len(raw_pairs),
                             raw_added_small_tp=sum(size(residual[g], tile) == "small" for _, g in raw_pairs),
                             added=len(added), added_tp=len(pairs),
                             added_small_tp=sum(size(residual[g], tile) == "small" for _, g in pairs))
    gate = {name: qualifies(totals[f"kept/{name}/positive_increment"])
            for name in ARMS if name != "baseline"}
    reproduced = reproduction["equivalent_tiles"] == SAMPLE
    return {"counts": {name: dict(c) for name, c in totals.items()},
            "baseline_reproduction": dict(reproduction),
            "gate_by_arm": gate, "training_candidate": reproduced and any(gate.values()),
            "scope": "all 128 tiles; existing kept/excluded tiles reported separately",
            "limitations": [
                "Hidden GT is diagnostic only; no training labels exported.",
                "All-floor candidate coverage is not accepted-pseudo-label quality.",
                "IoU>=0.50 descending-IoU matching over all GT sizes jointly.",
                "Increment locks original positive GT assignments; not standard precision.",
                "Overlap rejection may discard distinct crowded objects; it is conservative.",
                "NMS/max-det effects at conf>=0.05 only; no claim about below-floor predictions.",
                "Single sample/teacher/split; not student mAP or official AP_small.",
            ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify hashes and replay without GPU inference")
    parser.add_argument("--reuse-inference", action="store_true",
                        help="Freeze a refined analysis of the original audit predictions")
    args = parser.parse_args()
    if args.check and args.reuse_inference:
        parser.error("--check and --reuse-inference are mutually exclusive")
    records, protocol, inputs = selection()
    if args.reuse_inference or (args.check and (OUT / "inference_provenance.json").exists()):
        original = read(PRIOR / "manifest.json")
        for filename, expected in original["input_sha256"].items():
            if Path(filename).resolve() != Path(__file__).resolve() and digest(filename) != expected:
                raise ValueError(f"Original inference input changed: {filename}")
        if digest(PRIOR / "source_snapshot.py") != original["input_sha256"][str(Path(__file__).resolve())]:
            raise ValueError("Original inference source snapshot differs")
        for name, expected in original["output_sha256"].items():
            if digest(PRIOR / name) != expected:
                raise ValueError(f"Original inference output changed: {name}")
        old_protocol = read(PRIOR / "protocol.json")
        if any(protocol[k] != value for k, value in old_protocol.items()):
            raise ValueError("Original inference settings or sample changed")
        for path in (PRIOR / "manifest.json", PRIOR / "source_snapshot.py"):
            inputs[str(path)] = digest(path)
    if args.check:
        receipt = read(OUT / "manifest.json")
        if inputs != receipt["input_sha256"] or protocol != read(OUT / "protocol.json"):
            raise ValueError("Frozen inputs or audit protocol changed")
        for name, expected in receipt["output_sha256"].items():
            if digest(OUT / name) != expected:
                raise ValueError(f"Audit output changed: {name}")
        analysis = analyse(records)
        if analysis != read(OUT / "analysis.json"):
            raise ValueError("Offline replay differs")
        print(f"Verified {len(inputs)} input hashes and complete {SAMPLE}-tile replay")
    else:
        if OUT.exists():
            raise FileExistsError("Refusing to overwrite frozen audit")
        OUT.mkdir(parents=True)
        write(OUT / "protocol.json", protocol)
        write(OUT / "input_sha256.json", inputs)
        if args.reuse_inference:
            for name in [f"{arm}.jsonl" for arm in ARMS] + ["prediction_images.txt"]:
                shutil.copyfile(PRIOR / name, OUT / name)
            write(OUT / "inference_provenance.json", {
                "original_manifest": str(PRIOR / "manifest.json"),
                "sha256": digest(PRIOR / "manifest.json"),
                "analysis_refinement": "Retain overlapping proposals for raw matching; separate fixed positive-only overlap rejection. No new inference.",
            })
            costs = read(PRIOR / "costs.json")
        else:
            costs = infer(records, protocol)
        write(OUT / "costs.json", costs)
        analysis = analyse(records)
        write(OUT / "analysis.json", analysis)
        if any(digest(path) != expected for path, expected in inputs.items()):
            raise ValueError("Input changed during audit")
        write(OUT / "manifest.json", {
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "input_sha256": inputs,
            "output_sha256": {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        })
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
