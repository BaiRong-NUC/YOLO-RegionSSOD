"""Bounded full-image context diagnostic; no training or label export."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from audit_paper1_gaps import DATA, PSEUDO, match_iou, overlaps, size
from regionssod.data.annotations import (
    Tile, YoloBox, clip_box_to_tile, label_path_for_image, read_label_file,
    replace_images_component,
)
from regionssod.pseudo.context import map_context, rectify_categories
from run_p1_scale_seed_pair import ROOT, digest, read, write

CONFIG = ROOT / "configs/context_rectification_diagnostic.yaml"
CODE = (Path(__file__).resolve(), ROOT / "src/regionssod/pseudo/context.py",
        ROOT / "src/regionssod/pseudo/consistency.py",
        ROOT / "src/regionssod/data/annotations.py",
        ROOT / "scripts/audit_paper1_gaps.py", ROOT / "scripts/run_p1_scale_seed_pair.py")


def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_lines(path, rows):
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows),
                    encoding="utf-8")


def qualifies(counts, rule):
    corrected = counts["corrected"]
    before = counts["corrected_correct_before"] / corrected if corrected else None
    after = counts["corrected_correct_after"] / corrected if corrected else None
    checks = {
        "enough_corrected": corrected >= rule["min_corrected"],
        "corrected_quality": after is not None and after >= rule["min_corrected_quality"],
        "corrected_quality_gain": after is not None
        and after - before >= rule["min_corrected_quality_gain"] - 1e-12,
        "unique_matches_gain": counts["matches_after"] - counts["matches_before"]
        >= rule["min_unique_matches_gain"],
        "small_matches_gain": counts["small_matches_after"] - counts["small_matches_before"]
        >= rule["min_small_matches_gain"],
        "small_matches_lost": counts["small_matches_lost"] <= rule["max_small_matches_lost"],
    }
    return {"before_quality": before, "after_quality": after,
            "quality_gain": after - before if after is not None else None,
            "checks": checks, "passed": all(checks.values())}


def verify_receipt(output):
    receipt = read(output / "manifest.json")
    for path, checksum in receipt["input_sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"changed input: {path}")
    for name, checksum in receipt["output_sha256"].items():
        if digest(output / name) != checksum:
            raise ValueError(f"changed output: {name}")
    return receipt


def analyse(selected, records, metadata, original, predictions, source, split, config):
    counts = Counter({name: 0 for name in (
        "tiles", "accepted_tiles", "excluded_tiles", "positives", "gt", "small_gt",
        "corrected", "corrected_correct_before", "corrected_correct_after",
        "matches_before", "matches_after", "matches_gained", "matches_lost",
        "small_matches_before", "small_matches_after",
        "small_matches_gained", "small_matches_lost")})
    transitions, diagnostics = Counter(), []
    if len(predictions) != len(selected):
        raise ValueError("incomplete context predictions")
    for index, predicted in zip(selected, predictions, strict=True):
        record = records[index]
        if (Path(predicted["source_image"]).resolve() != Path(record["source_image"]).resolve()
                or predicted["shape"] != list(reversed(record["source_size"]))):
            raise ValueError("context image or native shape differs")
        if len(record["unlabeled_tiles"]) != 1:
            raise ValueError("expected one unlabeled tile per source")
        counts["tiles"] += 1
        entry = record["unlabeled_tiles"][0]
        image, tile = entry["image"], Tile(*entry["tile"])
        if image not in metadata:
            counts["excluded_tiles"] += 1
            continue
        counts["accepted_tiles"] += 1
        positive = [p for p in original[image]["candidates"] if p["state"] == "positive"]
        boxes = read_label_file(
            replace_images_component(Path(metadata[image]["image"])).with_suffix(".txt"))
        if (positive != metadata[image]["positive"] or not positive
                or len(boxes) != len(positive)
                or original[image]["shape"] != [tile.height, tile.width]):
            raise ValueError("saved training positives differ")
        for p, b in zip(positive, boxes, strict=True):
            if (p["class_id"] != b.class_id or size(b, tile) != p["scale"]
                    or not np.allclose(p["xywhn"], [b.x, b.y, b.w, b.h], atol=1e-9, rtol=0)):
                raise ValueError("positive coordinates/order/scale differ")
        context = map_context(predicted["candidates"], tile, record["source_size"],
                              min_visibility=split["min_box_visibility"])
        corrected, changes = rectify_categories(
            positive, context, **{k: config[k] for k in (
                "context_conf", "original_below", "match_iou", "only_scale")})
        # Only now reconstruct hidden GT for offline scoring of the frozen decision.
        gt_path = label_path_for_image(Path(record["source_image"]),
                                      Path(source["images_train"]), Path(source["labels_train"]))
        gt = [g for b in read_label_file(gt_path)
              if (g := clip_box_to_tile(b, tile, *record["source_size"],
                                       split["min_box_visibility"])) is not None]
        if len(gt) != entry["visible_gt_boxes_hidden_from_training"]:
            raise ValueError("hidden GT reconstruction differs")
        after_boxes = [YoloBox(p["class_id"], *p["xywhn"]) for p in corrected]
        before_hits = {g for _, g in match_iou(boxes, gt)}
        after_hits = {g for _, g in match_iou(after_boxes, gt)}
        small = {g for g, b in enumerate(gt) if size(b, tile) == "small"}
        counts.update(
            positives=len(positive), gt=len(gt), small_gt=len(small), corrected=len(changes),
            matches_before=len(before_hits), matches_after=len(after_hits),
            matches_gained=len(after_hits - before_hits), matches_lost=len(before_hits - after_hits),
            small_matches_before=len(before_hits & small), small_matches_after=len(after_hits & small),
            small_matches_gained=len((after_hits - before_hits) & small),
            small_matches_lost=len((before_hits - after_hits) & small))
        iou = overlaps(boxes, gt)
        classes = np.asarray([g.class_id for g in gt])
        for change in changes:
            p = change["positive_index"]
            before = bool(((classes == positive[p]["class_id"]) & (iou[p] >= .5)).any())
            after = bool(((classes == corrected[p]["class_id"]) & (iou[p] >= .5)).any())
            counts["corrected_correct_before"] += before
            counts["corrected_correct_after"] += after
            transitions[f"{positive[p]['class_id']}->{corrected[p]['class_id']}"] += 1
            diagnostics.append({
                "tile_index": index, **change,
                "source_prediction_index": context[change["context_index"]]["source_prediction_index"],
                "original_confidence": positive[p]["confidence"],
                "before_same_class_iou_ge_050": before, "after_same_class_iou_ge_050": after,
            })
    return {"counts": dict(counts), "class_transitions": dict(sorted(transitions.items())),
            "gate": qualifies(counts, config["gate"]),
            "scope": "non-blind hidden-GT diagnostic only; fixed GT-free corrections; no labels exported",
            "new_inference_images": len(selected), "new_training_epochs": 0}, diagnostics


def run(*, confirm=False, check=False):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output"] / ("confirmation" if confirm else "discovery")
    if output.exists() and not check:
        raise FileExistsError(output)
    if not 1 <= config["sample_size"] <= 128:
        raise ValueError("sample exceeds preregistered budget")
    if confirm:
        # Replay discovery, rather than trusting a saved pass flag.
        discovery = run(check=True)
        if not discovery["gate"]["passed"]:
            raise ValueError("discovery did not pass; confirmation forbidden")
    records = jsonlines(DATA / "manifest.jsonl")
    source, split = read(DATA / "source.json"), read(DATA / "summary.json")
    settings = read(PSEUDO / "summary.json")
    original = {r["source_image"]: r for r in jsonlines(PSEUDO / "candidates.jsonl")}
    metadata = {r["source_image"]: r for r in jsonlines(PSEUDO / "metadata.jsonl")}
    if (version("ultralytics") != settings["ultralytics_version"]
            or digest(settings["weights"]) != settings["teacher_sha256"]):
        raise ValueError("teacher/environment differs")
    if settings["teacher_sha256"] != config["teacher_sha256"]:
        raise ValueError("not the preregistered B1 teacher")
    if split["min_box_visibility"] != config["min_visibility"]:
        raise ValueError("visibility differs")
    excluded, exclusion_paths = set(), [ROOT / p for p in config["exclude_caches"]]
    for path in exclusion_paths:
        excluded.update(read(path)["sample_indices"])
    if confirm:
        discovery_output = output.parent / "discovery"
        excluded.update(read(discovery_output / "protocol.json")["sample_indices"])
        exclusion_paths.extend(discovery_output / name for name in (
            "manifest.json", "protocol.json", "analysis.json"))
    if len({r["source_image"] for r in records}) != len(records):
        raise ValueError("duplicate source images in split")
    seed = config["confirmation_seed" if confirm else "discovery_seed"]
    selected = sorted(random.Random(seed).sample(
        [i for i in range(len(records)) if i not in excluded], config["sample_size"]))
    tracked = [CONFIG, ROOT / "docs/paper/context_rectification_analysis.md", *CODE]
    if not check:
        relative = [str(p.relative_to(ROOT)) for p in tracked]
        subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                       cwd=ROOT, check=True, capture_output=True)
        subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *relative], cwd=ROOT, check=True)
    receipt = verify_receipt(output) if check else None
    source_commit = (read(output / "protocol.json")["source_commit"] if check else subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    protocol = {**config, "sample": "confirmation" if confirm else "discovery",
                "sample_indices": selected, "excluded_source_count": len(excluded),
                "source_commit": source_commit, "ultralytics_version": settings["ultralytics_version"]}
    paths = [*tracked, *exclusion_paths, Path(settings["weights"])]
    paths.extend(DATA / n for n in ("manifest.jsonl", "summary.json", "source.json"))
    paths.extend(PSEUDO / n for n in ("candidates.jsonl", "metadata.jsonl", "summary.json"))
    for index in selected:
        record = records[index]
        paths.extend((Path(record["source_image"]), label_path_for_image(
            Path(record["source_image"]), Path(source["images_train"]), Path(source["labels_train"]))))
        image = record["unlabeled_tiles"][0]["image"]
        if image in metadata:
            paths.append(replace_images_component(Path(metadata[image]["image"])).with_suffix(".txt"))
    inputs = {str(p.resolve()): digest(p) for p in paths}
    if check:
        if inputs != receipt["input_sha256"] or protocol != read(output / "protocol.json"):
            raise ValueError("input scope or protocol differs")
    else:
        from ultralytics import YOLO

        output.mkdir(parents=True)
        write(output / "protocol.json", protocol)
        images = [records[i]["source_image"] for i in selected]
        image_list = output / "images.txt"
        image_list.write_text("\n".join(images) + "\n", encoding="utf-8")
        model = YOLO(settings["weights"])
        start = time.perf_counter()
        predictions = []
        for i, result in enumerate(model.predict(
                source=str(image_list), stream=True, imgsz=config["imgsz"], conf=config["conf"],
                iou=config["nms_iou"], max_det=config["max_det"], batch=config["batch"],
                device="0", verbose=False)):
            if i >= len(images) or Path(result.path).resolve() != Path(images[i]).resolve():
                raise ValueError("prediction order differs")
            predictions.append({
                "source_image": images[i], "shape": list(result.orig_shape),
                "candidates": [
                    {"class_id": int(c), "confidence": float(s), "xywhn": b.tolist()}
                    for c, s, b in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                       result.boxes.xywhn.cpu().numpy(), strict=True)]})
        if len(predictions) != config["sample_size"]:
            raise ValueError("incomplete predictions")
        write_lines(output / "context.jsonl", predictions)
        write(output / "costs.json", {
            "inference_seconds_including_warmup": time.perf_counter() - start,
            "inference_images": len(images), "training_epochs": 0})
    result, diagnostics = analyse(selected, records, metadata, original,
                                  jsonlines(output / "context.jsonl"), source, split, config)
    if check:
        if (result != read(output / "analysis.json")
                or diagnostics != jsonlines(output / "diagnostics.jsonl")):
            raise ValueError("replayed decisions/statistics differ")
        print(f"Verified {len(inputs)} inputs, {len(selected)} source images, "
              f"{len(diagnostics)} corrections", flush=True)
    else:
        write(output / "analysis.json", result)
        write_lines(output / "diagnostics.jsonl", diagnostics)
        if any(digest(p) != expected for p, expected in inputs.items()):
            raise ValueError("input changed during diagnostic")
        write(output / "manifest.json", {
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "input_sha256": inputs,
            "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    run(confirm=args.confirm, check=args.check)


if __name__ == "__main__":
    main()
