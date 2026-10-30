"""Learn weights from open tiles; test frozen maps on disjoint hidden-GT samples."""
from __future__ import annotations

import argparse
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

from audit_paper1_gaps import DATA, PSEUDO, match_iou, size
from regionssod.data.annotations import (
    Tile, YoloBox, clip_box_to_tile, label_path_for_image, read_label_file,
    replace_images_component,
)
from regionssod.pseudo.calibration import (
    calibrated_weights, fit_calibrator, reliability_metrics,
)
from regionssod.pseudo.scale import ScalePolicy
from run_p1_scale_seed_pair import ROOT, digest, read, write

CONFIG = ROOT / "configs/confidence_calibration_diagnostic.yaml"
SOURCES = [
    CONFIG, ROOT / "docs/paper/confidence_calibration_protocol.md", Path(__file__).resolve(),
    ROOT / "src/regionssod/pseudo/calibration.py", ROOT / "src/regionssod/pseudo/scale.py",
    ROOT / "src/regionssod/pseudo/generator.py", ROOT / "src/regionssod/data/annotations.py",
    ROOT / "src/regionssod/data/dataset.py", ROOT / "scripts/audit_paper1_gaps.py",
    ROOT / "scripts/run_p1_scale_seed_pair.py",
]


def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_lines(path, rows):
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


def hashes(paths):
    return {str(Path(p).resolve()): digest(p) for p in paths}


def verify(output):
    receipt = read(output / "manifest.json")
    if hashes(receipt["input_sha256"]) != receipt["input_sha256"]:
        raise ValueError(f"changed inputs: {output}")
    if {name: digest(output / name) for name in receipt["output_sha256"]} != receipt["output_sha256"]:
        raise ValueError(f"changed outputs: {output}")
    return receipt


def save_receipt(output, inputs):
    if hashes(inputs) != inputs:
        raise ValueError("inputs changed during diagnostic")
    write(output / "manifest.json", {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "input_sha256": inputs,
        "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})


def fixed_samples(records, config):
    excluded = set()
    for path in config["exclude_caches"]:
        excluded.update(read(ROOT / path)["sample_indices"])
    result = {}
    for name, count in (("fit", config["fit_size"]), ("discovery", config["sample_size"]),
                        ("confirmation", config["sample_size"])):
        result[name] = sorted(random.Random(config[f"{name}_seed"]).sample(
            [i for i in range(len(records)) if i not in excluded], count))
        excluded.update(result[name])
    return result


def collect_fit(config, records, selected, settings, common, *, check=False):
    output = ROOT / config["output"] / "fit"
    if output.exists() and not check:
        raise FileExistsError(output)
    images = [records[i]["labeled_tile"]["image"] for i in selected]
    labels = [replace_images_component(Path(p)).with_suffix(".txt") for p in images]
    inputs = hashes([*common, *images, *labels])
    if check:
        if verify(output)["input_sha256"] != inputs:
            raise ValueError("fit source scope differs")
    else:
        from ultralytics import YOLO

        output.mkdir(parents=True)
        write(output / "protocol.json", {
            **config, "sample_indices": selected,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "fit_labels": "only existing labeled/labels; no full-source GT"})
        image_list = output / "images.txt"
        image_list.write_text("\n".join(images) + "\n", encoding="utf-8")
        policy = ScalePolicy(**settings["policy"])
        model = YOLO(settings["weights"])
        start = time.perf_counter()
        predictions = []
        for i, result in enumerate(model.predict(
                source=str(image_list), stream=True, imgsz=config["imgsz"], conf=config["conf"],
                iou=config["nms_iou"], max_det=config["max_det"], batch=config["batch"],
                device="0", verbose=False)):
            if i >= len(images) or Path(result.path).resolve() != Path(images[i]).resolve():
                raise ValueError("fit prediction ordering differs")
            predictions.append({"source_image": images[i], "shape": list(result.orig_shape),
                                "candidates": policy.select(
                                    result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                    result.boxes.xywhn.cpu().numpy(), result.orig_shape)})
        if len(predictions) != len(selected):
            raise ValueError("incomplete fit inference")
        write_lines(output / "predictions.jsonl", predictions)
        write(output / "costs.json", {
            "inference_images": len(images), "inference_seconds_including_warmup": time.perf_counter()-start,
            "training_epochs": 0})
    if read(output / "protocol.json")["sample_indices"] != selected:
        raise ValueError("fit indices changed")
    predictions = jsonlines(output / "predictions.jsonl")
    if len(predictions) != len(selected):
        raise ValueError("fit prediction count differs")
    items, targets = [], []
    for index, path, label, predicted in zip(selected, images, labels, predictions, strict=True):
        tile = Tile(*records[index]["labeled_tile"]["tile"])
        if (predicted["source_image"] != path
                or predicted["shape"] != [tile.height, tile.width]):
            raise ValueError("fit native tile differs")
        gt = read_label_file(label)
        if len(gt) != records[index]["labeled_tile"]["labels"]:
            raise ValueError("open-label count differs")
        positive = [p for p in predicted["candidates"] if p["state"] == "positive"]
        matched = {p for p, _ in match_iou([YoloBox(p["class_id"], *p["xywhn"]) for p in positive], gt)}
        items.extend(positive)
        targets.extend(int(i in matched) for i in range(len(positive)))
    model = fit_calibrator(items, targets, **config["calibrator"])
    if check:
        if model != read(output / "model.json"):
            raise ValueError("open-label fit replay differs")
        print(f"Verified fit: {len(inputs)} inputs, {len(selected)} open tiles, {len(items)} positives",
              flush=True)
    else:
        write(output / "model.json", model)
        save_receipt(output, inputs)
    return model


def evaluate(items, targets, model):
    full, support = calibrated_weights(items, model)
    global_weights, _ = calibrated_weights(items, model, conditional=False)
    raw = [p["confidence"] for p in items]
    metrics = {}
    for scale in ("all", "small", "medium", "large"):
        selected = [i for i, p in enumerate(items) if scale == "all" or p["scale"] == scale]
        metrics[scale] = {
            name: reliability_metrics([targets[i] for i in selected], [weights[i] for i in selected])
            for name, weights in (("raw", raw), ("global", global_weights), ("conditional", full))}
    rows = [{"class_id": p["class_id"], "scale": p["scale"], "raw": p["confidence"],
             "global": g, "conditional": f, "conditional_supported": s, "matched": t}
            for p, g, f, s, t in zip(items, global_weights, full, support, targets, strict=True)]
    return {"metrics": metrics, "conditional_coverage": float(np.mean(support)) if support else 0}, rows


def qualifies(result, model, rule):
    metrics = result["metrics"]
    all_raw, all_global, all_full = (metrics["all"][name] for name in ("raw", "global", "conditional"))
    small_raw, small_full = metrics["small"]["raw"], metrics["small"]["conditional"]
    enough = all_raw["positives"] >= rule["min_positives"]
    small_enough = small_raw["positives"] >= rule["min_small_positives"]
    checks = {
        "enough_fit": model["fit_positives"] >= rule["min_fit_positives"],
        "enough_positives": enough, "enough_small": small_enough,
        "conditional_coverage": result["conditional_coverage"] >= rule["min_conditional_coverage"],
        "brier_vs_raw": enough and all_raw["brier"] - all_full["brier"] >= rule["min_brier_gain_vs_raw"],
        "brier_vs_global": enough and all_global["brier"] - all_full["brier"]
        >= rule["min_brier_gain_vs_global"],
        "small_brier_vs_raw": small_enough and small_raw["brier"] - small_full["brier"]
        >= rule["min_small_brier_gain_vs_raw"],
        "error_weight_share": enough and all_full["error_weight_share"] is not None
        and all_raw["error_weight_share"] is not None and all_full["error_weight_share"]
        <= all_raw["error_weight_share"] + rule["max_error_weight_share_increase"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def collect_sample(config, records, selected, model, common, name, *, check=False):
    output = ROOT / config["output"] / name
    if output.exists() and not check:
        raise FileExistsError(output)
    original = {r["source_image"]: r for r in jsonlines(PSEUDO / "candidates.jsonl")}
    metadata = {r["source_image"]: r for r in jsonlines(PSEUDO / "metadata.jsonl")}
    source, split = read(DATA / "source.json"), read(DATA / "summary.json")
    paths = [*common]
    fit = output.parent / "fit"
    paths.extend(fit / n for n in ("model.json", "manifest.json", "protocol.json"))
    for index in selected:
        record = records[index]
        paths.append(label_path_for_image(Path(record["source_image"]),
                                         Path(source["images_train"]), Path(source["labels_train"])))
        image = record["unlabeled_tiles"][0]["image"]
        if image in metadata:
            paths.append(replace_images_component(Path(metadata[image]["image"])).with_suffix(".txt"))
    inputs = hashes(paths)
    if check and verify(output)["input_sha256"] != inputs:
        raise ValueError("diagnostic input scope differs")
    items, targets, indices, excluded = [], [], [], 0
    for index in selected:
        record = records[index]
        entry = record["unlabeled_tiles"][0]
        image, tile = entry["image"], Tile(*entry["tile"])
        if image not in metadata:
            excluded += 1
            continue
        positive = [p for p in original[image]["candidates"] if p["state"] == "positive"]
        boxes = read_label_file(
            replace_images_component(Path(metadata[image]["image"])).with_suffix(".txt"))
        if positive != metadata[image]["positive"] or len(positive) != len(boxes):
            raise ValueError("positive metadata differs")
        for p, box in zip(positive, boxes, strict=True):
            if (p["class_id"] != box.class_id or size(box, tile) != p["scale"]
                    or not np.allclose(p["xywhn"], [box.x, box.y, box.w, box.h], atol=1e-9, rtol=0)):
                raise ValueError("saved training labels differ")
        # Apply the frozen model before reconstructing diagnostic hidden labels.
        calibrated_weights(positive, model)
        gt_path = label_path_for_image(Path(record["source_image"]),
                                      Path(source["images_train"]), Path(source["labels_train"]))
        gt = [g for b in read_label_file(gt_path)
              if (g := clip_box_to_tile(b, tile, *record["source_size"],
                                       split["min_box_visibility"])) is not None]
        if len(gt) != entry["visible_gt_boxes_hidden_from_training"]:
            raise ValueError("hidden GT reconstruction differs")
        matched = {p for p, _ in match_iou(boxes, gt)}
        items.extend(positive)
        targets.extend(int(p in matched) for p in range(len(positive)))
        indices.extend((index, p) for p in range(len(positive)))
    result, rows = evaluate(items, targets, model)
    for row, (index, p) in zip(rows, indices, strict=True):
        row.update(tile_index=index, positive_index=p)
    result.update(tiles=len(selected), accepted_tiles=len(selected)-excluded, excluded_tiles=excluded,
                  scope="frozen open-label calibration; hidden GT diagnostic only; no label export",
                  new_inference_images=0, new_training_epochs=0)
    result["gate"] = qualifies(result, model, config["gate"])
    protocol = {**config, "sample_indices": selected}
    if check:
        if (protocol != read(output / "protocol.json") or result != read(output / "analysis.json")
                or rows != jsonlines(output / "diagnostics.jsonl")):
            raise ValueError("calibration diagnostic replay differs")
        print(f"Verified {name}: {len(inputs)} inputs, {len(rows)} positives", flush=True)
    else:
        output.mkdir(parents=True)
        write(output / "protocol.json", protocol)
        write(output / "analysis.json", result)
        write_lines(output / "diagnostics.jsonl", rows)
        save_receipt(output, inputs)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["fit_size"] != 512 or config["sample_size"] != 256:
        raise ValueError("outside preregistered diagnostic budget")
    if args.confirm and not (ROOT / config["output"] / "discovery/manifest.json").is_file():
        raise ValueError("completed discovery required before confirmation")
    if not args.check:
        relative = [p.relative_to(ROOT).as_posix() for p in SOURCES]
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                                 cwd=ROOT, text=True, capture_output=True)
        if tracked.returncode:
            raise RuntimeError(tracked.stderr)
        subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *relative], cwd=ROOT, check=True)
    settings = read(PSEUDO / "summary.json")
    if (settings["teacher_sha256"] != config["teacher_sha256"]
            or digest(settings["weights"]) != config["teacher_sha256"]
            or version("ultralytics") != settings["ultralytics_version"]):
        raise ValueError("teacher or environment differs")
    records = jsonlines(DATA / "manifest.jsonl")
    if len({r["source_image"] for r in records}) != len(records):
        raise ValueError("duplicate sources")
    samples = fixed_samples(records, config)
    common = [*SOURCES, Path(settings["weights"])]
    common.extend(ROOT / p for p in config["exclude_caches"])
    common.extend(DATA / n for n in ("manifest.jsonl", "source.json", "summary.json"))
    common.extend(PSEUDO / n for n in ("summary.json", "candidates.jsonl", "metadata.jsonl"))
    fit_exists = (ROOT / config["output"] / "fit").exists()
    model = collect_fit(config, records, samples["fit"], settings, common,
                        check=args.check or fit_exists)
    discovery = collect_sample(config, records, samples["discovery"], model, common, "discovery",
                               check=args.check or args.confirm)
    if args.confirm:
        if not discovery["gate"]["passed"]:
            raise ValueError("discovery did not pass; confirmation forbidden")
        collect_sample(config, records, samples["confirmation"], model, common, "confirmation",
                       check=args.check)


if __name__ == "__main__":
    main()
