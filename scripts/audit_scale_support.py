"""Screen complementary teacher input scale with fixed GT-free promotion rules."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random
import time
from zoneinfo import ZoneInfo

from audit_candidate_retention import GATE, qualifies, write
from audit_paper1_gaps import DATA, PSEUDO, match_iou, size
from regionssod.data.annotations import (
    Tile, YoloBox, clip_box_to_tile, label_path_for_image, read_label_file,
    replace_images_component,
)
from regionssod.pseudo.consistency import promote_consistent
from regionssod.pseudo.scale import ScalePolicy
from run_p1_scale_seed_pair import ROOT, digest, read

RULES = {f"floor{int(f*100)}_support{int(c*100)}": {"floor": f, "support_conf": c}
         for f in (.15, .25) for c in (.3, .5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--confirm", action="store_true", help="Use disjoint 256-tile confirmation sample")
    args = parser.parse_args()
    output = ROOT / ("outputs/scale_support_confirm_20261030" if args.confirm
                     else "outputs/scale_support_discovery_20261030")
    if output.exists() and not args.check:
        raise FileExistsError(output)
    flip = ROOT / "outputs/cross_view_flip_20261030/analysis.json"
    nms = ROOT / "outputs/candidate_retention_128_20261030_v2/protocol.json"
    records = [json.loads(line) for line in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    prior = read(flip)["sample_indices"]
    if args.confirm:
        excluded = set(prior) | set(read(nms)["sample_indices"])
        indices = sorted(random.Random(20261031).sample(
            [i for i in range(len(records)) if i not in excluded], 256))
    else:
        indices = prior
    selected = [records[i] for i in indices]
    settings = read(PSEUDO / "summary.json")
    source, split = read(DATA / "source.json"), read(DATA / "summary.json")
    original = {r["source_image"]: r for r in map(json.loads, (PSEUDO / "candidates.jsonl").read_text(encoding="utf-8").splitlines())}
    metadata = {r["source_image"]: r for r in map(json.loads, (PSEUDO / "metadata.jsonl").read_text(encoding="utf-8").splitlines())}
    paths = [Path(__file__).resolve(), DATA / "manifest.jsonl", DATA / "source.json",
             DATA / "summary.json", PSEUDO / "summary.json", PSEUDO / "candidates.jsonl",
             PSEUDO / "metadata.jsonl", Path(settings["weights"]), flip,
             ROOT / "src/regionssod/pseudo/consistency.py",
             ROOT / "src/regionssod/pseudo/scale.py", ROOT / "src/regionssod/data/annotations.py",
             ROOT / "scripts/audit_paper1_gaps.py", ROOT / "scripts/audit_candidate_retention.py"]
    if args.confirm:
        paths.append(nms)
    for record in selected:
        image = record["unlabeled_tiles"][0]["image"]
        paths.append(Path(image))
        paths.append(label_path_for_image(Path(record["source_image"]),
                                         Path(source["images_train"]), Path(source["labels_train"])))
        if image in metadata:
            paths.append(replace_images_component(Path(metadata[image]["image"])).with_suffix(".txt"))
    inputs = {str(p): digest(p) for p in paths}
    if digest(settings["weights"]) != settings["teacher_sha256"]:
        raise ValueError("Teacher changed")
    protocol = {"sample_indices": indices, "sample_size": 256,
                "view_imgsz": 640, "original_imgsz": 960, "conf": .05, "iou": .7,
                "max_det": 500, "batch": 8, "rules": RULES, "continuation_gate": GATE,
                "duplicate_iou": .5, "support_iou": .5,
                "scope": "existing accepted P1 tiles only; excluded tiles reported by count"}
    if args.check:
        receipt = read(output / "manifest.json")
        if inputs != receipt["input_sha256"] or protocol != read(output / "protocol.json"):
            raise ValueError("Inputs or protocol changed")
        for name, expected in receipt["output_sha256"].items():
            if digest(output / name) != expected:
                raise ValueError(f"Changed output: {name}")
    else:
        import ultralytics
        from ultralytics import YOLO

        if ultralytics.__version__ != settings["ultralytics_version"]:
            raise ValueError("Ultralytics version differs")
        output.mkdir(parents=True)
        write(output / "protocol.json", protocol)
        image_list = output / "images.txt"
        images = [r["unlabeled_tiles"][0]["image"] for r in selected]
        image_list.write_text("\n".join(images) + "\n", encoding="utf-8")
        policy = ScalePolicy(**settings["policy"])
        model = YOLO(settings["weights"])
        start = time.perf_counter()
        predictions = []
        for i, result in enumerate(model.predict(
                source=str(image_list), stream=True, imgsz=640, conf=.05, iou=.7,
                max_det=500, batch=8, device="0", verbose=False)):
            if Path(result.path).resolve() != Path(images[i]).resolve():
                raise ValueError("Prediction ordering differs")
            predictions.append({"source_image": images[i], "candidates": policy.select(
                result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                result.boxes.xywhn.cpu().numpy(), result.orig_shape)})
        if len(predictions) != 256:
            raise ValueError("Incomplete view predictions")
        (output / "view.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in predictions), encoding="utf-8")
        write(output / "costs.json", {"inference_seconds_including_model_warmup": time.perf_counter()-start})
    view = {r["source_image"]: r for r in map(json.loads, (output / "view.jsonl").read_text(encoding="utf-8").splitlines())}
    totals = {name: Counter() for name in RULES}
    excluded = 0
    for record in selected:
        entry = record["unlabeled_tiles"][0]
        image = entry["image"]
        if image not in metadata:
            excluded += 1
            continue
        tile = Tile(*entry["tile"])
        label = label_path_for_image(Path(record["source_image"]),
                                    Path(source["images_train"]), Path(source["labels_train"]))
        gt = [g for b in read_label_file(label)
              if (g := clip_box_to_tile(b, tile, *record["source_size"],
                                       split["min_box_visibility"])) is not None]
        if len(gt) != entry["visible_gt_boxes_hidden_from_training"]:
            raise ValueError("GT reconstruction mismatch")
        positive = read_label_file(replace_images_component(Path(metadata[image]["image"])).with_suffix(".txt"))
        locked = {g for _, g in match_iou(positive, gt)}
        residual = [g for i, g in enumerate(gt) if i not in locked]
        for name, kwargs in RULES.items():
            promoted = promote_consistent(original[image]["candidates"], view[image]["candidates"], **kwargs)
            extra = [YoloBox(item["class_id"], *item["xywhn"]) for item in promoted]
            pairs = match_iou(extra, residual)
            totals[name].update(tiles=1, positives=len(positive), baseline_tp=len(locked),
                                gt=len(gt), small_gt=sum(size(g, tile) == "small" for g in gt),
                                baseline_small_tp=sum(size(gt[g], tile) == "small" for g in locked),
                                added=len(extra), added_tp=len(pairs),
                                added_small_tp=sum(size(residual[g], tile) == "small" for _, g in pairs))
    result = {"counts": {name: dict(c) for name, c in totals.items()},
              "gate_by_rule": {name: qualifies(c) for name, c in totals.items()},
              "excluded_tiles": excluded,
              "scope": "exploratory hidden-GT diagnostic; unchanged original boxes/confidence; no training-label export"}
    if args.check:
        if result != read(output / "analysis.json"):
            raise ValueError("Saved analysis differs from replay")
        print(f"Verified {len(inputs)} input hashes and 256-tile replay")
    else:
        write(output / "analysis.json", result)
        if any(digest(p) != expected for p, expected in inputs.items()):
            raise ValueError("Input changed during inference")
        write(output / "manifest.json", {
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "input_sha256": inputs,
            "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
