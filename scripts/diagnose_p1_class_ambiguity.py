"""Bounded teacher inference on training tiles; no hidden GT, validation, or training."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from regionssod.data.annotations import YoloBox, read_label_file, replace_images_component
from regionssod.evaluation.class_ambiguity import (
    ambiguity_ratio, features, fit_logistic, image_bootstrap, metrics, predict,
)
from regionssod.pseudo.scale import ScalePolicy
from audit_paper1_gaps import overlaps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/visdrone_area10"
PSEUDO = DATA / "pseudo/p1_s030_m050_l050"
OUT = ROOT / "outputs/p1_class_ambiguity_diagnostic_20261102"
PROTOCOL = ROOT / "docs/paper/candidates/class_ambiguity/diagnostic_protocol.md"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write(path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(mapping):
    for path, expected in mapping.items():
        if sha(path) != expected:
            raise ValueError(f"Changed frozen file: {path}")


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    records = lines(DATA / "manifest.jsonl")
    chosen = random.Random(20261102).sample(range(len(records)), 512)
    labeled = []
    for index, record_index in enumerate(chosen):
        row = records[record_index]
        image = Path(row["labeled_tile"]["image"])
        label = replace_images_component(image).with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(label)
        labeled.append({"image": str(image), "label": str(label),
                        "source_image": row["source_image"], "image_id": record_index,
                        "group": "design" if index < 256 else "confirmation"})
    if len({row["source_image"] for row in labeled}) != 512:
        raise ValueError("Original-image sample overlap")
    # The standard predictor sorts file-list inputs; retain group membership.
    labeled.sort(key=lambda row: row["image"])
    old_images = (PSEUDO / "prediction_images.txt").read_text(encoding="utf-8").splitlines()
    batch_ids = sorted(random.Random(20261103).sample(range(len(old_images) // 8), 16))
    unlabeled = [old_images[batch * 8 + offset] for batch in batch_ids for offset in range(8)]
    settings = read(PSEUDO / "summary.json")
    if sha(settings["weights"]) != settings["teacher_sha256"]:
        raise ValueError("Original teacher hash differs")
    import ultralytics
    import ultralytics.utils.nms as nms
    import ultralytics.models.yolo.detect.predict as predictor
    import torch
    import inspect

    if ultralytics.__version__ != "8.4.172":
        raise ValueError("Pinned Ultralytics version required")
    paths = [PROTOCOL, Path(__file__), DATA / "manifest.jsonl", PSEUDO / "summary.json",
             PSEUDO / "prediction_images.txt", PSEUDO / "candidates.jsonl",
             ROOT / "src/regionssod/pseudo/class_scores.py",
             ROOT / "src/regionssod/evaluation/class_ambiguity.py",
             ROOT / "src/regionssod/pseudo/scale.py", ROOT / "scripts/audit_paper1_gaps.py",
             ROOT / "src/regionssod/data/annotations.py",
             Path(inspect.getfile(nms)), Path(inspect.getfile(predictor)), Path(settings["weights"])]
    paths += [Path(row[field]) for row in labeled for field in ("image", "label")]
    paths += list(map(Path, unlabeled))
    input_hashes = {str(path.resolve()): sha(path) for path in paths}
    OUT.mkdir(parents=True)
    write(OUT / "plan.json", {
        "protocol": str(PROTOCOL), "labeled": labeled, "unlabeled": unlabeled,
        "original_batch_ids": batch_ids, "settings": settings, "input_sha256": input_hashes,
        "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__, "numpy": np.__version__},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "training_steps": 0, "hidden_labels_read": False,
    })
    print("Frozen 512 labeled originals (256/256) and 16 original unlabeled batches.", flush=True)


def capture():
    plan = read(OUT / "plan.json")
    verify(plan["input_sha256"])
    if (OUT / "capture_receipt.json").exists() or (OUT / "labeled_predictions.jsonl").exists():
        raise FileExistsError("Prediction outputs already exist")
    from ultralytics import YOLO
    from regionssod.pseudo.class_scores import ClassScorePredictor

    settings = plan["settings"]
    policy = ScalePolicy(**settings["policy"])
    model = YOLO(settings["weights"])
    old = {row["source_image"]: row for row in lines(PSEUDO / "candidates.jsonl")}
    start = time.monotonic()
    identity = {"images": 0, "candidates": 0, "max_conf_error": 0., "max_xywhn_error": 0.}
    for mode in ("labeled", "unlabeled"):
        images = [row["image"] for row in plan["labeled"]] if mode == "labeled" else plan["unlabeled"]
        listing = OUT / f"{mode}_images.txt"
        listing.write_text("\n".join(images) + "\n", encoding="utf-8")
        with (OUT / f"{mode}_predictions.jsonl").open("x", encoding="utf-8") as output:
            count = 0
            for index, result in enumerate(model.predict(
                source=str(listing), predictor=ClassScorePredictor, stream=True,
                imgsz=960, batch=8, device="0", half=False, conf=.05, iou=.7,
                max_det=500, verbose=False, save=False,
            )):
                if Path(result.path).resolve() != Path(images[index]).resolve():
                    raise ValueError("Prediction order mismatch")
                candidates = policy.select(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                           result.boxes.xywhn.cpu().numpy(), result.orig_shape)
                if len(candidates) != len(result.teacher_class_scores):
                    raise ValueError("ScalePolicy removed a degenerate box; fail without reindexing")
                scores = result.teacher_class_scores
                ratio = ambiguity_ratio(scores)
                for row, full, anchor, r in zip(candidates, scores, result.teacher_anchor_indices, ratio):
                    row.update(class_scores=full.tolist(), anchor_index=int(anchor), ratio=float(r))
                if mode == "unlabeled":
                    previous = old[str(Path(result.path).resolve())]["candidates"]
                    if len(previous) != len(candidates):
                        raise ValueError("Original candidate count changed")
                    for a, b in zip(previous, candidates):
                        if any(a[key] != b[key] for key in ("class_id", "state", "scale")):
                            raise ValueError("Original candidate class/state/scale changed")
                        dc = abs(a["confidence"] - b["confidence"])
                        db = float(np.max(np.abs(np.asarray(a["xywhn"]) - b["xywhn"])))
                        if dc > 1e-6 or db > 1e-6:
                            raise ValueError(f"Original candidate identity changed: {dc}, {db}")
                        identity["max_conf_error"] = max(identity["max_conf_error"], dc)
                        identity["max_xywhn_error"] = max(identity["max_xywhn_error"], db)
                    identity["images"] += 1
                    identity["candidates"] += len(previous)
                payload = {"image": str(Path(result.path).resolve()), "shape": list(result.orig_shape),
                           "candidates": candidates}
                output.write(json.dumps(payload, allow_nan=False) + "\n")
                count += 1
                if count % 64 == 0:
                    print(f"{mode}: {count}/{len(images)}", flush=True)
            if count != len(images):
                raise ValueError("Incomplete capture")
    verify(plan["input_sha256"])
    write(OUT / "capture_receipt.json", {
        "accepted": True, "labeled_images": 512, "unlabeled_images": 128,
        "seconds": time.monotonic() - start, "original_candidate_identity": identity,
        "indexed_nms_exact_each_batch": True, "training_steps": 0,
        "output_sha256": {str(OUT / f"{name}_predictions.jsonl"): sha(OUT / f"{name}_predictions.jsonl")
                          for name in ("labeled", "unlabeled")},
    })


def events(plan):
    rows = []
    for expected, captured in zip(plan["labeled"], lines(OUT / "labeled_predictions.jsonl"), strict=True):
        if captured["image"] != expected["image"]:
            raise ValueError("Labeled source mismatch")
        gt = read_label_file(Path(expected["label"]))
        positives = [p for p in captured["candidates"] if p["state"] == "positive"]
        pred = [YoloBox(p["class_id"], *p["xywhn"]) for p in positives]
        ious = overlaps(pred, gt)
        for k, p in enumerate(positives):
            connected = [j for j, value in enumerate(ious[k]) if value >= .5]
            gt_classes = {gt[j].class_id for j in connected}
            target = None if not connected else gt[int(np.argmax(ious[k]))].class_id
            row = {key: p[key] for key in ("class_id", "confidence", "area", "scale", "ratio")}
            row.update(group=expected["group"], image_id=expected["image_id"],
                       target_class=target, geometry_ambiguous=len(gt_classes) > 1,
                       localizable=bool(connected), correct_same_class=p["class_id"] in gt_classes,
                       class_error=None if target is None else int(target != p["class_id"]),
                       true_class_rank=None if target is None else
                       int(1 + np.sum(np.asarray(p["class_scores"]) > p["class_scores"][target])))
            rows.append(row)
    return rows


def describe(rows):
    if not rows:
        return {"n": 0}
    return {"n": len(rows), "errors": sum(row["class_error"] for row in rows),
            "error_rate": float(np.mean([row["class_error"] for row in rows])),
            "ratio_quantiles": np.quantile([row["ratio"] for row in rows], [0, .25, .5, .75, .9, 1]).tolist()}


def analyze():
    plan = read(OUT / "plan.json")
    verify(plan["input_sha256"])
    verify(read(OUT / "capture_receipt.json")["output_sha256"])
    rows = events(plan)
    selected = [row for row in rows if row["localizable"] and not row["geometry_ambiguous"]]
    design = [row for row in selected if row["group"] == "design"]
    confirmation = [row for row in selected if row["group"] == "confirmation"]
    unlabeled = [p for row in lines(OUT / "unlabeled_predictions.jsonl")
                 for p in row["candidates"] if p["state"] == "positive"]
    result = {
        "all_accepted_visible": len(rows), "localizable": sum(row["localizable"] for row in rows),
        "excluded_multi_class_overlap": sum(row["geometry_ambiguous"] for row in rows),
        "all_accepted_same_class_iou050_fraction": float(np.mean([row["correct_same_class"] for row in rows])),
        "primary_design": describe(design), "primary_confirmation": describe(confirmation),
        "unlabeled_positive": len(unlabeled),
        "unlabeled_high_ratio": sum(row["ratio"] >= .25 for row in unlabeled),
        "groups": {}, "class_strata": {}, "confidence_strata": {}, "scale_strata": {},
    }
    for group, sample in (("design", design), ("confirmation", confirmation)):
        result["groups"][group] = {
            "low": describe([r for r in sample if r["ratio"] < .1]),
            "middle": describe([r for r in sample if .1 <= r["ratio"] < .25]),
            "high": describe([r for r in sample if r["ratio"] >= .25]),
        }
    for cls in range(10):
        result["class_strata"][str(cls)] = describe([r for r in confirmation if r["class_id"] == cls])
    for lo, hi in ((.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.00001)):
        subset = [r for r in confirmation if lo <= r["confidence"] < hi]
        result["confidence_strata"][str(lo)] = {
            "low": describe([r for r in subset if r["ratio"] < .1]),
            "high": describe([r for r in subset if r["ratio"] >= .25]),
        }
    for scale in ("small", "medium", "large"):
        result["scale_strata"][scale] = describe([r for r in confirmation if r["scale"] == scale])
    y_design = np.asarray([r["class_error"] for r in design])
    y_confirm = np.asarray([r["class_error"] for r in confirmation])
    if set(np.unique(y_design)) != {0, 1} or set(np.unique(y_confirm)) != {0, 1}:
        result.update(status="INSUFFICIENT_SUPPORT", worth_student_training=False)
    else:
        models, loss_vectors = {}, {}
        for name, with_ratio in (("baseline", False), ("candidate", True)):
            model = fit_logistic(features(design, with_ratio), y_design)
            score = predict(model, features(confirmation, with_ratio))
            metric, loss = metrics(y_confirm, score)
            models[name] = {"model": model, "confirmation": metric}
            loss_vectors[name] = loss
        result["models"] = models
        baseline, candidate = [models[key]["confirmation"] for key in ("baseline", "candidate")]
        bootstrap = image_bootstrap(
            [r["image_id"] for r in confirmation], loss_vectors["baseline"] - loss_vectors["candidate"],
            all_images=[r["image_id"] for r in plan["labeled"] if r["group"] == "confirmation"],
        )
        result["log_loss_improvement_bootstrap"] = bootstrap
        error_classes = Counter(r["class_id"] for r in confirmation if r["class_error"])
        support = (len(confirmation) >= 500 and int(y_confirm.sum()) >= 30
                   and sum(r["ratio"] >= .25 for r in confirmation) >= 30
                   and result["unlabeled_high_ratio"] >= 30
                   and sum(count >= 10 for count in error_classes.values()) >= 2)
        gates = {
            "support": support,
            "fit_converged": all(item["model"]["converged"] for item in models.values()),
            "ratio_coefficient_positive": models["candidate"]["model"]["beta"][-1] > 0,
            "relative_log_loss_reduction_ge_001": 1 - candidate["log_loss"] / baseline["log_loss"] >= .01,
            "brier_improvement_positive": baseline["brier"] > candidate["brier"],
            "auc_improvement_ge_001": candidate["auc"] - baseline["auc"] >= .01,
            "image_bootstrap_lower_positive": bootstrap["ci95"][0] > 0,
        }
        result.update(gates=gates, worth_student_training=all(gates.values()),
                      status="SUPPORTED_FOR_PREPARATION" if all(gates.values()) else
                      "INSUFFICIENT_SUPPORT" if not support else "NOT_SUPPORTED")
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--capture", action="store_true")
    action.add_argument("--analyze", action="store_true")
    action.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.capture:
        capture()
    else:
        result, rows = analyze()
        if args.audit:
            if result != read(OUT / "analysis.json"):
                raise ValueError("Numeric diagnostic replay differs")
            if rows != lines(OUT / "events.jsonl"):
                raise ValueError("Diagnostic events replay differs")
            print("Accepted exact class-ambiguity diagnostic replay; no new inference or training.")
        else:
            write(OUT / "analysis.json", result)
            with (OUT / "events.jsonl").open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
