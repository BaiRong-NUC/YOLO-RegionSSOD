"""Export full teacher scores bound to every ORIGINAL P1 positive, without training."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from diagnose_p1_class_ambiguity import ROOT, PSEUDO, lines, read, sha, verify, write

OUT = ROOT / "outputs/p1_class_scores_20261102"


def run_export():
    from ultralytics import YOLO
    import ultralytics
    from regionssod.pseudo.class_scores import ClassScorePredictor
    from regionssod.pseudo.scale import ScalePolicy

    if OUT.exists():
        raise FileExistsError(f"Export is exclusive; inspect existing output: {OUT}")
    if ultralytics.__version__ != "8.4.172":
        raise ValueError("Pinned Ultralytics required")
    settings = read(PSEUDO / "summary.json")
    if sha(settings["weights"]) != settings["teacher_sha256"]:
        raise ValueError("Teacher differs")
    images = (PSEUDO / "prediction_images.txt").read_text(encoding="utf-8").splitlines()
    if len(images) != 6471 or images != sorted(images):
        raise ValueError("Original batch order or population differs")
    old = {r["source_image"]: r for r in lines(PSEUDO / "candidates.jsonl")}
    metadata = {r["source_image"]: r for r in lines(PSEUDO / "metadata.jsonl")}
    paths = [Path(__file__), ROOT / "src/regionssod/pseudo/class_scores.py",
             ROOT / "src/regionssod/pseudo/scale.py", Path(settings["weights"])]
    paths += [PSEUDO / name for name in ("summary.json", "prediction_images.txt",
                                       "metadata.jsonl", "candidates.jsonl")]
    paths += list(map(Path, images))
    hashes = {str(p.resolve()): sha(p) for p in paths}
    OUT.mkdir(parents=True)
    write(OUT / "plan.json", {"input_sha256": hashes, "settings": settings,
                              "training_steps": 0, "hidden_gt_read": False})
    policy = ScalePolicy(**settings["policy"])
    model = YOLO(settings["weights"])
    count = positive_count = candidate_count = 0
    max_conf = max_box = 0.
    started = time.monotonic()
    seen = set()
    with (OUT / "class_scores.jsonl").open("x", encoding="utf-8") as stream:
        import json
        for i, result in enumerate(model.predict(
            source=str(PSEUDO / "prediction_images.txt"), predictor=ClassScorePredictor,
            stream=True, imgsz=960, batch=8, device="0", half=False, conf=.05,
            iou=.7, max_det=500, verbose=False, save=False,
        )):
            key = str(Path(result.path).resolve())
            if key != str(Path(images[i]).resolve()):
                raise ValueError("Original inference order changed")
            candidates = policy.select(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                       result.boxes.xywhn.cpu().numpy(), result.orig_shape)
            previous = old[key]["candidates"]
            if len(candidates) != len(previous) or len(candidates) != len(result.teacher_class_scores):
                raise ValueError("Original candidate count changed")
            positives = []
            for a, b, scores in zip(previous, candidates, result.teacher_class_scores, strict=True):
                if any(a[k] != b[k] for k in ("class_id", "state", "scale")):
                    raise ValueError("Candidate identity changed")
                dc = abs(a["confidence"] - b["confidence"])
                db = float(np.max(np.abs(np.asarray(a["xywhn"]) - b["xywhn"])))
                if dc > 1e-6 or db > 1e-6:
                    raise ValueError(f"Candidate identity differs: {dc}, {db}")
                max_conf, max_box = max(max_conf, dc), max(max_box, db)
                if a["state"] == "positive":
                    positives.append({**a, "teacher_class_scores": scores.tolist()})
            if positives:
                meta = metadata[key]
                if [{k: p[k] for k in meta["positive"][0]} for p in positives] != meta["positive"]:
                    raise ValueError("Original metadata positive order changed")
                stream.write(json.dumps({"image": meta["image"], "positive": positives},
                                        allow_nan=False) + "\n")
                seen.add(key)
                positive_count += len(positives)
            count += 1
            candidate_count += len(candidates)
            if count % 512 == 0:
                print(f"Verified {count}/6471 original tiles, {positive_count} original positives", flush=True)
    if count != 6471 or positive_count != 35067 or seen != set(metadata):
        raise ValueError("Incomplete original P1 coverage")
    verify(hashes)
    write(OUT / "receipt.json", {
        "accepted": True, "metadata_sha256": sha(PSEUDO / "metadata.jsonl"),
        "sidecar_sha256": sha(OUT / "class_scores.jsonl"), "plan_sha256": sha(OUT / "plan.json"),
        "images": count, "pseudo_images": len(seen), "positives": positive_count,
        "candidates": candidate_count, "max_conf_error": max_conf, "max_xywhn_error": max_box,
        "indexed_nms_exact": True, "training_steps": 0, "hidden_gt_read": False,
        "seconds": time.monotonic() - started,
    })
    print(read(OUT / "receipt.json"))


def audit():
    receipt = read(OUT / "receipt.json")
    if receipt["accepted"] is not True or receipt["plan_sha256"] != sha(OUT / "plan.json"):
        raise ValueError("Invalid receipt")
    verify(read(OUT / "plan.json")["input_sha256"])
    if receipt["sidecar_sha256"] != sha(OUT / "class_scores.jsonl"):
        raise ValueError("Changed score sidecar")
    print(f"Accepted full score export: {receipt['positives']} positives; no inference/training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    audit() if args.audit else run_export()
