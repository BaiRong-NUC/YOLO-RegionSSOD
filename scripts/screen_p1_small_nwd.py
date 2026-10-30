"""One new five-epoch NWD arm, paired with an audited cached P1 reference."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import traceback
from zoneinfo import ZoneInfo

import yaml

from regionssod.data.annotations import replace_images_component
from regionssod.training.nwd import estimate_normalizer, run_nwd_training
from regionssod.training.runner import load_experiment_config
from run_p1_scale_seed_pair import ROOT, digest, parameter_differences, read, replay, write

CONFIG = ROOT / "configs/screen_p1_small_nwd_5ep.yaml"
NAME = "p1_small_nwd_area10_subset1024_e5_s0"
OUT = ROOT / "outputs/screens/p1_small_nwd_1024_e5"
RUN = ROOT / "runs/screen" / NAME
VAL = ROOT / "runs/screen/val" / NAME
RECALL = OUT / "size_recall"
REF_NAME = "p1_full_area10_subset1024_e5_s0"
REF = ROOT / "runs/screen" / REF_NAME
REF_VAL = ROOT / "runs/screen/val" / REF_NAME
REF_RECALL = ROOT / "outputs/screens/p1_soft_weights_1024_e5" / REF_NAME
DATA = ROOT / "data/screens/p1_soft_weights_1024"
SOURCE = Path("D:/Python/YOLO/data/visdrone/dataset.yaml")
OWNED = False


def stage(name, **extra):
    value = {"stage": name, "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra}
    write(OUT / "status.json", value)
    print(json.dumps(value), flush=True)


def five_epochs(run):
    with (run / "results.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if ([int(float(r["epoch"])) for r in rows] != list(range(1, 6))
            or not all(math.isfinite(float(v)) for r in rows for v in r.values())):
        raise ValueError(f"expected finite contiguous five epochs: {run}")
    return rows


def qualifies(delta):
    return delta["mAP50_95"] >= .30 and delta["small_recall"] >= 0


def preflight():
    import ultralytics

    if ultralytics.__version__ != "8.4.172" or Path.cwd().resolve() != ROOT:
        raise ValueError("run from repository root in the pinned yolo environment")
    if any(p.exists() for p in (RUN, VAL, RECALL, OUT / "dispatch.json", OUT / "status.json")):
        raise FileExistsError("existing NWD screen state; inspect instead of restarting")
    config = load_experiment_config(CONFIG)
    base = load_experiment_config(ROOT / "configs/screen_p1_soft_weights_5ep.yaml")
    if {k: v for k, v in config.items() if k != "nwd"} != base | {"name": NAME}:
        raise ValueError("only run name and NWD extension may change")
    if config["nwd"] != {"normalizer": "open_small_mean", "mix": .5, "small_side": 32.}:
        raise ValueError("outside fixed NWD rule")
    own_paths = [CONFIG, Path(__file__).resolve(), ROOT / "src/regionssod/training/nwd.py",
                 ROOT / "docs/paper/small_nwd_protocol.md",
                 ROOT / "scripts/audit_small_nwd_screen.py", ROOT / "scripts/run_small_nwd_background.py"]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--",
         *[p.relative_to(ROOT).as_posix() for p in own_paths]],
        cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode:
        raise RuntimeError(tracked.stderr)
    subprocess.run(["git", "diff", "HEAD", "--quiet", "--",
                    *[p.relative_to(ROOT).as_posix() for p in own_paths]], cwd=ROOT, check=True)
    hashes = dict(read(ROOT / "outputs/gate3_20261027/preflight.json")["sha256"])
    reference = read(REF / "experiment.json")
    hashes.update(reference["input_sha256"])
    hashes.update(read(ROOT / "outputs/screens/p1_soft_weights_1024_e5/preflight.json")["sha256"])
    expected_weight = read(ROOT / "outputs/screens/p1_soft_weights_1024_e5/comparison.json")[
        "runs"][0]["weight_sha256"]
    if digest(REF / "weights/best.pt") != expected_weight:
        raise ValueError("cached reference weight changed")
    for filename, checksum in hashes.items():
        if digest(filename) != checksum:
            raise ValueError(f"changed frozen reference input: {filename}")
    five_epochs(REF)
    if reference["train_args"] != config["train"] or reference["p1"] != config["p1"]:
        raise ValueError("cached reference training settings differ")
    if not read(ROOT / "outputs/screens/p1_soft_weights_1024_e5/acceptance_audit.json")[
            "acceptance_checks_passed"]:
        raise ValueError("reference has no accepted screen audit")
    images = (DATA / "train.txt").read_text(encoding="utf-8").splitlines()
    opened = [p for p in images if "labeled" in Path(p).parts]
    if len(images) != 1765 or len(opened) != 1024:
        raise ValueError("not the frozen 1024-source, 1765-tile screen")
    normalizer = estimate_normalizer(opened)
    paths = [*own_paths, DATA / "summary.json", DATA / "train.txt", DATA / "dataset.yaml",
             DATA / "metadata.jsonl", REF / "weights/best.pt", REF / "args.yaml",
             REF / "results.csv", REF / "experiment.json", REF_VAL / "metrics.json",
             REF_RECALL / "metrics.json", REF_RECALL / "images.txt", REF_RECALL / "predictions.jsonl"]
    paths.extend(Path(p) for p in images)
    paths.extend(replace_images_component(Path(p)).with_suffix(".txt") for p in images)
    hashes.update({str(p): digest(p) for p in paths})
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
               "sha256": hashes, "normalizer": normalizer,
               "train_tiles": 1765, "new_epochs": 5, "reference_run": str(REF),
               "continuation": "candidate minus cached P1: mAP50-95 >= +0.30pp and small recall >= 0"}
    write(OUT / "preflight.json", receipt)
    print(f"Verified {len(hashes)} frozen inputs; C={normalizer['normalizer']:.8f} input pixels", flush=True)
    return receipt


def gpu_check(receipt):
    import gc
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from regionssod.training.nwd import NWDDetectionModel

    torch.manual_seed(0)
    pretrained = YOLO(load_experiment_config(CONFIG)["model"])
    model = NWDDetectionModel(pretrained.model.yaml, nc=10, ch=3, verbose=False)
    model.load(pretrained.model)
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    model.nwd_options = {"normalizer": receipt["normalizer"]["normalizer"],
                         "mix": .5, "small_side": 32.}
    model.args = get_cfg()
    model = model.cuda().train()
    batch = {
        "img": torch.rand(2, 3, 640, 640, device="cuda"),
        "batch_idx": torch.tensor([0., 1.], device="cuda"),
        "cls": torch.tensor([[0.], [3.]], device="cuda"),
        "bboxes": torch.tensor([[.5, .5, .025, .025], [.5, .5, .15, .15]], device="cuda"),
        "pseudo_weights": torch.tensor([.4, 1.], device="cuda"),
        "ignore_bboxes": torch.tensor([[.2, .2, .1, .1]], device="cuda"),
        "ignore_batch_idx": torch.tensor([0], device="cuda")}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, _ = model(batch)
    loss.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not torch.isfinite(loss).all() or not grads or not all(torch.isfinite(g).all() for g in grads):
        raise ValueError("non-finite real YOLO11s AMP forward/backward")
    result = {"loss": loss.detach().float().cpu().tolist(), "finite_gradient_tensors": len(grads),
              "optimizer_steps": 0, "training_epochs": 0, "input_shape": [2, 3, 640, 640],
              "amp": True, "scope": "synthetic engineering backward only; no student metrics"}
    write(OUT / "geometry_gpu_check.json", result)
    del model, pretrained, grads, loss, batch
    gc.collect()
    torch.cuda.empty_cache()
    print("Real YOLO11s AMP forward/backward passed; zero optimizer steps", flush=True)


def collect(*, check=False):
    receipt = read(OUT / "preflight.json")
    for path, checksum in receipt["sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"input changed during screen: {path}")
    reference, experiment = read(REF / "experiment.json"), read(RUN / "experiment.json")
    if (reference["model"] != experiment["model"] or reference["data"] != experiment["data"]
            or reference["p1"] != experiment["p1"] or reference["train_args"] != experiment["train_args"]
            or reference["pseudo_summary"] != experiment["pseudo_summary"]
            or experiment["nwd"] != {"normalizer": receipt["normalizer"]["normalizer"],
                                     "mix": .5, "small_side": 32.}):
        raise ValueError("paired run metadata differs beyond the NWD term")
    if any(digest(p) != h for p, h in experiment["input_sha256"].items()):
        raise ValueError("training code/config changed")
    actual = yaml.safe_load((RUN / "args.yaml").read_text(encoding="utf-8"))
    old_actual = yaml.safe_load((REF / "args.yaml").read_text(encoding="utf-8"))
    differences = parameter_differences(old_actual, actual, {"name", "save_dir"})
    if differences:
        raise ValueError(f"actual hyperparameters differ: {differences}")
    rows = five_epochs(RUN)
    before_val, after_val = read(REF_VAL / "metrics.json"), read(VAL / "metrics.json")
    before_rec, after_rec = read(REF_RECALL / "metrics.json"), read(RECALL / "metrics.json")
    if any(after_rec[k] != before_rec[k] for k in (
            "data", "images", "imgsz", "conf", "match_iou", "nms_iou", "max_det",
            "ultralytics_version", "size_pixel_space", "matching")):
        raise ValueError("recall protocol differs")
    if ((RECALL / "images.txt").read_bytes() != (REF_RECALL / "images.txt").read_bytes()
            or after_rec["images"] != 548 or after_rec["imgsz"] != 960
            or after_val["imgsz"] != 960 or after_val["data"] != before_val["data"]
            or Path(after_val["weights"]) != RUN / "weights/best.pt"
            or after_rec["weights_sha256"] != digest(RUN / "weights/best.pt")):
        raise ValueError("full-val inputs or weights differ")
    replay(REF_RECALL)
    replay(RECALL)
    if [after_rec["per_size"][s]["gt"] for s in ("small", "medium", "large")] != [26577, 11115, 1067]:
        raise ValueError("unexpected full-image GT counts")
    before, after = {}, {}
    for name, key in (("mAP50", "metrics/mAP50(B)"), ("mAP50_95", "metrics/mAP50-95(B)")):
        before[name], after[name] = before_val["results"][key], after_val["results"][key]
    for size in ("small", "medium", "large"):
        before[f"{size}_recall"] = before_rec["per_size"][size]["recall"]
        after[f"{size}_recall"] = after_rec["per_size"][size]["recall"]
    if not all(math.isfinite(x) for x in [*before.values(), *after.values()]):
        raise ValueError("non-finite metrics")
    delta = {key: 100*(after[key]-before[key]) for key in before}
    comparison = {"reference": before, "candidate": after, "delta_nwd_minus_p1_pp": delta,
                  "worth_full30_confirmation": qualifies(delta), "new_epochs": 5,
                  "new_student_seconds": float(rows[-1]["time"]),
                  "scope": "exploratory 1024-source subset only; not paper results or full30 evidence"}
    if check:
        if comparison != read(OUT / "comparison.json"):
            raise ValueError("comparison replay differs")
        for line in (RUN / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            checksum, path = line.split("  ", 1)
            if digest(path) != checksum:
                raise ValueError(f"changed result artifact: {path}")
    else:
        write(OUT / "comparison.json", comparison)
        artifacts = [RUN / p for p in ("weights/best.pt", "weights/last.pt", "args.yaml",
                                      "results.csv", "experiment.json")]
        artifacts += [VAL / "metrics.json", RECALL / "metrics.json", RECALL / "images.txt",
                      RECALL / "predictions.jsonl", OUT / "preflight.json",
                      OUT / "geometry_gpu_check.json", OUT / "comparison.json"]
        (RUN / "SHA256SUMS.txt").write_text(
            "".join(f"{digest(p)}  {p}\n" for p in artifacts), encoding="utf-8")
    print(json.dumps(comparison, indent=2), flush=True)
    return comparison


def main():
    global OWNED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    receipt = preflight()
    if args.check_only:
        return
    gpu_check(receipt)
    if args.geometry_only:
        return
    with (OUT / "dispatch.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, handle)
    OWNED = True
    stage("training", new_epochs=5, train_tiles=1765)
    if run_nwd_training(CONFIG, normalizer=receipt["normalizer"]["normalizer"]) != RUN:
        raise ValueError("unexpected training directory")
    five_epochs(RUN)
    from regionssod.evaluation.runner import run_validation
    from regionssod.evaluation.size_recall import run_size_recall

    stage("validation960")
    if run_validation(RUN / "weights/best.pt", DATA / "dataset.yaml", imgsz=960,
                      project=str(VAL.parent), name=NAME) != VAL:
        raise ValueError("unexpected validation directory")
    stage("size_recall")
    run_size_recall(RUN / "weights/best.pt", SOURCE, RECALL, imgsz=960,
                    batch=8, device="0", conf=.25, match_iou=.5)
    stage("comparing")
    collect()
    stage("completed", comparison=str(OUT / "comparison.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if OWNED:
            stage("failed", error=traceback.format_exc())
        raise
