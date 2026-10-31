"""One full-data coverage-guarded assignment student, 30 epochs and full-image evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime
import inspect
import math
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from regionssod.training.coverage_guard import run_coverage_guard_training
from regionssod.training.runner import load_experiment_config
from run_p1_scale_seed_pair import ROOT, check_epochs, digest, parameter_differences, read, replay, write


def qualifies(delta):
    ap, recall = delta["mAP50_95"], delta["small_recall"]
    return (math.isfinite(ap) and math.isfinite(recall)
            and ((ap >= .1 and recall >= 0) or (recall >= .5 and ap >= 0)))

NAME = "partial10_p1_coverage_guard_pilot30_s0"
OUT = ROOT / "outputs/p1_coverage_guard_pilot30_s0"
RUN = ROOT / "runs" / NAME
VAL = ROOT / "runs/val" / f"{NAME}_val960"
RECALL = ROOT / "runs/size_recall/p1_coverage_guard_pilot30_s0"
REF = ROOT / "runs/partial10_p1_pilot30_s0"
REF_VAL = ROOT / "runs/val/partial10_p1_pilot30_s0_val960"
REF_RECALL = ROOT / "runs/size_recall/p1_pilot30_s0"
CONFIG = ROOT / "configs/train_p1_coverage_guard_10.yaml"
STARTED = False


def stage(name, **extra):
    write(OUT / "status.json", {"stage": name,
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra})
    print(name, extra, flush=True)


def preflight():
    import ultralytics
    from ultralytics.utils import loss, tal

    if Path.cwd() != ROOT or ultralytics.__version__ != "8.4.172":
        raise ValueError("use repository root and pinned yolo environment")
    if any(p.exists() for p in (RUN, VAL, RECALL)) or any(
            (OUT / n).exists() for n in ("status.json", "dispatch.json")):
        raise FileExistsError("existing run or dispatch; inspect instead of restarting")
    config = load_experiment_config(CONFIG)
    base = load_experiment_config(ROOT / "configs/train_p1_10.yaml")
    if config != base | {"name": NAME, "assignment": "image_coverage_guard"}:
        raise ValueError("only run name and the assignment extension may differ")
    diagnostic = ROOT / "outputs/assignment_coverage_20261031"
    result = read(diagnostic / "analysis.json")["fresh_confirmation"]
    if result["retention"] != {"all": 1.0, "small": 1.0}:
        raise ValueError("diagnostic target retention differs")
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_assignment_coverage.py"), "--check"],
                   check=True, capture_output=True)
    paths = [CONFIG, Path(__file__), ROOT / "scripts/run_coverage_guard_background.py",
             ROOT / "scripts/audit_p1_coverage_guard.py",
             ROOT / "docs/paper/assignment_coverage30_protocol.md",
             ROOT / "src/regionssod/training/assignment.py",
             ROOT / "tests/test_coverage_guard_loss.py",
             Path(inspect.getfile(run_coverage_guard_training))]
    relative = [p.relative_to(ROOT).as_posix() for p in paths]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                   capture_output=True, check=True)
    subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *relative], check=True)
    frozen = read(ROOT / "outputs/p1_small_nwd_pilot30_s0/preflight.json")
    hashes = dict(frozen["sha256"])
    for path, checksum in hashes.items():
        if digest(path) != checksum:
            raise ValueError(f"changed frozen input: {path}")
    reference = read(REF / "experiment.json")
    if reference["train_args"] != config["train"] or reference["p1"] != config["p1"]:
        raise ValueError("reference schedule and P1 switches differ")
    check_epochs(REF)
    paths += [diagnostic / "analysis.json", diagnostic / "receipt.json",
              ROOT / "scripts/audit_assignment_coverage.py",
              ROOT / "docs/paper/assignment_coverage_protocol.md",
              ROOT / "outputs/p1_small_nwd_pilot30_s0/preflight.json",
              Path(inspect.getfile(loss)), Path(inspect.getfile(tal))]
    if (OUT / "engineering_check.json").exists():
        paths.append(OUT / "engineering_check.json")
    for path in paths:
        hashes[str(path)] = digest(path)
    receipt = {"source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "sha256": hashes, "epochs": 30, "train_tiles": 11310,
        "only_method_change": "original-point preference with whole-image effective-target coverage fallback; unchanged P1 loss",
        "reference_run": str(REF), "continuation":
        "(mAP50-95 >= +.10pp and small recall >= 0) OR (small recall >= +.50pp and mAP50-95 >= 0)"}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "preflight.json", receipt)
    print(f"Preflight passed: {len(hashes)} sources, 11310 tiles, exactly 30 epochs", flush=True)


def collect(check=False):
    receipt = read(OUT / "preflight.json")
    for path, checksum in receipt["sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"changed frozen input: {path}")
    before, after = read(REF / "experiment.json"), read(RUN / "experiment.json")
    if (after["assignment"] != "image_coverage_guard"
            or any(before[k] != after[k] for k in ("model", "data", "p1", "train_args", "pseudo_summary"))):
        raise ValueError("paired metadata differ beyond the intended assignment rule")
    for path, checksum in after["input_sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"changed run provenance: {path}")
    actual = yaml.safe_load((RUN / "args.yaml").read_text(encoding="utf-8"))
    original = yaml.safe_load((REF / "args.yaml").read_text(encoding="utf-8"))
    if parameter_differences(original, actual, {"name", "save_dir"}):
        raise ValueError("actual training hyperparameters differ")
    rows, reference_rows = check_epochs(RUN), check_epochs(REF)
    metrics = {}
    paths = []
    for label, run, val, rec in (("reference", REF, REF_VAL, REF_RECALL),
                                  ("candidate", RUN, VAL, RECALL)):
        validation, recall = read(val / "metrics.json"), read(rec / "metrics.json")
        if (validation["imgsz"] != 960 or Path(validation["weights"]) != run / "weights/best.pt"
                or recall["images"] != 548 or recall["imgsz"] != 960
                or recall["conf"] != .25 or recall["match_iou"] != .5
                or recall["nms_iou"] != .7 or recall["max_det"] != 500
                or recall["weights_sha256"] != digest(run / "weights/best.pt")):
            raise ValueError("full-image validation protocol or weights differ")
        if (validation["class_indices"] != list(range(10))
                or len(validation["class_maps"]) != 10
                or not all(math.isfinite(v) for v in validation["class_maps"])
                or not math.isclose(sum(validation["class_maps"])/10,
                    validation["results"]["metrics/mAP50-95(B)"], abs_tol=1e-12)):
            raise ValueError("per-class AP values do not reconstruct mAP")
        if [recall["per_size"][s]["gt"] for s in ("small", "medium", "large")] != [26577, 11115, 1067]:
            raise ValueError("full-val GT counts differ")
        replay(rec)
        metrics[label] = {"mAP50": validation["results"]["metrics/mAP50(B)"],
                          "mAP50_95": validation["results"]["metrics/mAP50-95(B)"],
                          **{f"{s}_recall": recall["per_size"][s]["recall"]
                             for s in ("small", "medium", "large")}}
        paths += [val / "metrics.json", rec / "metrics.json", rec / "images.txt", rec / "predictions.jsonl"]
    if (RECALL / "images.txt").read_bytes() != (REF_RECALL / "images.txt").read_bytes():
        raise ValueError("validation image order differs")
    before_rec, after_rec = read(REF_RECALL / "metrics.json"), read(RECALL / "metrics.json")
    if any(before_rec[k] != after_rec[k] for k in
           ("data", "ultralytics_version", "size_pixel_space", "matching")):
        raise ValueError("recall inputs or matching settings differ")
    if read(REF_VAL / "metrics.json")["data"] != read(VAL / "metrics.json")["data"]:
        raise ValueError("validation dataset differs")
    if not all(math.isfinite(v) for m in metrics.values() for v in m.values()):
        raise ValueError("non-finite metrics")
    delta = {key: 100*(metrics["candidate"][key]-metrics["reference"][key])
             for key in metrics["reference"]}
    result = {**metrics, "delta_coverage_guard_minus_p1_pp": delta,
        "worth_larger_training": qualifies(delta), "epochs": 30, "train_tiles": 11310,
        "student_time_seconds": {"reference": float(reference_rows[-1]["time"]),
                                 "candidate": float(rows[-1]["time"])},
        "scope": "single split, teacher and student seed; 30-epoch development comparison; "
                 "not significance or official AP_small; exploratory user-authorized continuation criteria"}
    if check:
        if read(OUT / "comparison.json") != result:
            raise ValueError("comparison differs on independent replay")
        for path, checksum in read(OUT / "artifact_sha256.json").items():
            if digest(path) != checksum:
                raise ValueError(f"changed result artifact: {path}")
    else:
        write(OUT / "comparison.json", result)
        paths += [RUN / p for p in ("weights/best.pt", "weights/last.pt", "results.csv",
                                    "args.yaml", "experiment.json")]
        paths += [OUT / "preflight.json", OUT / "comparison.json"]
        write(OUT / "artifact_sha256.json", {str(p): digest(p) for p in paths})
    print(result, flush=True)
    return result


def engineering_check():
    from copy import deepcopy
    from types import SimpleNamespace

    import torch
    from torch.utils.data import DataLoader, Subset
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.detect import DetectionValidator
    from regionssod.training.coverage_guard import CoverageGuardLoss, CoverageGuardModel

    torch.manual_seed(0)
    config = load_experiment_config(CONFIG)
    pretrained_model: Any = YOLO(config["model"]).model
    model: Any = CoverageGuardModel(pretrained_model.yaml, nc=10, ch=3, verbose=False)
    model.load(pretrained_model)
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    model.args = get_cfg()
    model = model.cuda().train()
    batch = {"img": torch.rand(2, 3, 640, 640, device="cuda"),
             "batch_idx": torch.tensor([0., 1.], device="cuda"),
             "cls": torch.tensor([[0.], [3.]], device="cuda"),
             "bboxes": torch.tensor([[.5, .5, .025, .025], [.5, .5, .15, .15]], device="cuda"),
             "pseudo_weights": torch.tensor([.4, 1.], device="cuda"),
             "ignore_bboxes": torch.tensor([[.2, .2, .1, .1]], device="cuda"),
             "ignore_batch_idx": torch.tensor([0], device="cuda")}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, loss_items = model(batch)
    loss.sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not torch.isfinite(loss).all() or not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError("non-finite AMP forward/backward")
    ema = deepcopy(model).float()
    if not isinstance(ema.criterion, CoverageGuardLoss):
        raise ValueError("engineering validation must reuse a cached training criterion")
    validator: Any = DetectionValidator(
        save_dir=OUT / "engineering_validation",
        args=get_cfg(overrides={"data": config["data"], "imgsz": 640, "batch": 2,
                                "workers": 0, "plots": False, "save_json": False,
                                "conf": .25, "max_det": 500}))
    validator.data = check_det_dataset(config["data"])
    validator.device = torch.device("cuda:0")
    validator.stride = int(ema.stride.max())
    loader = validator.get_dataloader(validator.data["val"], batch_size=2)
    validator.dataloader = DataLoader(
        Subset(loader.dataset, [0, 1]), batch_size=2, num_workers=0,
        collate_fn=loader.dataset.collate_fn)
    plain = next(iter(validator.dataloader))
    if any(key in plain for key in ("pseudo_weights", "ignore_bboxes", "ignore_batch_idx")):
        raise ValueError("validation must exercise standard fully labeled data")
    probe = SimpleNamespace(
        ema=SimpleNamespace(ema=ema), device=validator.device, data=validator.data,
        amp=True, loss_items=loss_items, stopper=SimpleNamespace(possible_stop=False),
        epoch=0, epochs=30, world_size=1,
        label_loss_items=lambda values, prefix: {f"{prefix}/{key}": value for key, value in values.items()})
    validation = validator(trainer=probe)
    validation_loss = {key: value for key, value in validation.items() if key.startswith("val/")}
    if (validator.seen != 2 or len(validation_loss) != 3
            or not all(math.isfinite(value) for value in validation_loss.values())):
        raise ValueError("real validation path did not complete with finite losses")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        plain = validator.preprocess(plain)
        preds = ema(plain["img"])
        observed = ema.loss(plain, preds)[0]
        from regionssod.training.p1 import P1DetectionLoss
        expected = P1DetectionLoss(ema)(preds, plain)[0]
    if not torch.equal(observed, expected):
        raise ValueError("fully labeled validation loss must remain identical to P1")
    rejected = False
    try:
        model.loss({key: value for key, value in batch.items() if key != "pseudo_weights"})
    except ValueError as error:
        if "provenance" not in str(error):
            raise
        rejected = True
    if not rejected:
        raise ValueError("training must still reject missing pseudo provenance")
    write(OUT / "engineering_check.json", {
        "loss": loss.detach().float().cpu().tolist(), "finite_gradient_tensors": len(gradients),
        "optimizer_steps": 0, "epochs": 0, "synthetic_inputs": 2, "amp": True,
        "validation_images": validator.seen, "validation_batches": 1,
        "validation_loss": validation_loss, "validation_matches_p1": True,
        "validation_uses_cached_training_criterion": True,
        "training_missing_provenance_rejected": rejected,
        "source_sha256": {str(p): digest(p) for p in
            (CONFIG, Path(__file__), Path(inspect.getfile(run_coverage_guard_training)))}})
    print("Real YOLO11s AMP backward and two-image training-validator path passed; zero optimizer steps", flush=True)


def main():
    global STARTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--engineering-only", action="store_true")
    options = parser.parse_args()
    preflight()
    if options.check_only:
        return
    if options.engineering_only:
        engineering_check()
        return
    engineering = read(OUT / "engineering_check.json")
    if (engineering["optimizer_steps"] != 0 or engineering["epochs"] != 0
            or engineering["amp"] is not True or engineering["finite_gradient_tensors"] <= 0
            or engineering["validation_images"] != 2 or engineering["validation_batches"] != 1
            or engineering["validation_matches_p1"] is not True
            or engineering["validation_uses_cached_training_criterion"] is not True
            or engineering["training_missing_provenance_rejected"] is not True
            or len(engineering["validation_loss"]) != 3
            or not all(math.isfinite(v) for v in engineering["validation_loss"].values())
            or not all(math.isfinite(v) for v in engineering["loss"])):
        raise ValueError("a passing zero-step AMP engineering check is required")
    if any(digest(p) != h for p, h in engineering["source_sha256"].items()):
        raise ValueError("engineering check source changed")
    with (OUT / "dispatch.json").open("x", encoding="utf-8") as handle:
        handle.write(datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()+"\n")
    STARTED = True
    stage("training", epochs=30, train_tiles=11310)
    if run_coverage_guard_training(CONFIG) != RUN:
        raise ValueError("unexpected training directory")
    check_epochs(RUN)
    from regionssod.evaluation.runner import run_validation
    from regionssod.evaluation.size_recall import run_size_recall
    stage("validation960")
    if run_validation(RUN / "weights/best.pt", ROOT / "data/visdrone_area10/dataset_p1.yaml",
                      imgsz=960, batch=8, device="0", name=f"{NAME}_val960") != VAL:
        raise ValueError("unexpected full validation directory")
    stage("size_recall")
    run_size_recall(RUN / "weights/best.pt", Path("D:/Python/YOLO/data/visdrone/dataset.yaml"),
                   RECALL, imgsz=960, batch=8, device="0", conf=.25, match_iou=.5)
    stage("comparing")
    collect()
    stage("completed", comparison=str(OUT / "comparison.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if STARTED:
            stage("failed", error=traceback.format_exc())
        raise
