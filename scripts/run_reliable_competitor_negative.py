"""Fresh reference/candidate/control x30; explicit dispatch, no retry or resume.

Check/audit never instantiate a model, dataset or loader and never run inference.
The only executable training entry is an authenticated one-shot child worker.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True  # --audit must not create even Python bytecode.

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import yaml

from evaluate_paper1_small_pr import INFERENCE, digest as sha, ground_truth, read, verify_hashes as verify, write
from regionssod.data.annotations import replace_images_component
from regionssod.evaluation.error_diagnosis import ERROR_NAMES, box_iou_matrix, classify_false_predictions
from regionssod.evaluation.precision_recall import assign_predictions
from regionssod.training.runner import load_experiment_config
from run_p1_scale_seed_pair import check_epochs, parameter_differences

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("reference", "candidate", "control")
IMPLEMENTATION = "reliable_competitor_negative_v1"
OUT = ROOT / "outputs/reliable_competitor_negative_three30_s0"
BASE = ROOT / "configs/train_p1_10.yaml"
CONFIGS = {a: ROOT / f"configs/candidates/reliable_competitor_negative/{a}30.yaml" for a in ARMS}
RUNS = {a: ROOT / f"runs/reliable_competitor_negative_{a}30_s0" for a in ARMS}
DOC = ROOT / "docs/paper/candidates/reliable_competitor_negative"
VALIDATION = ROOT / "outputs/reliable_competitor_negative_engineering/validation_final.json"
ORIGIN = ROOT / "outputs/gate3_20261027/preflight.json"
ORIGIN_SHA = "63d5b96d59f3c4209fcd35566e8cc64fe21bcb284a4b49b2df1783e3172f3b5a"
BASE_SHA = "fe0dc21fd9c0c289b1461cf293b12a92dedee9dc7ab8e1676df27d3140f221d6"
GATES = {
    "reference_AP50_95_pp": ("ge", .10),
    "reference_AP_small_pp": ("ge", 0.),
    "reference_small_conf025_pp": ("ge", -.20),
    "reference_small_iou050_P70_pp": ("ge", 0.),
    "reference_small_iou050_P90_pp": ("ge", 0.),
    "reference_small_iou075_P90_pp": ("ge", 0.),
    "reference_FP_conf025_excess": ("le", 0.),
    "reference_wrong_class_conf025_excess": ("le", 0.),
    "control_AP50_95_pp": ("ge", .10),
    "control_AP_small_pp": ("ge", 0.),
}
LEGACY_NOTICE = {
    "applicable_to_this_candidate": False,
    "legacy_files": ["comparison.json", "paired_pr.png", "paired_pr.pdf"],
    "reason": "Reused evaluator's class-ambiguity title/five gates are NOT this candidate's decision.",
    "authoritative_decision": "../decision.json",
}


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def merge_hashes(*mappings):
    result = {}
    for mapping in mappings:
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Empty/nonmapping hash evidence")
        for path, checksum in mapping.items():
            key = str(Path(path).resolve())
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError("Invalid SHA256 evidence")
            if key in result and result[key] != checksum:
                raise ValueError(f"Conflicting frozen input: {key}")
            result[key] = checksum
    return result


def hashes(paths):
    return {str(Path(p).resolve()): sha(p) for p in paths}


def configurations():
    if sha(BASE) != BASE_SHA:
        raise ValueError("Original P1 config changed")
    base = load_experiment_config(BASE)
    result = {}
    for arm in ARMS:
        expected = base | {"name": RUNS[arm].name, "reliable_competitor_negative": {"mode": arm}}
        result[arm] = load_experiment_config(CONFIGS[arm])
        if result[arm] != expected:
            raise ValueError(f"{arm}: only run name and declared mode may differ")
    return result


def engineering():
    receipt = read(VALIDATION)
    if (receipt.get("accepted") is not True or receipt.get("optimizer_steps") != 0
            or receipt.get("parameter_hash_unchanged") is not True
            or receipt.get("implementation") != IMPLEMENTATION):
        raise ValueError("Engineering receipt not accepted/zero-step/parameter-preserving")
    batch = receipt.get("real_batch", {})
    if (batch.get("images") != 8 or batch.get("amp") is not True
            or batch.get("device") not in ("cuda:0", "cuda0")
            or receipt.get("torch_version") != version("torch")
            or receipt.get("ultralytics_version") != version("ultralytics")):
        raise ValueError("Engineering batch/runtime differs")
    sources = receipt["source_sha256"]
    required = [*list((ROOT / "src/regionssod").rglob("*.py")), BASE,
                ROOT / "scripts/validate_reliable_competitor_negative.py", DOC / "engineering_protocol.md"]
    normalized = merge_hashes(sources)
    if any(str(p.resolve()) not in normalized for p in required):
        raise ValueError("Engineering source coverage incomplete")
    mapping = merge_hashes(normalized, receipt["artifact_sha256"], hashes([VALIDATION]))
    verify(mapping)
    return mapping


def expected_args(config):
    from ultralytics.cfg import DEFAULT_CFG_DICT
    return canonical(DEFAULT_CFG_DICT | config["train"] | {
        "task": "detect", "mode": "train", "model": config["model"],
        "data": str((ROOT / config["data"]).resolve()),
        "project": str((ROOT / config["project"]).resolve()), "name": config["name"],
        "exist_ok": False, "resume": False,
        "save_dir": str((ROOT / config["project"] / config["name"]).resolve()),
    })


def runtime():
    import ultralytics
    import torch
    import PIL
    import pycocotools
    if version("ultralytics") != "8.4.172" or Path(sys.prefix).name != "yolo":
        raise ValueError("Pinned yolo Conda environment required")
    paths = [Path(sys.executable)]
    # Freeze complete installed Python sources, including loss/augment/TAL and
    # default YAMLs; compiled COCO implementation is part of numeric replay.
    for module in (ultralytics, torch, np, PIL, yaml, pycocotools):
        for p in Path(module.__file__).parent.rglob("*"):
            if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".pyd", ".so"}:
                paths.append(p)
    return hashes(paths), {n: version(n) for n in (
        "ultralytics", "torch", "numpy", "pillow", "PyYAML", "pycocotools", "matplotlib")}


def label_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if (len(values) != 5 or not all(math.isfinite(v) for v in values)
                or int(values[0]) != values[0] or not 0 <= values[0] < 10
                or not all(0 <= v <= 1 for v in values[1:]) or min(values[3:]) <= 0):
            raise ValueError(f"Illegal YOLO label: {path}")
        rows.append(values)
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def validate_population(images, human, metadata):
    """Pure CPU file validation; no YOLODataset, hidden labels, batch or forward."""
    keys = [str(Path(p).resolve()) for p in images]
    human = {str(Path(p).resolve()) for p in human}
    sidecar = {}
    for row in metadata:
        key = str(Path(row["image"]).resolve())
        if key in sidecar or key in human:
            raise ValueError("Duplicated/overlapping human/pseudo provenance")
        sidecar[key] = row
    if len(set(keys)) != len(keys) or set(keys) != human | set(sidecar):
        raise ValueError("Training population/provenance mismatch")
    paths, counts = [], Counter()
    for key in keys:
        image = Path(key)
        label = replace_images_component(image).with_suffix(".txt")
        if image.with_suffix(".npy").exists():
            raise ValueError("Unbound image npy cache is forbidden")
        with Image.open(image) as pixels:
            pixels.verify()
        values = label_rows(label)
        paths.extend([image, label])
        counts["human_rows" if key in human else "pseudo_rows"] += len(values)
        if key in human:
            if "labeled" not in image.parts:
                raise ValueError("Human image is not a materialized labeled tile")
            continue
        row = sidecar[key]
        expected = []
        for kind in ("positive", "ignore"):
            counts[kind] += len(row[kind])
            for item in row[kind]:
                cls, conf, box = item["class_id"], item["confidence"], item["xywhn"]
                if (type(cls) is not int or not 0 <= cls < 10 or not math.isfinite(conf)
                        or not 0 < conf <= 1 or len(box) != 4
                        or not all(math.isfinite(v) and 0 <= v <= 1 for v in box)
                        or min(box[2:]) <= 0 or item.get("state") != kind):
                    raise ValueError("Illegal pseudo class/confidence/box/state")
                if kind == "positive":
                    expected.append([cls, *box])
        expected = np.asarray(expected, np.float32).reshape(-1, 5)
        if (values.shape != expected.shape or not np.array_equal(values[:, 0], expected[:, 0])
                or not np.allclose(values[:, 1:], expected[:, 1:], atol=1e-6, rtol=0)):
            raise ValueError("Pseudo metadata/label binding differs")
        paths.append(Path(row["source_image"]))
    return paths, dict(counts)


def validate_label_cache(images):
    """Read the one aggregate cache selected by YOLODataset's first label."""
    cache = replace_images_component(Path(images[0])).parent.with_suffix(".cache")
    if not cache.exists():
        return {}
    content = np.load(cache, allow_pickle=True).item()
    expected_images = {str(Path(p).resolve()) for p in images}
    cached_images = [str(Path(row["im_file"]).resolve()) for row in content["labels"]]
    if len(cached_images) != len(expected_images) or set(cached_images) != expected_images:
        raise ValueError("Label cache population differs")
    for row in content["labels"]:
        image = Path(row["im_file"])
        labels = label_rows(replace_images_component(image).with_suffix(".txt"))
        _, indices = np.unique(labels, axis=0, return_index=True)
        labels = labels[indices] if len(indices) != len(labels) else labels
        cached = np.concatenate([row["cls"], row["bboxes"]], axis=1)
        with Image.open(image) as im:
            shape = (im.height, im.width)
        if (not np.array_equal(labels, cached) or tuple(row["shape"]) != shape
                or row.get("normalized") is not True or row.get("bbox_format") != "xywh"
                or len(row.get("segments", [])) or row.get("keypoints") is not None):
            raise ValueError("Label cache content differs")
    return hashes([cache])


def data_inputs(config):
    if sha(ORIGIN) != ORIGIN_SHA:
        raise ValueError("Original split/teacher provenance changed")
    original = merge_hashes(read(ORIGIN)["sha256"])
    data_path = ROOT / config["data"]
    spec = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    area = data_path.parent
    metadata_path = Path(spec["p1_metadata"])
    summary_path = metadata_path.parent / "summary.json"
    summary = read(summary_path)
    split = read(area / "summary.json")
    if split["seed"] != 0 or split["area_ratio"] != .1:
        raise ValueError("split0 and 10% labeled area required")
    listing = Path(spec["train"])
    human_list = area / "splits/labeled_train.txt"
    pseudo_list = metadata_path.parent / "pseudo_train.txt"
    images = listing.read_text(encoding="utf-8").splitlines()
    human = human_list.read_text(encoding="utf-8").splitlines()
    pseudo = pseudo_list.read_text(encoding="utf-8").splitlines()
    if len(human) != 6471 or len(pseudo) != 4839 or len(images) != 11310 or images != human + pseudo:
        raise ValueError("Frozen train list/order differs")
    metadata = [json.loads(s) for s in metadata_path.read_text(encoding="utf-8").splitlines()]
    paths, counts = validate_population(images, human, metadata)
    if counts != {"human_rows": 105058, "pseudo_rows": 35067, "positive": 35067, "ignore": 52117}:
        raise ValueError("Training object counts changed")
    gt, validation_hashes, gt_counts = ground_truth()
    val_images = [im["file_name"] for im in gt["images"]]
    for image in val_images:
        label_rows(replace_images_component(Path(image)).with_suffix(".txt"))
        if Path(image).with_suffix(".npy").exists():
            raise ValueError("Unbound validation npy cache is forbidden")
    if (Path(spec["val"]).resolve() != Path(gt["images"][0]["file_name"]).parent.resolve()
            or spec["names"] != {c["id"]-1: c["name"] for c in gt["categories"]}):
        raise ValueError("Train/evaluation val/class axes differ")
    teacher = Path(summary["weights"])
    if sha(teacher) != summary["teacher_sha256"]:
        raise ValueError("Teacher binding differs")
    paths += [data_path, listing, human_list, pseudo_list, metadata_path, summary_path,
              area / "summary.json", area / "source.json", area / "manifest.jsonl",
              area / "splits/unlabeled_train.txt", area / "splits/val.txt",
              area / "dataset_p1.combine_summary.json", teacher, Path(config["model"]), BASE,
              ROOT / "configs/pseudo_p1_10.yaml"]
    # Match all inputs already present in original P1 receipt, without freezing
    # unrelated historic source revisions or reusing any old student checkpoint.
    mapping = merge_hashes(hashes(paths), validation_hashes)
    for path, checksum in mapping.items():
        if path in original and original[path] != checksum:
            raise ValueError(f"Original P1 data/input drift: {path}")
    for path in (BASE, Path(config["model"]), teacher, metadata_path, area / "manifest.jsonl", human_list):
        if str(path.resolve()) not in original:
            raise ValueError("Missing original data identity")
    mapping.update(hashes([ORIGIN]))
    # Freeze source image provenance (without reading hidden training GT).
    provenance_images = []
    for line in (area / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        provenance_images.append(Path(row["source_image"]))
        provenance_images.extend(Path(t["image"]) for t in row["unlabeled_tiles"])
    mapping = merge_hashes(mapping, hashes(provenance_images))
    # Each split's cache is aggregate, including labels in other directories.
    for population in (images, val_images):
        mapping.update(validate_label_cache(population))
    return mapping, {"training_images": len(images), "counts": counts, "gt_counts": gt_counts,
                     "ground_truth_sha256": hashlib.sha256(json.dumps(gt, sort_keys=True).encode()).hexdigest()}


def preflight():
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from repository root")
    if OUT.exists() or any(p.exists() for p in RUNS.values()):
        raise FileExistsError("Existing output/run; no overwrite, retry or resume")
    configs = configurations()
    evidence = engineering()
    runtime_hashes, versions = runtime()
    from regionssod.training.reliable_competitor_negative import IMPLEMENTATION as actual
    if actual != IMPLEMENTATION:
        raise ValueError("Wrong implementation")
    inputs, population = data_inputs(configs["reference"])
    sources = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    sources += list((ROOT / "tests").glob("*.py")) + list((ROOT / "configs").rglob("*.yaml"))
    sources += [*DOC.glob("*.md"), ROOT / "pyproject.toml", ROOT / "AGENTS.md"]
    mapping = merge_hashes(evidence, runtime_hashes, inputs, hashes(sources))
    verify(mapping)
    return {"implementation": IMPLEMENTATION, "train_arms": list(ARMS), "epochs_per_arm": 30,
        "arms": {a: str(p) for a, p in RUNS.items()}, "configurations": configs,
        "expected_args": {a: expected_args(c) for a, c in configs.items()},
        "input_sha256": mapping, "population": population, "versions": versions,
        "inference": INFERENCE, "gates": canonical(GATES),
        "split_seed": 0, "student_seed": 0, "reused_students": [],
        "optimizer_steps_this_check": 0, "forward_calls_this_check": 0,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}


def verify_dispatch():
    receipt = read(OUT / "preflight.json")
    dispatch = read(OUT / "dispatch.json")
    if (dispatch["preflight_sha256"] != sha(OUT / "preflight.json")
            or dispatch["train_arms"] != list(ARMS) or receipt["train_arms"] != list(ARMS)
            or receipt["arms"] != {a: str(p) for a, p in RUNS.items()}
            or receipt["implementation"] != IMPLEMENTATION or receipt["epochs_per_arm"] != 30
            or receipt["gates"] != canonical(GATES) or receipt["inference"] != INFERENCE
            or receipt["configurations"] != configurations()
            or receipt["expected_args"] != {a: expected_args(c) for a, c in configurations().items()}):
        raise ValueError("Dispatch/config/decision identity differs")
    verify(receipt["input_sha256"])
    return receipt


def audit_arm(arm, receipt):
    run = RUNS[arm]
    exited = read(OUT / f"{arm}_exit.json")
    if exited["exit_code"] != 0 or exited["arm"] != arm:
        raise ValueError("Worker did not exit successfully")
    seal = read(OUT / f"{arm}_seal.json")
    verify(seal["artifact_sha256"])
    required = [run / n for n in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "experiment.json")]
    if any(str(p) not in seal["artifact_sha256"] for p in required):
        raise ValueError("Incomplete arm seal")
    rows = check_epochs(run)
    actual = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    if parameter_differences(actual, receipt["expected_args"][arm], set()):
        raise ValueError(f"{arm}: actual training args differ from frozen defaults/overrides")
    experiment = read(run / "experiment.json")
    if (experiment["implementation"] != IMPLEMENTATION or experiment["mode"] != arm
            or experiment["config"] != receipt["configurations"][arm]):
        raise ValueError("Training provenance mode/config differs")
    verify(experiment["input_sha256"])
    expected_sources = [CONFIGS[arm], ROOT / receipt["configurations"][arm]["data"],
                        Path(receipt["configurations"][arm]["model"])]
    if any(str(p.resolve()) not in experiment["input_sha256"] for p in expected_sources):
        raise ValueError("Training input provenance incomplete")
    if any(receipt["input_sha256"].get(str(Path(p).resolve())) != h for p, h in experiment["input_sha256"].items()):
        raise ValueError("Training inputs not frozen by preflight")
    claim = read(OUT / f"{arm}_claim.json")
    if (claim["arm"] != arm or claim["preflight_sha256"] != sha(OUT / "preflight.json")
            or seal["claim_sha256"] != sha(OUT / f"{arm}_claim.json")):
        raise ValueError("Worker claim binding differs")
    return {"epochs": 30, "student_seconds": float(rows[-1]["time"]),
            "wall_seconds": exited["worker_wall_seconds"],
            "best_sha256": sha(run / "weights/best.pt"), "actual_args": actual}


def audit_training():
    receipt = verify_dispatch()
    return {"accepted": True, "arms": {a: audit_arm(a, receipt) for a in ARMS},
            "reused_students": [], "epochs_per_arm": 30}


def worker(arm):
    # Parent PID + secret passed only through environment; disk holds its digest.
    # The x-mode claim is atomic and survives failure, forbidding retry/replay.
    receipt = verify_dispatch()
    dispatch = read(OUT / "dispatch.json")
    token = os.environ.pop("REGIONSSOD_RCN_WORKER_TOKEN", "")
    state = read(OUT / "status.json")
    if (os.getppid() != dispatch["parent_pid"] or not token
            or hashlib.sha256(token.encode()).hexdigest() != dispatch["token_sha256"][arm]
            or state.get("stage") != "training" or state.get("arm") != arm):
        raise ValueError("Internal worker requires live matching parent dispatch")
    for earlier in ARMS[:ARMS.index(arm)]:
        audit_arm(earlier, receipt)
    if RUNS[arm].exists():
        raise FileExistsError("Existing arm; worker may not resume")
    claim = {"arm": arm, "pid": os.getpid(), "parent_pid": os.getppid(),
             "preflight_sha256": sha(OUT / "preflight.json")}
    with (OUT / f"{arm}_claim.json").open("x", encoding="utf-8") as stream:
        json.dump(claim, stream)
    from regionssod.training.reliable_competitor_negative import run_training
    run = run_training(CONFIGS[arm])
    if Path(run).resolve() != RUNS[arm]:
        raise ValueError("Unexpected training destination")
    check_epochs(RUNS[arm])
    verify_dispatch()
    paths = [RUNS[arm] / n for n in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "experiment.json")]
    write(OUT / f"{arm}_seal.json", {"claim_sha256": sha(OUT / f"{arm}_claim.json"),
                                    "artifact_sha256": hashes(paths)})


def fp_decomposition(records, gt):
    by_name = {r["image"]: r for r in records}
    if len(by_name) != len(records) or set(by_name) != {im["file_name"] for im in gt["images"]}:
        raise ValueError("FP cache population differs")
    annotations = {im["id"]: [] for im in gt["images"]}
    for item in gt["annotations"]:
        annotations[item["image_id"]].append(item)
    totals = Counter({name: 0 for name in ERROR_NAMES})
    per_image = {}
    for im in gt["images"]:
        record = by_name[im["file_name"]]
        scores = np.asarray(record["scores"], float)
        classes = np.asarray(record["classes"])
        boxes = np.asarray(record["xyxy"], float).reshape(-1, 4)
        if len(scores) != len(classes) or len(scores) != len(boxes) or not np.isfinite(scores).all():
            raise ValueError("Invalid cached predictions")
        selected = scores >= .25
        scores, classes, boxes = scores[selected], classes[selected], boxes[selected]
        truth = annotations[im["id"]]
        gt_classes = [a["category_id"]-1 for a in truth]
        gt_boxes = np.asarray([a["bbox"] for a in truth], float).reshape(-1, 4)
        gt_boxes[:, 2:] += gt_boxes[:, :2]
        matched = assign_predictions(classes, scores, boxes, gt_classes, gt_boxes, .5)
        labels = classify_false_predictions(matched, box_iou_matrix(boxes, gt_boxes), classes, gt_classes, .5)
        counts = {n: int((labels == i).sum()) for i, n in enumerate(ERROR_NAMES)}
        per_image[im["file_name"]] = counts
        totals.update(counts)
    return {"conf": .25, "iou": .5, "counts": dict(totals), "per_image": per_image,
            "fp": sum(totals[n] for n in ERROR_NAMES if n != "matched"),
            "scope": "fixed error_diagnosis priority, all-size FP, no GT-size attribution"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metrics(model):
    def point(iou, p):
        v = model["precision_constraints"][iou][p]
        return None if v is None else v["small_recall"]
    fixed = model["conf025"]["0.5"]
    return {"AP50": model["ultralytics"]["metrics/mAP50(B)"],
        "AP50_95": model["ultralytics"]["metrics/mAP50-95(B)"],
        "AP_small": model["coco"]["metrics"]["AP_small"],
        "small_conf025": None if fixed is None else fixed["small_recall"],
        "small_iou050_P70": point("0.5", "0.7"), "small_iou050_P90": point("0.5", "0.9"),
        "small_iou075_P90": point("0.75", "0.9")}


def comparison_and_decision(models, errors):
    if set(models) != set(ARMS) or set(errors) != set(ARMS):
        raise ValueError("All three fresh arms are mandatory")
    absolute = {a: metrics(models[a]) for a in ARMS}
    delta = {a: {k: 100*(absolute["candidate"][k]-v) if finite(v) and finite(absolute["candidate"][k]) else None
                 for k, v in absolute[a].items()} for a in ("reference", "control")}
    classes = {a: models[a]["coco"]["per_class_AP_small"] for a in ARMS}
    if any(len(c) != 10 or set(c) != set(classes["reference"]) for c in classes.values()):
        raise ValueError("All ten class AP_small values must be reported")
    counts = {}
    for a in ARMS:
        point = models[a]["conf025"]["0.5"]
        fp = 0 if point is None else point["fp"]
        if errors[a]["fp"] != fp or sum(errors[a]["counts"][n] for n in ERROR_NAMES[1:]) != fp:
            raise ValueError("FP decomposition and PR cache disagree")
        counts[a] = {"FP": fp, "wrong_class": errors[a]["counts"]["wrong_class"]}
    observed = {f"reference_{k}_pp": delta["reference"][k] for k in (
        "AP50_95", "AP_small", "small_conf025", "small_iou050_P70", "small_iou050_P90", "small_iou075_P90")}
    observed.update(reference_FP_conf025_excess=counts["candidate"]["FP"]-.97*counts["reference"]["FP"],
        reference_wrong_class_conf025_excess=counts["candidate"]["wrong_class"]-.95*counts["reference"]["wrong_class"],
        control_AP50_95_pp=delta["control"]["AP50_95"], control_AP_small_pp=delta["control"]["AP_small"])
    gates = {k: {"observed": observed[k], "operator": op, "threshold": limit,
                "passed": finite(observed[k]) and (observed[k] >= limit if op == "ge" else observed[k] <= limit)}
             for k, (op, limit) in GATES.items()}
    passed = all(g["passed"] for g in gates.values())
    comparison = {"absolute_fraction": absolute, "candidate_minus_other_pp": delta, "FP_counts": counts,
        "per_class_AP_small_fraction": classes,
        "per_class_candidate_minus_other_pp": {a: {k: 100*(classes["candidate"][k]-v)
            if finite(v) and finite(classes["candidate"][k]) else None for k, v in classes[a].items()}
            for a in ("reference", "control")},
        "scope": "three fresh arms, same split0/teacher/COCO/student seed0; 30 epochs; all signs retained"}
    decision = {"implementation": IMPLEMENTATION, "gates": gates, "all_ten_passed": passed,
        "outcome": "eligible_for_separate_confirmation_review" if passed else "stop",
        "automatic_further_training": False, "automatic_retry": False,
        "semantic_specificity": "development_criteria_passed" if all(gates[k]["passed"] for k in GATES if k.startswith("control_")) else "NOT_ESTABLISHED",
        "limitations": ["investment criteria, not significance or causal proof", "one student seed, fixed teacher/split, development val",
            "control matches local negative-logit L1 only, not parameter gradients/optimization",
            "COCO-style AP_small is not official VisDrone AP", "precision thresholds estimated separately per arm on val",
            "conf>=.001, maxDet500, NMS .7 bound the available predictions", "all pass does not authorize 100 epochs"]}
    return comparison, decision


def models_and_errors(receipt):
    out = OUT / "evaluation"
    manifest = read(out / "manifest.json")
    expected = {a: str(p / "weights/best.pt") for a, p in RUNS.items()}
    if manifest["weights"] != expected or manifest["inference"] != receipt["inference"]:
        raise ValueError("Evaluation weight/protocol binding differs")
    for p, h in manifest["input_sha256"].items():
        if p not in expected.values() and receipt["input_sha256"].get(str(Path(p).resolve())) != h:
            raise ValueError("Evaluation input not frozen at preflight")
    gt = read(out / "ground_truth.json")
    if (len(gt["images"]) != 548 or manifest["counts"] != receipt["population"]["gt_counts"]
            or hashlib.sha256(json.dumps(gt, sort_keys=True).encode()).hexdigest() != receipt["population"]["ground_truth_sha256"]):
        raise ValueError("Ground truth identity differs")
    models, errors = {}, {}
    for arm in ARMS:
        model = read(out / arm / "analysis.json")
        predictions = out / arm / "predictions.jsonl"
        if (model["weights_sha256"] != sha(expected[arm]) or model["predictions_sha256"] != sha(predictions)
                or model["manifest_sha256"] != sha(out / "manifest.json")):
            raise ValueError("Weight/prediction binding differs")
        models[arm] = model
        errors[arm] = fp_decomposition([json.loads(s) for s in predictions.read_text(encoding="utf-8").splitlines()], gt)
    return models, errors


def finish():
    from evaluate_class_ambiguity_pair import evaluate, audit as evaluation_audit
    training = audit_training()
    write(OUT / "training_audit.json", training)
    write(OUT / "status.json", {"stage": "evaluating", "automatic_retry": False})
    evaluate({a: p / "weights/best.pt" for a, p in RUNS.items()}, OUT / "evaluation")
    write(OUT / "evaluation/LEGACY_NOT_THIS_CANDIDATE.json", LEGACY_NOTICE)
    write(OUT / "evaluation_audit.json", evaluation_audit(OUT / "evaluation"))
    models, errors = models_and_errors(verify_dispatch())
    comparison, decision = comparison_and_decision(models, errors)
    write(OUT / "wrong_class_decomposition.json", errors)
    write(OUT / "comparison.json", comparison)
    write(OUT / "decision.json", decision)
    write(OUT / "status.json", {"stage": "completed", "outcome": decision["outcome"],
                               "automatic_further_training": False, "automatic_retry": False})


def seal_terminal():
    paths = [p for p in OUT.rglob("*") if p.is_file() and p != OUT / "artifact_sha256.json"]
    for run in RUNS.values():
        if run.exists():
            paths.extend(p for p in run.rglob("*") if p.is_file())
    write(OUT / "artifact_sha256.json", hashes(paths))


def train():
    receipt = preflight()
    OUT.mkdir(parents=True, exist_ok=False)
    tokens = {a: secrets.token_hex(32) for a in ARMS}
    write(OUT / "preflight.json", receipt)
    write(OUT / "dispatch.json", {"preflight_sha256": sha(OUT / "preflight.json"),
        "parent_pid": os.getpid(), "train_arms": list(ARMS),
        "token_sha256": {a: hashlib.sha256(t.encode()).hexdigest() for a, t in tokens.items()}})
    try:
        for arm in ARMS:
            write(OUT / "status.json", {"stage": "training", "arm": arm, "epochs": 30, "automatic_retry": False})
            start = time.monotonic()
            env = os.environ | {"REGIONSSOD_RCN_WORKER_TOKEN": tokens[arm], "PYTHONDONTWRITEBYTECODE": "1"}
            with (OUT / f"{arm}_stdout.log").open("x", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", arm],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(OUT / f"{arm}_exit.json", {"arm": arm, "exit_code": result.returncode,
                                          "worker_wall_seconds": time.monotonic()-start})
            result.check_returncode()
            audit_arm(arm, verify_dispatch())
        finish()
    except BaseException:
        write(OUT / "status.json", {"stage": "failed", "error": traceback.format_exc(), "automatic_retry": False})
        raise
    finally:
        seal_terminal()


def audit():
    """Read-only terminal verification; never repairs, rewrites or runs a model."""
    status = read(OUT / "status.json")
    if status["stage"] not in ("completed", "failed"):
        raise ValueError("Not terminal; audit does not monitor or resume")
    artifacts = read(OUT / "artifact_sha256.json")
    present = {str(p) for p in OUT.rglob("*") if p.is_file() and p != OUT / "artifact_sha256.json"}
    for run in RUNS.values():
        if run.exists():
            present.update(str(p) for p in run.rglob("*") if p.is_file())
    if set(artifacts) != present:
        raise ValueError("Terminal artifact inventory changed")
    verify(artifacts)
    receipt = verify_dispatch()
    if status["stage"] == "failed":
        return {"accepted": False, "terminal_evidence_verified": True, "stage": "failed",
                "performance": "NOT_ACCEPTED", "automatic_retry": False}
    training = audit_training()
    if training != read(OUT / "training_audit.json"):
        raise ValueError("Training replay differs")
    from evaluate_class_ambiguity_pair import audit as evaluation_audit
    evaluated = evaluation_audit(OUT / "evaluation")
    if evaluated != read(OUT / "evaluation_audit.json"):
        raise ValueError("Numeric AP/PR cache replay differs")
    if read(OUT / "evaluation/LEGACY_NOT_THIS_CANDIDATE.json") != LEGACY_NOTICE:
        raise ValueError("Missing legacy-decision disclaimer")
    models, errors = models_and_errors(receipt)
    comparison, decision = comparison_and_decision(models, errors)
    if (errors != read(OUT / "wrong_class_decomposition.json") or comparison != read(OUT / "comparison.json")
            or decision != read(OUT / "decision.json") or status["outcome"] != decision["outcome"]):
        raise ValueError("FP/comparison/ten-gate decision replay differs")
    return {"accepted": True, "stage": "completed", "training": training,
            "evaluation": evaluated, "decision": decision, "inference_this_audit": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--train", action="store_true")
    group.add_argument("--audit", action="store_true")
    group.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_only:
        receipt = preflight()
        print(json.dumps({"accepted": True, "sources": len(receipt["input_sha256"]),
            "train_arms": receipt["train_arms"], "population": receipt["population"],
            "training": False, "forward": False, "created_files": False}, ensure_ascii=False))
    elif args.train:
        train()
    elif args.worker:
        worker(args.worker)
    else:
        print(json.dumps(audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
