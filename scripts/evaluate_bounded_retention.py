"""Single-pass full-val capture and cache-only numeric evaluation, without gates.

Import/analysis do not load a detector, dataset, checkpoints, or old caches.
Only the explicit evaluate command performs inference. Audit never does so.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np

from regionssod.evaluation.precision_recall import assign_predictions
from regionssod.evaluation.error_diagnosis import (
    ERROR_NAMES, box_iou_matrix, classify_false_predictions, paired_geometry,
)

SCHEMA = "bounded-retention-evaluation-v1"
POPULATION = 548
IOUS = (.5, .75)
INFERENCE = dict(imgsz=960, batch=8, workers=0, device="0", conf=.001, iou=.7,
    max_det=500, rect=True, quantize=None, augment=False, plots=False,
    save_json=False, verbose=False, split="val")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _prepare(records, gt):
    images, categories = gt["images"], gt["categories"]
    names = [im["file_name"] for im in images]
    image_ids = {im["id"] for im in images}
    if (len(images) != POPULATION or len(set(names)) != POPULATION or len(image_ids) != POPULATION
            or len(records) != POPULATION or len({r["image"] for r in records}) != POPULATION
            or {r["image"] for r in records} != set(names)):
        raise ValueError("exactly all 548 unique validation images required")
    if len(categories) != 10 or {c["id"] for c in categories} != set(range(1, 11)):
        raise ValueError("all ten categories 1..10 required")
    if len({c["name"] for c in categories}) != 10:
        raise ValueError("unique category names required")
    truth, annotation_ids = defaultdict(list), set()
    for ann in gt["annotations"]:
        box = np.asarray(ann["bbox"], float)
        if (ann["id"] in annotation_ids or ann["image_id"] not in image_ids
                or ann["category_id"] not in range(1, 11) or ann.get("iscrowd", 0) != 0 or ann.get("ignore", 0) != 0
                or box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any()
                or not np.isfinite(ann["area"]) or ann["area"] <= 0
                or not np.isclose(ann["area"], box[2]*box[3], rtol=1e-10, atol=0)):
            raise ValueError("invalid canonical validation GT")
        annotation_ids.add(ann["id"])
        truth[ann["image_id"]].append(ann)
    by_name = {r["image"]: r for r in records}
    prepared = []
    for im in sorted(images, key=lambda x: x["file_name"]):
        row = by_name[im["file_name"]]
        classes, scores = np.asarray(row["classes"], float), np.asarray(row["scores"], float)
        boxes = np.asarray(row["xyxy"], float).reshape(-1, 4)
        indices = row.get("original_indices", list(range(len(scores))))
        if (row["shape"] != [im["height"], im["width"]] or min(im["height"], im["width"]) <= 0
                or classes.ndim != 1 or scores.ndim != 1 or len(boxes) != len(scores)
                or len(classes) != len(scores) or len(scores) > INFERENCE["max_det"]
                or len(indices) != len(scores) or len(set(indices)) != len(indices)
                or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in indices)
                or not np.isfinite(scores).all() or not np.isfinite(classes).all()
                or not np.isfinite(boxes).all() or (scores < .001).any() or (scores > 1).any()
                or (classes != classes.astype(int)).any() or (classes < 0).any() or (classes >= 10).any()
                or (boxes[:, 2:] < boxes[:, :2]).any()):
            raise ValueError("invalid captured prediction/shape/index")
        annotations = sorted(truth[im["id"]], key=lambda a: a["id"])
        gt_boxes = np.asarray([a["bbox"] for a in annotations], float).reshape(-1, 4)
        gt_boxes[:, 2:] += gt_boxes[:, :2]
        prepared.append(dict(image=im, classes=classes.astype(int), scores=scores, boxes=boxes,
            original_indices=np.asarray(indices, int), annotations=annotations, gt_boxes=gt_boxes,
            gt_classes=np.asarray([a["category_id"]-1 for a in annotations], int)))
    return prepared


def _counts(gt):
    areas = np.asarray([a["area"] for a in gt["annotations"]], float)
    return {"all": len(areas), "small": int((areas < 1024).sum()),
            "medium": int(((areas >= 1024) & (areas < 9216)).sum()), "large": int((areas >= 9216).sum())}


def _ratio(n, d):
    return n/d if d else None


def _selection(prepared, threshold=None):
    return {p["image"]["file_name"]: [int(i) for i, s in zip(p["original_indices"], p["scores"])
                                      if threshold is None or s >= threshold]
            for p in prepared}


def _match(prepared, gt, selection, iou):
    """Re-match each selected image population; never filter stale TP flags."""
    if set(selection) != {p["image"]["file_name"] for p in prepared}:
        raise ValueError("selected population must retain all 548 image keys")
    errors = dict.fromkeys(ERROR_NAMES, 0)
    per_class = {c["name"]: dict(category_id=c["id"], predictions=0, tp=0, fp=0,
        gt=sum(a["category_id"] == c["id"] for a in gt["annotations"]), errors=dict.fromkeys(ERROR_NAMES, 0))
        for c in gt["categories"]}
    class_name = {c["id"]-1: c["name"] for c in gt["categories"]}
    matches, geometry, flags = {}, {}, {}
    size_tp = dict.fromkeys(("all", "small", "medium", "large"), 0)
    predictions = 0
    for p in prepared:
        name = p["image"]["file_name"]
        requested = selection[name]
        if len(set(requested)) != len(requested) or not set(requested) <= set(p["original_indices"]):
            raise ValueError("selection contains duplicate/unknown indices")
        # Stable original-index order governs equal-score per-image matching.
        lookup = {int(i): k for k, i in enumerate(p["original_indices"])}
        positions = np.asarray([lookup[i] for i in sorted(requested)], dtype=int)
        cls, scores, boxes = p["classes"][positions], p["scores"][positions], p["boxes"][positions]
        local = assign_predictions(cls, scores, boxes, p["gt_classes"], p["gt_boxes"], iou)
        labels = classify_false_predictions(local, box_iou_matrix(boxes, p["gt_boxes"]), cls, p["gt_classes"], iou)
        predictions += len(positions)
        for j, (position, label) in enumerate(zip(positions, labels)):
            original = int(p["original_indices"][position])
            flags[name, original] = bool(local[j] >= 0)
            error = ERROR_NAMES[int(label)]
            errors[error] += 1
            per_class[class_name[int(cls[j])]]["predictions"] += 1
            per_class[class_name[int(cls[j])]]["errors"][error] += 1
            if local[j] >= 0:
                ann = p["annotations"][int(local[j])]
                matches[ann["id"]] = (name, original)
                area = ann["area"]
                size_tp["all"] += 1
                size_tp["small" if area < 1024 else "medium" if area < 9216 else "large"] += 1
                values = paired_geometry(boxes[j:j+1], p["gt_boxes"][local[j]:local[j]+1])
                geometry[ann["id"]] = {k: float(v[0]) for k, v in values.items()}
        for entry in per_class.values():
            entry["tp"] = entry["errors"]["matched"]
            entry["fp"] = entry["predictions"]-entry["tp"]
    counts = _counts(gt)
    for entry in per_class.values():
        entry["recall"] = _ratio(entry["tp"], entry["gt"])
        entry["fn"] = entry["gt"]-entry["tp"]
    summary = dict(predictions=predictions, tp=size_tp["all"], fp=predictions-size_tp["all"],
        precision=_ratio(size_tp["all"], predictions), errors=errors, per_class=per_class,
        sizes={k: dict(gt=counts[k], tp=size_tp[k], fn=counts[k]-size_tp[k], recall=_ratio(size_tp[k], counts[k])) for k in counts})
    return summary, dict(matches=matches, geometry=geometry, flags=flags)


def _coco(prepared, gt):
    """One COCO tensor, all areas and all ten classes; no official-VisDrone claim."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    detections = []
    for p in prepared:
        for j in np.argsort(p["original_indices"], kind="stable"):
            x1, y1, x2, y2 = p["boxes"][j]
            detections.append(dict(image_id=p["image"]["id"], category_id=int(p["classes"][j])+1,
                score=float(p["scores"][j]), bbox=[x1, y1, x2-x1, y2-y1]))
    with redirect_stdout(io.StringIO()):
        truth = COCO(); truth.dataset = gt; truth.createIndex()
        if detections:
            predicted = truth.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = dict(images=gt["images"], categories=gt["categories"], annotations=[])
            predicted.createIndex()
        evaluator = COCOeval(truth, predicted, "bbox")
        evaluator.params.maxDets = [1, 100, 500]
        evaluator.evaluate(); evaluator.accumulate()
    tensor = evaluator.eval["precision"]
    def mean(values):
        valid = values[values >= 0]
        return float(valid.mean()) if len(valid) else None
    def metrics(values):
        return {f"{tag}_{area}": mean(values[index, ..., ai, -1])
                for ai, area in enumerate(evaluator.params.areaRngLbl)
                for tag, index in (("AP", slice(None)), ("AP50", 0), ("AP75", 5))}
    per_class = {c["name"]: dict(category_id=c["id"], **metrics(tensor[:, :, i, :, :]))
                 for i, c in enumerate(sorted(gt["categories"], key=lambda c: c["id"]))}
    return dict(metrics=metrics(tensor), per_class=per_class, max_dets=[1, 100, 500],
                area_ranges=evaluator.params.areaRng, convention="COCO interpolation on YOLO val GT; not VisDrone official AP")


def _analyze_prepared(prepared, gt):
    points = {name: {str(iou): _match(prepared, gt, _selection(prepared, threshold), iou) for iou in IOUS}
              for name, threshold in (("conf025", .25), ("saved_floor", None))}
    summary = dict(coco=_coco(prepared, gt), gt_counts=_counts(gt),
        operating_points={name: {key: pair[0] for key, pair in values.items()} for name, values in points.items()},
        prediction_count=sum(len(p["scores"]) for p in prepared),
        images_capped_at_500=sum(len(p["scores"]) == 500 for p in prepared),
        zero_area_predictions=sum(int(np.any(p["boxes"][:, 2:] == p["boxes"][:, :2], axis=1).sum()) for p in prepared))
    return summary, points


def analyze_records(records, gt, counts=None):
    prepared = _prepare(records, gt)
    actual = _counts(gt)
    if counts is not None and any(actual[k] != v for k, v in counts.items()):
        raise ValueError("GT count binding changed")
    return _analyze_prepared(prepared, gt)[0]


def _ranked(prepared):
    return sorted((-float(score), p["image"]["file_name"], int(index))
                  for p in prepared for score, index in zip(p["scores"], p["original_indices"]))


def _prefix_selection(prepared, ranked, n):
    result = {p["image"]["file_name"]: [] for p in prepared}
    for _, image, original in ranked[:n]:
        result[image].append(original)
    return result


def _cap_ambiguity(prepared, cutoff, last_key=None):
    affected = []
    if cutoff is None:
        return affected
    for p in prepared:
        if len(p["scores"]) != 500:
            continue
        missing_key = (-float(min(p["scores"])), p["image"]["file_name"], int(max(p["original_indices"]))+1)
        if (last_key is None and cutoff <= -missing_key[0]) or (last_key is not None and missing_key < last_key):
            affected.append(p["image"]["file_name"])
    return affected


def select_same_k(prepared, k):
    """Exact stable top-K in the frozen saved universe; truncation is separate."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("nonnegative K required")
    ranked = _ranked(prepared)
    n = min(k, len(ranked))
    cutoff = -ranked[n-1][0] if n else None
    capped = _cap_ambiguity(prepared, cutoff, ranked[n-1] if n else None)
    reasons, warnings = [], []
    if len(ranked) < k:
        reasons.append("saved_floor_or_cap_has_fewer_than_K")
    if n and cutoff <= .001:
        warnings.append("selection_reaches_inference_floor_tie")
    if capped:
        warnings.append("max_det_can_change_uncapped_topK")
    return _prefix_selection(prepared, ranked, n), dict(available=not reasons,
        available_within_frozen_inference=not reasons, uncapped_population_certified=not reasons and not warnings,
        reasons=reasons, scope_warnings=warnings,
        requested_k=k, selected_count=n, cutoff_score=cutoff, capped_images=capped,
        selection_rule="exact topK by (-score,imagepath,originalindex); no GT tie selection")


def select_same_fp(prepared, gt, budget, iou, *, full_flags=None):
    """Longest whole-score-group saved prefix; an unspent budget remains valid."""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("nonnegative FP budget required")
    if full_flags is None:
        _, full = _match(prepared, gt, _selection(prepared), iou)
        full_flags = full["flags"]
    ranked = _ranked(prepared)
    chosen, actual_fp, offset, running_fp = 0, 0, 0, 0
    while offset < len(ranked):
        end = offset+1
        while end < len(ranked) and ranked[end][0] == ranked[offset][0]:
            end += 1
        running_fp += sum(not full_flags[image, index] for _, image, index in ranked[offset:end])
        if running_fp > budget:
            break
        chosen, actual_fp, offset = end, running_fp, end
    threshold = -ranked[chosen-1][0] if chosen else None
    capped = _cap_ambiguity(prepared, threshold)
    warnings = []
    if chosen == len(ranked):
        warnings.append("saved_prediction_universe_exhausted")
    if chosen and threshold <= .001:
        warnings.append("threshold_reaches_inference_floor_tie")
    if capped:
        warnings.append("max_det_can_change_uncapped_threshold_population")
    return _prefix_selection(prepared, ranked, chosen), dict(available=True,
        available_within_frozen_inference=True, uncapped_population_certified=not warnings,
        reasons=[], scope_warnings=warnings,
        fp_budget=budget, actual_fp=actual_fp, slack=budget-actual_fp, budget_saturated=actual_fp == budget, selected_count=chosen,
        threshold=threshold, empty_prefix=chosen == 0, exhausted_saved_predictions=chosen == len(ranked),
        capped_images=capped, selection_rule="longest whole-score-group threshold prefix with FP<=budget")


def _describe(values):
    values = np.asarray(values, float)
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                q10_q50_q90=np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None)


def _common_geometry(left, right, gt, scope):
    small = {a["id"] for a in gt["annotations"] if a["area"] < 1024}
    l, r = set(left["matches"]) & small, set(right["matches"]) & small
    common = sorted(l & r)
    return dict(diagnostic_only=True, selection_scope=scope,
        population="all 548 images; strict native GT area<1024; class-correct IoU>=0.5 matches in BOTH arms",
        transitions=dict(total_small=len(small), both=len(common), reference_only=len(l-r),
                         other_only=len(r-l), neither=len(small-(l|r)), net_gain=len(r)-len(l)),
        common_gt_ids=common,
        geometry={key: dict(reference=_describe([left["geometry"][i][key] for i in common]),
                            other=_describe([right["geometry"][i][key] for i in common]),
                            other_minus_reference=_describe([right["geometry"][i][key]-left["geometry"][i][key] for i in common]))
                  for key in ("iou", "normalized_center_error", "normalized_edge_error")})


def compare_records(arms, gt, reference="reference"):
    if reference not in arms or len(arms) < 2:
        raise ValueError("reference and at least one comparison arm required")
    prepared = {name: _prepare(rows, gt) for name, rows in arms.items()}
    computed = {name: _analyze_prepared(data, gt) for name, data in prepared.items()}
    summaries = {name: value[0] for name, value in computed.items()}
    order = sorted((name for name in arms if name != reference), key=lambda n: (n != "candidate", n != "control", n))
    directions = [(reference, name) for name in order]
    if "control" in arms and "candidate" in arms and ("control", "candidate") not in directions:
        directions.append(("control", "candidate"))
    pairs = {}
    for baseline, name in directions:
        data = prepared[name]
        ref_points = computed[baseline][1]["conf025"]
        same_fp, geometry = {}, {}
        for iou in IOUS:
            key = str(iou)
            selected, metadata = select_same_fp(data, gt, ref_points[key][0]["fp"], iou,
                full_flags=computed[name][1]["saved_floor"][key][1]["flags"])
            metric, detail = _match(data, gt, selected, iou)
            if metric["fp"] != metadata["actual_fp"] or metric["fp"] > metadata["fp_budget"]:
                raise AssertionError("threshold prefix/rematching disagreement")
            same_fp[key] = dict(**metadata, reference_conf025=ref_points[key][0], selected_metrics=metric)
            if iou == .5:
                geometry["same_fp_iou0.5"] = _common_geometry(ref_points[key][1], detail, gt,
                    dict(reference="conf>=0.25", other=metadata))
        selected, metadata = select_same_k(data, ref_points["0.5"][0]["predictions"])
        k_points = {str(iou): _match(data, gt, selected, iou) for iou in IOUS}
        geometry["same_k"] = _common_geometry(ref_points["0.5"][1], k_points["0.5"][1], gt,
                                              dict(reference="conf>=0.25", other=metadata))
        conf_detail = computed[name][1]["conf025"]["0.5"][1]
        geometry["conf025"] = _common_geometry(ref_points["0.5"][1], conf_detail, gt,
                                               dict(reference="conf>=0.25", other="conf>=0.25"))
        pairs[f"{baseline}->{name}"] = dict(baseline_arm=baseline, other_arm=name, same_fp=same_fp,
            same_k=dict(**metadata, reference_conf025=ref_points["0.5"][0],
                        selected_metrics={k: v[0] for k, v in k_points.items()}),
            common_small_geometry=geometry,
            delta_coco_absolute={k: None if v is None or summaries[baseline]["coco"]["metrics"][k] is None
                                 else v-summaries[baseline]["coco"]["metrics"][k]
                                 for k, v in summaries[name]["coco"]["metrics"].items()})
    return dict(schema=SCHEMA, performance_gates=None, reference=reference,
        scope=dict(images=548, inference=INFERENCE, area_diagnostics="strict <1024 and <9216; COCO retains its default area boundaries",
            matching="class-aware score-greedy one-to-one per image, all GT sizes jointly; equal scores use original index",
            error_labels=list(ERROR_NAMES), budgets="each pair's baseline_arm at conf>=.25; empirical descriptive comparisons, not deployment thresholds",
            fp_priority="same-class IoU>=threshold: duplicate; else other-class >=threshold: wrong_class; else same-class >=.1: localization_near; else other-class >=.1: near_other_class; else background",
            geometry_normalization="center: sqrt((dx/GTwidth)^2+(dy/GTheight)^2); edges: mean absolute xyxy error divided by corresponding GT width/height",
            pair_role_names="reference fields within each pair mean that pair's baseline_arm, including control->candidate",
            zero_area_predictions="retained as FP and included in budgets/AP; reversed xyxy is invalid; GT must have positive area",
            budget_universe="all captured predictions under frozen conf>=.001/maxDet500; no extension below floor or beyond cap",
            uncapped_population="separate flag addresses confidence/maxDet truncation of post-NMS score-ranked predictions, not pre-NMS proposals",
            unavailable="sameK only when fewer than K captured records; sameFP is a valid <=budget prefix even with slack"),
        arms=summaries, pairs=pairs,
        comparisons={name: pairs[f"{reference}->{name}"] for name in order})


def _source_files():
    root = Path(__file__).resolve().parents[1]
    return {"evaluator": Path(__file__).resolve(),
            "matching": root/"src/regionssod/evaluation/precision_recall.py",
            "error_geometry": root/"src/regionssod/evaluation/error_diagnosis.py"}


def _seal(out):
    _write(out/"artifact_sha256.json", {str(p.relative_to(out)).replace("\\", "/"): digest(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "artifact_sha256.json"})


def _capture_validator():
    from ultralytics.models.yolo.detect.val import DetectionValidator
    class CaptureValidator(DetectionValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.captured = []

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for si, prediction in enumerate(preds):
                prepared = self._prepare_batch(si, batch)
                native = self.scale_preds(self._prepare_pred(prediction), prepared)
                scores = native["conf"].cpu().tolist()
                self.captured.append(dict(image=str(Path(prepared["im_file"]).resolve()),
                    shape=list(prepared["ori_shape"]), classes=native["cls"].cpu().tolist(), scores=scores,
                    xyxy=native["bboxes"].cpu().tolist(), original_indices=list(range(len(scores)))))

        def finalize_metrics(self):
            super().finalize_metrics()
            with (self.save_dir.parent/"predictions.jsonl").open("x", encoding="utf-8") as stream:
                for row in self.captured:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
    return CaptureValidator


def evaluate(weights, out):
    """Explicit future real-evaluation entrypoint; never called by audit."""
    from importlib.metadata import version
    if "reference" not in weights or len(weights) < 2 or any(not re.fullmatch(r"[A-Za-z0-9_-]+", k) for k in weights):
        raise ValueError("named reference/comparison weights required")
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("fresh output directory required; audit never reruns inference")
    if version("ultralytics") != "8.4.172":
        raise ValueError("pinned ultralytics 8.4.172 required")
    # This source only loads the full validation population when evaluate is
    # explicitly invoked. No class-ambiguity or bottleneck cache is accessed.
    if __package__:
        from .evaluate_paper1_small_pr import ground_truth, DATA
    else:
        from evaluate_paper1_small_pr import ground_truth, DATA
    from ultralytics import YOLO
    gt, inputs, counts = ground_truth()
    sources = _source_files()
    manifest = dict(schema=SCHEMA, inference=INFERENCE, population=548, counts=counts,
        versions={package: version(package) for package in ("ultralytics", "torch", "numpy", "pycocotools")},
        weights={k: dict(path=str(Path(p).resolve()), sha256=digest(p)) for k, p in weights.items()},
        capture_input_sha256=inputs, source_sha256={k: digest(p) for k, p in sources.items()},
        capture_source_sha256=digest(Path(__file__).with_name("evaluate_paper1_small_pr.py")),
        audit_policy="only this cache plus current pure-source hashes; no original data/weights and no inference fallback")
    _write(out/"manifest.json", manifest); _write(out/"ground_truth.json", gt)
    for key, path in sources.items():
        target = out/"source"/f"{key}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    arms = {}
    validator = _capture_validator()
    for arm, checkpoint in weights.items():
        started = time.monotonic()
        model = YOLO(str(checkpoint))
        result = model.val(validator=validator, data=str(DATA), project=str(out/arm),
                           name="validator", exist_ok=False, **INFERENCE)
        arms[arm] = _records(out/arm/"predictions.jsonl")
        _prepare(arms[arm], gt)
        if digest(checkpoint) != manifest["weights"][arm]["sha256"]:
            raise ValueError("weight changed during capture")
        _write(out/arm/"inference.json", dict(ultralytics=result.results_dict,
            predictions_sha256=digest(out/arm/"predictions.jsonl"),
            weights_sha256=manifest["weights"][arm]["sha256"], inference_passes=1,
            seconds=time.monotonic()-started))
        del model
    # External inputs are checked only during capture, never during audit.
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError("validation input changed during capture")
    for key, path in sources.items():
        if digest(path) != manifest["source_sha256"][key]:
            raise ValueError("evaluation source changed during capture")
    for arm, path in weights.items():
        if digest(path) != manifest["weights"][arm]["sha256"]:
            raise ValueError("weight changed before capture completed")
    report = compare_records(arms, gt)
    _write(out/"report.json", report)
    _seal(out)
    return report


def audit(out):
    """Replay only this tool's new cache; missing files fail, never infer."""
    out = Path(out).resolve()
    manifest = _read(out/"manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("inference") != INFERENCE or manifest.get("population") != 548:
        raise ValueError("not a compatible bounded-retention evaluation cache")
    artifacts = _read(out/"artifact_sha256.json")
    required = {"manifest.json", "ground_truth.json", "report.json"}
    for arm in manifest["weights"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", arm):
            raise ValueError("invalid cached arm")
        required.update((f"{arm}/predictions.jsonl", f"{arm}/inference.json"))
    required.update(f"source/{key}.py" for key in _source_files())
    if not required <= set(artifacts):
        raise ValueError("incomplete artifact binding")
    for relative, expected in artifacts.items():
        path = (out/relative).resolve()
        if not path.is_relative_to(out) or path == out:
            raise ValueError("artifact path escapes this cache")
        if digest(path) != expected:
            raise ValueError("cached artifact changed")
    for key, path in _source_files().items():
        if digest(path) != manifest["source_sha256"][key] or digest(out/"source"/f"{key}.py") != manifest["source_sha256"][key]:
            raise ValueError("replay source changed; use the frozen pure-source version")
    arms = {}
    for arm, binding in manifest["weights"].items():
        receipt = _read(out/arm/"inference.json")
        if (receipt["weights_sha256"] != binding["sha256"] or receipt["inference_passes"] != 1
                or receipt["predictions_sha256"] != digest(out/arm/"predictions.jsonl")):
            raise ValueError("arm/weight/prediction binding changed")
        arms[arm] = _records(out/arm/"predictions.jsonl")
    gt = _read(out/"ground_truth.json")
    if any(_counts(gt)[k] != v for k, v in manifest["counts"].items()):
        raise ValueError("cached GT counts changed")
    replayed = compare_records(arms, gt)
    if replayed != _read(out/"report.json"):
        raise ValueError("numeric replay differs")
    return dict(audit_ok=True, numeric_replay=True, images=548, arms=list(arms), inference_calls=0,
                external_data_or_weights_read=False, native_ultralytics_metrics="hash-bound, not recomputed", performance_gates=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("evaluate")
    capture.add_argument("--reference", required=True); capture.add_argument("--candidate", required=True)
    capture.add_argument("--control"); capture.add_argument("--out", type=Path, required=True)
    replay = sub.add_parser("audit"); replay.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit(args.out)
    else:
        weights = {k: getattr(args, k) for k in ("reference", "candidate", "control") if getattr(args, k)}
        result = evaluate(weights, args.out)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
