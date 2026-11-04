"""Replay bound training tensors; no model, loader, inference or training entry.

This is a diagnostic, not a student loss. Mixed-source legacy tensors cannot
establish a human-only mechanism. Outputs refuse overwrite; audit is read-only.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/positive_quality_transport_20261104"
PROTOCOL = ROOT / "docs/paper/diagnostics/positive_quality_transport/protocol.md"
SETS = ("assignment_geometry_20261031", "assignment_coverage_20261031")
CACHE = ROOT / "outputs/reliable_competitor_negative_engineering/real_batch_cache.pt"
CACHE_SHA = "8b239178522f0aa176195d5e7e781d0139efc4dffa7a070c4230ddcf355c4db9"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def verify(mapping):
    for path, expected in mapping.items():
        if sha(path) != expected:
            raise ValueError(f"SHA256 drift: {path}")


def permute_targets(target, quality):
    """Increasing targets by quality, preserving values and equal-quality groups.

    Ties inherit the original target rank, avoiding changes for equal quality.
    No clipping, renormalization, threshold, coefficient or fabricated target.
    """
    if (target.ndim != 1 or quality.shape != target.shape
            or not torch.isfinite(target).all() or not torch.isfinite(quality).all()
            or (target < 0).any() or (target > 1).any()):
        raise ValueError("invalid diagnostic target/quality")
    result = target.clone()
    active = torch.where(target > 0)[0]
    if len(active) < 2:
        return result
    by_target = active[torch.argsort(target[active], stable=True)]
    order = by_target[torch.argsort(quality[by_target], stable=True)]
    result[order] = target[by_target]
    return result


def object_row(target, scores, iou, ciou, points, box):
    target = target.double()
    moved = permute_targets(target, iou)
    assert torch.equal(target.sort().values, moved.sort().values)
    difference = iou[:, None] - iou[None, :]
    target_difference = target[:, None] - target[None, :]
    score_difference = scores[:, None] - scores[None, :]
    # Each unordered pair contributes once via the upper triangle.
    upper = torch.ones_like(difference, dtype=torch.bool).triu(1)
    inversions = (difference * target_difference < 0) & upper
    score_inversions = (difference * score_difference < 0) & upper
    geometric_pairs = (difference != 0) & upper
    outside = ~((points >= box[:2]) & (points <= box[2:])).all(1)
    return {
        "anchors": len(target), "multi": len(target) >= 2,
        "small": bool((box[2:] - box[:2]).prod() < 32**2),
        "target_mass": float(target.sum()),
        "moved_mass": float((target - moved).abs().sum() / 2),
        "changed_anchors": int((target != moved).sum()),
        "inverse_pairs": int(inversions.sum()),
        "large_gap_inverse_pairs": int((inversions & (difference.abs() > .05)).sum()),
        "score_inverse_pairs": int(score_inversions.sum()),
        "geometric_pairs": int(geometric_pairs.sum()),
        "iou_ciou_disagree_pairs": int(((difference * (ciou[:, None] - ciou[None, :]) < 0) & upper).sum()),
        "best_target_iou_gap": float(iou.max() - iou[target.argmax()]),
        "outside_mass_before": float(target[outside].sum()),
        "outside_mass_after": float(moved[outside].sum()),
        "weighted_iou_before": float((target * iou).sum()),
        "weighted_iou_after": float((moved * iou).sum()),
        "values": {"target": target.tolist(), "score": scores.tolist(),
                   "iou": iou.tolist(), "ciou": ciou.tolist(), "rearranged": moved.tolist()},
    }


def replay(cache, *, source_flags=None):
    tensor = lambda name: torch.from_numpy(cache[name])
    scores, boxes, points, labels, targets, valid = [tensor(k) for k in
        ("scores", "boxes", "points", "labels", "targets", "valid")]
    _, assigned_boxes, quality, fg, indices = TaskAlignedAssigner(
        topk=10, num_classes=10, alpha=.5, beta=6., stride=[8, 16, 32])(
            scores, boxes, points, labels, targets, valid)
    rows = []
    for image in range(len(scores)):
        for gt in torch.where(valid[image, :, 0])[0].tolist():
            cls = int(labels[image, gt, 0])
            chosen = fg[image] & (indices[image] == gt) & (quality[image, :, cls] > 0)
            if not chosen.any():
                rows.append({"image": int(cache["sample_ids"][image]), "gt": gt, "class": cls,
                             "source": "mixed_unknown" if source_flags is None else source_flags[image][gt],
                             "anchors": 0, "multi": False,
                             "small": bool((targets[image, gt, 2:] - targets[image, gt, :2]).prod() < 32**2)})
                continue
            pred = boxes[image, chosen]
            truth = assigned_boxes[image, chosen]
            row = object_row(quality[image, chosen, cls], scores[image, chosen, cls],
                bbox_iou(pred, truth, xywh=False).flatten(),
                bbox_iou(pred, truth, xywh=False, CIoU=True).flatten(),
                points[chosen], targets[image, gt])
            row.update(image=int(cache["sample_ids"][image]), gt=gt, **{"class": cls},
                       source="mixed_unknown" if source_flags is None else source_flags[image][gt])
            rows.append(row)
    return rows


def summarize(rows):
    result = {}
    for group, sub in (("all", rows), ("small", [r for r in rows if r["small"]])):
        active = [r for r in sub if r["anchors"] > 0]
        multi = [r for r in active if r["multi"]]
        total = lambda key: sum(r[key] for r in active)
        n = len(multi)
        mass = total("target_mass")
        result[group] = {
            "objects": len(sub), "active": len(active), "multi": n,
            "inverse_objects": sum(r["inverse_pairs"] > 0 for r in multi),
            "inverse_fraction": sum(r["inverse_pairs"] > 0 for r in multi) / n if n else 0.,
            "large_gap_inverse_objects": sum(r["large_gap_inverse_pairs"] > 0 for r in multi),
            "large_gap_inverse_fraction": sum(r["large_gap_inverse_pairs"] > 0 for r in multi) / n if n else 0.,
            "score_inverse_objects": sum(r["score_inverse_pairs"] > 0 for r in multi),
            **{k: total(k) for k in ("target_mass", "moved_mass", "changed_anchors", "inverse_pairs",
                "geometric_pairs", "iou_ciou_disagree_pairs", "outside_mass_before", "outside_mass_after",
                "weighted_iou_before", "weighted_iou_after")},
            "moved_fraction": total("moved_mass") / mass if mass else 0.,
        }
    s = result["small"]
    result["investment_checks"] = {"multi_small_ge100": s["multi"] >= 100,
        "inverse_small_ge10pct": s["inverse_fraction"] >= .10,
        "large_gap_small_ge5pct": s["large_gap_inverse_fraction"] >= .05,
        "mass_transport_ge1pct": s["moved_fraction"] >= .01}
    return result


def sources():
    paths = [Path(__file__).resolve(), PROTOCOL, Path(inspect.getfile(TaskAlignedAssigner)),
             Path(inspect.getfile(bbox_iou))]
    mapping = {str(p): sha(p) for p in paths}
    for name, count in zip(SETS, (32, 16)):
        root = ROOT / "outputs" / name
        manifest = read(root / "cache_manifest.json")
        actual = {str(p.resolve()) for p in (root / "raw").glob("batch_*.npz")}
        if set(manifest) != actual or len(actual) != count:
            raise ValueError("legacy cache set mismatch")
        receipt = read(root / "receipt.json")
        verify(receipt["sha256"])
        mapping.update(receipt["sha256"])
        mapping.update(manifest)
        for p in (root / "receipt.json", root / "cache_manifest.json"):
            mapping[str(p)] = sha(p)
    capture = ROOT / "outputs/reliable_competitor_negative_engineering/capture.json"
    validation = ROOT / "outputs/reliable_competitor_negative_engineering/validation_final.json"
    if sha(CACHE) != CACHE_SHA or read(capture)["cache_sha256"] != CACHE_SHA:
        raise ValueError("initialization batch cache drift")
    old = read(validation)
    if old.get("accepted") is not True or old["artifact_sha256"].get(str(CACHE)) != CACHE_SHA:
        raise ValueError("unbound initialization cache")
    for p in (CACHE, capture, validation):
        mapping[str(p)] = sha(p)
    verify(mapping)
    return mapping


def calculate():
    groups, details = {}, {}
    for name in SETS:
        rows = []
        for path in sorted((ROOT / "outputs" / name / "raw").glob("batch_*.npz")):
            with np.load(path, allow_pickle=False) as cache:
                rows.extend(replay(cache))
        groups[name] = summarize(rows)
        details[name] = rows
    # Existing AMP initialization tensors are inspected as provenance inventory
    # only; no trained-human diagnostic is inferred from them.
    batch = torch.load(CACHE, map_location="cpu", weights_only=True)["batch"]
    from regionssod.training.reliable_competitor_negative import validate_batch_provenance
    validate_batch_provenance(batch, 10)
    initialization = {"images": len(batch["img"]), "human_objects": int(batch["object_is_human"].sum()),
        "pseudo_objects": int((~batch["object_is_human"]).sum()),
        "trained_checkpoint": False, "used_for_investment_statistics": False}
    checks = {name: all(value["investment_checks"].values()) for name, value in groups.items()}
    summary = {"groups": groups, "initialization_cache": initialization,
        "checks": checks, "trained_explicit_human_source_available": False,
        "decision": "SIGNAL_INSUFFICIENT" if all(checks.values()) else "REJECT_GEOMETRIC_REARRANGEMENT",
        "implementation_authorized_by_diagnostic": False,
        "new_model_forwards": 0, "new_real_batches": 0, "optimizer_steps": 0,
        "student_performance": "NOT_TESTED", "mechanism": "NOT_ESTABLISHED"}
    return summary, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    if args.audit:
        manifest = read(OUT / "manifest.json")
        verify(manifest["sources"])
        verify(manifest["artifacts"])
        # Exact new-source set equality rejects silent cache injection/removal.
        if sources() != manifest["sources"]:
            raise ValueError("source inventory drift")
        result, rows = calculate()
        if result != read(OUT / "summary.json") or rows != read(OUT / "objects.json"):
            raise ValueError("deterministic diagnostic replay mismatch")
        text = {"accepted": True, "source_bindings": len(manifest["sources"]),
                "decision": result["decision"], "readonly": True}
    else:
        if OUT.exists():
            raise FileExistsError(OUT)
        mapping = sources()
        result, rows = calculate()
        verify(mapping)
        OUT.mkdir()
        write_new(OUT / "summary.json", result)
        write_new(OUT / "objects.json", rows)
        write_new(OUT / "manifest.json", {"sources": mapping,
            "artifacts": {str(OUT / f): sha(OUT / f) for f in ("summary.json", "objects.json")},
            "elapsed_seconds": time.perf_counter() - started})
        text = {"accepted": True, "decision": result["decision"], "groups": result["groups"]}
    print(json.dumps(text, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
