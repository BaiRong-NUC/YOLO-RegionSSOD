"""Complete frozen-P1 loss replay on two existing caches, no model forward.

Autograd is through saved head tensors, not model parameters. Never constructs a
dataset, optimizer or training loop. Does not estimate student performance.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import inspect
from importlib.metadata import version
import json
from pathlib import Path

import torch
from ultralytics.utils.metrics import bbox_iou

from diagnose_positive_quality_transport import ROOT, CACHE, sha, read, verify, write_new, object_row, summarize
from validate_reliable_competitor_negative import nn_fixture
from regionssod.training.p1 import P1DetectionLoss
from regionssod.training.positive_quality_transport import IMPLEMENTATION, PositiveQualityLoss, PositiveQualityModel

OUT = ROOT / "outputs/positive_quality_transport_engineering"
TRAINED = ROOT / "outputs/positive_quality_transport_source_20261104"


def evidence():
    for path in (ROOT / "outputs/positive_quality_transport_20261104/manifest.json", TRAINED / "manifest.json"):
        m = read(path)
        verify(m["sources"])
        verify(m["artifacts"])
    capture = read(TRAINED / "capture.json")
    if (capture["new_model_forwards"] != 1 or capture["optimizer_steps"] != 0
            or capture["new_dataloader_batches"] != 0 or capture["state_sha_before"] != capture["state_sha_after"]):
        raise ValueError("single-forward budget or model-state invariance failed")


def clone_predictions(pred):
    return {key: value.detach().clone().requires_grad_(True) if key in ("scores", "boxes") else value
            for key, value in pred.items()}


def check_cache(path):
    cache = torch.load(path, map_location="cuda:0", weights_only=True)
    model = nn_fixture(cache)
    model.positive_quality_diagnostics = True
    batch = cache["batch"]
    results, gradients, criteria = {}, {}, {}
    for mode in ("p1", "reference", "candidate", "control"):
        pred = clone_predictions(cache["predictions"])
        model.positive_quality_mode = mode
        criterion = P1DetectionLoss(model) if mode == "p1" else PositiveQualityLoss(model)
        with torch.autocast("cuda", dtype=torch.float16):
            result = criterion.get_assigned_targets_and_loss(pred, batch)
            objective = result[1].sum()
        gradients[mode] = tuple(g.detach() for g in torch.autograd.grad(objective, (pred["scores"], pred["boxes"])))
        results[mode], criteria[mode] = result, criterion
        assert torch.isfinite(result[1]).all()
        assert all(torch.isfinite(g).all() for g in gradients[mode])
    original = results["p1"]
    for mode in ("reference", "candidate", "control"):
        assert all(torch.equal(a, b) for a, b in zip(original[0], results[mode][0], strict=True))
        assert torch.equal(original[1][[0, 2]], results[mode][1][[0, 2]])
        assert torch.equal(gradients["p1"][1], gradients[mode][1])
    assert torch.equal(original[1], results["reference"][1])
    assert torch.equal(gradients["p1"][0], gradients["reference"][0])
    human = batch["object_is_human"]
    labels, _, raw, foreground, local = criteria["candidate"].last_assignment
    original_targets = criteria["candidate"].last_original_targets
    allowed = torch.zeros_like(original_targets, dtype=torch.bool)
    human_rows = []
    per_mode = {mode: {"transport_L1": 0., "signed_gradient_sum_max_error": 0., "changed_anchors": 0}
                for mode in ("candidate", "control")}
    max_transport_error = 0.
    max_added_gradient_error = 0.
    for image in range(len(foreground)):
        objects = batch["batch_idx"] == image
        for gt in local[image, foreground[image]].unique(sorted=True).tolist():
            if not human[objects][gt]:
                continue
            cls = int(batch["object_class"][objects][gt])
            at = torch.where(foreground[image] & (local[image] == gt) & (original_targets[image, :, cls] > 0))[0]
            if not len(at):
                continue
            allowed[image, at, cls] = True
            values = original_targets[image, at, cls]
            displacements = []
            gradient_displacements = []
            for mode in per_mode:
                actual = criteria[mode].last_targets[image, at, cls]
                if mode == "candidate":
                    assert torch.equal(values.sort().values, actual.sort().values)
                else:
                    assert actual.min() >= values.min() - 1e-7 and actual.max() <= values.max() + 1e-7
                    torch.testing.assert_close(actual.double().sum(), values.double().sum(), rtol=1e-6, atol=2e-6)
                delta = actual.double() - values.double()
                displacements.append(delta.abs().sum())
                dg = gradients[mode][0][image, cls, at].double() - gradients["p1"][0][image, cls, at].double()
                gradient_displacements.append(dg.abs().sum())
                per_mode[mode]["transport_L1"] += float(delta.abs().sum())
                per_mode[mode]["changed_anchors"] += int((actual != values).sum())
                per_mode[mode]["signed_gradient_sum_max_error"] = max(
                    per_mode[mode]["signed_gradient_sum_max_error"], abs(float(delta.sum())))
            # Eligibility statistics now follow actual CUDA AMP TAL and actual
            # target_scores.to(logits.dtype), addressing the FP32 proxy gap.
            torch.testing.assert_close(*displacements, rtol=2e-6, atol=2e-6)
            max_transport_error = max(max_transport_error, float((displacements[0] - displacements[1]).abs()))
            # FP16 leaf gradients round independently, so actual autograd is
            # checked at a declared half-precision absolute/relative tolerance.
            torch.testing.assert_close(*gradient_displacements, rtol=.02, atol=2e-6)
            max_added_gradient_error = max(max_added_gradient_error,
                float((gradient_displacements[0] - gradient_displacements[1]).abs()))
            q = criteria["candidate"].last_quality[image, at]
            truth = original[0][2][image, at].float()
            predicted = criteria["candidate"].last_predicted_boxes[image, at].float()
            ciou = bbox_iou(predicted, truth, xywh=False, CIoU=True).flatten()
            points = original[0][3][at] * original[0][4][at]
            row = object_row(values.cpu(), cache["predictions"]["scores"][image, cls, at].float().sigmoid().cpu(),
                q.cpu(), ciou.cpu(), points.cpu(), truth[0].cpu())
            human_rows.append(row)
    base_grad = gradients["p1"][0].permute(0, 2, 1)
    for mode in ("candidate", "control"):
        target = criteria[mode].last_targets
        assert torch.equal(target[~allowed], original_targets[~allowed])
        assert torch.equal(target > 0, original_targets > 0)
        actual_grad = gradients[mode][0].permute(0, 2, 1)
        assert torch.equal(actual_grad[~allowed], base_grad[~allowed])
    # Eval with already-cached training criterion must use original P1 fallback.
    ordinary = {k: v for k, v in batch.items() if not k.startswith("object_") and k != "pseudo_weights"}
    model.eval()
    model.criterion = criteria["candidate"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        actual = PositiveQualityModel.loss(model, ordinary, cache["predictions"])
        expected = P1DetectionLoss(model)(cache["predictions"], ordinary)
    assert torch.equal(actual[0], expected[0])
    return {"cache_sha256": sha(path), "assignment_exact": True, "original_denominator_retained": True,
        "box_DFL_and_distribution_gradients_exact": True, "reference_loss_and_gradients_exact": True,
        "untriggered_gradients_exact": True, "pseudo_and_negative_targets_exact": True,
        "candidate_per_object_target_multisets_exact": True, "effective_support_exact": True,
        "control_target_L1_max_abs_error": max_transport_error,
        "control_added_head_gradient_L1_max_abs_error": max_added_gradient_error,
        "control_preserves_multiset": False,
        "BCE_target_dtype": str(original_targets.dtype), "local_head_tensor_autograd_only": True,
        "human_actual_amp": summarize(human_rows), "actions": per_mode,
        "loss_components": {m: r[1].detach().float().cpu().tolist() for m, r in results.items()}}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    evidence()
    torch.set_num_threads(4)
    caches = [CACHE, TRAINED / "full_loss_cache.pt"]
    results = {"initialization": check_cache(caches[0]), "trained": check_cache(caches[1])}
    if not all(results["trained"]["human_actual_amp"]["investment_checks"].values()):
        raise ValueError("actual AMP path does not meet the prior human-signal lines")
    paths = list((ROOT / "src/regionssod").rglob("*.py"))
    paths += [Path(__file__).resolve(), ROOT / "scripts/validate_reliable_competitor_negative.py",
        ROOT / "scripts/diagnose_positive_quality_transport.py",
        ROOT / "docs/paper/candidates/positive_quality_transport/engineering_protocol.md"]
    paths += [Path(inspect.getfile(P1DetectionLoss))]
    sources = {str(p): sha(p) for p in paths}
    artifacts = {str(p): sha(p) for p in caches}
    for root in (ROOT / "outputs/positive_quality_transport_20261104", TRAINED):
        artifacts[str(root / "manifest.json")] = sha(root / "manifest.json")
    capture = read(TRAINED / "capture.json")
    receipt = {"accepted": True, "implementation": IMPLEMENTATION, "optimizer_steps": 0,
        "parameter_hash_unchanged": capture["state_sha_before"] == capture["state_sha_after"],
        "checks_actual_amp_path": True, "new_dataloader_batches": 0, "new_model_forwards_this_script": 0,
        "torch_version": version("torch"), "ultralytics_version": version("ultralytics"),
        "real_batch": {"images": 8, "amp": True, "device": "cuda:0", "source": "previously_bound_cached_batch"},
        "source_sha256": sources, "artifact_sha256": artifacts, "results": results,
        "student_performance": "NOT_TESTED", "mechanism": "NOT_ESTABLISHED"}
    OUT.mkdir()
    write_new(OUT / "validation.json", receipt)
    print(json.dumps({"accepted": True, "checks_actual_amp_path": True,
        "human_actual_amp": results["trained"]["human_actual_amp"], "actions": results["trained"]["actions"]}))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
