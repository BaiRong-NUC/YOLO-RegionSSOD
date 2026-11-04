"""One forward of an existing bound 8-image tensor batch; no dataset or optimizer.

Subsequent --audit only replays the saved tensors. No automatic retry/resume.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.tal import make_anchors

from diagnose_positive_quality_transport import CACHE, ROOT, sha, read, write_new, verify, replay, summarize, sources
from regionssod.training.p1 import P1DetectionLoss
from regionssod.training.reliable_competitor_negative import validate_batch_provenance

OUT = ROOT / "outputs/positive_quality_transport_source_20261104"
WEIGHTS = ROOT / "runs/partial10_p1_pilot30_s0/weights/best.pt"
WEIGHT_SHA = "8d20c9645328d9f1a18429c7223d40e3a123f211ec04e86b2f697a844a51e115"
PROTOCOL = ROOT / "docs/paper/diagnostics/positive_quality_transport/source_check_protocol.md"


def state_sha(model):
    import hashlib
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        h.update(key.encode())
        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def collect():
    if OUT.exists():
        raise FileExistsError(OUT)
    mapping = sources()
    if sha(WEIGHTS) != WEIGHT_SHA:
        raise ValueError("trained P1 checkpoint changed")
    for p in (Path(__file__).resolve(), PROTOCOL, WEIGHTS,
              Path(inspect.getfile(validate_batch_provenance)), Path(inspect.getfile(P1DetectionLoss))):
        mapping[str(p)] = sha(p)
    source = torch.load(CACHE, map_location="cpu", weights_only=True)
    batch = source["batch"]
    validate_batch_provenance(batch, 10)
    if batch["img"].shape != (8, 3, 640, 640):
        raise ValueError("requires the original eight-image cache")
    model = YOLO(str(WEIGHTS)).model.float().cuda().eval()
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    model.args = source["args"]
    # Detection head raw outputs; all BN layers remain in eval mode.
    model.model[-1].training = True
    criterion = P1DetectionLoss(model)
    before = state_sha(model)
    OUT.mkdir()
    write_new(OUT / "started.json", {"source_sha256": mapping, "forward_budget": 1,
        "note": "Directory is an exclusive claim; never auto-retry an interrupted capture."})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pred = model(batch["img"].cuda())
        points, stride = make_anchors(pred["feats"], model.model[-1].stride, .5)
        boxes = criterion.bbox_decode(points, pred["boxes"].permute(0, 2, 1).contiguous()) * stride
        scores = pred["scores"].permute(0, 2, 1).contiguous().sigmoid()
    after = state_sha(model)
    if before != after:
        raise ValueError("model parameters or buffers changed")
    combined = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = criterion.preprocess(combined.cuda(), 8,
        scale_tensor=torch.tensor([640.] * 4, device="cuda")).cpu()
    flags = [["human" if h else "pseudo" for h in batch["object_is_human"][batch["batch_idx"] == i].tolist()]
             for i in range(8)]
    # Store raw prediction dictionaries too, enabling later complete-loss replay
    # without spending another real forward. This is not a student checkpoint.
    def cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: cpu(v) for k, v in x.items()}
        if isinstance(x, list):
            return [cpu(v) for v in x]
        return x
    torch.save(source | {"predictions": cpu(pred)}, OUT / "full_loss_cache.pt")
    np.savez_compressed(OUT / "tensors.npz", scores=scores.float().cpu().numpy(),
        boxes=boxes.float().cpu().numpy(), points=(points * stride).cpu().numpy(),
        labels=targets[..., :1].numpy(), targets=targets[..., 1:].numpy(),
        valid=(targets[..., 1:].sum(-1, keepdim=True) > 0).numpy(), sample_ids=np.arange(8))
    write_new(OUT / "capture.json", {"source_flags": flags, "new_model_forwards": 1,
        "new_dataloader_batches": 0, "cached_input_images": 8, "amp": "float16",
        "state_sha_before": before, "state_sha_after": after, "optimizer_steps": 0,
        "student_training_iterations": 0, "val_forwards": 0})
    verify(mapping)
    summary, rows = calculate()
    write_new(OUT / "summary.json", summary)
    write_new(OUT / "objects.json", rows)
    artifacts = [OUT / n for n in ("started.json", "full_loss_cache.pt", "tensors.npz", "capture.json", "summary.json", "objects.json")]
    write_new(OUT / "manifest.json", {"sources": mapping, "artifacts": {str(p): sha(p) for p in artifacts}})
    return summary


def calculate():
    with np.load(OUT / "tensors.npz", allow_pickle=False) as cache:
        rows = replay(cache, source_flags=read(OUT / "capture.json")["source_flags"])
    groups = {key: summarize([r for r in rows if r["source"] == key]) for key in ("human", "pseudo")}
    passed = all(groups["human"]["investment_checks"].values())
    return {"groups": groups, "human_signal_passed": passed,
        "decision": "ELIGIBLE_FOR_MINIMAL_ENGINEERING" if passed else "INSUFFICIENT_HUMAN_SIGNAL",
        "student_performance": "NOT_TESTED", "mechanism": "NOT_ESTABLISHED",
        "sampling": "existing convenience batch; not independent confirmation"}, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.audit:
        manifest = read(OUT / "manifest.json")
        verify(manifest["sources"])
        verify(manifest["artifacts"])
        summary, rows = calculate()
        if summary != read(OUT / "summary.json") or rows != read(OUT / "objects.json"):
            raise ValueError("source diagnostic replay drift")
        print(json.dumps({"accepted": True, "readonly": True, "decision": summary["decision"]}))
    else:
        print(json.dumps(collect()))


if __name__ == "__main__":
    main()
