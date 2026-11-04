"""Independent repair receipt: synthetic dataloaders + existing head-cache replay.

Never creates a detector/optimizer or loads a new real batch. Output is exclusive.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import torch

from validate_positive_quality_transport import check_cache, evidence
from diagnose_positive_quality_transport import ROOT, CACHE, sha, read, verify, write_new
from regionssod.training.positive_quality_pipeline_check import check_pipeline
from regionssod.training.positive_quality_transport_v2 import IMPLEMENTATION

OUT = ROOT / "outputs/positive_quality_transport_v2_engineering"
LEGACY = ROOT / "outputs/positive_quality_transport_engineering/validation.json"
FAILED = ROOT / "outputs/positive_quality_transport_three30_s0"
TERMINAL = ROOT / "outputs/pqt_failure_terminal_receipt_20261104_1048.json"


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    evidence()
    old = read(LEGACY)
    if old.get("accepted") is not True or old.get("optimizer_steps") != 0:
        raise ValueError("frozen loss receipt not accepted")
    verify(old["source_sha256"])
    verify(old["artifact_sha256"])
    failed_files = read(FAILED / "artifact_sha256.json")
    verify(failed_files)
    terminal = read(TERMINAL)
    verify(terminal["frozen_file_sha256"])
    if (terminal["completed_epochs"] != 0 or terminal["audit_process_exit_code"] != 0
            or terminal["audit"].get("terminal_evidence_verified") is not True
            or terminal["audit"].get("accepted") is not False
            or read(FAILED / "status.json")["stage"] != "failed"
            or read(FAILED / "reference_exit.json")["exit_code"] != 1
            or sha(terminal["audit_log"]) != terminal["audit_log_sha256"]):
        raise ValueError("V1 failure evidence differs")
    OUT.mkdir()
    pipeline = check_pipeline(OUT / "synthetic_fixture")
    write_new(OUT / "trainer_pipeline.json", pipeline)
    torch.set_num_threads(4)
    trained = ROOT / "outputs/positive_quality_transport_source_20261104/full_loss_cache.pt"
    results = {"initialization": check_cache(CACHE), "trained": check_cache(trained)}
    if not all(results["trained"]["human_actual_amp"]["investment_checks"].values()):
        raise ValueError("same prior AMP signal lines not met")
    # The repaired Trainer inherits get_model/criterion from the frozen module;
    # replay therefore tests precisely its unchanged scientific loss/model.
    paths = list((ROOT / "src/regionssod").rglob("*.py"))
    paths += [Path(__file__).resolve(), ROOT / "scripts/validate_positive_quality_transport.py",
        ROOT / "scripts/validate_reliable_competitor_negative.py",
        ROOT / "scripts/diagnose_positive_quality_transport.py",
        ROOT / "docs/paper/candidates/positive_quality_transport_v2/engineering_protocol.md"]
    sources = {str(p.resolve()): sha(p) for p in paths}
    artifacts = {str(p): sha(p) for p in (CACHE, trained, LEGACY,
        OUT / "trainer_pipeline.json", FAILED / "artifact_sha256.json", TERMINAL)}
    artifacts.update(failed_files)
    capture = read(trained.parent / "capture.json")
    receipt = {"accepted": True, "implementation": IMPLEMENTATION, "optimizer_steps": 0,
        "parameter_hash_unchanged": capture["state_sha_before"] == capture["state_sha_after"],
        "checks_actual_amp_path": True, "trainer_pipeline_checks": True,
        "new_dataloader_batches": 0, "new_model_forwards_this_script": 0,
        "new_synthetic_batches": 6, "pipeline_evidence": str(OUT / "trainer_pipeline.json"),
        "torch_version": version("torch"), "ultralytics_version": version("ultralytics"),
        "real_batch": {"images": 8, "amp": True, "device": "cuda:0", "source": "previously_bound_cached_batch"},
        "source_sha256": sources, "artifact_sha256": artifacts, "results": results,
        "student_performance": "NOT_TESTED", "mechanism": "NOT_ESTABLISHED",
        "limitation": "Synthetic loader check and saved-head loss replay, not full trainer setup or optimization"}
    write_new(OUT / "validation.json", receipt)
    print(json.dumps({"accepted": True, "trainer_pipeline_checks": True, "new_real_batches": 0,
        "new_synthetic_batches": 6, "model_forwards": 0, "optimizer_steps": 0}))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
