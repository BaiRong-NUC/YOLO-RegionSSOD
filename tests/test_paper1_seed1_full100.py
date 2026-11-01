import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_paper1_seed1_full100 as repeat


def test_repeat_changes_only_student_seed_and_output_name():
    for arm in repeat.ARMS:
        old = repeat.seed0.configuration(arm)
        new = repeat.configuration(arm)
        assert new["train"] == old["train"] | {"seed": 1}
        assert new == old | {"train": new["train"], "name": new["name"]}
        assert new["name"].endswith("_s1")
        assert repeat.paths(arm)["out"] != repeat.seed0.paths(arm)["out"]
        assert new["train"]["epochs"] == 100


def test_incomplete_baseline_blocks_second_arm(monkeypatch):
    def reject(arm):
        assert arm == "b2"
        raise ValueError("incomplete baseline")

    monkeypatch.setattr(repeat, "audit", reject)
    monkeypatch.setattr(repeat, "preflight", lambda arm: pytest.fail("must wait for baseline"))
    with pytest.raises(ValueError, match="incomplete baseline"):
        repeat.train("p1")


def test_existing_output_blocks_before_expensive_evidence_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(repeat, "ROOT", Path.cwd().resolve())
    monkeypatch.setattr(repeat, "paths", lambda arm: {"out": tmp_path})
    monkeypatch.setattr(repeat, "basis_hashes", lambda: pytest.fail("must reject existing output first"))
    with pytest.raises(FileExistsError):
        repeat.preflight("b2")


@pytest.mark.parametrize("exit_code,stage", [(None, "completed"), (1, "completed"), (0, "training")])
def test_audit_requires_actual_successful_exit(tmp_path, monkeypatch, exit_code, stage):
    monkeypatch.setattr(repeat, "paths", lambda arm: {"out": tmp_path})
    (tmp_path / "launch_receipt.json").write_text(json.dumps({"exit_code": exit_code}))
    (tmp_path / "status.json").write_text(json.dumps({"stage": stage}))
    monkeypatch.setattr(repeat, "measure", lambda arm: pytest.fail("must reject before measuring"))
    with pytest.raises(ValueError, match="real exit0"):
        repeat.audit("b2")


def test_pilot_duration_cannot_be_mixed_into_formal_pair():
    formal = {"epochs": 100, "metrics": {"mAP50_95": .25, "small_recall": .40}}
    with pytest.raises(ValueError, match="100-epoch"):
        repeat.paired_summary(formal | {"epochs": 30}, formal)
    result = repeat.paired_summary(formal, formal | {
        "metrics": {"mAP50_95": .251, "small_recall": .42}})
    assert result["delta_p1_minus_b2_pp"]["mAP50_95"] == pytest.approx(.1)
    assert result["delta_p1_minus_b2_pp"]["small_recall"] == pytest.approx(2.)


def test_reauditing_seed0_does_not_freeze_mutable_acceptance_timestamp(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("frozen source")
    outputs = {arm: tmp_path / arm for arm in repeat.ARMS}
    for out in outputs.values():
        out.mkdir()
        (out / "metrics.json").write_text("{}")
    (outputs["p1"] / "preflight.json").write_text(json.dumps({
        "sha256": {str(source): repeat.digest(source)}}))
    (outputs["p1"] / "paired_comparison.json").write_text("{}")
    calls = []

    def audit(arm):
        calls.append(arm)
        (outputs[arm] / "independent_audit.json").write_text(str(len(calls)))

    monkeypatch.setattr(repeat.seed0, "audit", audit)
    monkeypatch.setattr(repeat.seed0, "paths", lambda arm: {"out": outputs[arm]})
    monkeypatch.setattr(repeat.seed0, "artifacts", lambda arm: [outputs[arm] / "metrics.json"])
    first = repeat.basis_hashes()
    second = repeat.basis_hashes()
    assert first == second
    assert str(outputs["p1"] / "independent_audit.json") not in first
