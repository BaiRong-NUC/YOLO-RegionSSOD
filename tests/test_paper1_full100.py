import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_paper1_full100 as formal
from audit_training_investment import paired_counts


def test_main_formal_pair_preserves_pilot_methods_and_initialization():
    b2, p1 = (formal.configuration(arm) for arm in formal.ARMS)
    assert b2["train"] == p1["train"]
    assert b2["train"]["epochs"] == 100 and b2["train"]["seed"] == 0
    assert b2["model"] == p1["model"]
    assert b2.get("method", "baseline") == "baseline" and "p1" not in b2
    assert p1["method"] == "p1"
    assert p1["p1"] == {"soft_weights": True, "uncertain_ignore": True}
    assert "assignment" not in p1 and "assignment" not in b2
    assert formal.paths("p1")["run"] != formal.ROOT / "runs/partial10_p1_full100_s0"


def decision_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(formal, "ROOT", tmp_path)
    decision = tmp_path / "decision.json"
    monkeypatch.setattr(formal, "DECISION", decision)
    out = tmp_path / "outputs/b2_pilot30_s1"
    out.mkdir(parents=True)
    (out / "comparison.json").write_text('{"historical_gate3": false}')
    (out / "preflight.json").write_text('{"sha256": {}}')
    decision.write_text(json.dumps({
        "recommendation": "user_run_b2_p1_full100_seed0",
        "paired_evidence_sha256": formal.digest(out / "comparison.json"),
    }))
    monkeypatch.setattr(formal, "load_main_pair", lambda hashes: None)
    return out


def test_research_decision_requires_independent_audit(tmp_path, monkeypatch):
    decision_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="independent paired"):
        formal.eligible_evidence()


def test_new_research_decision_can_keep_old_gate_failed(tmp_path, monkeypatch):
    out = decision_fixture(tmp_path, monkeypatch)
    (out / "independent_audit.json").write_text("{}")
    assert formal.eligible_evidence() == {}
    assert json.loads((out / "comparison.json").read_text())["historical_gate3"] is False
    (out / "comparison.json").write_text('{"historical_gate3": true}')
    with pytest.raises(ValueError, match="paired evidence differs"):
        formal.eligible_evidence()


@pytest.mark.parametrize("exit_code,stage", [(None, "completed"), (1, "completed"), (0, "training")])
def test_audit_requires_real_successful_exit(tmp_path, monkeypatch, exit_code, stage):
    monkeypatch.setattr(formal, "paths", lambda arm: {"out": tmp_path})
    (tmp_path / "launch_receipt.json").write_text(json.dumps({"exit_code": exit_code}))
    (tmp_path / "status.json").write_text(json.dumps({"stage": stage}))
    monkeypatch.setattr(formal, "measure", lambda arm: pytest.fail("must reject before reading metrics"))
    with pytest.raises(ValueError, match="real exit0"):
        formal.audit("b2")


def test_p1_training_waits_for_completed_b2_audit(monkeypatch):
    def fail_audit(arm):
        assert arm == "b2"
        raise ValueError("B2 incomplete")

    monkeypatch.setattr(formal, "audit", fail_audit)
    monkeypatch.setattr(formal, "preflight", lambda arm: pytest.fail("must wait for B2"))
    with pytest.raises(ValueError, match="B2 incomplete"):
        formal.train("p1")


def test_net_gain_does_not_hide_lost_objects():
    import numpy as np

    before = [np.array([True, False, False])]
    after = [np.array([False, True, True])]
    result = paired_counts(before, after, {"small": [np.ones(3, dtype=bool)]})["small"]
    assert result == {
        "gained": 2, "lost": 1, "net": 1,
        "images_helped": 1, "images_harmed": 0, "images_tied": 0,
    }
