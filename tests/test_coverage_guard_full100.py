import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_coverage_guard_full100 as formal


def test_generated_formal_pair_is_fair_and_independent_of_pilot_duration():
    baseline = formal.configuration("p1")
    candidate = formal.configuration("coverage_guard")
    assert candidate == baseline | {
        "name": "partial10_coverage_guard_full100_s0", "assignment": "image_coverage_guard"}
    assert baseline["train"]["epochs"] == 100
    assert baseline["train"]["seed"] == 0
    baseline["train"]["epochs"] = 7
    assert formal.configuration("p1")["train"]["epochs"] == 100
    assert all(formal.paths(arm)["run"].name.endswith("full100_s0") for arm in formal.ARMS)


def test_missing_pilot_cannot_start_large_training(tmp_path, monkeypatch):
    monkeypatch.setattr(formal, "PILOT", tmp_path)
    with pytest.raises(ValueError, match="audited positive"):
        formal.pilot_eligible()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("epochs,nonfinite", [(30, False), (99, False), (100, True), (100, False)])
def test_formal_audit_accepts_only_complete_finite_training(tmp_path, epochs, nonfinite):
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "time"])
        writer.writerows((epoch, "nan" if nonfinite and epoch == 100 else epoch * 300)
                         for epoch in range(1, epochs + 1))
    if epochs == 100 and not nonfinite:
        assert len(formal.complete_epochs(path)) == 100
    else:
        with pytest.raises(ValueError, match="100 continuous finite"):
            formal.complete_epochs(path)
