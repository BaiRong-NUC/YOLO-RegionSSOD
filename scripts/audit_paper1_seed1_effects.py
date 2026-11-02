"""Replay the accepted seed1 pair with the frozen seed0 diagnostic procedure."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path

import audit_paper1_full100_effects as effects
import run_paper1_seed1_full100 as repeat
from run_p1_scale_seed_pair import ROOT, digest, read, write

OUT = ROOT / "outputs/paper1_full100_effects_s1"


def main():
    for arm in ("b2", "p1"):
        record = read(repeat.paths(arm)["out"] / "independent_audit.json")
        if (record.get("accepted") is not True or record.get("epochs") != 100
                or record.get("student_seed") != 1 or record.get("split_seed") != 0):
            raise ValueError("accepted paired student seed1/split0 full100 results required")
    # Reuse every matching, threshold, image and bootstrap rule in a separate
    # process. Only the arm locations/output change; seed0 files stay untouched.
    effects.paths = repeat.paths
    effects.OUT = OUT
    with redirect_stdout(io.StringIO()):
        effects.main()
    result = read(OUT / "analysis.json")
    result["student_seed"] = 1
    result["split_seed"] = 0
    result["limits"][-1] = (
        "seed1 B2 uninterrupted; seed1 P1 resumed after epoch55; "
        "seed0 B2 resumed after epoch63; equal epochs have unequal data and compute")
    result["sha256"][str(Path(__file__).resolve())] = digest(__file__)
    for filename in ("scripts/audit_training_investment.py",
                     "src/regionssod/evaluation/size_recall.py"):
        result["sha256"][str(ROOT / filename)] = digest(ROOT / filename)
    write(OUT / "analysis.json", result)
    print({key: value for key, value in result.items() if key != "sha256"}, flush=True)


if __name__ == "__main__":
    main()
