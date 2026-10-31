"""Independently accept only a terminal, complete confidence-regression run."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from audit_p1_small_nwd import original_process_running
import run_p1_regression_confidence as trial
from run_p1_scale_seed_pair import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-fix", action="store_true")
    options = parser.parse_args()
    trial.select_revision(validation_fix=options.validation_fix)
    OUT = trial.OUT
    launch = read(OUT / "launch_receipt.json")
    if launch.get("exit_code") != 0 or read(OUT / "status.json")["stage"] != "completed":
        raise ValueError("exit0 and all complete stages are required")
    if any(original_process_running(launch[key]) for key in ("wrapper_identity", "training_identity")):
        raise ValueError("original training process remains active")
    result = trial.collect(check=True)
    write(OUT / "independent_audit.json", {
        "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "accepted": True, "epochs": 30, "train_tiles": 11310,
        "worth_larger_training": result["worth_larger_training"],
        "scope": "terminal state, frozen sources/weights/actual arguments, two full-image prediction replays"})


if __name__ == "__main__":
    main()
