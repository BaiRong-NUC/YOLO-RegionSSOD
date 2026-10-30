"""Audit the terminal full-data NWD30 run without training or inference."""
from datetime import datetime
from zoneinfo import ZoneInfo

import psutil

from run_p1_scale_seed_pair import read, write
from run_p1_small_nwd import OUT, collect


def original_process_running(identity):
    try:
        process = psutil.Process(identity["pid"])
        expected = datetime.fromisoformat(identity["created_at"]).timestamp()
        return (abs(process.create_time() - expected) < 1.
                and process.cmdline() == identity["command"])
    except psutil.NoSuchProcess:
        return False


def main():
    launch = read(OUT / "launch_receipt.json")
    if launch.get("exit_code") != 0 or read(OUT / "status.json")["stage"] != "completed":
        raise ValueError("wrapper exit0 and all stages completed are required")
    if any(original_process_running(launch[key])
           for key in ("wrapper_identity", "training_identity")):
        raise ValueError("original wrapper/training process is still running")
    result = collect(check=True)
    write(OUT / "independent_audit.json", {
        "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "accepted": True, "epochs": 30, "train_tiles": 11310,
        "worth_larger_training": result["worth_larger_training"],
        "scope": "terminal-state, input/artifact SHA256, actual arguments and two full-val replays",
    })


if __name__ == "__main__":
    main()
