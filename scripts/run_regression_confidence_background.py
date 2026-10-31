"""One-shot wrapper retaining process identity, logs and actual exit code."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/p1_regression_confidence_pilot30_s0"


def identity(pid):
    process = psutil.Process(pid)
    return {"pid": pid, "created_at": datetime.fromtimestamp(
        process.create_time(), timezone.utc).isoformat(), "command": process.cmdline()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"wrapper_pid": os.getpid(),
               "wrapper_identity": identity(os.getpid()),
               "started_at": datetime.now(timezone.utc).isoformat(),
               "command": [sys.executable, "-u", str(ROOT / "scripts/run_p1_regression_confidence.py")],
               "cwd": str(ROOT), "scope": "one independent full30 student; never retry or resume"}
    path = OUT / "launch_receipt.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)

    def update():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
        temporary.replace(path)

    with (OUT / "stdout.log").open("x", encoding="utf-8") as stdout, \
            (OUT / "stderr.log").open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(receipt["command"], cwd=ROOT, stdout=stdout, stderr=stderr)
        receipt.update(training_pid=process.pid, training_identity=identity(process.pid))
        update()
        exit_code = process.wait()
    receipt.update(exit_code=exit_code, finished_at=datetime.now(timezone.utc).isoformat())
    update()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
