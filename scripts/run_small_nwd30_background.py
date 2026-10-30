"""One-shot 30-epoch wrapper; retain process identities/logs and never retry."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/p1_small_nwd_pilot30_s0"


def identity(pid):
    process = psutil.Process(pid)
    return {"pid": pid, "created_at": datetime.fromtimestamp(
        process.create_time(), timezone.utc).isoformat(), "command": process.cmdline()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "launch_receipt.json"
    value = {"wrapper_pid": os.getpid(), "wrapper_identity": identity(os.getpid()),
             "started_at": datetime.now(timezone.utc).isoformat(), "cwd": str(ROOT),
             "command": [sys.executable, "-u", str(ROOT / "scripts/run_p1_small_nwd.py")],
             "scope": "one full-data 30-epoch student from COCO; cached full P1; no restart"}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)

    def update():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    with (OUT / "stdout.log").open("x", encoding="utf-8") as stdout, \
            (OUT / "stderr.log").open("x", encoding="utf-8") as stderr:
        child = subprocess.Popen(value["command"], cwd=ROOT, stdout=stdout, stderr=stderr)
        value.update(training_pid=child.pid, training_identity=identity(child.pid))
        update()
        exit_code = child.wait()
    value.update(exit_code=exit_code, finished_at=datetime.now(timezone.utc).isoformat())
    update()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
