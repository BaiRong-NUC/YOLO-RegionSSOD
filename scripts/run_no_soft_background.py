"""One-shot background wrapper; dispatched via WMI outside the tool shell."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/p1_no_soft_weights_pilot30_s0"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / "launch_receipt.json"
    receipt = {"wrapper_pid": os.getpid(),
               "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "cwd": str(ROOT),
               "command": [sys.executable, "-u", str(ROOT / "scripts/run_p1_no_soft_weights.py")],
               "mechanism": "WMI hidden process; one child; no automatic restart"}
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    with (OUT / "stdout.log").open("x", encoding="utf-8") as stdout, \
            (OUT / "stderr.log").open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(receipt["command"], cwd=ROOT, stdout=stdout, stderr=stderr)
        receipt["training_pid"] = process.pid
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        result = process.wait()
    receipt.update(exit_code=result, finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(result)


if __name__ == "__main__":
    main()
