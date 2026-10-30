"""Independent terminal-state audit of the one-arm NWD screen; no inference."""
from datetime import datetime
from zoneinfo import ZoneInfo

from screen_p1_small_nwd import OUT, collect
from run_p1_scale_seed_pair import read, write


def main():
    launch = read(OUT / "launch_receipt.json")
    if launch.get("exit_code") != 0 or read(OUT / "status.json")["stage"] != "completed":
        raise ValueError("wrapper must have exited successfully and all stages completed")
    result = collect(check=True)
    write(OUT / "independent_audit.json", {
        "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "accepted": True, "scope": "single five-epoch subset screen; no inference",
        "worth_full30_confirmation": result["worth_full30_confirmation"]})


if __name__ == "__main__":
    main()
