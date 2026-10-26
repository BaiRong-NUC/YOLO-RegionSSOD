from __future__ import annotations

import sys

from regionssod.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["pseudo-label", *sys.argv[1:]]))
