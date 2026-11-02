"""Replay frozen analyses while writing new audit receipts to a separate directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "outputs/paper1_reaudits"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def redirect_receipts(module, destinations):
    """Only redirect the known audit receipt writes; reject any other mutation."""
    mapping = {Path(source).resolve(): Path(target) for source, target in destinations.items()}
    before = {source: digest(source) for source in mapping}
    for target in mapping.values():
        if target.exists() or target.resolve() in mapping:
            raise FileExistsError(f"Receipt destination is not fresh: {target}")
    original_write = module.write
    emitted = []

    def write(path, data):
        source = Path(path).resolve()
        if source not in mapping:
            raise ValueError(f"Unexpected write during frozen audit: {path}")
        target = mapping[source]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=True, indent=2)
            stream.write("\n")
        emitted.append(str(target))

    module.write = write
    try:
        yield emitted
    finally:
        module.write = original_write
        if any(digest(source) != sha for source, sha in before.items()):
            raise RuntimeError("A historical audit receipt changed")


def run(stage, directory):
    import diagnose_paper1_high_precision as diagnostic
    import evaluate_paper1_small_pr as base
    import evaluate_scaled_operating_point as scale

    if not directory.resolve().is_relative_to(STORE.resolve()):
        raise ValueError(f"Output must be inside {STORE}")
    directory.mkdir(parents=True, exist_ok=False)
    modules = {"small-pr": base, "high-precision": diagnostic, "scale": scale}
    selected = list(modules) if stage == "all" else [stage]
    mapping = {
        module.OUT / "independent_audit.json": directory / name / "independent_audit.json"
        for name, module in modules.items()
    }
    report = {"stage": stage, "accepted": False, "training_steps": 0, "new_inference": 0}
    try:
        with redirect_receipts(base, mapping) as emitted:
            for name in selected:
                modules[name].audit()
        if len(emitted) != len(selected):
            raise ValueError("Missing audit receipt")
        report.update(accepted=True, historical_receipts_unchanged=True, receipts=emitted)
        report["verification_scope"] = {
            "small-pr": "Exact stored COCO/PR aggregate replay, source and artifact hashes",
            "high-precision": "Exact stored diagnostic and table replay, source and artifact hashes",
            "scale": "Exact summary replay excluding fold records; artifact hashes cover stored records",
        }
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        with (directory / "run.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "small-pr", "high-precision", "scale"), default="all")
    parser.add_argument(
        "--output", type=Path,
        default=STORE / datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )
    args = parser.parse_args()
    run(args.stage, (ROOT / args.output).resolve())


if __name__ == "__main__":
    main()
