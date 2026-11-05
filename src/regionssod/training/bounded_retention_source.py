"""Bind the conditional numerical observer to existing legal provenance.

Opens named reports and reviewed source only. No manifest/container, images,
hidden labels, original-source GT or old prediction/quality cache access.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .bounded_retention_observation import build_observation

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "outputs/bounded_retention_engineering_20261105/source_audit.json"
AUDIT_SHA = "c8f2d57e5b652c1bfd79893404175aea7ce1e8aa64610b49f3690084e375f95a"
GEOMETRY_SHA = "b8bc076aa73f6a1d19a932e8a908f6bcace7d4bebbed92fe0e52d1dfc38e45cd"
LEGAL = (ROOT / "data/visdrone_area10/labeled/labels/train").resolve()


class BoundSourceFactory:
    def __init__(self):
        self.reads, self.counts, self.bindings = {}, Counter(), {}
        def read(path):
            raw = path.read_bytes()
            self.reads[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
            return json.loads(raw)
        audit = read(AUDIT)
        if self.reads[str(AUDIT.resolve())] != AUDIT_SHA:
            raise ValueError("initial source audit changed")
        previous = ROOT / "outputs/reliable_region_ranking_feasibility_20261105"
        source, receipt = read(previous / "results.json"), read(previous / "receipt.json")
        if (self.reads[str((previous / "results.json").resolve())] != receipt["results_sha256"]
                or receipt["results_sha256"] != audit["input_sha256"][str((previous / "results.json").resolve())]):
            raise ValueError("legal source provenance chain changed")
        geometry_dir = ROOT / "outputs/human_multiscale_observation_review_20261105"
        geometry, geometry_receipt = read(geometry_dir / "geometry.json"), read(geometry_dir / "geometry_receipt.json")
        if (self.reads[str((geometry_dir / "geometry.json").resolve())] != GEOMETRY_SHA
                or geometry_receipt["geometry_sha256"] != GEOMETRY_SHA
                or not geometry_receipt["exact_replay"]
                or geometry["input_sha256"][str((previous / "results.json").resolve())] != receipt["results_sha256"]):
            raise ValueError("public geometry report binding changed")
        manifest_key = str((ROOT / "data/visdrone_area10/manifest.jsonl").resolve())
        # Compare the historical identities as strings; never open manifest.
        if geometry["input_sha256"][manifest_key] != source["input_sha256"][manifest_key]:
            raise ValueError("public source reports describe different populations")
        expected = {"area_ratio": .1, "seed": 0, "min_box_visibility": .35, "jitter_fraction": .2}
        if source["parameters"] != expected or audit["parameters"] != expected:
            raise ValueError("public selection parameters changed")
        if source["gates"]["G1_human_identity_and_export_consistency"] != "PASS":
            raise ValueError("human provenance gate absent")
        if (geometry["human_tiles"] != 6471 or geometry["non_square_tiles"] != 0
                or geometry["tile_long_side_min_max"] != [131, 548]):
            raise ValueError("public numeric domain differs")
        for relative in ("src/regionssod/data/annotations.py", "src/regionssod/data/partial.py"):
            path = (ROOT / relative).resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.reads[str(path)] = digest
            if digest != source["input_sha256"][str(path)]:
                raise ValueError("reviewed source exporter changed")
        self.labels = {Path(p).resolve(): digest for p, digest in audit["input_sha256"].items()
                       if Path(p).resolve().parent == LEGAL and Path(p).suffix == ".txt"}
        if len(self.labels) != 6471:
            raise ValueError("legal label allowlist differs")
        alpha_lower = math.nextafter(source["inventory"]["actual_area_ratio_min"], -math.inf)
        self.ratio = math.nextafter(1 / alpha_lower, math.inf)
        self.premises = {"source_ratio_upper": self.ratio, "tile_size_lower": 131,
            "tile_size_upper": 548, "min_visibility": .35, "retention_tolerance": 1e-12}

    def __call__(self, *, label_path, label_sha256, audit_sha256, line_number, text, source_object_id):
        path = Path(label_path).resolve()
        if (path.parent != LEGAL or self.labels.get(path) != label_sha256 or audit_sha256 != AUDIT_SHA
                or not isinstance(line_number, int) or line_number <= 0):
            raise ValueError("source factory requires an audited human-row binding")
        tokens = text.split()
        if len(tokens) != 5:
            raise ValueError("original five-column source required")
        observation = build_observation(tokens[1:], **self.premises)
        intervals = np.array([[np.nextafter(float(i.lo), -np.inf), np.nextafter(float(i.hi), np.inf)]
                              for i in observation.serialized_intervals])
        self.counts["objects"] += 1
        self.counts[f"possible_unknown_sides_{int(observation.unknown.sum())}"] += 1
        self.bindings[source_object_id] = {"label_path": str(path), "label_sha256": label_sha256,
            "line_number": line_number, "source_object_id": source_object_id}
        return {"physical": observation, "original_xywh": np.asarray(tokens[1:], dtype=np.float64),
            "original_serialized_intervals": intervals, "original_unknown": observation.unknown}
