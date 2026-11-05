"""Read legal human exports only; fail closed before any ranking experiment.

No model/dataset construction, pixels, predictions, raw full-image labels or
unlabeled label files. JSON containers are projected immediately; hidden counts
are neither extracted nor used. --audit replays this new review only.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import yaml

from regionssod.data.annotations import Tile, YoloBox, clip_box_to_tile, read_label_file

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/visdrone_area10"
DOC = ROOT / "docs/paper/diagnostics/reliable_region_ranking_feasibility/protocol.md"
OUT = ROOT / "outputs/reliable_region_ranking_feasibility_20261105"
TOP_KEYS = {"source_image", "source_size", "requested_area_ratio", "actual_area_ratio", "labeled_tile", "unlabeled_tiles"}
LABELED_KEYS = {"image", "tile", "labels", "source_gt_boxes"}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def within(path, directory):
    """Resolve before authorization, including symlinks and '..' escapes."""
    path, directory = Path(path).resolve(), Path(directory).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f"outside authorized human directory: {path}")
    return path


def project_manifest(row):
    # Refuse unknown schemas instead of silently discarding a possible new
    # completeness certificate. Only key names, never hidden values, are tested.
    if set(row) != TOP_KEYS or set(row["labeled_tile"]) != LABELED_KEYS:
        raise ValueError("manifest schema differs; manual scoped review required")
    labeled = row["labeled_tile"]
    return {"source_image": row["source_image"], "source_size": row["source_size"],
            "requested_area_ratio": row["requested_area_ratio"], "actual_area_ratio": row["actual_area_ratio"],
            "labeled_tile": {key: labeled[key] for key in ("image", "tile", "labels")}}


def synthetic_counterexample():
    """A format-valid synthetic witness; absolutely not an observed hidden box."""
    tile = Tile(300, 300, 616, 616)
    retained = YoloBox(0, .45, .45, .04, .04)
    censored = YoloBox(1, .5, .5, 1., 1.)
    def exported(boxes):
        clipped = [clip_box_to_tile(box, tile, 1000, 1000, min_visibility=.35) for box in boxes]
        return [asdict(box) for box in clipped if box is not None]
    left, right = exported([retained]), exported([retained, censored])
    discarded = clip_box_to_tile(censored, tile, 1000, 1000, min_visibility=.35) is None
    if not (left == right and len(left) == 1 and discarded):
        raise ValueError("synthetic clipping counterexample no longer holds")
    return {"synthetic_only": True, "source_size": [1000, 1000], "tile": asdict(tile),
            "retained_source_box": asdict(retained), "extra_source_box": asdict(censored),
            "extra_box_visibility": tile.width * tile.height / 1_000_000,
            "extra_box_filtered": discarded, "same_export": left,
            "extra_box_intersection_covers_entire_tile": True,
            "meaning": "non-injective label export; not an estimate or observation of real missing GT"}


def review():
    mapping = {}
    def bind(path):
        path = Path(path).resolve()
        mapping[str(path)] = sha(path)
        return path
    def container(path):
        return json.loads(bind(path).read_text(encoding="utf-8"))
    summary = container(DATA / "summary.json")
    parameters = {key: summary[key] for key in ("area_ratio", "seed", "min_box_visibility", "jitter_fraction")}
    del summary
    if parameters["area_ratio"] != .1 or parameters["min_box_visibility"] != .35:
        raise ValueError("different prospective source population/visibility rule")
    labeled_images = DATA / "labeled/images/train"
    labeled_labels = DATA / "labeled/labels/train"
    manifest = bind(DATA / "manifest.jsonl")
    rows = [project_manifest(json.loads(line)) for line in manifest.read_text(encoding="utf-8").splitlines()]
    expected, ratios, total_boxes = {}, [], 0
    for row in rows:
        item = row["labeled_tile"]
        image = within(item["image"], labeled_images)
        if image in expected or not image.is_file():
            raise ValueError("duplicate or absent human image identity")
        width, height = row["source_size"]
        x1, y1, x2, y2 = item["tile"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("invalid source/tile geometry")
        ratio = (x2 - x1) * (y2 - y1) / (width * height)
        if ratio != row["actual_area_ratio"] or row["requested_area_ratio"] != .1:
            raise ValueError("area-ratio provenance differs")
        label = within(labeled_labels / f"{image.stem}.txt", labeled_labels)
        if not label.is_file():
            raise ValueError("missing human label file is not an empty annotation")
        bind(label)
        boxes = read_label_file(label)
        if len(boxes) != item["labels"] or any(box.class_id >= 10 for box in boxes):
            raise ValueError("manifest/human-label format or count differs")
        total_boxes += len(boxes)
        ratios.append(ratio)
        expected[image] = label
    exported = {p.resolve() for p in labeled_labels.glob("*.txt")}
    if exported != set(expected.values()):
        raise ValueError("human label inventory differs from manifest")
    human_list = bind(DATA / "splits/labeled_train.txt")
    humans = [within(line, labeled_images) for line in human_list.read_text().splitlines() if line.strip()]
    if len(humans) != len(set(humans)) or set(humans) != set(expected):
        raise ValueError("human split/manifest identity differs")
    dataset_path = bind(DATA / "dataset_p1.yaml")
    dataset = yaml.safe_load(dataset_path.read_text())
    train_file = Path(dataset["train"]).resolve()
    if train_file != (DATA / "combined_p1.txt").resolve():
        raise ValueError("unexpected combined training list")
    bind(train_file)
    training = [Path(line).resolve() for line in train_file.read_text().splitlines() if line.strip()]
    if len(training) != len(set(training)) or not set(humans).issubset(training):
        raise ValueError("human export not represented uniquely in actual P1 input list")
    if any(p.is_relative_to(labeled_images.resolve()) and p not in expected for p in training):
        raise ValueError("unmanifested human training input")
    sources = [ROOT / path for path in (
        "src/regionssod/data/annotations.py", "src/regionssod/data/partial.py",
        "src/regionssod/training/p1.py", "src/regionssod/training/reliable_competitor_negative.py",
        "configs/candidates/positive_quality_transport_v2/reference30.yaml")]
    augment = Path(sys.prefix) / "Lib/site-packages/ultralytics/data/augment.py"
    sources += [augment, DOC, Path(__file__).resolve()]
    for source in sources:
        bind(source)
    source_file = ROOT / "src/regionssod/training/reliable_competitor_negative.py"
    tree = ast.parse(source_file.read_text())
    provenance = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "PROVENANCE" for t in n.targets))
    if provenance != {"object_is_human", "object_id", "object_class", "object_table"}:
        raise ValueError("changed provenance schema; manual review needed")
    # These are evidence anchors, not a generic proof checker for arbitrary code.
    needles = {sources[0]: ["if visibility + 1e-12 < min_visibility:", "return None"],
        sources[1]: ["write_label_file(labeled_label_path, tile_labels)", '"labels": len(tile_labels)',
                     '"source_gt_boxes": len(source_labels)'],
        sources[2]: ['self.p1_ignores[key]'],
        source_file: ["self.p1_ignores[key] = np.empty((0, 4), dtype=np.float32)", 'sample["object_is_human"]'],
        augment: ['good = final_labels["instances"].remove_zero_area_boxes()',
                  'labels["instances"] = new_instances[i]', 'labels["cls"] = cls[i]',
                  'area_thr=0.01 if len(segments) else 0.10']}
    locations = {}
    for path, texts in needles.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        locations[str(path)] = {text: next((i for i, line in enumerate(lines, 1) if text in line), None)
                                for text in texts}
        if any(value is None for value in locations[str(path)].values()):
            raise ValueError("reviewed source anchor differs")
    witness = synthetic_counterexample()
    # A synthetic full-source-image box contains each tile, with visibility
    # equal to this legal geometry ratio. No real source annotation is opened.
    all_allow_witness = all(ratio + 1e-12 < parameters["min_box_visibility"] for ratio in ratios)
    if not all_allow_witness:
        raise ValueError("not all tiles admit the predeclared witness; no post-hoc subset search")
    result = {"scope": "current P1 area10 split0 legal human exports only",
        "parameters": parameters,
        "inventory": {"human_tiles": len(humans), "human_exported_boxes": total_boxes,
            "combined_training_images": len(training), "human_label_files": len(exported),
            "actual_area_ratio_min": min(ratios), "actual_area_ratio_max": max(ratios)},
        "manifest_schema": {"top_key_names": sorted(TOP_KEYS), "labeled_key_names": sorted(LABELED_KEYS),
            "hidden_counts_projected_or_used": False, "completeness_certificate_in_schema": False},
        "source_provenance_field_names": sorted(provenance), "source_locations": locations,
        "synthetic_witness": witness, "all_export_geometries_admit_full_source_box_witness": all_allow_witness,
        "gates": {"G1_human_identity_and_export_consistency": "PASS",
                  "G2_complete_foreground_or_censored_footprints": "NOT_ESTABLISHED",
                  "G3_augmented_spatial_completeness_transport": "NOT_ESTABLISHED"},
        "decision": "STOP_BEFORE_RANKING_DIAGNOSTICS",
        "limits": ["not an observed missing-label count", "not proof that every actual unmatched location is foreground",
                   "not proof all possible object-level nontrue-class targets are unreliable",
                   "static source anchors plus human review, not a general program verifier"],
        "actions": {"training": False, "model_forward": False, "dataset_construction": False,
                    "image_pixels": False, "raw_full_image_labels": False, "unlabeled_labels": False,
                    "hidden_gt_statistics": False, "validation_data": False, "ranking_statistics": False,
                    "new_loss": False, "old_terminal_audit": False},
        "input_sha256": mapping}
    for path, digest in mapping.items():
        if sha(path) != digest:
            raise ValueError("input changed during review")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if args.audit:
        saved = json.loads((OUT / "results.json").read_text())
        receipt = json.loads((OUT / "receipt.json").read_text())
        if sha(OUT / "results.json") != receipt["results_sha256"] or review() != saved:
            raise ValueError("review replay differs")
        print(json.dumps({"accepted": True, "replay_exact": True, "decision": saved["decision"],
                          "seconds": time.monotonic() - started}))
    else:
        if OUT.exists():
            raise FileExistsError("existing review is immutable; use --audit")
        result = review()
        OUT.mkdir()
        (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (OUT / "receipt.json").write_text(json.dumps({"results_sha256": sha(OUT / "results.json"),
            "input_count": len(result["input_sha256"]), "seconds": time.monotonic() - started}, indent=2) + "\n")
        print(json.dumps({"saved": str(OUT), "decision": result["decision"], "inventory": result["inventory"]}))


if __name__ == "__main__":
    main()
