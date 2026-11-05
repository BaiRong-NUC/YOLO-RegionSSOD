"""Access-boundary and identifiability counterexamples, using synthetic data only."""
import importlib.util
from pathlib import Path

import pytest

from regionssod.data.annotations import Tile, YoloBox, clip_box_to_tile

spec = importlib.util.spec_from_file_location(
    "region_feasibility", Path(__file__).parents[1] / "scripts/audit_reliable_region_ranking_feasibility.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ProtectedValues(dict):
    def __getitem__(self, key):
        if key in {"unlabeled_tiles", "source_gt_boxes"}:
            raise AssertionError("forbidden hidden metadata value accessed")
        return super().__getitem__(key)


def manifest_row():
    return ProtectedValues(source_image="identity-only.jpg", source_size=[1000, 1000],
        requested_area_ratio=.1, actual_area_ratio=.1,
        labeled_tile=ProtectedValues(image="human.jpg", tile=[0, 0, 316, 316], labels=1,
                                     source_gt_boxes=object()), unlabeled_tiles=object())


def test_projection_never_reads_hidden_count_or_unlabeled_values():
    projected = review.project_manifest(manifest_row())
    assert set(projected["labeled_tile"]) == {"image", "tile", "labels"}
    assert "unlabeled_tiles" not in projected


def test_unknown_schema_cannot_silently_hide_a_new_certificate():
    row = manifest_row()
    row["completeness_certificate"] = "requires a new scoped review"
    with pytest.raises(ValueError, match="schema differs"):
        review.project_manifest(row)


@pytest.mark.parametrize("escape", ["../outside.txt", "../human-lookalike/label.txt"])
def test_path_guard_rejects_traversal_and_prefix_lookalikes(tmp_path, escape):
    allowed = tmp_path / "human"
    with pytest.raises(ValueError, match="outside authorized"):
        review.within(allowed / escape, allowed)
    assert review.within(allowed / "valid.txt", allowed) == (allowed / "valid.txt").resolve()


def test_same_label_export_and_even_same_source_count_do_not_determine_background():
    witness = review.synthetic_counterexample()
    tile = Tile(**witness["tile"])
    retained = YoloBox(**witness["retained_source_box"])
    covering = YoloBox(**witness["extra_source_box"])
    outside = YoloBox(1, .05, .05, .02, .02)
    world_a, world_b = [retained, outside], [retained, covering]
    def export(boxes):
        values = [clip_box_to_tile(box, tile, 1000, 1000, min_visibility=.35) for box in boxes]
        return [value for value in values if value is not None]
    assert len(world_a) == len(world_b) == 2
    assert export(world_a) == export(world_b)
    assert len(export(world_a)) == 1
    assert clip_box_to_tile(outside, tile, 1000, 1000, min_visibility=.35) is None
    assert witness["extra_box_intersection_covers_entire_tile"]
