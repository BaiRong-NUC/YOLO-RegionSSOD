from pathlib import Path

from regionssod.data.annotations import Tile, YoloBox, clip_box_to_tile


def test_clip_box_to_tile_preserves_fully_inside_box() -> None:
    box = YoloBox(class_id=3, x=0.5, y=0.5, w=0.2, h=0.2)
    tile = Tile(x1=40, y1=40, x2=140, y2=140)
    clipped = clip_box_to_tile(box, tile, image_width=200, image_height=200)

    assert clipped is not None
    assert clipped.class_id == 3
    assert clipped.x == 0.6
    assert clipped.y == 0.6
    assert clipped.w == 0.4
    assert clipped.h == 0.4


def test_clip_box_to_tile_clips_boundary_and_keeps_visibility() -> None:
    box = YoloBox(class_id=0, x=0.1, y=0.5, w=0.2, h=0.2)
    tile = Tile(x1=20, y1=40, x2=120, y2=140)
    clipped = clip_box_to_tile(
        box,
        tile,
        image_width=200,
        image_height=200,
        min_visibility=0.5,
    )

    assert clipped is not None
    assert clipped.x == 0.1
    assert clipped.w == 0.2


def test_clip_box_to_tile_rejects_low_visibility() -> None:
    box = YoloBox(class_id=0, x=0.05, y=0.5, w=0.1, h=0.1)
    tile = Tile(x1=40, y1=40, x2=140, y2=140)
    clipped = clip_box_to_tile(
        box,
        tile,
        image_width=200,
        image_height=200,
        min_visibility=0.5,
    )

    assert clipped is None


def test_write_and_read_label_file(tmp_path: Path) -> None:
    from regionssod.data.annotations import read_label_file, write_label_file

    path = tmp_path / "labels" / "sample.txt"
    expected = [YoloBox(class_id=2, x=0.25, y=0.5, w=0.1, h=0.2)]
    write_label_file(path, expected)

    assert read_label_file(path) == expected
