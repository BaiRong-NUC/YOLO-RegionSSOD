from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_cross_view_flip import support
from regionssod.data.annotations import YoloBox


def test_view_support_requires_same_class_and_unique_matching():
    original = [YoloBox(0, .5, .5, .2, .2), YoloBox(0, .5, .5, .2, .2), YoloBox(1, .5, .5, .2, .2)]
    flipped = [YoloBox(0, .5, .5, .2, .2)]
    assert support(original, flipped, .5) == {0}
    assert support([], flipped, .5) == set()
    assert support(original, [], .5) == set()


def test_view_support_honors_iou_threshold():
    original = [YoloBox(0, .5, .5, .2, .2)]
    flipped = [YoloBox(0, .57, .5, .2, .2)]
    assert support(original, flipped, .3) == {0}
    assert support(original, flipped, .5) == set()
