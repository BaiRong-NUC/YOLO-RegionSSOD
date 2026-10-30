from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_paper1_gaps import REASONS, match_iou, miss_evidence, overlaps
from regionssod.data.annotations import YoloBox


def test_quality_matching_is_joint_class_aware_and_one_to_one():
    box = YoloBox(0, .5, .5, .1, .1)
    other = YoloBox(1, .5, .5, .1, .1)
    assert match_iou([box, box, other], [box, other]) == [(0, 0), (2, 1)]


def test_empty_candidate_pool_is_distinguished_from_near_localization():
    gt = [YoloBox(0, .5, .5, .1, .1)]
    assert overlaps([], gt).shape == (0, 1)
    assert match_iou([], gt) == []
    assert miss_evidence(0, np.array([], dtype=int), np.array([], dtype=bool), np.array([])) == REASONS[-1]
    assert miss_evidence(0, np.array([0]), np.array([True]), np.array([.49])) == REASONS[3]


def test_overlap_evidence_preserves_competition_and_ignore_priority():
    classes = np.array([0, 0, 1])
    positive = np.array([True, False, True])
    assert miss_evidence(0, classes, positive, np.array([.6, .9, .8])) == REASONS[0]
    assert miss_evidence(0, classes, positive, np.array([.2, .9, .8])) == REASONS[1]
    assert miss_evidence(0, classes, positive, np.array([.2, .4, .8])) == REASONS[2]
