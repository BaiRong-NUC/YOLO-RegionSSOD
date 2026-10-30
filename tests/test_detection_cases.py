from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_detection_cases import crop_for, match_details


def test_case_predictions_use_joint_class_aware_matching_and_flag_duplicates():
    gt = [[.5, .5, .1, .1], [.5, .5, .2, .2]]
    prediction = {"classes": [0, 0, 1, 2], "confidences": [.9, .8, .7, .6],
                  "xywhn": [gt[0], gt[0], gt[1], gt[1]]}
    gt_hit, pred_hit, pairs = match_details(prediction, [0, 1], gt)
    assert gt_hit.tolist() == [True, True]
    assert pred_hit.tolist() == [True, False, True, False]
    assert pairs == [[0, 0], [2, 1]]


def test_case_matching_empty_inputs():
    gt_hit, pred_hit, pairs = match_details({"classes": [], "confidences": [], "xywhn": []}, [], [])
    assert not len(gt_hit) and not len(pred_hit) and pairs == []


def test_case_crop_stays_in_image_at_boundary():
    assert crop_for([.99, .01, .01, .01], 1920, 1080) == [1536, 0, 1920, 256]
