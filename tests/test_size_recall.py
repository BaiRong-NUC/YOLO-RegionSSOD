import numpy as np

from regionssod.evaluation.size_recall import match_by_confidence


def test_size_recall_matches_all_sizes_jointly_and_never_reuses_gt():
    gt = np.array([[.5, .5, .1, .1], [.5, .5, .11, .11], [.8, .8, .1, .1]])
    predictions = np.array([[.5, .5, .11, .11], [.5, .5, .11, .11], [.8, .8, .1, .1]])
    matched = match_by_confidence([0, 0, 1], [.8, .9, .99], predictions, [0, 0, 0], gt)
    assert matched.tolist() == [True, True, False]
    assert not match_by_confidence([], [], np.empty((0, 4)), [0], gt[:1]).any()
    assert len(match_by_confidence([0], [.9], predictions[:1], [], np.empty((0, 4)))) == 0
