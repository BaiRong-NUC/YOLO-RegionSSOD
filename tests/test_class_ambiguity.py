import numpy as np
import json
import pytest
import torch

from regionssod.evaluation.class_ambiguity import (
    ambiguity_ratio, fit_logistic, image_bootstrap, metrics, predict,
)
from regionssod.pseudo.class_scores import indexed_nms


def test_full_scores_keep_original_anchor_identity_after_nms():
    raw = torch.zeros((1, 7, 8))
    raw[0, :4] = torch.tensor([[10, 10, 50, 80, 110, 140, 170, 200],
                              [10] * 8, [10] * 8, [10] * 8])
    raw[0, 4:] = torch.tensor([[.9, .8, .1, .05, .02, .03, .01, .01],
                               [.2, .3, .95, .05, .01, .01, .01, .01],
                               [.1, .2, .2, .05, .01, .01, .01, .01]])
    output, anchors, scores = indexed_nms(raw, conf=.1, iou=.7, max_det=500, nc=3)
    assert anchors[0].tolist() == [2, 0]
    assert torch.equal(scores[0], raw[0, 4:, anchors[0]].T)
    assert output[0][:, 5].tolist() == [1, 0]


def test_empty_nms_shape():
    _, _, scores = indexed_nms(torch.zeros((1, 7, 8)), conf=.1, iou=.7, max_det=500, nc=3)
    assert scores[0].shape == (0, 3)


def test_ratio_not_softmax_or_top1_relabel():
    assert np.allclose(ambiguity_ratio([[.6, .59, .01], [.6, .05, .01]]), [.59/.6, .05/.6])
    with pytest.raises(ValueError):
        ambiguity_ratio([[0, 0]])


def test_logistic_and_auc_known_independent_signal():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 1/(1+np.exp(-3*x[:, 1]))).astype(int)
    baseline = fit_logistic(x[:1000, :1], y[:1000])
    candidate = fit_logistic(x[:1000], y[:1000])
    b, _ = metrics(y[1000:], predict(baseline, x[1000:, :1]))
    c, _ = metrics(y[1000:], predict(candidate, x[1000:]))
    assert candidate["converged"]
    assert c["log_loss"] < b["log_loss"]
    assert c["auc"] > .85
    assert metrics([0, 1], [.5, .5])[0]["auc"] == .5
    json.dumps({"auc_gate": c["auc"] - b["auc"] >= .01, "metrics": c})


def test_bootstrap_samples_images_including_zero_event_images():
    result = image_bootstrap([0, 0, 1], [.1, .1, .1], all_images=[0, 1], repetitions=100)
    assert np.allclose(result["ci95"], [.1, .1])
    with pytest.raises(ValueError):
        image_bootstrap([3], [.1], all_images=[0, 1])
