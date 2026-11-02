"""Preserve full teacher class scores using the original NMS anchor indices."""

from __future__ import annotations

import torch
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms


def indexed_nms(raw, *, conf, iou, max_det, nc):
    if raw.ndim != 3 or raw.shape[1] != 4 + nc:
        raise ValueError("Expected non-end-to-end B,(4+nc),N predictions")
    options = dict(conf_thres=conf, iou_thres=iou, max_det=max_det, nc=nc)
    plain = nms.non_max_suppression(raw.clone(), **options)
    detections, indices = nms.non_max_suppression(raw.clone(), return_idxs=True, **options)
    scores = []
    for i, (reference, selected, anchors) in enumerate(zip(plain, detections, indices)):
        if not torch.equal(reference, selected):
            raise ValueError("Indexed NMS changed original detections")
        anchors = anchors.reshape(-1).long()
        full = raw[i, 4:4 + nc, anchors].T.contiguous()
        if len(full):
            top, cls = full.max(1)
            if not torch.equal(top, selected[:, 4]) or not torch.equal(cls, selected[:, 5].long()):
                raise ValueError("NMS class-score identity mismatch")
        scores.append(full)
    return detections, indices, scores


class ClassScorePredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        if self.args.classes is not None or self.args.agnostic_nms or getattr(self.model, "end2end", False):
            raise ValueError("Class scores require original class-aware YOLO11 detection")
        raw = preds[0] if isinstance(preds, (tuple, list)) else preds
        detections, indices, scores = indexed_nms(
            raw, conf=self.args.conf, iou=self.args.iou, max_det=self.args.max_det,
            nc=len(self.model.names),
        )
        results = self.construct_results(detections, img, orig_imgs, **kwargs)
        for result, anchors, full in zip(results, indices, scores):
            result.teacher_class_scores = full.cpu().numpy()
            result.teacher_anchor_indices = anchors.reshape(-1).cpu().numpy()
        return results
