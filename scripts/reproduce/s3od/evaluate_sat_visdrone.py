import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from reproduce.s3od.sat import load_sat


def areas(boxes):
    return (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)


def intersections(boxes, targets):
    lower = np.maximum(boxes[:, None, :2], targets[None, :, :2])
    upper = np.minimum(boxes[:, None, 2:], targets[None, :, 2:])
    return np.maximum(upper - lower, 0).prod(axis=2)


def score_image(predictions, truth, ignored, selections):
    # 按分数从高到低做同类一对一匹配，再排除主要落在忽略区的未匹配预测。
    boxes, targets = predictions[:, :4], truth[:, :4]
    overlap = intersections(boxes, targets)
    union = areas(boxes)[:, None] + areas(targets)[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    iou[predictions[:, 5, None] != truth[None, :, 4]] = -1
    ignore = (intersections(boxes, ignored) > 0.5 * areas(boxes)[:, None]).any(axis=1)
    small = areas(targets) < 1024
    counts = {}
    for name, selected in selections.items():
        matched = np.zeros(len(targets), dtype=bool)
        false_positives = 0
        order = selected[np.argsort(-predictions[selected, 4], kind="stable")]
        for index in order:
            scores = np.where(matched, -1, iou[index])
            if scores.size and scores.max() >= 0.5:
                matched[scores.argmax()] = True
            elif not ignore[index]:
                false_positives += 1
        counts[name] = {
            "kept": len(selected),
            "true_positives": int(matched.sum()),
            "false_positives": false_positives,
            "small_true_positives": int((matched & small).sum()),
            "small_gt": int(small.sum()),
            "all_gt": len(targets),
        }
    return counts


def evaluate(checkpoint, images, metadata, batch, device):
    settings, estimate_thresholds, filter_boxes = load_sat()
    model = YOLO(str(checkpoint))
    candidates, thresholds_used, sat_kept = [], [], []
    for start in range(0, len(images), batch):
        # Ultralytics 会把列表中的图片作为一个批次读入，因此显式按批切分。
        batch_images = images[start:start + batch]
        predictions = model.predict(
            source=[str(image) for image in batch_images], imgsz=640, device=device,
            conf=settings["pseudo_label_initial_score_thr"], iou=0.7, max_det=1000,
            rect=False, stream=True, verbose=False,
        )
        group = []
        for prediction, image in zip(predictions, batch_images, strict=True):
            if Path(prediction.path).name != image.name:
                raise ValueError(f"预测与输入顺序不一致：{prediction.path}，{image}")
            boxes = prediction.boxes.data.cpu()
            boxes = boxes[(boxes[:, 2:4] > boxes[:, :2]).all(dim=1)]
            candidates.append(boxes.numpy())

            # 原图坐标用于评测，SAT 分组换算到 YOLO 的 640 输入尺度，水平框角度取零。
            dimensions = (boxes[:, 2:4] - boxes[:, :2]) * (640 / max(prediction.orig_shape))
            group.append(torch.cat((boxes[:, :2], dimensions, torch.zeros_like(boxes[:, :1]), boxes[:, 4:5]), dim=1))

        # 沿用作者的每图预筛选：按分数和估计保留数量，再用这些候选拟合两组阈值。
        preliminary = []
        for boxes in group:
            scores, order = boxes[:, 5].sort(descending=True)
            cutoff = min(int(scores.sum() + 0.5), len(scores) - 1)
            preliminary.append(boxes[order[scores > scores[cutoff] - 1e-5]] if len(scores) else boxes)
        labels = [torch.zeros(len(boxes), dtype=torch.long) for boxes in preliminary]
        thresholds = estimate_thresholds(
            preliminary, labels, vaild_len=settings["cls_thr_vaild_len"],
            percent=settings["cls_thr_percent"], min=settings["cls_thr_min"],
            max=settings["cls_thr_max"], small_size=settings["small_size"],
        )
        thresholds_used.append({name: float(value) for name, value in thresholds.items()})
        for boxes in group:
            # 作者实际调用省略 small_size，沿用筛框函数的 800 像素²默认值。
            _, indices = filter_boxes(boxes[:, :5], torch.arange(len(boxes)), boxes[:, 5], thresholds)
            sat_kept.append(indices.numpy())

    # 等量对照选全验证集置信度最高的同样多候选，检验尺度排序能否提高伪框利用率。
    offsets = np.cumsum([0, *(len(boxes) for boxes in candidates)])
    ranking = np.argsort(-np.concatenate([boxes[:, 4] for boxes in candidates]), kind="stable")
    budget = np.zeros(len(ranking), dtype=bool)
    budget[ranking[:sum(map(len, sat_kept))]] = True
    rows = []
    for index, (image, boxes, sat) in enumerate(zip(images, candidates, sat_kept, strict=True)):
        targets = json.loads((metadata / f"{image.stem}.json").read_text(encoding="utf-8"))
        truth = np.array([[*obj["box_xyxy"], obj["class_id"]] for obj in targets["valid_objects"]]).reshape(-1, 5)
        ignored = np.array(targets["ignore_boxes_xyxy"]).reshape(-1, 4)
        selections = {
            "uniform_050": np.flatnonzero(boxes[:, 4] > 0.5),
            "uniform_090": np.flatnonzero(boxes[:, 4] > 0.9),
            "uniform_equal_count": np.flatnonzero(budget[offsets[index]:offsets[index + 1]]),
            "sat": sat,
        }
        rows.append({"image": image.name, "counts": score_image(boxes, truth, ignored, selections)})
    totals = {
        name: {key: sum(row["counts"][name][key] for row in rows) for key in first}
        for name, first in rows[0]["counts"].items()
    }
    return {
        "checkpoint": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "totals": totals,
        "threshold_batches": thresholds_used,
        "images": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="在 VisDrone 验证集离线评测移植的 S³OD SAT 伪框筛选")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/visdrone"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    images = sorted((args.data / "images/val").glob("*.jpg"))
    if not images or args.batch < 1:
        parser.error("验证集必须有图像，且批量大小必须为正")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for checkpoint in args.checkpoints:
        result = evaluate(checkpoint, images, args.data / "metadata/val", args.batch, args.device)
        results.append(result)
        print(f"{checkpoint}: {result['totals']}", flush=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"batch": args.batch, "device": args.device, "runs": results}, output, ensure_ascii=False)


if __name__ == "__main__":
    main()
