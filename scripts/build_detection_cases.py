"""Render deterministic success/failure examples from saved seed0 predictions.

No training or inference. Four event types, one image per type, first eligible
filename prefix in the frozen validation order and first eligible box in source order.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from regionssod.data.annotations import iter_images, label_path_for_image, read_label_file
from regionssod.data.dataset import load_source_dataset
from regionssod.evaluation.size_recall import match_by_confidence
from run_p1_scale_seed_pair import ROOT, SOURCE, digest, read, write

OUT = ROOT / "outputs/paper_cases_b2_p1_s0"
TYPES = ("p1_recovers_small", "p1_loses_small", "both_miss_small", "p1_unmatched_prediction")
TITLES = ("A | P1 recovers a small target", "B | P1 misses a target detected by B2",
          "C | Both models miss a small target", "D | P1 has an unmatched prediction")
COLORS = {"tp": "#16a085", "fp": "#e67e22", "miss": "#d14b9a", "focus": "#ffe45c"}


def match_details(prediction, gt_classes, gt_boxes, iou=.5):
    """Return GT/prediction matches using the existing joint greedy protocol."""
    gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    pred = np.asarray(prediction["xywhn"], dtype=float).reshape(-1, 4)
    hit_gt, hit_pred = np.zeros(len(gt), bool), np.zeros(len(pred), bool)
    pairs = []
    lo_gt, hi_gt = gt[:, :2] - gt[:, 2:] / 2, gt[:, :2] + gt[:, 2:] / 2
    for i in np.argsort(-np.asarray(prediction["confidences"]), kind="stable"):
        box = pred[i]
        lo, hi = box[:2] - box[2:] / 2, box[:2] + box[2:] / 2
        intersection = np.maximum(0, np.minimum(hi, hi_gt) - np.maximum(lo, lo_gt)).prod(axis=1)
        overlaps = intersection / np.maximum(box[2:].prod() + gt[:, 2:].prod(axis=1) - intersection, 1e-12)
        eligible = (np.asarray(gt_classes) == prediction["classes"][i]) & ~hit_gt & (overlaps >= iou)
        if eligible.any():
            j = int(np.where(eligible, overlaps, -1).argmax())
            hit_gt[j], hit_pred[i] = True, True
            pairs.append([int(i), j])
    expected = match_by_confidence(prediction["classes"], prediction["confidences"], pred, gt_classes, gt, iou)
    if not np.array_equal(hit_gt, expected):
        raise RuntimeError("case matcher differs from evaluation protocol")
    return hit_gt, hit_pred, pairs


def crop_for(box, width, height):
    x, y = box[0] * width, box[1] * height
    cw, ch = min(384, width), min(256, height)
    left, top = int(np.clip(x - cw / 2, 0, width - cw)), int(np.clip(y - ch / 2, 0, height - ch))
    return [left, top, left + cw, top + ch]


def select_cases():
    source = load_source_dataset(SOURCE)
    directories = {m: ROOT / f"runs/size_recall/{m.lower()}_pilot30_s0" for m in ("B2", "P1")}
    predictions = {m: [__import__("json").loads(line) for line in (d / "predictions.jsonl").read_text().splitlines()]
                   for m, d in directories.items()}
    paths = [str(p.resolve()) for p in iter_images(source.images_val)]
    hashes = {str(SOURCE): digest(SOURCE)}
    for m, directory in directories.items():
        summary = read(directory / "metrics.json")
        if (summary["images"], summary["imgsz"], summary["conf"], summary["match_iou"], summary["nms_iou"], summary["max_det"]) != (548, 960, .25, .5, .7, 500):
            raise RuntimeError("unexpected saved prediction protocol")
        if digest(summary["weights"]) != summary["weights_sha256"]:
            raise RuntimeError("saved prediction weight hash differs")
        if len(predictions[m]) != 548 or [row["image"] for row in predictions[m]] != paths:
            raise RuntimeError("incomplete or unordered predictions")
        for p in (directory / "predictions.jsonl", directory / "metrics.json", directory / "images.txt", Path(summary["weights"])):
            hashes[str(p)] = digest(p)
    # Selection is frozen by type/order; no ranking by how large an improvement looks.
    selected, used, used_hashes, counts = {}, set(), set(), dict.fromkeys(TYPES, 0)
    for index, filename in enumerate(paths):
        image = Path(filename)
        label = label_path_for_image(image, source.images_val, source.labels_val)
        gt = read_label_file(label)
        boxes = np.array([[b.x, b.y, b.w, b.h] for b in gt]).reshape(-1, 4)
        classes = [b.class_id for b in gt]
        with Image.open(image) as pixels:
            width, height = pixels.size
        image_hash = digest(image)
        small = boxes[:, 2] * width * boxes[:, 3] * height < 1024
        details = {m: match_details(predictions[m][index], classes, boxes) for m in directories}
        b2_gt, p1_gt = details["B2"][0], details["P1"][0]
        eligible = {"p1_recovers_small": np.where(small & ~b2_gt & p1_gt)[0],
                    "p1_loses_small": np.where(small & b2_gt & ~p1_gt)[0],
                    "both_miss_small": np.where(small & ~b2_gt & ~p1_gt)[0],
                    "p1_unmatched_prediction": np.where(~details["P1"][1])[0]}
        for kind in TYPES:
            counts[kind] += len(eligible[kind])
            prefix = image.name.split("_")[0]
            if kind in selected or prefix in used or image_hash in used_hashes or not len(eligible[kind]):
                continue
            # Keep the focus clear of image edges so its arrow/box is legible.
            possible = predictions["P1"][index]["xywhn"] if kind == TYPES[-1] else boxes
            interior = [int(i) for i in eligible[kind]
                        if 32 <= possible[i][0]*width <= width-32
                        and 32 <= possible[i][1]*height <= height-32]
            if not interior:
                continue
            chosen = interior[0]
            anchor = predictions["P1"][index]["xywhn"][chosen] if kind == TYPES[-1] else boxes[chosen].tolist()
            selected[kind] = {"type": kind, "image": filename, "image_index": index,
                              "width": width, "height": height, "crop_xyxy": crop_for(anchor, width, height),
                              "anchor_index": chosen, "anchor_kind": "prediction" if kind == TYPES[-1] else "gt",
                              "anchor_xywhn": anchor, "anchor_class": int(predictions["P1"][index]["classes"][chosen]) if kind == TYPES[-1] else classes[chosen],
                              "class_name": source.names[int(predictions["P1"][index]["classes"][chosen]) if kind == TYPES[-1] else classes[chosen]],
                              "anchor_confidence": predictions["P1"][index]["confidences"][chosen] if kind == TYPES[-1] else None,
                              "image_sha256": image_hash, "label_sha256": digest(label)}
            used.add(prefix)
            used_hashes.add(image_hash)
    if set(selected) != set(TYPES):
        raise RuntimeError("could not find all four types with distinct images")
    # Freeze image/GT inputs across all 548 examples, not just displayed crops.
    for filename in paths:
        p = Path(filename)
        for f in (p, label_path_for_image(p, source.images_val, source.labels_val)):
            hashes[str(f)] = digest(f)
    return selected, counts, predictions, source, hashes


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def draw_panel(pixels, record, boxes, classes, prediction=None):
    crop = record["crop_xyxy"]
    scale = 1.5
    panel = pixels.crop(crop).resize((576, 384), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    width, height = pixels.size

    def rect(box, color, line=2, label=None):
        x, y, w, h = box
        p = [(x - w/2) * width - crop[0], (y - h/2) * height - crop[1],
             (x + w/2) * width - crop[0], (y + h/2) * height - crop[1]]
        if p[2] < 0 or p[0] > crop[2]-crop[0] or p[3] < 0 or p[1] > crop[3]-crop[1]:
            return False
        p = [v * scale for v in p]
        draw.rectangle(p, outline=color, width=line)
        if label:
            tx, ty = max(0, min(p[0], 530)), max(0, min(p[1]-17, 365))
            draw.rectangle([tx, ty, tx+max(35, len(label)*8), ty+17], fill="#17212b")
            draw.text((tx+2,ty), label, font=font(13), fill=color)
        return True

    visible = {"gt": 0, "missed_gt": 0, "matched_prediction": 0, "unmatched_prediction": 0}
    if prediction is None:
        for i, box in enumerate(boxes):
            visible["gt"] += rect(box, "#f7f7f7", 1)
        rect(record["anchor_xywhn"], COLORS["focus"], 3, "focus")
    else:
        hit_gt, hit_pred, _ = match_details(prediction, classes, boxes)
        for i, box in enumerate(boxes):
            if not hit_gt[i]:
                visible["missed_gt"] += rect(box, COLORS["miss"], 2)
        for i, box in enumerate(prediction["xywhn"]):
            color = COLORS["tp"] if hit_pred[i] else COLORS["fp"]
            key = "matched_prediction" if hit_pred[i] else "unmatched_prediction"
            visible[key] += rect(box, color, 2)
        # Yellow arrow marks the same full-image location in all three columns.
        x, y = record["anchor_xywhn"][:2]
        px, py = (x*width-crop[0])*scale, (y*height-crop[1])*scale
        draw.line([(px+30, py-30), (px+5, py-5)], fill=COLORS["focus"], width=3)
        draw.polygon([(px+5,py-5),(px+15,py-7),(px+7,py-15)], fill=COLORS["focus"])
    return panel, visible


def main(check=False):
    if check:
        manifest = read(OUT / "manifest.json")
        if any(digest(p) != h for p, h in manifest["input_sha256"].items()):
            raise RuntimeError("case input changed")
        if any(digest(OUT / p) != h for p, h in manifest["output_sha256"].items()):
            raise RuntimeError("case figure changed")
        selected, counts, _, _, _ = select_cases()
        if list(selected.values()) != manifest["selection"] or counts != manifest["all_val_event_counts"]:
            raise RuntimeError("case selection is not reproducible")
        print("Verified four deterministic cases, all-val event counts, and figure hashes")
        return
    if OUT.exists():
        raise FileExistsError("refusing overwrite of frozen case selection")
    selected, counts, predictions, source, hashes = select_cases()
    OUT.mkdir(parents=True)
    sheet = Image.new("RGB", (1776, 1910), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 16), "B2 vs P1 | seed 0 | full-image val960 | confidence 0.25 | matching IoU 0.50", font=font(26), fill="#17212b")
    draw.text((24, 56), "Green: matched prediction   Orange: unmatched prediction   Magenta: missed GT   Yellow: focus", font=font(21), fill="#17212b")
    display = []
    for row_index, kind in enumerate(TYPES):
        record = selected[kind]
        image = Path(record["image"])
        gt = read_label_file(label_path_for_image(image, source.images_val, source.labels_val))
        boxes = np.array([[b.x,b.y,b.w,b.h] for b in gt]).reshape(-1,4)
        classes = [b.class_id for b in gt]
        top = 103 + row_index * 442
        draw.text((24,top), TITLES[row_index] + f" | {record['class_name']} | {image.stem}", font=font(20), fill="#17212b")
        crop_counts = {}
        with Image.open(image) as original:
            pixels = original.convert("RGB")
            for col, name in enumerate(("GT", "B2", "P1")):
                x = 16 + col * 588
                draw.text((x,top+26), name, font=font(20), fill="#17212b")
                pred = None if name == "GT" else predictions[name][record["image_index"]]
                panel, visible = draw_panel(pixels, record, boxes, classes, pred)
                sheet.paste(panel, (x,top+53))
                crop_counts[name] = visible
        display.append({"type": kind, "visible_counts": crop_counts})
    draw.text((24, 1880), "Diagnostic examples from saved predictions; selection includes failures. Not official VisDrone error categories.", font=font(18), fill="#17212b")
    sheet.save(OUT / "cases_overview.png", dpi=(300,300))
    for i, kind in enumerate(TYPES):
        sheet.crop((0,103+i*442,1776,103+(i+1)*442)).save(OUT / f"{kind}.png", dpi=(300,300))
    hashes[str(Path(__file__).resolve())] = digest(__file__)
    files = ["cases_overview.png"] + [f"{kind}.png" for kind in TYPES]
    manifest = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "split_seed": 0, "student_seed": 0, "validation_images": 548,
                "selection_rule": "scan validation images in order and event types in fixed order; for each unfilled type use first eligible GT/prediction index with center >=32 native pixels from image borders, an unused filename prefix before first underscore and unique image SHA256; no effect-size ranking",
                "crop_native_pixels": [384,256], "display_scale": 1.5,
                "selection": list(selected.values()), "all_val_event_counts": counts,
                "display_counts": display, "input_sha256": hashes,
                "output_sha256": {name: digest(OUT / name) for name in files},
                "limitations": ["illustrative seed0 cases, not statistical evidence or seed1 selection",
                                "matching uses all native full-image GT before cropping; TP/FP are diagnostic at fixed threshold",
                                "all val selected before examining new B2 seed1 result; output images stay ignored"]}
    write(OUT / "manifest.json", manifest)
    print("Generated", OUT / "cases_overview.png")
    print("All-val event counts:", counts)
    for r in selected.values():
        print(r["type"],Path(r["image"]).name,r["anchor_index"],r["class_name"],r["crop_xyxy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args().check)
