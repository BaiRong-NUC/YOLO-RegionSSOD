"""通过作者 test.test 的匹配和积分逻辑重评 VisDrone val。"""

import argparse
import hashlib
import json
from multiprocessing.pool import ThreadPool
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from .evaluate_visdrone import LOCAL, ROOT, SHA256, UPSTREAM, WEIGHT, read_visdrone


def prepare(source, target, *, split="val", paths=None, output_split="", on_prepared=None):
    """按作者规则转换指定图片；默认仍准备完整的 548 张验证图。"""
    images = target / "images" / output_split
    labels = target / "labels" / output_split
    images.mkdir(exist_ok=True, parents=True)
    labels.mkdir(exist_ok=True, parents=True)
    if paths is None:
        paths = sorted((source / "images" / split).glob("*.jpg"))
        if split == "val" and len(paths) != 548:
            raise ValueError(f"预期 548 张验证图，实际 {len(paths)}")
    for path in paths:
        image, boxes, masked = read_visdrone(path, source / "original_annotations" / split / (path.stem + ".txt"))
        height, width = image.shape[:2]
        rows = []
        for category, (left, top, box_w, box_h) in boxes:
            rows.append(f"{category - 1} {(left + box_w / 2) / width:.6f} "
                        f"{(top + box_h / 2) / height:.6f} {box_w / width:.6f} {box_h / height:.6f}")
        stem = path.stem + ("_masked" if masked else "")
        if masked:
            if not cv2.imwrite(str(images / (stem + ".jpg")), image):
                raise OSError(f"写入失败 {stem}")
        else:
            shutil.copyfile(path, images / path.name)
        label_path = labels / (stem + ".txt")
        label_path.write_text("\n".join(rows), encoding="ascii")
        if on_prepared is not None:
            # 复用已解码、已涂灰的原尺寸图，供训练数据入口生成监督热图。
            on_prepared(label_path, image)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/visdrone")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--head", choices=["dense", "sparse"], default="dense")
    args = parser.parse_args()
    sys.path.insert(0, str(UPSTREAM))
    from utils import datasets
    from test import test

    with WEIGHT.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != SHA256:
            raise ValueError("作者权重校验失败")
    images = prepare(args.data, ROOT / "data/esod/val")
    options = SimpleNamespace(task="val", single_cls=False)
    # 仅将缓存扫描改成线程，避免 Windows 为每个 worker 重复导入 CUDA。
    allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype("float32"))]
    with patch.object(datasets, "Pool", ThreadPool), torch.serialization.safe_globals(allowed):
        dataset = datasets.LoadImagesAndLabels(str(images), 1536, args.batch, rect=True, stride=32)
    if len(dataset) != 548:
        raise ValueError(f"作者加载器接收了 {len(dataset)} 张图，需核查标签")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, collate_fn=dataset.collate_fn)
    checkpoint = torch.load(WEIGHT, weights_only=False, map_location="cpu")
    model = (checkpoint.get("ema") or checkpoint["model"]).cuda().float().fuse()
    # 传入模型和 DataLoader 使用原 test.test 计算 AP，显式关闭 GT 热图。
    values, _, _ = test({"nc": 10, "val": str(images)}, model=model, dataloader=loader,
                        imgsz=1536, batch_size=args.batch, plots=False, half_precision=False,
                        sparse_head=args.head == "sparse", use_gt=False, opt=options)
    report = {"images": len(dataset), "head": args.head, "batch": args.batch, "sha256": SHA256,
              "AP": float(values[3]) * 100, "AP50": float(values[2]) * 100}
    output = LOCAL / f"author-{args.head}-b{args.batch}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
