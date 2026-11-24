"""准备 ESOD 配对训练用的 VisDrone 初筛子集或完整官方划分。"""

import argparse
from functools import partial
import hashlib
from pathlib import Path
import random
import runpy
import sys
from tempfile import TemporaryDirectory

import numpy as np
import yaml

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.reproduce.esod.author_val import ROOT, UPSTREAM, prepare

sys.path.insert(0, str(UPSTREAM))


def choose_splits(source, full, seed):
    def official(split, expected):
        paths = sorted((source / "images" / split).glob("*.jpg"))
        if len(paths) != expected:
            raise ValueError(f"官方 {split} 预期 {expected} 张图，实际 {len(paths)}")
        return paths

    train = official("train", 6471)
    if full:
        return {"train": ("train", train), "val": ("val", official("val", 548))}
    # 一次无放回抽样，后 16 张作内部验证，官方 val 不参与初筛。
    sample = random.Random(seed).sample(train, 48)
    return {"train": ("train", sorted(sample[:32])), "val": ("train", sorted(sample[32:]))}


def gaussian_generator():
    # 作者脚本在导入时探测并加载 SAM；显式选择其 predictor=None 分支。
    previous = sys.modules.get("segment_anything", ...)
    sys.modules["segment_anything"] = None
    try:
        return runpy.run_path(str(UPSTREAM / "scripts/data_prepare.py"))["gen_mask"]
    finally:
        if previous is ...:
            del sys.modules["segment_anything"]
        else:
            sys.modules["segment_anything"] = previous


def write_mask(generate, label, image):
    rows = label.read_text(encoding="ascii").splitlines()
    # 上游遇到重复框会跳过整张图；零面积框没有可生成高斯监督的区域。
    seen = set()
    valid = []
    for row in rows:
        reason = ""
        if row in seen:
            reason = "重复框，保留首次标注"
        elif any(float(side) <= 0 for side in row.split()[3:5]):
            reason = "零面积框，跳过此框"
        if reason:
            print(f"{label.parent.name}/{label.name}：{reason}，YOLO 行 {row}", flush=True)
        else:
            valid.append(row)
        seen.add(row)
    if valid != rows:
        label.write_text("\n".join(valid), encoding="ascii")
    if not valid:
        print(f"{label.parent.name}/{label.name}：无有效目标，按作者规则保留背景图", flush=True)
    # 强制正斜杠，让作者脚本在 Windows 也写入 masks/<split>/。
    generate(label.as_posix(), image, cls_ratio=True, thresh=0.5, sam_only=False)


def progress(paths, description):
    for index, path in enumerate(paths, 1):
        yield path
        if len(paths) >= 500 and (index % 500 == 0 or index == len(paths)):
            print(f"{datetime.now():%H:%M:%S} {description} {index}/{len(paths)}", flush=True)


def verify_split(root, split, source_paths):
    from utils.datasets import img2label_paths, verify_image_label

    images = sorted((root / "images" / split).glob("*.jpg"))
    if sorted(path.stem.removesuffix("_masked") for path in images) != sorted(path.stem for path in source_paths):
        raise ValueError(f"{split} 输出图片与固定抽样清单不一致")
    stems = {path.stem for path in images}
    for folder, extension in (("labels", "*.txt"), ("masks", "*.npy")):
        if {path.stem for path in (root / folder / split).glob(extension)} != stems:
            raise ValueError(f"{folder}/{split} 与图片未逐一配对")

    boxes = 0
    positive_images = 0
    for path in progress(images, f"核验 {split}"):
        label = img2label_paths([str(path)])[0]
        mask = root / "masks" / split / (path.stem + ".npy")
        _, targets, size, _, missing, found, _, corrupt = verify_image_label((str(path), label, ""))
        if missing or not found or corrupt:
            raise ValueError(f"作者加载器无法接收图像或标注：{path}，{label}")
        count = len(targets)
        if count and not np.isin(targets[:, 0], np.arange(10)).all():
            raise ValueError(f"类别编号超出 VisDrone 十类：{label}")
        shape = (size[1], size[0], 2)
        data = np.load(mask, allow_pickle=False)
        if data.shape != shape or data.dtype != np.float16 or not np.isfinite(data).all():
            raise ValueError(f"热图应为有限 float16 且维度为 {shape}：{mask}")
        foreground, weight = data[..., 0], data[..., 1]
        positive = bool(np.any(foreground > 0))
        if positive != bool(count) or np.any(foreground < 0) or np.any(foreground > 1) or np.any(weight < 1):
            raise ValueError(f"前景、权重或标签数量异常：{mask}")
        boxes += count
        positive_images += positive
    digest = hashlib.sha256("\n".join(path.name for path in source_paths).encode()).hexdigest()
    print(f"{split}：{len(images)} 张，{boxes} 框，{positive_images} 张有前景，"
          f"{len(images) - positive_images} 张背景图。逐张通过原图 H×W×2、float16、前景和权重核验。"
          f"清单 SHA-256 {digest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/visdrone")
    parser.add_argument("--target", type=Path, help="默认 data/esod/pair32，全量默认 data/esod/pair-full")
    parser.add_argument("--seed", type=int, default=42, help="初筛抽样种子")
    parser.add_argument("--full", action="store_true", help="准备官方 train 6471 张和 val 548 张")
    args = parser.parse_args()
    target = (args.target or ROOT / "data/esod" / ("pair-full" if args.full else "pair32")).resolve()
    yaml_path = target / "visdrone.yaml"
    splits = choose_splits(args.data, args.full, args.seed)
    author = yaml.safe_load((UPSTREAM / "data/visdrone.yaml").read_text(encoding="utf-8"))
    config = {split: (target / "images" / split).as_posix() for split in splits}
    config.update(nc=author["nc"], names=author["names"],
                  source_splits={split: source for split, (source, _) in splits.items()},
                  sample_seed=None if args.full else args.seed)
    if target.exists():
        if not yaml_path.is_file() or yaml.safe_load(yaml_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"目标已有其他配置；请通过 --target 指定独立目录：{target}")
        for split, (_, paths) in splits.items():
            verify_split(target, split, paths)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_gaussian = partial(write_mask, gaussian_generator())
        # 同盘临时目录完成后整目录改名，失败或中断不会发布半成品。
        with TemporaryDirectory(dir=target.parent, prefix=f".{target.name}-") as directory:
            staging = Path(directory)
            for split, (source, paths) in splits.items():
                prepare(args.data, staging, split=source, paths=progress(paths, f"生成 {split}"),
                        output_split=split, on_prepared=write_gaussian)
                verify_split(staging, split, paths)
            (staging / yaml_path.name).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            staging.rename(target)
    print(f"已就绪 {yaml_path}", flush=True)


if __name__ == "__main__":
    main()
