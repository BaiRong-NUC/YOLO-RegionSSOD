"""从同一份 COCO 权重独立训练普通 YOLOv5m 或 ESOD-YOLOv5m。"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from .evaluate_visdrone import LOCAL, REVISION, UPSTREAM


INITIAL_SHA256 = "317d74254b2958ff33b3c42f1a0fd06ad5f2fa613a0e3e23bdf835a9579a99a8"


def prepare_source(directory):
    """在临时目录应用本机兼容改动，保留作者子模块原样。"""
    archive = subprocess.check_output(["git", "-C", str(UPSTREAM), "archive", "--format=zip", REVISION,
                                       "train.py", "test.py", "models", "utils", "data"])
    with zipfile.ZipFile(io.BytesIO(archive)) as snapshot:
        snapshot.extractall(directory)

    # 旧版训练器硬编码种子和 worker 数，且只识别 Unix 格式的热图路径。
    edits = {
        "train.py": [
            ("plots = not opt.evolve", "plots = False"),
            ("init_seeds(2 + rank)", "init_seeds(opt.seed)"),
            ("    opt = parser.parse_args()", "    parser.add_argument('--seed', type=int, required=True)\n"
                                            "    opt = parser.parse_args()"),
        ],
        "utils/datasets.py": [
            ("    nw = 8\n", "    nw = workers\n"),
            ("with Pool(num_threads) as pool:", "with ThreadPool(num_threads) as pool:"),
            ("path.replace('/images/', '/masks/').replace('.jpg', '.npy')",
             "Path(path.replace(os.sep + 'images' + os.sep, os.sep + 'masks' + os.sep)).with_suffix('.npy')"),
            ("            mask = np.zeros((h0, w0, 2), dtype=np.float32)\n            mask[..., 1] += 1.",
             "            raise FileNotFoundError(mask_path)"),
        ],
        # 新版 PyTorch 要求对整数索引执行 clamp 时也提供整数网格上界。
        "utils/loss.py": [("gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)",
                           "gj.clamp_(0, p[i].shape[2] - 1), gi.clamp_(0, p[i].shape[3] - 1)")],
    }
    for name, replacements in edits.items():
        path = directory / name
        source = path.read_text(encoding="utf-8")
        for original, replacement in replacements:
            if source.count(original) != 1:
                raise ValueError(f"上游 {name} 的兼容位置不唯一或已变化：{original}")
            source = source.replace(original, replacement)
        path.write_text(source, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["vanilla", "esod"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--long-edge", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    if min(args.long_edge, args.epochs, args.batch) <= 0:
        parser.error("长边、轮数和 batch 均须为正整数")
    # ESOD 的 P3 步长 8、每边切 8 块、块边长对齐 4；普通基线没有切片约束。
    if args.arm == "esod" and args.long_edge % 256:
        parser.error("ESOD 训练长边须为 256 的倍数，例如 768 或 1536")
    weights = LOCAL / "yolov5m-v5.0.pt"
    with weights.open("rb") as checkpoint:
        if hashlib.file_digest(checkpoint, "sha256").hexdigest() != INITIAL_SHA256:
            raise ValueError("COCO 初始权重与 YOLOv5 v5.0 下载文件不一致")
    data = args.data.resolve(strict=True)
    output = args.output.resolve()
    # 每个种子和方法单独保存，防止重跑覆盖已有权重。
    output.mkdir(parents=True, exist_ok=False)
    protocol = {**vars(args), "data": str(data), "output": str(output),
                "upstream": REVISION, "initial_sha256": INITIAL_SHA256}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="visdrone-train-") as temp:
        source = Path(temp)
        prepare_source(source)
        # 作者依据 cfg 路径中的 esod 决定如何映射预训练层，必须保留相对路径。
        command = [sys.executable, "-u", "train.py", "--cfg", f"models/cfg/{args.arm}/visdrone_yolov5m.yaml",
                   "--data", str(data), "--hyp", "data/hyps/hyp.visdrone.yaml", "--weights", str(weights),
                   "--epochs", str(args.epochs), "--img-size", str(args.long_edge),
                   "--batch-size", str(args.batch), "--seed", str(args.seed), "--workers", "0",
                   "--device", "0", "--project", str(output.parent), "--name", output.name, "--exist-ok"]
        # 作者权重和本机标签缓存采用旧版 pickle。只在这次受控训练进程启用兼容模式。
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "WANDB_MODE": "disabled",
               "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "MPLBACKEND": "Agg"}
        with (output / "train.log").open("wb") as log:
            subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
