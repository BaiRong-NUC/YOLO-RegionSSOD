from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from regionssod.data.dataset import load_source_dataset
from regionssod.data.partial import build_partial_dataset
from regionssod.evaluation.runner import run_validation
from regionssod.pseudo.generator import (
    combine_image_lists,
    generate_pseudo_labels,
)
from regionssod.training.runner import run_training


def _print_summary(summary: dict[str, object]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regionssod",
        description="Region-partial semi-supervised YOLO experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_partial = subparsers.add_parser(
        "make-partial",
        help="materialize object-centered labeled tiles and unlabeled tiles",
    )
    make_partial.add_argument("--source-data", type=Path, required=True)
    make_partial.add_argument("--output", type=Path, required=True)
    make_partial.add_argument("--ratio", type=float, required=True)
    make_partial.add_argument("--seed", type=int, default=0)
    make_partial.add_argument("--unlabeled-per-image", type=int, default=1)
    make_partial.add_argument("--min-box-visibility", type=float, default=0.35)
    make_partial.add_argument("--jitter-fraction", type=float, default=0.2)
    make_partial.add_argument("--limit", type=int, default=None)

    train = subparsers.add_parser("train", help="train a YOLO experiment")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--model", type=str, default=None)
    train.add_argument("--data", type=Path, default=None)
    train.add_argument("--name", type=str, default=None)
    train.add_argument("--project", type=str, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch", type=int, default=None)
    train.add_argument("--imgsz", type=int, default=None)
    train.add_argument("--device", type=str, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--exist-ok", action="store_true")

    validate = subparsers.add_parser("validate", help="evaluate on full-image val")
    validate.add_argument("--weights", type=Path, required=True)
    validate.add_argument("--data", type=Path, required=True)
    validate.add_argument("--imgsz", type=int, default=960)
    validate.add_argument("--batch", type=int, default=8)
    validate.add_argument("--device", type=str, default="0")
    validate.add_argument("--project", type=str, default="runs/val")
    validate.add_argument("--name", type=str, default=None)

    size_recall = subparsers.add_parser("size-recall", help="full-image recall at a fixed confidence, by native GT size")
    size_recall.add_argument("--weights", type=Path, required=True)
    size_recall.add_argument("--data", type=Path, required=True)
    size_recall.add_argument("--output", type=Path, required=True)
    size_recall.add_argument("--imgsz", type=int, default=960)
    size_recall.add_argument("--batch", type=int, default=8)
    size_recall.add_argument("--device", type=str, default="0")
    size_recall.add_argument("--conf", type=float, default=.25)
    size_recall.add_argument("--match-iou", type=float, default=.5)

    pseudo = subparsers.add_parser(
        "pseudo-label",
        help="generate global-threshold pseudo labels for unlabeled tiles",
    )
    pseudo.add_argument("--weights", type=Path, required=True)
    pseudo.add_argument("--images-list", type=Path, required=True)
    pseudo.add_argument("--output", type=Path, required=True)
    pseudo.add_argument("--threshold", type=float, default=0.5)
    pseudo.add_argument("--conf-floor", type=float, default=0.05)
    pseudo.add_argument("--imgsz", type=int, default=960)
    pseudo.add_argument("--batch", type=int, default=8)
    pseudo.add_argument("--device", type=str, default="0")
    pseudo.add_argument("--max-det", type=int, default=500)
    pseudo.add_argument("--limit", type=int, default=None)

    p1 = subparsers.add_parser("pseudo-label-p1", help="generate scale-aware positives and uncertain-ignore metadata")
    p1.add_argument("--weights", type=Path, required=True)
    p1.add_argument("--images-list", type=Path, required=True)
    p1.add_argument("--output", type=Path, required=True)
    p1.add_argument("--policy", type=Path, default=Path("configs/pseudo_p1_10.yaml"))
    p1.add_argument("--imgsz", type=int, default=960)
    p1.add_argument("--batch", type=int, default=8)
    p1.add_argument("--device", type=str, default="0")
    p1.add_argument("--max-det", type=int, default=500)
    p1.add_argument("--limit", type=int, default=None)

    combine = subparsers.add_parser(
        "combine-lists",
        help="combine labeled and pseudo image lists into a dataset YAML",
    )
    combine.add_argument("--inputs", type=Path, nargs="+", required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--dataset-yaml", type=Path, required=True)
    combine.add_argument("--source-data", type=Path, required=True)
    combine.add_argument("--p1-metadata", type=Path, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "make-partial":
        source = load_source_dataset(args.source_data)
        summary = build_partial_dataset(
            source=source,
            output_dir=args.output,
            area_ratio=args.ratio,
            seed=args.seed,
            unlabeled_per_image=args.unlabeled_per_image,
            min_box_visibility=args.min_box_visibility,
            jitter_fraction=args.jitter_fraction,
            limit=args.limit,
        )
        _print_summary(summary)
        return 0

    if args.command == "train":
        save_dir = run_training(
            args.config,
            model=args.model,
            data=args.data,
            name=args.name,
            project=args.project,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            seed=args.seed,
            exist_ok=args.exist_ok,
        )
        print(f"run_dir: {save_dir}")
        return 0

    if args.command == "validate":
        save_dir = run_validation(
            weights=args.weights,
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
        )
        print(f"validation_dir: {save_dir}")
        return 0

    if args.command == "size-recall":
        from regionssod.evaluation.size_recall import run_size_recall

        _print_summary(run_size_recall(args.weights, args.data, args.output, imgsz=args.imgsz,
                                      batch=args.batch, device=args.device, conf=args.conf,
                                      match_iou=args.match_iou))
        return 0

    if args.command == "pseudo-label":
        summary = generate_pseudo_labels(
            weights=args.weights,
            images_list=args.images_list,
            output_dir=args.output,
            threshold=args.threshold,
            conf_floor=args.conf_floor,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            max_det=args.max_det,
            limit=args.limit,
        )
        _print_summary(summary)
        return 0

    if args.command == "pseudo-label-p1":
        from regionssod.pseudo.scale import generate_scale_labels

        _print_summary(generate_scale_labels(
            args.weights, args.images_list, args.output, policy_path=args.policy,
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            max_det=args.max_det, limit=args.limit))
        return 0

    if args.command == "combine-lists":
        summary = combine_image_lists(
            inputs=args.inputs,
            output=args.output,
            dataset_yaml=args.dataset_yaml,
            source_data=args.source_data,
            p1_metadata=args.p1_metadata,
        )
        _print_summary(summary)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2
