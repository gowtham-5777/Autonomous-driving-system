"""Prepare BDD100K traffic-light annotations as a YOLO training dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.traffic_signal.bdd100k_converter import (
    YOLO_CLASS_NAMES,
    convert_bdd100k_annotation,
    load_bdd100k_annotation,
    write_dataset_yaml,
)
from training.traffic_signal.config import TrafficSignalTrainingConfig

logger = logging.getLogger(__name__)


def _resolve_image_size(image_path: Path, verify: bool) -> tuple[int, int]:
    if not verify:
        return 1280, 720
    try:
        with Image.open(image_path) as image:
            return image.size
    except OSError:
        return 1280, 720


def _link_or_copy(source: Path, destination: Path, use_copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if use_copy:
        shutil.copy2(source, destination)
        return

    try:
        os.link(source, destination)
        return
    except OSError:
        pass

    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def _init_stats() -> dict:
    return {
        "splits": {},
        "class_counts": {"red": 0, "yellow": 0, "green": 0},
        "skipped": {
            "none_color": 0,
            "invalid_color": 0,
            "invalid_box": 0,
        },
        "total_traffic_lights_seen": 0,
        "total_boxes_kept": 0,
        "images_with_labels": 0,
        "images_without_valid_labels": 0,
        "missing_images": 0,
    }


def _process_split(
    split: str,
    config: TrafficSignalTrainingConfig,
    use_copy: bool,
    verify_dimensions: bool,
) -> dict:
    label_dir = config.bdd100k_labels / split
    image_dir = config.bdd100k_images / split
    out_image_dir = config.dataset_dir / "images" / split
    out_label_dir = config.dataset_dir / "labels" / split
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_image_dir.mkdir(parents=True, exist_ok=True)

    split_stats = {
        "source_label_files": 0,
        "images_exported": 0,
        "boxes_kept": 0,
        "class_counts": {"red": 0, "yellow": 0, "green": 0},
        "skipped": {"none_color": 0, "invalid_color": 0, "invalid_box": 0},
        "missing_images": 0,
        "images_without_valid_labels": 0,
    }

    label_files = sorted(label_dir.glob("*.json"))
    split_stats["source_label_files"] = len(label_files)

    for label_path in tqdm(label_files, desc=f"prepare-{split}", unit="file"):
        stem = label_path.stem
        image_path = image_dir / f"{stem}.jpg"
        if not image_path.is_file():
            split_stats["missing_images"] += 1
            continue

        annotation = load_bdd100k_annotation(label_path)
        img_width, img_height = _resolve_image_size(image_path, verify_dimensions)
        yolo_lines, delta = convert_bdd100k_annotation(
            annotation,
            img_width=img_width,
            img_height=img_height,
        )

        split_stats["skipped"]["none_color"] += delta["skipped_none_color"]
        split_stats["skipped"]["invalid_color"] += delta["skipped_invalid_color"]
        split_stats["skipped"]["invalid_box"] += delta["skipped_invalid_box"]

        if not yolo_lines:
            split_stats["images_without_valid_labels"] += 1
            continue

        for line in yolo_lines:
            class_id = int(line.split()[0])
            class_name = YOLO_CLASS_NAMES[class_id]
            split_stats["class_counts"][class_name] += 1
            split_stats["boxes_kept"] += 1

        out_image = out_image_dir / image_path.name
        out_label = out_label_dir / f"{stem}.txt"
        _link_or_copy(image_path, out_image, use_copy=use_copy)
        out_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        split_stats["images_exported"] += 1

    return split_stats


def prepare_dataset(
    config: TrafficSignalTrainingConfig,
    use_copy: bool = False,
    verify_dimensions: bool = False,
) -> dict:
    """Build ``traffic_signal_yolo`` dataset and return aggregate statistics."""
    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    stats = _init_stats()

    for split in ("train", "val"):
        split_stats = _process_split(
            split, config, use_copy=use_copy, verify_dimensions=verify_dimensions
        )
        stats["splits"][split] = split_stats
        stats["images_with_labels"] += split_stats["images_exported"]
        stats["images_without_valid_labels"] += split_stats["images_without_valid_labels"]
        stats["missing_images"] += split_stats["missing_images"]
        stats["total_boxes_kept"] += split_stats["boxes_kept"]
        for color, count in split_stats["class_counts"].items():
            stats["class_counts"][color] += count
        for key in stats["skipped"]:
            stats["skipped"][key] += split_stats["skipped"][key]

    stats["total_traffic_lights_seen"] = (
        stats["total_boxes_kept"]
        + stats["skipped"]["none_color"]
        + stats["skipped"]["invalid_color"]
        + stats["skipped"]["invalid_box"]
    )

    write_dataset_yaml(config.dataset_yaml, config.dataset_dir)
    config.stats_path.parent.mkdir(parents=True, exist_ok=True)
    config.stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _print_summary(stats: dict, config: TrafficSignalTrainingConfig) -> None:
    print("\n=== Traffic Signal YOLO Dataset ===")
    print(f"Output: {config.dataset_dir}")
    print(f"Dataset YAML: {config.dataset_yaml}")
    print(f"Stats JSON: {config.stats_path}")
    for split, split_stats in stats["splits"].items():
        print(
            f"\n[{split}] images={split_stats['images_exported']} "
            f"boxes={split_stats['boxes_kept']} "
            f"missing_images={split_stats['missing_images']}"
        )
        print(f"  class counts: {split_stats['class_counts']}")
        print(f"  skipped: {split_stats['skipped']}")
    print(f"\nTotal boxes kept: {stats['total_boxes_kept']}")
    print(f"Global class distribution: {stats['class_counts']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract BDD100K traffic-light boxes and convert to YOLO format.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=TrafficSignalTrainingConfig().bdd100k_images,
        help="BDD100K images root (contains train/ and val/).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=TrafficSignalTrainingConfig().bdd100k_labels,
        help="BDD100K labels root (contains train/ and val/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TrafficSignalTrainingConfig().dataset_dir,
        help="Output dataset directory (traffic_signal_yolo).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    parser.add_argument(
        "--verify-dimensions",
        action="store_true",
        help="Read each image with PIL to verify size (slower).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = TrafficSignalTrainingConfig(
        bdd100k_images=args.images,
        bdd100k_labels=args.labels,
        dataset_dir=args.output,
        stats_path=args.output / "dataset_stats.json",
    )

    if not config.bdd100k_images.is_dir():
        logger.error("Images directory not found: %s", config.bdd100k_images)
        return 1
    if not config.bdd100k_labels.is_dir():
        logger.error("Labels directory not found: %s", config.bdd100k_labels)
        return 1

    stats = prepare_dataset(
        config,
        use_copy=args.copy,
        verify_dimensions=args.verify_dimensions,
    )
    _print_summary(stats, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
