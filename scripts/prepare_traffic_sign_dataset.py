"""Prepare GTSRB sign crops as a YOLO training dataset (7-class ADAS subset)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.traffic_sign.config import TrafficSignTrainingConfig
from training.traffic_sign.gtsrb_converter import (
    GTSRB_IMAGE_EXTENSIONS,
    GTSRB_TO_YOLO_CLASS,
    MAPPED_GTSRB_CLASS_IDS,
    YOLO_CLASS_NAMES,
    gtsrb_class_id_to_yolo_class,
    load_gtsrb_test_csv,
    gtsrb_test_row_to_yolo_line,
    train_crop_to_yolo_line,
    write_dataset_yaml,
)

logger = logging.getLogger(__name__)


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


def _export_jpg(source: Path, destination: Path, use_copy: bool) -> None:
    """Convert GTSRB PPM/PNG to JPG for Ultralytics compatibility."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if source.suffix.lower() in {".jpg", ".jpeg"}:
        _link_or_copy(source, destination, use_copy=use_copy)
        return

    with Image.open(source) as image:
        rgb = image.convert("RGB")
        rgb.save(destination, format="JPEG", quality=95)


def _init_stats() -> dict:
    return {
        "splits": {},
        "class_counts": {name: 0 for name in YOLO_CLASS_NAMES},
        "gtsrb_class_counts": {str(gtsrb_id): 0 for gtsrb_id in sorted(MAPPED_GTSRB_CLASS_IDS)},
        "skipped": {
            "unmapped_gtsrb_class": 0,
            "missing_image": 0,
            "invalid_test_row": 0,
        },
        "total_boxes_kept": 0,
        "images_exported": 0,
        "val_source": "",
    }


def _collect_train_images(train_dir: Path) -> list[tuple[Path, int, int]]:
    """Return ``(image_path, gtsrb_class_id, yolo_class_id)`` for mapped classes."""
    entries: list[tuple[Path, int, int]] = []
    for gtsrb_id in sorted(MAPPED_GTSRB_CLASS_IDS):
        class_dir = train_dir / f"{gtsrb_id:05d}"
        if not class_dir.is_dir():
            logger.warning("Missing GTSRB train folder: %s", class_dir)
            continue

        yolo_id = GTSRB_TO_YOLO_CLASS[gtsrb_id]
        for ext in GTSRB_IMAGE_EXTENSIONS:
            for image_path in sorted(class_dir.glob(f"*{ext}")):
                entries.append((image_path, gtsrb_id, yolo_id))
    return entries


def _split_train_entries(
    entries: list[tuple[Path, int, int]],
    val_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, int, int]], list[tuple[Path, int, int]]]:
    """Stratified train/val split when official GTSRB test set is unavailable."""
    by_class: dict[int, list[tuple[Path, int, int]]] = {}
    for entry in entries:
        by_class.setdefault(entry[2], []).append(entry)

    train_split: list[tuple[Path, int, int]] = []
    val_split: list[tuple[Path, int, int]] = []
    rng = random.Random(seed)

    for class_entries in by_class.values():
        shuffled = list(class_entries)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * val_fraction))
        val_split.extend(shuffled[:val_count])
        train_split.extend(shuffled[val_count:])

    return train_split, val_split


def _export_train_split(
    entries: list[tuple[Path, int, int]],
    split: str,
    config: TrafficSignTrainingConfig,
    use_copy: bool,
    convert_jpg: bool,
) -> dict:
    out_image_dir = config.dataset_dir / "images" / split
    out_label_dir = config.dataset_dir / "labels" / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    split_stats = {
        "images_exported": 0,
        "boxes_kept": 0,
        "class_counts": {name: 0 for name in YOLO_CLASS_NAMES},
        "gtsrb_class_counts": {str(gtsrb_id): 0 for gtsrb_id in sorted(MAPPED_GTSRB_CLASS_IDS)},
    }

    for index, (image_path, gtsrb_id, yolo_id) in enumerate(
        tqdm(entries, desc=f"prepare-train-{split}", unit="img"),
    ):
        stem = f"gtsrb_train_{gtsrb_id:05d}_{index:06d}"
        out_image = out_image_dir / f"{stem}.jpg"
        out_label = out_label_dir / f"{stem}.txt"

        if convert_jpg:
            _export_jpg(image_path, out_image, use_copy=use_copy)
        else:
            _link_or_copy(image_path, out_image, use_copy=use_copy)

        class_name = YOLO_CLASS_NAMES[yolo_id]
        out_label.write_text(train_crop_to_yolo_line(yolo_id) + "\n", encoding="utf-8")

        split_stats["images_exported"] += 1
        split_stats["boxes_kept"] += 1
        split_stats["class_counts"][class_name] += 1
        split_stats["gtsrb_class_counts"][str(gtsrb_id)] += 1

    return split_stats


def _export_official_test_split(
    config: TrafficSignTrainingConfig,
    use_copy: bool,
    convert_jpg: bool,
) -> dict:
    """Export GTSRB official test images with ROI boxes from ``Test.csv``."""
    out_image_dir = config.dataset_dir / "images" / "val"
    out_label_dir = config.dataset_dir / "labels" / "val"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    split_stats = {
        "images_exported": 0,
        "boxes_kept": 0,
        "class_counts": {name: 0 for name in YOLO_CLASS_NAMES},
        "gtsrb_class_counts": {str(gtsrb_id): 0 for gtsrb_id in sorted(MAPPED_GTSRB_CLASS_IDS)},
        "skipped": {"unmapped_gtsrb_class": 0, "missing_image": 0, "invalid_test_row": 0},
    }

    rows = load_gtsrb_test_csv(config.gtsrb_test_csv)
    for row in tqdm(rows, desc="prepare-test-val", unit="row"):
        try:
            gtsrb_id = int(row["ClassId"])
        except (KeyError, TypeError, ValueError):
            split_stats["skipped"]["invalid_test_row"] += 1
            continue

        if gtsrb_class_id_to_yolo_class(gtsrb_id) is None:
            split_stats["skipped"]["unmapped_gtsrb_class"] += 1
            continue

        parsed = gtsrb_test_row_to_yolo_line(row)
        if parsed is None:
            split_stats["skipped"]["invalid_test_row"] += 1
            continue

        yolo_line, _width, _height = parsed
        class_id = int(yolo_line.split()[0])
        filename = row.get("Filename", "")
        if not filename:
            split_stats["skipped"]["invalid_test_row"] += 1
            continue

        image_path = config.gtsrb_test_dir / filename
        if not image_path.is_file():
            split_stats["skipped"]["missing_image"] += 1
            continue

        stem = Path(filename).stem
        out_image = out_image_dir / f"{stem}.jpg"
        out_label = out_label_dir / f"{stem}.txt"

        if convert_jpg:
            _export_jpg(image_path, out_image, use_copy=use_copy)
        else:
            _link_or_copy(image_path, out_image, use_copy=use_copy)

        out_label.write_text(yolo_line + "\n", encoding="utf-8")

        class_name = YOLO_CLASS_NAMES[class_id]
        split_stats["images_exported"] += 1
        split_stats["boxes_kept"] += 1
        split_stats["class_counts"][class_name] += 1
        split_stats["gtsrb_class_counts"][str(gtsrb_id)] += 1

    return split_stats


def prepare_dataset(
    config: TrafficSignTrainingConfig,
    use_copy: bool = False,
    convert_jpg: bool = True,
    use_official_test_val: bool = True,
) -> dict:
    """Build ``traffic_sign_yolo`` dataset and return aggregate statistics."""
    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    stats = _init_stats()

    train_entries = _collect_train_images(config.gtsrb_train_dir)
    if not train_entries:
        raise FileNotFoundError(
            f"No mapped GTSRB training images found under {config.gtsrb_train_dir}. "
            "Download GTSRB and extract Train/ class folders."
        )

    has_official_test = (
        use_official_test_val
        and config.gtsrb_test_csv.is_file()
        and config.gtsrb_test_dir.is_dir()
    )

    if has_official_test:
        train_split = train_entries
        stats["val_source"] = "gtsrb_official_test"
        train_stats = _export_train_split(
            train_split,
            "train",
            config,
            use_copy=use_copy,
            convert_jpg=convert_jpg,
        )
        val_stats = _export_official_test_split(
            config,
            use_copy=use_copy,
            convert_jpg=convert_jpg,
        )
    else:
        train_split, val_split = _split_train_entries(
            train_entries,
            val_fraction=config.val_fraction,
            seed=config.seed,
        )
        stats["val_source"] = f"stratified_split_{config.val_fraction:.0%}"
        train_stats = _export_train_split(
            train_split,
            "train",
            config,
            use_copy=use_copy,
            convert_jpg=convert_jpg,
        )
        val_stats = _export_train_split(
            val_split,
            "val",
            config,
            use_copy=use_copy,
            convert_jpg=convert_jpg,
        )

    stats["splits"]["train"] = train_stats
    stats["splits"]["val"] = val_stats
    stats["images_exported"] = train_stats["images_exported"] + val_stats["images_exported"]
    stats["total_boxes_kept"] = train_stats["boxes_kept"] + val_stats["boxes_kept"]

    for split_stats in (train_stats, val_stats):
        for name, count in split_stats["class_counts"].items():
            stats["class_counts"][name] += count
        for gtsrb_id, count in split_stats["gtsrb_class_counts"].items():
            stats["gtsrb_class_counts"][gtsrb_id] += count

    if "skipped" in val_stats:
        for key, count in val_stats["skipped"].items():
            stats["skipped"][key] += count

    write_dataset_yaml(config.dataset_yaml, config.dataset_dir)
    config.stats_path.parent.mkdir(parents=True, exist_ok=True)
    config.stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _print_summary(stats: dict, config: TrafficSignTrainingConfig) -> None:
    print("\n=== Traffic Sign YOLO Dataset ===")
    print(f"GTSRB root : {config.gtsrb_root}")
    print(f"Output     : {config.dataset_dir}")
    print(f"Val source : {stats['val_source']}")
    print(f"Dataset YAML: {config.dataset_yaml}")
    print(f"Stats JSON : {config.stats_path}")
    for split, split_stats in stats["splits"].items():
        print(
            f"\n[{split}] images={split_stats['images_exported']} "
            f"boxes={split_stats['boxes_kept']}"
        )
        print(f"  class counts: {split_stats['class_counts']}")
    print(f"\nTotal boxes kept: {stats['total_boxes_kept']}")
    print(f"Global class distribution: {stats['class_counts']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GTSRB sign crops to YOLO format (7-class ADAS subset).",
    )
    parser.add_argument(
        "--gtsrb",
        type=Path,
        default=TrafficSignTrainingConfig().gtsrb_root,
        help="GTSRB root (contains Train/, Test/, Test.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TrafficSignTrainingConfig().dataset_dir,
        help="Output dataset directory (traffic_sign_yolo).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    parser.add_argument(
        "--no-jpg-convert",
        action="store_true",
        help="Keep original image format instead of converting to JPG.",
    )
    parser.add_argument(
        "--stratified-val",
        action="store_true",
        help="Use 20%% train hold-out for val instead of official GTSRB test set.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=TrafficSignTrainingConfig().val_fraction,
        help="Val fraction when using stratified split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TrafficSignTrainingConfig().seed,
        help="Random seed for stratified split.",
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

    config = TrafficSignTrainingConfig(
        gtsrb_root=args.gtsrb,
        dataset_dir=args.output,
        stats_path=args.output / "dataset_stats.json",
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    if not config.gtsrb_train_dir.is_dir():
        logger.error("GTSRB Train directory not found: %s", config.gtsrb_train_dir)
        return 1

    try:
        stats = prepare_dataset(
            config,
            use_copy=args.copy,
            convert_jpg=not args.no_jpg_convert,
            use_official_test_val=not args.stratified_val,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    _print_summary(stats, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
