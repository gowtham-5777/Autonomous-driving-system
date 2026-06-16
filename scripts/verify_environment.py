#!/usr/bin/env python3
"""Verify ADAS project configuration, dataset paths, and model weight paths.

Loads config/default.yaml, resolves all paths, creates model directories,
and prints a summary table. Exits with code 0 even when assets are missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as: python scripts/verify_environment.py
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.model_paths import (  # noqa: E402
    _get_data_root,
    _load_config,
    _resolve_config_path,
    get_pretrained_models_dir,
    get_ssd_weights_path,
    get_traffic_light_cnn_path,
    get_trained_models_dir,
    get_unet_weights_path,
    get_yolop_weights_path,
    get_yolov5_weights_path,
)

DATASET_PATH_KEYS = ("coco", "cityscapes", "gtsrb", "videos", "processed")

MODEL_CHECKS = (
    ("YOLOP", get_yolop_weights_path),
    ("SSD MobileNetV2", get_ssd_weights_path),
    ("YOLOv5", get_yolov5_weights_path),
    ("U-Net", get_unet_weights_path),
    ("Traffic Light CNN", get_traffic_light_cnn_path),
)


def _status_exists(path: Path, *, is_dir: bool) -> str:
    """Return a human-readable existence status for a path."""
    if is_dir:
        return "EXISTS" if path.is_dir() else "MISSING"
    return "EXISTS" if path.is_file() else "MISSING"


def _print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _print_table(rows: list[tuple[str, str, str]]) -> None:
    """Print a fixed-width summary table."""
    col1, col2, col3 = "Check", "Path", "Status"
    w1 = max([len(col1), *(len(r[0]) for r in rows)])
    w2 = max([len(col2), *(len(r[1]) for r in rows)])
    w3 = max([len(col3), *(len(r[2]) for r in rows)])

    separator = f"+{'-' * (w1 + 2)}+{'-' * (w2 + 2)}+{'-' * (w3 + 2)}+"
    print(separator)
    print(f"| {col1:<{w1}} | {col2:<{w2}} | {col3:<{w3}} |")
    print(separator)
    for name, path_str, status in rows:
        print(f"| {name:<{w1}} | {path_str:<{w2}} | {status:<{w3}} |")
    print(separator)


def main() -> int:
    """Run environment verification and print a summary report."""
    config = _load_config()
    project = config.get("project", {})

    name = project.get("name", "N/A")
    version = project.get("version", "N/A")
    data_root = _get_data_root(config)

    _print_header("Project")
    print(f"  Name:      {name}")
    print(f"  Version:   {version}")
    print(f"  data_root: {data_root}")

    _print_header("Dataset Paths")
    dataset_rows: list[tuple[str, str, str]] = []
    for key in DATASET_PATH_KEYS:
        path = _resolve_config_path(config, key)
        status = _status_exists(path, is_dir=True)
        print(f"  {key:<12} {path}")
        dataset_rows.append((key, str(path), status))

    _print_header("Model Directories")
    pretrained_dir = get_pretrained_models_dir()
    trained_dir = get_trained_models_dir()
    print(f"  pretrained: {pretrained_dir}")
    print(f"  trained:    {trained_dir}")

    _print_header("Model Weight Paths")
    model_rows: list[tuple[str, str, str]] = []
    for label, path_fn in MODEL_CHECKS:
        path = path_fn()
        status = _status_exists(path, is_dir=False)
        print(f"  {label:<20} {path}")
        model_rows.append((label, str(path), status))

    dir_rows = [
        ("models_pretrained", str(pretrained_dir), _status_exists(pretrained_dir, is_dir=True)),
        ("models_trained", str(trained_dir), _status_exists(trained_dir, is_dir=True)),
    ]

    _print_header("Summary")
    _print_table(dataset_rows + dir_rows + model_rows)

    missing_datasets = sum(1 for _, _, s in dataset_rows if s == "MISSING")
    missing_models = sum(1 for _, _, s in model_rows if s == "MISSING")

    print()
    print(f"Dataset directories missing: {missing_datasets}/{len(dataset_rows)}")
    print(f"Model weight files missing:  {missing_models}/{len(model_rows)}")
    print()
    print("Verification complete. No downloads performed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
