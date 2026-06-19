#!/usr/bin/env python3
"""Download publicly available pretrained model weights for ADAS modules."""

from __future__ import annotations

import logging
import shutil
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure local project paths are used when ADAS_DATA_ROOT is unset in this shell.
import os

os.environ.setdefault("ADAS_DATA_ROOT", str(PROJECT_ROOT))

from src.utils.model_paths import (
    get_traffic_signal_weights_path,
    get_traffic_sign_weights_path,
    get_yolop_weights_path,
    get_yolov8_weights_path,
)

logger = logging.getLogger(__name__)

DOWNLOADS = {
    "yolop": {
        "url": "https://github.com/hustvl/YOLOP/raw/main/weights/End-to-end.pth",
        "path_fn": get_yolop_weights_path,
        "min_bytes": 50_000_000,
    },
    "yolov8": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        "path_fn": get_yolov8_weights_path,
        "min_bytes": 20_000_000,
    },
}

TRAINED_WEIGHTS = {
    "traffic_sign": get_traffic_sign_weights_path,
    "traffic_signal": get_traffic_signal_weights_path,
}


def _download(url: str, destination: Path) -> None:
    """Download a file to destination with a simple progress log."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    logger.info("Downloading %s", url)
    logger.info("Destination: %s", destination)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(100.0, downloaded * 100.0 / total_size)
        if block_num % 200 == 0 or downloaded >= total_size:
            logger.info("  %.1f%% (%0.1f MB)", percent, downloaded / 1_000_000)

    urllib.request.urlretrieve(url, temp_path, reporthook=_report)
    temp_path.replace(destination)


def _relocate_misplaced_yolov8(project_root: Path) -> bool:
    """Move yolov8s.pt from yolop folder to yolov8 folder if needed."""
    wrong_path = project_root / "models" / "pretrained" / "yolop" / "yolov8s.pt"
    correct_path = get_yolov8_weights_path()
    if not wrong_path.is_file():
        return False
    if correct_path.is_file():
        wrong_path.unlink()
        logger.info("Removed misplaced duplicate: %s", wrong_path)
        return True

    correct_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(wrong_path), str(correct_path))
    logger.info("Moved misplaced weight: %s -> %s", wrong_path.name, correct_path)
    return True


def ensure_downloadable_weights(*, force: bool = False) -> dict[str, str]:
    """Download missing pretrained weights. Returns status per module."""
    status: dict[str, str] = {}

    _relocate_misplaced_yolov8(PROJECT_ROOT)

    for name, spec in DOWNLOADS.items():
        destination = spec["path_fn"]()
        if destination.is_file() and not force:
            size = destination.stat().st_size
            if size >= spec["min_bytes"]:
                status[name] = f"present ({size / 1_000_000:.1f} MB)"
                continue
            logger.warning("Existing file too small, re-downloading: %s", destination)

        try:
            _download(spec["url"], destination)
            status[name] = f"downloaded ({destination.stat().st_size / 1_000_000:.1f} MB)"
        except Exception as exc:
            status[name] = f"failed: {exc}"

    for name, path_fn in TRAINED_WEIGHTS.items():
        destination = path_fn()
        if destination.is_file():
            status[name] = f"present ({destination.stat().st_size / 1_000_000:.1f} MB)"
        else:
            status[name] = (
                "missing — fine-tuned weights are not publicly hosted; "
                "train locally or copy from Colab/Drive"
            )

    return status


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download ADAS pretrained weights")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    status = ensure_downloadable_weights(force=args.force)

    print()
    print("Weight status:")
    for name, message in status.items():
        print(f"  {name:16s} {message}")

    missing_trained = [
        name
        for name, message in status.items()
        if name in TRAINED_WEIGHTS and message.startswith("missing")
    ]
    if missing_trained:
        print()
        print("Manual steps for fine-tuned weights:")
        print("  traffic_signal: python scripts/train_traffic_signal.py --device cpu --epochs 100")
        print("    or copy traffic_signals_yolov8n.pt from Colab to:")
        print(f"    {get_traffic_signal_weights_path()}")
        print("  traffic_sign: train/obtain traffic_signs_yolov8n.pt and place at:")
        print(f"    {get_traffic_sign_weights_path()}")


if __name__ == "__main__":
    main()
