"""Export the prepared traffic-signal YOLO dataset as a Colab-uploadable zip."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.traffic_signal.bdd100k_converter import YOLO_CLASS_NAMES, write_dataset_yaml
from training.traffic_signal.config import TrafficSignalTrainingConfig

logger = logging.getLogger(__name__)

# Default extract path used in docs/traffic_signal_colab_training.md
DEFAULT_COLAB_DATASET_PATH = "/content/traffic_signal_yolo"


def _validate_dataset(dataset_dir: Path) -> dict[str, int]:
    """Ensure images/labels splits exist and return file counts."""
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")

        images = sorted(image_dir.glob("*.jpg"))
        labels = sorted(label_dir.glob("*.txt"))
        counts[f"images_{split}"] = len(images)
        counts[f"labels_{split}"] = len(labels)

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        if image_stems != label_stems:
            missing_labels = len(image_stems - label_stems)
            missing_images = len(label_stems - image_stems)
            raise ValueError(
                f"Split '{split}' image/label mismatch: "
                f"{missing_labels} images without labels, "
                f"{missing_images} labels without images."
            )

    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"Missing dataset.yaml at {yaml_path}. "
            "Run scripts/prepare_traffic_signal_dataset.py first."
        )
    return counts


def _collect_files(dataset_dir: Path, include_stats: bool) -> list[tuple[Path, str]]:
    """Return (source_path, archive_name) pairs for the zip."""
    entries: list[tuple[Path, str]] = []
    prefix = "traffic_signal_yolo"

    for split in ("train", "val"):
        for folder, ext in (("images", ".jpg"), ("labels", ".txt")):
            source_dir = dataset_dir / folder / split
            for path in sorted(source_dir.glob(f"*{ext}")):
                archive_name = f"{prefix}/{folder}/{split}/{path.name}"
                entries.append((path, archive_name))

    stats_path = dataset_dir / "dataset_stats.json"
    if include_stats and stats_path.is_file():
        entries.append((stats_path, f"{prefix}/dataset_stats.json"))

    return entries


def export_dataset_zip(
    dataset_dir: Path,
    output_zip: Path,
    colab_dataset_path: str = DEFAULT_COLAB_DATASET_PATH,
    include_stats: bool = True,
    compresslevel: int = 1,
) -> dict:
    """Create a zip archive with images, labels, and Colab-ready dataset.yaml."""
    counts = _validate_dataset(dataset_dir)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    staging_yaml = output_zip.parent / "_colab_dataset.yaml"
    write_dataset_yaml(staging_yaml, Path(colab_dataset_path))

    file_entries = _collect_files(dataset_dir, include_stats=include_stats)
    file_entries.insert(0, (staging_yaml, "traffic_signal_yolo/dataset.yaml"))

    manifest = {
        "dataset_dir": str(dataset_dir.resolve()),
        "output_zip": str(output_zip.resolve()),
        "colab_dataset_path": colab_dataset_path,
        "class_names": YOLO_CLASS_NAMES,
        "file_counts": counts,
        "total_files_in_zip": len(file_entries),
    }

    logger.info("Writing zip to %s (%d files)", output_zip, len(file_entries))
    with zipfile.ZipFile(
        output_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compresslevel,
    ) as archive:
        for source_path, archive_name in tqdm(file_entries, desc="export-zip", unit="file"):
            archive.write(source_path, arcname=archive_name)

    staging_yaml.unlink(missing_ok=True)

    manifest["zip_size_mb"] = round(output_zip.stat().st_size / (1024 * 1024), 2)
    manifest_path = output_zip.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    defaults = TrafficSignalTrainingConfig()
    parser = argparse.ArgumentParser(
        description="Export traffic_signal_yolo dataset as a zip for Google Colab.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=defaults.dataset_dir,
        help="Path to prepared traffic_signal_yolo directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "data" / "traffic_signal_yolo_colab.zip",
        help="Output zip file path.",
    )
    parser.add_argument(
        "--colab-path",
        type=str,
        default=DEFAULT_COLAB_DATASET_PATH,
        help="Absolute path written into dataset.yaml for Colab extraction.",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Omit dataset_stats.json from the archive.",
    )
    parser.add_argument(
        "--compresslevel",
        type=int,
        default=1,
        choices=range(0, 10),
        help="Zip compression level (1=fast, 9=smallest).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.dataset.is_dir():
        logger.error("Dataset directory not found: %s", args.dataset)
        return 1

    try:
        manifest = export_dataset_zip(
            dataset_dir=args.dataset,
            output_zip=args.output,
            colab_dataset_path=args.colab_path,
            include_stats=not args.no_stats,
            compresslevel=args.compresslevel,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    print("\n=== Traffic Signal Dataset Export ===")
    print(f"Zip file     : {manifest['output_zip']}")
    print(f"Size         : {manifest['zip_size_mb']} MB")
    print(f"Colab path   : {manifest['colab_dataset_path']}")
    print(f"Train images : {manifest['file_counts']['images_train']}")
    print(f"Val images   : {manifest['file_counts']['images_val']}")
    print(f"Class names  : {manifest['class_names']}")
    print(f"Manifest     : {manifest['manifest_path']}")
    print("\nUpload the zip to Google Drive, then follow docs/traffic_signal_colab_training.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
