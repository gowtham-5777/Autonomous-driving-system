"""Train YOLOv8n on the BDD100K-derived traffic-signal dataset."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.traffic_signal.config import TrafficSignalTrainingConfig

logger = logging.getLogger(__name__)


def train_model(config: TrafficSignalTrainingConfig, resume: bool = False) -> Path:
    """Run Ultralytics YOLOv8n training and copy best weights to project path."""
    from ultralytics import YOLO

    if not config.dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset config not found: {config.dataset_yaml}. "
            "Run scripts/prepare_traffic_signal_dataset.py first."
        )

    model_name = f"yolov8{config.model_variant}.pt"
    model = YOLO(model_name)

    train_kwargs = {
        "data": str(config.dataset_yaml),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "patience": config.patience,
        "workers": config.workers,
        "seed": config.seed,
        "project": str(_PROJECT_ROOT / "runs" / "traffic_signal"),
        "name": "yolov8n_bdd100k",
        "exist_ok": True,
        "hsv_h": config.hsv_h,
        "hsv_s": config.hsv_s,
        "hsv_v": config.hsv_v,
        "degrees": config.degrees,
        "translate": config.translate,
        "scale": config.scale,
        "fliplr": config.fliplr,
        "mosaic": config.mosaic,
        "save": True,
        "plots": True,
        "verbose": True,
    }
    if config.device:
        train_kwargs["device"] = config.device
    if resume:
        train_kwargs["resume"] = True

    results = model.train(**train_kwargs)
    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training finished but best weights missing: {best_weights}")

    config.model_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, config.model_output)
    logger.info("Saved trained model to %s", config.model_output)
    return config.model_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8n traffic-signal detector.")
    parser.add_argument(
        "--data",
        type=Path,
        default=TrafficSignalTrainingConfig().dataset_yaml,
        help="Path to dataset.yaml.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or 0/1/...")
    parser.add_argument(
        "--output",
        type=Path,
        default=TrafficSignalTrainingConfig().model_output,
        help="Destination for traffic_signals_yolov8n.pt.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume last training run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = TrafficSignalTrainingConfig(
        dataset_dir=args.data.parent,
        model_output=args.output,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
    )

    try:
        output_path = train_model(config, resume=args.resume)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    print(f"\nTraining complete. Model saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
