"""Evaluate a trained traffic-signal YOLOv8 model on the validation split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.traffic_signal.bdd100k_converter import YOLO_CLASS_NAMES, YOLO_TO_ADAS_LABEL
from training.traffic_signal.config import TrafficSignalTrainingConfig

logger = logging.getLogger(__name__)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_model(
    weights: Path,
    dataset_yaml: Path,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    conf: float = 0.25,
    iou: float = 0.45,
) -> dict:
    """Run validation and return precision, recall, F1, mAP50, mAP50-95."""
    from ultralytics import YOLO

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")

    model = YOLO(str(weights))
    val_kwargs = {
        "data": str(dataset_yaml),
        "imgsz": imgsz,
        "batch": batch,
        "conf": conf,
        "iou": iou,
        "split": "val",
        "plots": True,
        "verbose": True,
    }
    if device:
        val_kwargs["device"] = device

    metrics = model.val(**val_kwargs)
    box = metrics.box

    precision = _safe_float(getattr(box, "mp", 0.0))
    recall = _safe_float(getattr(box, "mr", 0.0))
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    results = {
        "weights": str(weights.resolve()),
        "dataset": str(dataset_yaml.resolve()),
        "class_mapping": {str(i): name for i, name in enumerate(YOLO_CLASS_NAMES)},
        "adas_class_mapping": {str(k): v for k, v in YOLO_TO_ADAS_LABEL.items()},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map50": round(_safe_float(getattr(box, "map50", 0.0)), 4),
        "map50_95": round(_safe_float(getattr(box, "map", 0.0)), 4),
        "per_class_ap50": {},
        "per_class_ap50_adas": {},
    }

    names = getattr(metrics, "names", {}) or {}
    ap50_array = getattr(box, "ap50", None)
    if ap50_array is not None:
        for class_id, class_name in names.items():
            if int(class_id) < len(ap50_array):
                ap50_value = round(_safe_float(ap50_array[int(class_id)]), 4)
                results["per_class_ap50"][str(class_name)] = ap50_value
                adas_label = YOLO_TO_ADAS_LABEL.get(int(class_id))
                if adas_label:
                    results["per_class_ap50_adas"][adas_label] = ap50_value

    return results


def parse_args() -> argparse.Namespace:
    defaults = TrafficSignalTrainingConfig()
    parser = argparse.ArgumentParser(
        description="Evaluate traffic-signal YOLOv8 weights on the val split.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=defaults.model_output,
        help="Path to traffic_signals_yolov8n.pt.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=defaults.dataset_yaml,
        help="Path to dataset.yaml.",
    )
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument("--batch", type=int, default=defaults.batch)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "runs" / "traffic_signal" / "evaluation_metrics.json",
        help="JSON file for metric output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        results = evaluate_model(
            weights=args.weights,
            dataset_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Traffic Signal Evaluation ===")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1        : {results['f1']:.4f}")
    print(f"mAP@0.5   : {results['map50']:.4f}")
    print(f"mAP@0.5:95: {results['map50_95']:.4f}")
    if results["per_class_ap50"]:
        print(f"Per-class AP@0.5 (YOLO): {results['per_class_ap50']}")
    if results["per_class_ap50_adas"]:
        print(f"Per-class AP@0.5 (ADAS): {results['per_class_ap50_adas']}")
    print(f"\nMetrics saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
