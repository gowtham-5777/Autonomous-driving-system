"""Run inference demo on images or video using a trained traffic-signal model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.traffic_signal.bdd100k_converter import YOLO_CLASS_NAMES
from training.traffic_signal.config import TrafficSignalTrainingConfig

logger = logging.getLogger(__name__)

CLASS_COLORS = {
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
}


def _draw_predictions(frame, result) -> None:
    names = result.names
    for box in result.boxes:
        class_id = int(box.cls.item())
        label = names.get(class_id, YOLO_CLASS_NAMES[class_id])
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {confidence:.2f}"
        cv2.putText(
            frame,
            caption,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def predict_source(
    source: Path,
    weights: Path,
    output_dir: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "",
    show: bool = False,
) -> list[Path]:
    """Run YOLO prediction on an image, directory, or video path."""
    from ultralytics import YOLO

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    model = YOLO(str(weights))
    output_dir.mkdir(parents=True, exist_ok=True)

    predict_kwargs = {
        "source": str(source),
        "imgsz": imgsz,
        "conf": conf,
        "save": False,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    saved_paths: list[Path] = []

    for index, result in enumerate(results):
        frame = result.orig_img.copy()
        _draw_predictions(frame, result)

        if source.is_file():
            suffix = source.suffix.lower()
            if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
                out_name = f"{source.stem}_pred_{index:04d}.jpg"
            else:
                out_name = f"{source.stem}_pred{source.suffix}"
        else:
            out_name = f"pred_{index:04d}.jpg"

        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), frame)
        saved_paths.append(out_path)

        if show:
            cv2.imshow("traffic_signal_prediction", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()

    return saved_paths


def parse_args() -> argparse.Namespace:
    defaults = TrafficSignalTrainingConfig()
    parser = argparse.ArgumentParser(
        description="Traffic-signal detection inference demo.",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Image file, directory of images, or video path.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=defaults.model_output,
        help="Path to traffic_signals_yolov8n.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "outputs" / "traffic_signal_predictions",
        help="Directory for annotated outputs.",
    )
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--show", action="store_true", help="Display predictions live.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        saved = predict_source(
            source=args.source,
            weights=args.weights,
            output_dir=args.output,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            show=args.show,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    print(f"\nSaved {len(saved)} prediction(s) to: {args.output}")
    for path in saved[:10]:
        print(f"  - {path}")
    if len(saved) > 10:
        print(f"  ... and {len(saved) - 10} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
