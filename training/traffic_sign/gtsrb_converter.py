"""Convert GTSRB sign crops to YOLO detection format (7-class ADAS subset)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

# Training class order — aligned with ADAS runtime (src/modules/yolov8_sign/class_map.py).
YOLO_CLASS_NAMES: list[str] = [
    "stop",
    "speed_limit_30",
    "speed_limit_60",
    "turn_left",
    "turn_right",
    "keep_right",
    "pedestrian_crossing",
]

YOLO_TO_ADAS_LABEL: dict[int, str] = {
    class_id: name for class_id, name in enumerate(YOLO_CLASS_NAMES)
}

# GTSRB benchmark class ID → YOLO class index (via ADAS label order above).
GTSRB_TO_YOLO_CLASS: dict[int, int] = {
    14: 0,  # stop
    1: 1,   # speed limit 30
    5: 2,   # speed limit 60
    38: 3,  # turn left ahead
    34: 4,  # turn right ahead
    36: 5,  # pass on right / keep right
    12: 6,  # pedestrians
}

MAPPED_GTSRB_CLASS_IDS = frozenset(GTSRB_TO_YOLO_CLASS.keys())

# Cropped GTSRB train patches: sign fills the frame — full-image box.
TRAIN_FULL_FRAME_BBOX = (0.5, 0.5, 1.0, 1.0)

GTSRB_IMAGE_EXTENSIONS = (".ppm", ".png", ".jpg", ".jpeg")


def gtsrb_class_id_to_yolo_class(gtsrb_class_id: int) -> int | None:
    """Map a GTSRB class ID to a YOLO class index, or ``None`` if unmapped."""
    return GTSRB_TO_YOLO_CLASS.get(gtsrb_class_id)


def yolo_class_to_name(class_id: int) -> str:
    """Return the YOLO class name for a class index."""
    return YOLO_CLASS_NAMES[class_id]


def _clip_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    return (
        max(0.0, min(x1, float(img_width))),
        max(0.0, min(y1, float(img_height))),
        max(0.0, min(x2, float(img_width))),
        max(0.0, min(y2, float(img_height))),
    )


def bbox_to_yolo_line(
    class_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int,
) -> str | None:
    """Convert pixel bbox to normalized YOLO ``class cx cy w h`` line."""
    if img_width <= 0 or img_height <= 0:
        return None

    x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, img_width, img_height)
    if x2 <= x1 or y2 <= y1:
        return None

    cx = ((x1 + x2) / 2.0) / img_width
    cy = ((y1 + y2) / 2.0) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def train_crop_to_yolo_line(class_id: int) -> str:
    """Return a full-frame YOLO label for a GTSRB training crop."""
    cx, cy, w, h = TRAIN_FULL_FRAME_BBOX
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def gtsrb_test_row_to_yolo_line(row: dict[str, Any]) -> tuple[str, int, int] | None:
    """Convert a GTSRB ``Test.csv`` row to ``(yolo_line, width, height)``."""
    try:
        gtsrb_id = int(row["ClassId"])
        width = int(row["Width"])
        height = int(row["Height"])
        x1 = float(row["Roi.X1"])
        y1 = float(row["Roi.Y1"])
        x2 = float(row["Roi.X2"])
        y2 = float(row["Roi.Y2"])
    except (KeyError, TypeError, ValueError):
        return None

    class_id = gtsrb_class_id_to_yolo_class(gtsrb_id)
    if class_id is None:
        return None

    line = bbox_to_yolo_line(class_id, x1, y1, x2, y2, width, height)
    if line is None:
        return None
    return line, width, height


def load_gtsrb_test_csv(csv_path: Path) -> list[dict[str, str]]:
    """Load GTSRB official test annotations."""
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_dataset_yaml(output_path: Path, dataset_root: Path) -> None:
    """Write Ultralytics-compatible ``dataset.yaml``."""
    name_lines = [f"  {class_id}: {name}" for class_id, name in enumerate(YOLO_CLASS_NAMES)]
    content = "\n".join(
        [
            f"path: {dataset_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "nc: 7",
            "names:",
            *name_lines,
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
