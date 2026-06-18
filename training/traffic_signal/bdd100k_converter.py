"""Convert BDD100K traffic-light annotations to YOLO detection format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Training class order — aligned with ADAS runtime (src/modules/yolov8_signal/class_map.py).
BDD100K_TO_YOLO_CLASS: dict[str, int] = {
    "red": 0,
    "yellow": 1,
    "green": 2,
}

YOLO_CLASS_NAMES: list[str] = ["red", "yellow", "green"]

# YOLO class index → ADAS inference label (matches SIGNAL_CLASS_ID_TO_LABEL).
YOLO_TO_ADAS_LABEL: dict[int, str] = {
    0: "red_light",
    1: "yellow_light",
    2: "green_light",
}

VALID_TRAFFIC_LIGHT_COLORS = frozenset(BDD100K_TO_YOLO_CLASS.keys())
TRAFFIC_LIGHT_CATEGORY = "traffic light"

# BDD100K driving frames are 1280×720; used when image file is unavailable.
BDD100K_DEFAULT_WIDTH = 1280
BDD100K_DEFAULT_HEIGHT = 720


def is_valid_traffic_light_box(
    box2d: dict[str, Any] | None,
    img_width: int,
    img_height: int,
) -> bool:
    """Return True when box2d has positive area and lies within image bounds."""
    if not box2d:
        return False

    try:
        x1 = float(box2d["x1"])
        y1 = float(box2d["y1"])
        x2 = float(box2d["x2"])
        y2 = float(box2d["y2"])
    except (KeyError, TypeError, ValueError):
        return False

    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
        return False

    width = x2 - x1
    height = y2 - y1
    if width < 1.0 or height < 1.0:
        return False

    return True


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
) -> str:
    """Convert pixel bbox to normalized YOLO ``class cx cy w h`` line."""
    x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, img_width, img_height)
    cx = ((x1 + x2) / 2.0) / img_width
    cy = ((y1 + y2) / 2.0) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def extract_traffic_light_objects(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return traffic-light objects from a BDD100K frame annotation."""
    frames = annotation.get("frames") or []
    if not frames:
        return []

    objects = frames[0].get("objects") or []
    return [obj for obj in objects if obj.get("category") == TRAFFIC_LIGHT_CATEGORY]


def convert_bdd100k_annotation(
    annotation: dict[str, Any],
    img_width: int = BDD100K_DEFAULT_WIDTH,
    img_height: int = BDD100K_DEFAULT_HEIGHT,
) -> tuple[list[str], dict[str, int]]:
    """Extract valid YOLO label lines and per-split skip counters.

    Returns:
        (yolo_lines, stats_delta) where stats_delta tracks skipped boxes.
    """
    stats: dict[str, int] = {
        "total_traffic_lights": 0,
        "skipped_none_color": 0,
        "skipped_invalid_color": 0,
        "skipped_invalid_box": 0,
        "kept": 0,
    }
    yolo_lines: list[str] = []

    for obj in extract_traffic_light_objects(annotation):
        stats["total_traffic_lights"] += 1
        attrs = obj.get("attributes") or {}
        color = str(attrs.get("trafficLightColor", "none")).strip().lower()

        if color == "none":
            stats["skipped_none_color"] += 1
            continue
        if color not in VALID_TRAFFIC_LIGHT_COLORS:
            stats["skipped_invalid_color"] += 1
            continue

        box2d = obj.get("box2d")
        if not is_valid_traffic_light_box(box2d, img_width, img_height):
            stats["skipped_invalid_box"] += 1
            continue

        class_id = BDD100K_TO_YOLO_CLASS[color]
        line = bbox_to_yolo_line(
            class_id,
            float(box2d["x1"]),
            float(box2d["y1"]),
            float(box2d["x2"]),
            float(box2d["y2"]),
            img_width,
            img_height,
        )
        yolo_lines.append(line)
        stats["kept"] += 1

    return yolo_lines, stats


def load_bdd100k_annotation(label_path: Path) -> dict[str, Any]:
    """Load a BDD100K JSON annotation file."""
    with label_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_dataset_yaml(output_path: Path, dataset_root: Path) -> None:
    """Write Ultralytics-compatible ``dataset.yaml``."""
    name_lines = [f"  {class_id}: {name}" for class_id, name in enumerate(YOLO_CLASS_NAMES)]
    content = "\n".join(
        [
            f"path: {dataset_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "nc: 3",
            "names:",
            *name_lines,
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
