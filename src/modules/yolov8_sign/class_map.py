"""Traffic sign class ID mappings and GTSRB translation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CLASSES_CONFIG_PATH = _PROJECT_ROOT / "config" / "classes.yaml"

# Model class index → ADAS sign label (7-class v1 head).
SIGN_CLASS_ID_TO_LABEL: dict[int, str] = {
    0: "stop",
    1: "speed_limit_30",
    2: "speed_limit_60",
    3: "turn_left",
    4: "turn_right",
    5: "keep_right",
    6: "pedestrian_crossing",
}

SIGN_LABEL_TO_CLASS_ID: dict[str, int] = {
    label: class_id for class_id, label in SIGN_CLASS_ID_TO_LABEL.items()
}

ALLOWED_SIGN_CLASS_IDS = frozenset(SIGN_CLASS_ID_TO_LABEL.keys())

# GTSRB benchmark class ID → ADAS label (v1 subset).
GTSRB_CLASS_ID_TO_ADAS_LABEL: dict[int, str] = {
    14: "stop",
    1: "speed_limit_30",
    5: "speed_limit_60",
    38: "turn_left",
    34: "turn_right",
    36: "keep_right",
    12: "pedestrian_crossing",
}

REGULATORY_LABELS = frozenset({"stop", "speed_limit_30", "speed_limit_60", "keep_right"})
WARNING_LABELS = frozenset({"pedestrian_crossing"})


def load_traffic_sign_classes(config_path: Path | None = None) -> list[str]:
    """Load ``traffic_sign_classes`` from ``config/classes.yaml``."""
    path = config_path or _CLASSES_CONFIG_PATH
    if not path.is_file():
        return list(SIGN_CLASS_ID_TO_LABEL.values())

    with path.open(encoding="utf-8") as config_file:
        data: dict[str, Any] = yaml.safe_load(config_file) or {}

    classes = data.get("traffic_sign_classes", [])
    if not isinstance(classes, list):
        return list(SIGN_CLASS_ID_TO_LABEL.values())
    return [str(label) for label in classes]


def gtsrb_id_to_adas_label(gtsrb_class_id: int) -> str | None:
    """Map a GTSRB class ID to an ADAS sign label, or ``None`` if unmapped."""
    return GTSRB_CLASS_ID_TO_ADAS_LABEL.get(gtsrb_class_id)


def adas_label_to_class_id(label: str) -> int | None:
    """Map an ADAS sign label to the model class index."""
    return SIGN_LABEL_TO_CLASS_ID.get(label)


def is_regulatory_label(label: str) -> bool:
    """Return whether a sign label enforces a traffic rule."""
    return label in REGULATORY_LABELS


def is_warning_label(label: str) -> bool:
    """Return whether a sign label is a warning category."""
    return label in WARNING_LABELS


def extract_speed_limit_kmh(label: str) -> int | None:
    """Parse numeric speed limit from labels like ``speed_limit_30``."""
    if not label.startswith("speed_limit_"):
        return None
    suffix = label.removeprefix("speed_limit_")
    if suffix.isdigit():
        return int(suffix)
    return None
