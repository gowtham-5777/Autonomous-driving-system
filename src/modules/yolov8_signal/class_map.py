"""Traffic signal class ID mappings and Bdd100K translation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CLASSES_CONFIG_PATH = _PROJECT_ROOT / "config" / "classes.yaml"

# Model class index → ADAS signal state label (3-class v1 head).
SIGNAL_CLASS_ID_TO_LABEL: dict[int, str] = {
    0: "red_light",
    1: "yellow_light",
    2: "green_light",
}

SIGNAL_LABEL_TO_CLASS_ID: dict[str, int] = {
    label: class_id for class_id, label in SIGNAL_CLASS_ID_TO_LABEL.items()
}

ALLOWED_SIGNAL_CLASS_IDS = frozenset(SIGNAL_CLASS_ID_TO_LABEL.keys())

# Bdd100K / legacy source label → ADAS label.
BDD100K_LABEL_TO_ADAS_LABEL: dict[str, str] = {
    "red": "red_light",
    "yellow": "yellow_light",
    "green": "green_light",
}

LEGACY_LABEL_ALIASES: dict[str, str] = dict(BDD100K_LABEL_TO_ADAS_LABEL)

# Priority for conflict resolution (higher = more conservative).
STATE_PRIORITY: dict[str, int] = {
    "red_light": 3,
    "yellow_light": 2,
    "green_light": 1,
}

CONTROLLING_SIGNAL_UPPER_FRACTION = 0.60


def load_traffic_signal_classes(config_path: Path | None = None) -> list[str]:
    """Load ``traffic_signal_classes`` from ``config/classes.yaml``."""
    path = config_path or _CLASSES_CONFIG_PATH
    if not path.is_file():
        return list(SIGNAL_CLASS_ID_TO_LABEL.values())

    with path.open(encoding="utf-8") as config_file:
        data: dict[str, Any] = yaml.safe_load(config_file) or {}

    classes = data.get("traffic_signal_classes")
    if classes is None:
        # Legacy fallback during config migration.
        legacy = data.get("traffic_light_classes", [])
        if isinstance(legacy, list):
            return [
                LEGACY_LABEL_ALIASES.get(str(label), str(label))
                for label in legacy
            ]
        return list(SIGNAL_CLASS_ID_TO_LABEL.values())

    if not isinstance(classes, list):
        return list(SIGNAL_CLASS_ID_TO_LABEL.values())
    return [str(label) for label in classes]


def bdd100k_label_to_adas_label(source_label: str) -> str | None:
    """Map a Bdd100K state label to an ADAS signal label."""
    normalized = source_label.strip().lower()
    if normalized in SIGNAL_LABEL_TO_CLASS_ID:
        return normalized
    return BDD100K_LABEL_TO_ADAS_LABEL.get(normalized)


def adas_label_to_class_id(label: str) -> int | None:
    """Map an ADAS signal label to the model class index."""
    return SIGNAL_LABEL_TO_CLASS_ID.get(label)


def enrich_state_flags(signal_label: str) -> tuple[bool, bool, bool]:
    """Return ``(is_stop_state, is_caution_state, is_proceed_state)``."""
    return (
        signal_label == "red_light",
        signal_label == "yellow_light",
        signal_label == "green_light",
    )


def state_priority(signal_label: str) -> int:
    """Return conservative priority for a signal label."""
    return STATE_PRIORITY.get(signal_label, 0)
