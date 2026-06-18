"""Centralized model weight path management for the ADAS project.

Reads directory and filename settings from ``config/default.yaml``,
expands path variables, creates missing directories, and returns
``pathlib.Path`` objects for each module's weights.

No model downloading or inference is performed by this module.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "default.yaml"
_VARIABLE_PATTERN = re.compile(r"\$\{([^}]+)\}")

# Maps weight_locations config values to directory resolver functions.
_LOCATION_RESOLVERS: dict[str, str] = {
    "pretrained": "get_pretrained_models_dir",
    "trained": "get_trained_models_dir",
}


@lru_cache(maxsize=1)
def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and return the parsed YAML configuration.

    Args:
        config_path: Optional override for the config file location.
            Defaults to ``config/default.yaml`` at the project root.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config: dict[str, Any] = yaml.safe_load(config_file) or {}

    return config


def _get_data_root(config: dict[str, Any]) -> str:
    """Resolve the data root directory from environment or config.

    Environment variable ``ADAS_DATA_ROOT`` takes precedence over
    ``data_root`` in the configuration file.

    Args:
        config: Parsed configuration dictionary.

    Returns:
        Absolute or project-specific data root path string.
    """
    return os.environ.get("ADAS_DATA_ROOT", config.get("data_root", ""))


def _expand_variables(value: str, variables: dict[str, str]) -> str:
    """Recursively expand ``${variable}`` placeholders in a string.

    Args:
        value: String potentially containing ``${key}`` placeholders.
        variables: Mapping of placeholder names to replacement values.

    Returns:
        String with all resolvable placeholders substituted.
    """
    expanded = value
    for _ in range(len(variables) + 1):
        match = _VARIABLE_PATTERN.search(expanded)
        if match is None:
            break
        key = match.group(1)
        replacement = variables.get(key)
        if replacement is None:
            break
        expanded = expanded.replace(f"${{{key}}}", replacement, 1)
    return expanded


def _build_variables(config: dict[str, Any]) -> dict[str, str]:
    """Build a flat variable map for path expansion.

    Args:
        config: Parsed configuration dictionary.

    Returns:
        Dictionary containing at least ``data_root`` and any expanded
        entries from ``paths``.
    """
    variables: dict[str, str] = {"data_root": _get_data_root(config)}
    paths = config.get("paths", {})

    for key, raw_value in paths.items():
        if isinstance(raw_value, str):
            variables[key] = _expand_variables(raw_value, variables)

    return variables


def _resolve_config_path(config: dict[str, Any], path_key: str) -> Path:
    """Resolve a path entry from the configuration ``paths`` section.

    Args:
        config: Parsed configuration dictionary.
        path_key: Key under ``paths`` (e.g. ``models_pretrained``).

    Returns:
        Resolved ``Path`` object.

    Raises:
        KeyError: If the path key is missing from configuration.
    """
    paths = config.get("paths", {})
    if path_key not in paths:
        raise KeyError(f"Missing path configuration: paths.{path_key}")

    variables = _build_variables(config)
    raw_path = paths[path_key]
    if not isinstance(raw_path, str):
        raise TypeError(f"paths.{path_key} must be a string, got {type(raw_path)}")

    return Path(_expand_variables(raw_path, variables))


def _ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path, guaranteed to exist as a directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_pretrained_models_dir() -> Path:
    """Return the pretrained model weights directory.

    The directory is created automatically if it does not exist.
    Path is read from ``paths.models_pretrained`` in ``config/default.yaml``.

    Returns:
        Path to the pretrained models directory.
    """
    config = _load_config()
    directory = _ensure_directory(_resolve_config_path(config, "models_pretrained"))
    logger.info("Pretrained models directory: %s", directory)
    return directory


def get_trained_models_dir() -> Path:
    """Return the fine-tuned / trained model weights directory.

    The directory is created automatically if it does not exist.
    Path is read from ``paths.models_trained`` in ``config/default.yaml``.

    Returns:
        Path to the trained models directory.
    """
    config = _load_config()
    directory = _ensure_directory(_resolve_config_path(config, "models_trained"))
    logger.info("Trained models directory: %s", directory)
    return directory


def _get_weight_path(weight_key: str, model_label: str) -> Path:
    """Resolve the full path for a specific model weight file.

    Args:
        weight_key: Key under ``weight_files`` and ``weight_locations``
            in the configuration (e.g. ``yolop``).
        model_label: Human-readable model name for logging.

    Returns:
        Full path to the weight file. Parent directory is created if
        missing; the weight file itself is not created.

    Raises:
        KeyError: If required configuration keys are missing.
    """
    config = _load_config()
    weight_files = config.get("weight_files", {})
    weight_locations = config.get("weight_locations", {})

    if weight_key not in weight_files:
        raise KeyError(f"Missing weight file configuration: weight_files.{weight_key}")
    if weight_key not in weight_locations:
        raise KeyError(
            f"Missing weight location configuration: weight_locations.{weight_key}"
        )

    filename = weight_files[weight_key]
    location = weight_locations[weight_key]

    if location == "pretrained":
        base_dir = get_pretrained_models_dir()
    elif location == "trained":
        base_dir = get_trained_models_dir()
    else:
        raise ValueError(
            f"Invalid weight location '{location}' for '{weight_key}'. "
            "Expected 'pretrained' or 'trained'."
        )

    weight_path = base_dir / filename
    _ensure_directory(weight_path.parent)
    logger.info("%s weights will be loaded from: %s", model_label, weight_path)
    return weight_path


def get_yolop_weights_path() -> Path:
    """Return the path to YOLOP lane detection weights.

    Returns:
        Path to the YOLOP checkpoint file.
    """
    return _get_weight_path("yolop", "YOLOP")


def get_yolov8_weights_path() -> Path:
    """Return the path to YOLOv8 vehicle detection weights.

    Returns:
        Path to the YOLOv8 checkpoint file (``yolov8/yolov8s.pt`` by default).
    """
    return _get_weight_path("yolov8", "YOLOv8")


def get_yolov8_config() -> dict[str, Any]:
    """Return YOLOv8 settings merged with global object-detection thresholds.

    Returns:
        Dictionary with ``model_variant``, ``imgsz``, ``device``,
        ``max_detections``, ``confidence_threshold``, and ``iou_threshold``.
    """
    config = _load_config()
    yolov8 = dict(config.get("yolov8", {}))
    thresholds = config.get("thresholds", {})

    yolov8.setdefault("model_variant", "s")
    yolov8.setdefault("imgsz", 640)
    yolov8.setdefault("device", "cpu")
    yolov8.setdefault("max_detections", 100)
    yolov8["confidence_threshold"] = float(
        thresholds.get("object_confidence", 0.5)
    )
    yolov8["iou_threshold"] = float(thresholds.get("object_iou", 0.45))
    return yolov8


def get_ssd_weights_path() -> Path:
    """Return the path to SSD MobileNetV2 object detection weights.

    Returns:
        Path to the SSD MobileNetV2 checkpoint file.
    """
    return _get_weight_path("ssd_mobilenetv2", "SSD MobileNetV2")


def get_traffic_sign_weights_path() -> Path:
    """Return the path to YOLOv8 fine-tuned traffic sign weights.

    Returns:
        Path to the traffic sign checkpoint file.
    """
    return _get_weight_path("yolov8_sign", "YOLOv8 Sign")


def get_yolov8_sign_config() -> dict[str, Any]:
    """Return YOLOv8 sign settings merged with global sign thresholds.

    Returns:
        Dictionary with ``model_variant``, ``imgsz``, ``device``,
        ``max_detections``, ``num_classes``, ``confidence_threshold``,
        and ``iou_threshold``.
    """
    config = _load_config()
    yolov8_sign = dict(config.get("yolov8_sign", {}))
    thresholds = config.get("thresholds", {})

    yolov8_sign.setdefault("model_variant", "n")
    yolov8_sign.setdefault("imgsz", 640)
    yolov8_sign.setdefault("device", "cpu")
    yolov8_sign.setdefault("max_detections", 50)
    yolov8_sign.setdefault("num_classes", 7)
    yolov8_sign["confidence_threshold"] = float(
        thresholds.get("sign_confidence", 0.5)
    )
    yolov8_sign["iou_threshold"] = float(thresholds.get("sign_iou", 0.45))
    return yolov8_sign


def get_yolov5_weights_path() -> Path:
    """Return the path to YOLOv5 traffic sign recognition weights (legacy).

    Returns:
        Path to the YOLOv5 checkpoint file.
    """
    return _get_weight_path("yolov5", "YOLOv5")


def get_traffic_signal_weights_path() -> Path:
    """Return the path to YOLOv8 fine-tuned traffic signal weights.

    Returns:
        Path to the traffic signal checkpoint file.
    """
    return _get_weight_path("yolov8_signal", "YOLOv8 Signal")


def get_yolov8_signal_config() -> dict[str, Any]:
    """Return YOLOv8 signal settings merged with global signal thresholds.

    Returns:
        Dictionary with ``model_variant``, ``imgsz``, ``device``,
        ``max_detections``, ``num_classes``, ``confidence_threshold``,
        and ``iou_threshold``.
    """
    config = _load_config()
    yolov8_signal = dict(config.get("yolov8_signal", {}))
    thresholds = config.get("thresholds", {})

    yolov8_signal.setdefault("model_variant", "n")
    yolov8_signal.setdefault("imgsz", 640)
    yolov8_signal.setdefault("device", "cpu")
    yolov8_signal.setdefault("max_detections", 20)
    yolov8_signal.setdefault("num_classes", 3)
    yolov8_signal["confidence_threshold"] = float(
        thresholds.get(
            "signal_confidence",
            thresholds.get("traffic_light_confidence", 0.5),
        )
    )
    yolov8_signal["iou_threshold"] = float(thresholds.get("signal_iou", 0.45))
    return yolov8_signal


def get_unet_weights_path() -> Path:
    """Return the path to U-Net semantic segmentation weights.

    Returns:
        Path to the U-Net checkpoint file.
    """
    return _get_weight_path("unet", "U-Net")


def get_decision_config() -> dict[str, Any]:
    """Return decision engine threshold settings from configuration."""
    config = _load_config()
    decision = dict(config.get("decision", {}))
    decision.setdefault("red_light_confidence", 0.70)
    decision.setdefault("stop_sign_confidence", 0.70)
    decision.setdefault("stop_sign_lower_frame_fraction", 0.40)
    decision.setdefault("vulnerable_user_confidence", 0.60)
    decision.setdefault("large_vehicle_area_ratio", 0.08)
    decision.setdefault("lane_offset_warn_px", 35.0)
    decision.setdefault("drivable_overlap_threshold", 0.15)
    return decision


def get_pipeline_config() -> dict[str, Any]:
    """Return pipeline orchestrator settings from configuration."""
    config = _load_config()
    pipeline = dict(config.get("pipeline", {}))
    pipeline.setdefault("run_lane", True)
    pipeline.setdefault("run_vehicles", True)
    pipeline.setdefault("run_signs", True)
    pipeline.setdefault("run_signals", True)
    pipeline.setdefault("run_segmentation", False)
    pipeline.setdefault("auto_initialize", True)
    pipeline.setdefault("collect_timing", True)
    return pipeline


def get_traffic_light_cnn_path() -> Path:
    """Return the path to the traffic light CNN classifier weights (legacy).

    Deprecated: use :func:`get_traffic_signal_weights_path` for YOLOv8 signal weights.

    Returns:
        Path to the traffic light CNN checkpoint file.
    """
    return _get_weight_path("traffic_light_cnn", "Traffic Light CNN")
