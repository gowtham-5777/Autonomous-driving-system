"""YOLOv8 signal integration package for traffic signal detection."""

from __future__ import annotations

from .class_map import (
    ALLOWED_SIGNAL_CLASS_IDS,
    BDD100K_LABEL_TO_ADAS_LABEL,
    CONTROLLING_SIGNAL_UPPER_FRACTION,
    LEGACY_LABEL_ALIASES,
    SIGNAL_CLASS_ID_TO_LABEL,
    SIGNAL_LABEL_TO_CLASS_ID,
    STATE_PRIORITY,
    adas_label_to_class_id,
    bdd100k_label_to_adas_label,
    enrich_state_flags,
    load_traffic_signal_classes,
    state_priority,
)
from .inference import (
    DEFAULT_IMGSZ,
    InferenceExecutionError,
    InferenceNotReadyError,
    InvalidFrameError,
    YOLOv8SignalInferenceConfig,
    YOLOv8SignalInferenceEngine,
    YOLOv8SignalRawOutput,
)
from .model_loader import (
    DEFAULT_MODEL_VARIANT,
    WeightsLoadError,
    WeightsMetadata,
    WeightsNotFoundError,
    WeightsValidationError,
    YOLOv8SignalModelLoader,
    resolve_variant_name,
)
from .output_parser import ParserConfig, YOLOv8SignalOutputParser
from .output_schema import (
    ADAS_SIGNAL_LABELS,
    TRAFFIC_SIGNAL_OUTPUT_KEYS,
    DetectedSignal,
    ParsedYOLOv8SignalOutput,
    SignalBoundingBoxData,
    TrafficSignalDetectionResult,
    TrafficSignalSummary,
)

__all__ = [
    "ADAS_SIGNAL_LABELS",
    "ALLOWED_SIGNAL_CLASS_IDS",
    "BDD100K_LABEL_TO_ADAS_LABEL",
    "CONTROLLING_SIGNAL_UPPER_FRACTION",
    "DEFAULT_IMGSZ",
    "DEFAULT_MODEL_VARIANT",
    "DetectedSignal",
    "InferenceExecutionError",
    "InferenceNotReadyError",
    "InvalidFrameError",
    "LEGACY_LABEL_ALIASES",
    "ParsedYOLOv8SignalOutput",
    "ParserConfig",
    "SIGNAL_CLASS_ID_TO_LABEL",
    "SIGNAL_LABEL_TO_CLASS_ID",
    "STATE_PRIORITY",
    "TRAFFIC_SIGNAL_OUTPUT_KEYS",
    "SignalBoundingBoxData",
    "TrafficSignalDetectionResult",
    "TrafficSignalSummary",
    "WeightsLoadError",
    "WeightsMetadata",
    "WeightsNotFoundError",
    "WeightsValidationError",
    "YOLOv8SignalInferenceConfig",
    "YOLOv8SignalInferenceEngine",
    "YOLOv8SignalModelLoader",
    "YOLOv8SignalOutputParser",
    "YOLOv8SignalRawOutput",
    "adas_label_to_class_id",
    "bdd100k_label_to_adas_label",
    "enrich_state_flags",
    "load_traffic_signal_classes",
    "resolve_variant_name",
    "state_priority",
]
