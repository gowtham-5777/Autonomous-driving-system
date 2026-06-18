"""YOLOv8 sign integration package for traffic sign detection."""

from __future__ import annotations

from .class_map import (
    ALLOWED_SIGN_CLASS_IDS,
    GTSRB_CLASS_ID_TO_ADAS_LABEL,
    SIGN_CLASS_ID_TO_LABEL,
    SIGN_LABEL_TO_CLASS_ID,
    adas_label_to_class_id,
    extract_speed_limit_kmh,
    gtsrb_id_to_adas_label,
    is_regulatory_label,
    is_warning_label,
    load_traffic_sign_classes,
)
from .inference import (
    DEFAULT_IMGSZ,
    InferenceExecutionError,
    InferenceNotReadyError,
    InvalidFrameError,
    YOLOv8SignInferenceConfig,
    YOLOv8SignInferenceEngine,
    YOLOv8SignRawOutput,
)
from .model_loader import (
    DEFAULT_MODEL_VARIANT,
    WeightsLoadError,
    WeightsMetadata,
    WeightsNotFoundError,
    WeightsValidationError,
    YOLOv8SignModelLoader,
    resolve_variant_name,
)
from .output_parser import ParserConfig, YOLOv8SignOutputParser
from .output_schema import (
    ADAS_SIGN_LABELS,
    TRAFFIC_SIGN_OUTPUT_KEYS,
    DetectedSign,
    ParsedYOLOv8SignOutput,
    SignBoundingBoxData,
    TrafficSignDetectionResult,
    TrafficSignDetectionSummary,
)

__all__ = [
    "ADAS_SIGN_LABELS",
    "ALLOWED_SIGN_CLASS_IDS",
    "DEFAULT_IMGSZ",
    "DEFAULT_MODEL_VARIANT",
    "DetectedSign",
    "GTSRB_CLASS_ID_TO_ADAS_LABEL",
    "InferenceExecutionError",
    "InferenceNotReadyError",
    "InvalidFrameError",
    "ParsedYOLOv8SignOutput",
    "ParserConfig",
    "SIGN_CLASS_ID_TO_LABEL",
    "SIGN_LABEL_TO_CLASS_ID",
    "TRAFFIC_SIGN_OUTPUT_KEYS",
    "SignBoundingBoxData",
    "TrafficSignDetectionResult",
    "TrafficSignDetectionSummary",
    "WeightsLoadError",
    "WeightsMetadata",
    "WeightsNotFoundError",
    "WeightsValidationError",
    "YOLOv8SignInferenceConfig",
    "YOLOv8SignInferenceEngine",
    "YOLOv8SignModelLoader",
    "YOLOv8SignOutputParser",
    "YOLOv8SignRawOutput",
    "adas_label_to_class_id",
    "extract_speed_limit_kmh",
    "gtsrb_id_to_adas_label",
    "is_regulatory_label",
    "is_warning_label",
    "load_traffic_sign_classes",
    "resolve_variant_name",
]
