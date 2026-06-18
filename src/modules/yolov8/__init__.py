"""YOLOv8 integration package for vehicle detection."""

from __future__ import annotations

from .inference import (
    DEFAULT_IMGSZ,
    InferenceExecutionError,
    InferenceNotReadyError,
    InvalidFrameError,
    YOLOv8InferenceConfig,
    YOLOv8InferenceEngine,
    YOLOv8RawOutput,
)
from .model_loader import (
    DEFAULT_MODEL_VARIANT,
    WeightsLoadError,
    WeightsMetadata,
    WeightsNotFoundError,
    WeightsValidationError,
    YOLOv8ModelLoader,
    resolve_variant_name,
    variant_to_filename,
)
from .output_parser import (
    ALLOWED_COCO_CLASS_IDS,
    COCO_CLASS_ID_TO_LABEL,
    ParserConfig,
    YOLOv8OutputParser,
)
from .output_schema import (
    ADAS_VEHICLE_LABELS,
    BoundingBoxData,
    DetectedObject,
    ParsedYOLOv8Output,
    VehicleDetectionResult,
    VehicleDetectionSummary,
)

__all__ = [
    "ADAS_VEHICLE_LABELS",
    "ALLOWED_COCO_CLASS_IDS",
    "BoundingBoxData",
    "COCO_CLASS_ID_TO_LABEL",
    "DEFAULT_IMGSZ",
    "DEFAULT_MODEL_VARIANT",
    "DetectedObject",
    "InferenceExecutionError",
    "InferenceNotReadyError",
    "InvalidFrameError",
    "ParsedYOLOv8Output",
    "ParserConfig",
    "VehicleDetectionResult",
    "VehicleDetectionSummary",
    "WeightsLoadError",
    "WeightsMetadata",
    "WeightsNotFoundError",
    "WeightsValidationError",
    "YOLOv8InferenceConfig",
    "YOLOv8InferenceEngine",
    "YOLOv8ModelLoader",
    "YOLOv8OutputParser",
    "YOLOv8RawOutput",
    "resolve_variant_name",
    "variant_to_filename",
]
