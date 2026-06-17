"""YOLOP integration package for lane detection.

Provides three layers for future YOLOP integration:

- ``YOLOPModelLoader`` — checkpoint loading and device management
- ``YOLOPInferenceEngine`` — forward-pass inference wrapper
- ``YOLOPOutputParser`` — raw output to lane data conversion
- ``parse_yolop_output`` — legacy convenience function (utils)

No YOLOP code or model downloads are performed at this stage.
"""

from .inference import (
    InferenceConfig,
    InferenceExecutionError,
    InferenceNotReadyError,
    InvalidFrameError,
    YOLOPInference,
    YOLOPInferenceEngine,
    YOLOPRawOutput,
)
from .model_loader import (
    CheckpointLoadError,
    CheckpointMetadata,
    CheckpointNotFoundError,
    CheckpointValidationError,
    DEFAULT_YOLOP_CHECKPOINT,
    YOLOPModelLoader,
)
from .lane_geometry import (
    LaneGeometryExtractor,
    LanePixels,
    VehicleOffsetResult,
)
from .mask_resize import resize_mask_to_frame
from .postprocess import (
    ConnectedComponentsResult,
    connect_lane,
    connected_components_analysis,
    morphological_process,
    postprocess_lane_mask,
)
from .output_parser import ParserConfig, YOLOPOutputParser
from .output_schema import (
    DrivableAreaData,
    LaneCenterData,
    LaneDepartureData,
    LaneDetectionResult,
    LaneLineData,
    ParsedYOLOPOutput,
    VehicleOffsetData,
)

from .utils import (
    LaneParseResult,
    compute_vehicle_offset,
    parse_lane_mask,
    parse_yolop_output,
)

__all__ = [
    "YOLOPModelLoader",
    "CheckpointMetadata",
    "CheckpointNotFoundError",
    "CheckpointValidationError",
    "CheckpointLoadError",
    "DEFAULT_YOLOP_CHECKPOINT",
    "YOLOPInferenceEngine",
    "YOLOPInference",
    "InferenceConfig",
    "InferenceNotReadyError",
    "InvalidFrameError",
    "InferenceExecutionError",
    "YOLOPRawOutput",
    "YOLOPOutputParser",
    "ParserConfig",
    "LaneGeometryExtractor",
    "LanePixels",
    "VehicleOffsetResult",
    "morphological_process",
    "connect_lane",
    "connected_components_analysis",
    "ConnectedComponentsResult",
    "postprocess_lane_mask",
    "resize_mask_to_frame",
    "ParsedYOLOPOutput",
    "LaneLineData",
    "DrivableAreaData",
    "LaneCenterData",
    "VehicleOffsetData",
    "LaneDepartureData",
    "LaneDetectionResult",
    "LaneParseResult",
    "parse_lane_mask",
    "compute_vehicle_offset",
    "parse_yolop_output",
]
