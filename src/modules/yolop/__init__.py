"""YOLOP integration package for lane detection.

Provides three layers for future YOLOP integration:

- ``YOLOPModelLoader`` — checkpoint loading and device management
- ``YOLOPInferenceEngine`` — forward-pass inference wrapper
- ``YOLOPOutputParser`` — raw output to lane data conversion
- ``parse_yolop_output`` — legacy convenience function (utils)

No YOLOP code or model downloads are performed at this stage.
"""

from src.modules.yolop.inference import (
    InferenceConfig,
    InferenceExecutionError,
    InferenceNotReadyError,
    InvalidFrameError,
    YOLOPInference,
    YOLOPInferenceEngine,
    YOLOPRawOutput,
)
from src.modules.yolop.model_loader import (
    CheckpointLoadError,
    CheckpointMetadata,
    CheckpointNotFoundError,
    CheckpointValidationError,
    DEFAULT_YOLOP_CHECKPOINT,
    YOLOPModelLoader,
)
from src.modules.yolop.lane_geometry import (
    LaneGeometryExtractor,
    LanePixels,
    VehicleOffsetResult,
)
from src.modules.yolop.postprocess import (
    ConnectedComponentsResult,
    connect_lane,
    connected_components_analysis,
    morphological_process,
    postprocess_lane_mask,
)
from src.modules.yolop.output_parser import ParserConfig, YOLOPOutputParser
from src.modules.yolop.output_schema import (
    DrivableAreaData,
    LaneCenterData,
    LaneDepartureData,
    LaneLineData,
    ParsedYOLOPOutput,
    VehicleOffsetData,
)

from src.modules.yolop.utils import (
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
    "ParsedYOLOPOutput",
    "LaneLineData",
    "DrivableAreaData",
    "LaneCenterData",
    "VehicleOffsetData",
    "LaneDepartureData",
    "LaneParseResult",
    "parse_lane_mask",
    "compute_vehicle_offset",
    "parse_yolop_output",
]
