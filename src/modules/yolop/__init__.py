"""YOLOP integration package for lane detection.

Provides three layers for future YOLOP integration:

- ``YOLOPModelLoader`` — checkpoint loading and device management
- ``YOLOPInference`` — forward-pass inference wrapper
- ``parse_yolop_output`` — raw output to lane data conversion

No YOLOP code or model downloads are performed at this stage.
"""

from src.modules.yolop.inference import YOLOPInference, YOLOPRawOutput
from src.modules.yolop.model_loader import YOLOPModelLoader
from src.modules.yolop.utils import (
    LaneParseResult,
    compute_vehicle_offset,
    parse_lane_mask,
    parse_yolop_output,
)

__all__ = [
    "YOLOPModelLoader",
    "YOLOPInference",
    "YOLOPRawOutput",
    "LaneParseResult",
    "parse_lane_mask",
    "compute_vehicle_offset",
    "parse_yolop_output",
]
