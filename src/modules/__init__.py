"""ADAS perception modules package."""

from .base import BaseModule, Frame, PredictionResult
from .lane_detection import (
    LANE_DETECTION_PIPELINE_VERSION,
    LaneDetectionModule,
    LANE_OUTPUT_KEYS,
)
from .yolop.output_schema import LaneDetectionResult
from .segmentation import SegmentationModule
from .traffic_sign import TrafficSignModule
from .traffic_signal import TrafficSignalModule
from .vehicle_detection import VehicleDetectionModule

__all__ = [
    "BaseModule",
    "Frame",
    "PredictionResult",
    "LaneDetectionModule",
    "LANE_DETECTION_PIPELINE_VERSION",
    "LaneDetectionResult",
    "LANE_OUTPUT_KEYS",
    "VehicleDetectionModule",
    "TrafficSignModule",
    "TrafficSignalModule",
    "SegmentationModule",
]
