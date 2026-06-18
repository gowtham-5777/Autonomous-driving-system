"""ADAS perception modules package."""

from .base import BaseModule, Frame, PredictionResult
from .lane_detection import (
    LANE_DETECTION_PIPELINE_VERSION,
    LaneDetectionModule,
    LANE_OUTPUT_KEYS,
)
from .vehicle_detection import (
    VEHICLE_OUTPUT_KEYS,
    VehicleDetectionModule,
)
from .yolop.output_schema import LaneDetectionResult
from .yolov8.output_schema import VehicleDetectionResult
from .segmentation import SegmentationModule
from .traffic_sign import TRAFFIC_SIGN_OUTPUT_KEYS, TrafficSignModule
from .traffic_signal import TRAFFIC_SIGNAL_OUTPUT_KEYS, TrafficSignalModule
from .yolov8_sign.output_schema import TrafficSignDetectionResult
from .yolov8_signal.output_schema import TrafficSignalDetectionResult

__all__ = [
    "BaseModule",
    "Frame",
    "PredictionResult",
    "LaneDetectionModule",
    "LANE_DETECTION_PIPELINE_VERSION",
    "LaneDetectionResult",
    "LANE_OUTPUT_KEYS",
    "VehicleDetectionModule",
    "VehicleDetectionResult",
    "VEHICLE_OUTPUT_KEYS",
    "TrafficSignModule",
    "TrafficSignDetectionResult",
    "TRAFFIC_SIGN_OUTPUT_KEYS",
    "TrafficSignalModule",
    "TrafficSignalDetectionResult",
    "TRAFFIC_SIGNAL_OUTPUT_KEYS",
    "SegmentationModule",
]
