"""ADAS perception modules package."""

from src.modules.base import BaseModule, Frame, PredictionResult
from src.modules.lane_detection import LaneDetectionModule, LANE_OUTPUT_KEYS
from src.modules.yolop.output_schema import LaneDetectionResult
from src.modules.segmentation import SegmentationModule
from src.modules.traffic_sign import TrafficSignModule
from src.modules.traffic_signal import TrafficSignalModule
from src.modules.vehicle_detection import VehicleDetectionModule

__all__ = [
    "BaseModule",
    "Frame",
    "PredictionResult",
    "LaneDetectionModule",
    "LaneDetectionResult",
    "LANE_OUTPUT_KEYS",
    "VehicleDetectionModule",
    "TrafficSignModule",
    "TrafficSignalModule",
    "SegmentationModule",
]
