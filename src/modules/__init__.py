"""ADAS perception modules package."""

from src.modules.base import BaseModule, Frame, PredictionResult
from src.modules.lane_detection import LaneDetectionModule
from src.modules.segmentation import SegmentationModule
from src.modules.traffic_sign import TrafficSignModule
from src.modules.traffic_signal import TrafficSignalModule
from src.modules.vehicle_detection import VehicleDetectionModule

__all__ = [
    "BaseModule",
    "Frame",
    "PredictionResult",
    "LaneDetectionModule",
    "VehicleDetectionModule",
    "TrafficSignModule",
    "TrafficSignalModule",
    "SegmentationModule",
]
