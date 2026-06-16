"""Dataclass schemas for parsed YOLOP inference outputs.

Defines structured types consumed by ``YOLOPOutputParser`` and downstream
lane detection / visualization modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

LanePolyline = list[tuple[int, int]] | None


@dataclass
class LaneLineData:
    """Left and right lane boundary representations.

    Attributes:
        left_lane: Polyline vertices for the left lane boundary.
        right_lane: Polyline vertices for the right lane boundary.
        lane_mask: Binary lane segmentation mask from YOLOP head output.
    """

    left_lane: LanePolyline = None
    right_lane: LanePolyline = None
    lane_mask: np.ndarray | None = None


@dataclass
class DrivableAreaData:
    """Drivable area segmentation output.

    Attributes:
        mask: Binary or probability drivable-area mask.
        coverage_ratio: Fraction of frame classified as drivable.
    """

    mask: np.ndarray | None = None
    coverage_ratio: float | None = None


@dataclass
class LaneCenterData:
    """Estimated lane center geometry.

    Attributes:
        center_line: Polyline representing the lane center path.
        center_x_at_bottom: Lane center x-coordinate at the bottom of frame.
    """

    center_line: LanePolyline = None
    center_x_at_bottom: float | None = None


@dataclass
class VehicleOffsetData:
    """Lateral offset of the ego vehicle relative to lane center.

    Attributes:
        offset_pixels: Signed pixel offset (negative=left, positive=right).
        vehicle_x: Ego vehicle horizontal position proxy in pixels.
        lane_center_x: Lane center horizontal position at evaluation row.
    """

    offset_pixels: float | None = None
    vehicle_x: int | None = None
    lane_center_x: float | None = None


@dataclass
class LaneDepartureData:
    """Lane departure warning state.

    Attributes:
        is_departing: Whether a lane departure condition is active.
        direction: Departure direction (``"left"``, ``"right"``, or ``None``).
    """

    is_departing: bool = False
    direction: str | None = None


@dataclass
class ParsedYOLOPOutput:
    """Fully parsed YOLOP output for lane detection modules.

    Attributes:
        lane_lines: Extracted left/right lane boundaries.
        drivable_area: Drivable area mask and coverage statistics.
        lane_center: Computed lane center geometry.
        vehicle_offset: Ego vehicle lateral offset data.
        lane_departure: Lane departure warning state.
        raw_status: Inference status string from raw outputs.
        metadata: Additional non-sensitive diagnostic fields.
    """

    lane_lines: LaneLineData = field(default_factory=LaneLineData)
    drivable_area: DrivableAreaData = field(default_factory=DrivableAreaData)
    lane_center: LaneCenterData = field(default_factory=LaneCenterData)
    vehicle_offset: VehicleOffsetData = field(default_factory=VehicleOffsetData)
    lane_departure: LaneDepartureData = field(default_factory=LaneDepartureData)
    raw_status: str = "unparsed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prediction_dict(self) -> dict[str, Any]:
        """Convert to the prediction schema used by ``LaneDetectionModule``.

        Returns:
            Dictionary with keys: ``left_lane``, ``right_lane``,
            ``lane_center``, ``vehicle_offset``, ``lane_departure``.
        """
        return {
            "left_lane": self.lane_lines.left_lane,
            "right_lane": self.lane_lines.right_lane,
            "lane_center": self.lane_center.center_line,
            "vehicle_offset": self.vehicle_offset.offset_pixels,
            "lane_departure": self.lane_departure.is_departing,
        }

    @classmethod
    def empty(cls) -> ParsedYOLOPOutput:
        """Return an empty parsed output with default placeholder values."""
        return cls(raw_status="empty")
