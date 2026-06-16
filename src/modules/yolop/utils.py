"""YOLOP output parsing utilities — architecture placeholder.

This module will convert raw YOLOP multi-head outputs into structured
lane data compatible with ``LaneDetectionModule`` and the visualization
layer.  No parsing logic is implemented yet.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("adas.modules.yolop.utils")

YOLOPRawOutput = dict[str, Any]
LaneParseResult = dict[str, Any]


def parse_lane_mask(
    lane_mask: np.ndarray | None,
    frame_shape: tuple[int, ...],
) -> LaneParseResult:
    """Extract lane lines and lane center from a YOLOP lane segmentation mask.

    Placeholder: returns empty lane fields until mask parsing is implemented.

    Args:
        lane_mask: Binary or multi-class lane segmentation output from YOLOP.
        frame_shape: Shape of the original BGR frame ``(H, W, C)``.

    Returns:
        Dictionary with ``left_lane``, ``right_lane``, and ``lane_center`` keys.
    """
    logger.debug(
        "parse_lane_mask() placeholder — mask=%s, frame_shape=%s",
        "present" if lane_mask is not None else "none",
        frame_shape,
    )

    # TODO: Threshold and skeletonize lane_mask.
    # TODO: Fit left and right lane polynomials or polylines.
    # TODO: Compute lane center curve from left/right boundaries.
    return {
        "left_lane": None,
        "right_lane": None,
        "lane_center": None,
    }


def compute_vehicle_offset(
    lane_center: Any,
    frame_width: int,
    vehicle_x: int | None = None,
    departure_threshold_px: float = 50.0,
) -> dict[str, Any]:
    """Compute lateral vehicle offset and lane departure flag.

    Placeholder: returns neutral values until lane center is available.

    Args:
        lane_center: Lane center representation (polyline or x-coordinate).
        frame_width: Width of the frame in pixels.
        vehicle_x: Ego vehicle x-position proxy. Defaults to image center.
        departure_threshold_px: Pixel offset threshold for departure warning.

    Returns:
        Dictionary with ``vehicle_offset`` and ``lane_departure`` keys.
    """
    logger.debug(
        "compute_vehicle_offset() placeholder — lane_center=%s",
        "present" if lane_center is not None else "none",
    )

    # TODO: Sample lane_center at the bottom of the frame.
    # TODO: Compare to vehicle_x and compute signed offset.
    # TODO: Set lane_departure when abs(offset) > departure_threshold_px.
    _ = frame_width, vehicle_x, departure_threshold_px
    return {
        "vehicle_offset": None,
        "lane_departure": False,
    }


def parse_yolop_output(
    raw_output: YOLOPRawOutput,
    frame_shape: tuple[int, ...],
) -> LaneParseResult:
    """Parse full YOLOP raw output into lane detection results.

    Convenience function that orchestrates mask parsing and offset
    calculation from a single raw output dictionary.

    Args:
        raw_output: Raw output from ``YOLOPInferenceEngine.run()``.
        frame_shape: Shape of the original BGR frame.

    Returns:
        Combined dictionary with lane lines, center, offset, and departure.
    """
    logger.debug("parse_yolop_output() placeholder — parsing raw YOLOP output")

    lane_data = parse_lane_mask(
        raw_output.get("lane_mask"),
        frame_shape,
    )

    offset_data = compute_vehicle_offset(
        lane_data.get("lane_center"),
        frame_width=frame_shape[1],
    )

    return {
        **lane_data,
        **offset_data,
    }
