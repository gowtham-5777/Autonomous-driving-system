"""Visualization overlays for lane detection and other perception modules."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger("adas.visualization.overlays")

Frame = np.ndarray
LaneLine = Sequence[tuple[int, int]] | Sequence[Sequence[int]] | None
LaneCenter = tuple[int, int] | Sequence[int] | None
LaneResults = dict[str, Any]

# BGR color palette for lane overlays
COLOR_LEFT_LANE = (0, 255, 0)       # Green
COLOR_RIGHT_LANE = (255, 128, 0)    # Orange-blue
COLOR_LANE_CENTER = (0, 255, 255)   # Yellow
COLOR_VEHICLE_MARKER = (255, 0, 255)  # Magenta
COLOR_WARNING_BG = (0, 0, 200)      # Red (dark)
COLOR_WARNING_TEXT = (255, 255, 255)
COLOR_PLACEHOLDER = (180, 180, 180)  # Gray


def _ensure_bgr_frame(frame: Frame) -> Frame:
    """Return a writable BGR copy of the input frame.

    Args:
        frame: Input image.

    Returns:
        Copy of ``frame`` suitable for drawing operations.
    """
    return frame.copy()


def _as_point_sequence(line: LaneLine) -> list[tuple[int, int]] | None:
    """Convert a lane line representation to a list of integer points.

    Args:
        line: Lane polyline as a sequence of ``(x, y)`` pairs, or ``None``.

    Returns:
        List of vertex tuples, or ``None`` if input is ``None``.
    """
    if line is None:
        return None

    points: list[tuple[int, int]] = []
    for point in line:
        if len(point) < 2:
            continue
        points.append((int(point[0]), int(point[1])))

    return points if points else None


def _default_placeholder_lanes(frame: Frame) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Generate placeholder left/right lane polylines for demo rendering.

    Args:
        frame: Reference frame for scaling vertex positions.

    Returns:
        Tuple of ``(left_lane_points, right_lane_points)``.
    """
    height, width = frame.shape[:2]
    left_lane = [
        (int(width * 0.15), height),
        (int(width * 0.38), int(height * 0.62)),
        (int(width * 0.44), int(height * 0.55)),
    ]
    right_lane = [
        (int(width * 0.85), height),
        (int(width * 0.62), int(height * 0.62)),
        (int(width * 0.56), int(height * 0.55)),
    ]
    return left_lane, right_lane


def draw_lane_lines(
    frame: Frame,
    left_lane: LaneLine,
    right_lane: LaneLine,
) -> Frame:
    """Draw left and right lane boundary lines on the frame.

    When lane coordinates are ``None``, placeholder guide lines are drawn
    in the lower road region until YOLOP integration provides real data.

    Args:
        frame: BGR input image.
        left_lane: Left lane polyline vertices, or ``None``.
        right_lane: Right lane polyline vertices, or ``None``.

    Returns:
        Annotated BGR image with lane lines drawn.
    """
    output = _ensure_bgr_frame(frame)
    left_points = _as_point_sequence(left_lane)
    right_points = _as_point_sequence(right_lane)

    using_placeholder = left_points is None or right_points is None
    if using_placeholder:
        left_points, right_points = _default_placeholder_lanes(output)
        logger.debug("Drawing placeholder lane lines")

    if left_points is not None and len(left_points) >= 2:
        cv2.polylines(
            output,
            [np.array(left_points, dtype=np.int32)],
            isClosed=False,
            color=COLOR_PLACEHOLDER if using_placeholder else COLOR_LEFT_LANE,
            thickness=4,
            lineType=cv2.LINE_AA,
        )

    if right_points is not None and len(right_points) >= 2:
        cv2.polylines(
            output,
            [np.array(right_points, dtype=np.int32)],
            isClosed=False,
            color=COLOR_PLACEHOLDER if using_placeholder else COLOR_RIGHT_LANE,
            thickness=4,
            lineType=cv2.LINE_AA,
        )

    logger.debug("Lane lines drawn (placeholder=%s)", using_placeholder)
    return output


def draw_lane_center(frame: Frame, lane_center: LaneCenter) -> Frame:
    """Draw the estimated lane center on the frame.

    Placeholder behavior draws a vertical dashed-style center line at the
    image midpoint when ``lane_center`` is ``None``.

    Args:
        frame: BGR input image.
        lane_center: Lane center as ``(x, y)`` or ``None``.

    Returns:
        Annotated BGR image with lane center marker.
    """
    output = _ensure_bgr_frame(frame)
    height, width = output.shape[:2]

    if lane_center is None:
        center_x = width // 2
        logger.debug("Drawing placeholder lane center at x=%d", center_x)
    else:
        center_x = int(lane_center[0])
        logger.debug("Drawing lane center at x=%d", center_x)

    # Dashed vertical line effect using short segments
    segment_length = 20
    gap_length = 15
    y = int(height * 0.55)
    while y < height:
        y_end = min(y + segment_length, height)
        cv2.line(
            output,
            (center_x, y),
            (center_x, y_end),
            COLOR_LANE_CENTER,
            2,
            lineType=cv2.LINE_AA,
        )
        y += segment_length + gap_length

    cv2.circle(output, (center_x, height - 20), 6, COLOR_LANE_CENTER, -1, lineType=cv2.LINE_AA)
    return output


def draw_vehicle_offset(frame: Frame, vehicle_offset: float | int | None) -> Frame:
    """Display the vehicle's lateral offset from the lane center.

    Placeholder behavior shows ``"Offset: N/A"`` when offset is unknown.

    Args:
        frame: BGR input image.
        vehicle_offset: Lateral offset in pixels (negative=left, positive=right),
            or ``None`` if unavailable.

    Returns:
        Annotated BGR image with offset text overlay.
    """
    output = _ensure_bgr_frame(frame)

    if vehicle_offset is None:
        label = "Vehicle Offset: N/A (placeholder)"
        logger.debug("Drawing placeholder vehicle offset label")
    else:
        direction = "left" if vehicle_offset < 0 else "right" if vehicle_offset > 0 else "center"
        label = f"Vehicle Offset: {vehicle_offset:+.1f}px ({direction})"
        logger.debug("Drawing vehicle offset: %s", label)

    cv2.putText(
        output,
        label,
        (20, output.shape[0] - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_VEHICLE_MARKER,
        2,
        lineType=cv2.LINE_AA,
    )

    # Vehicle position marker at image horizontal center (ego proxy)
    center_x = output.shape[1] // 2
    cv2.drawMarker(
        output,
        (center_x, output.shape[0] - 30),
        COLOR_VEHICLE_MARKER,
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=16,
        thickness=2,
        line_type=cv2.LINE_AA,
    )

    return output


def draw_lane_departure_warning(frame: Frame, lane_departure: bool) -> Frame:
    """Draw a lane departure warning banner when departure is detected.

    Args:
        frame: BGR input image.
        lane_departure: Whether a lane departure condition is active.

    Returns:
        Annotated BGR image, with a warning banner when ``lane_departure``
        is ``True``.
    """
    output = _ensure_bgr_frame(frame)

    if not lane_departure:
        logger.debug("No lane departure — skipping warning banner")
        return output

    logger.info("Drawing lane departure warning banner")

    banner_height = 50
    cv2.rectangle(output, (0, 0), (output.shape[1], banner_height), COLOR_WARNING_BG, -1)
    cv2.putText(
        output,
        "WARNING: LANE DEPARTURE",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        COLOR_WARNING_TEXT,
        2,
        lineType=cv2.LINE_AA,
    )

    return output


# BGR colors for vehicle detection overlays
VEHICLE_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (0, 165, 255),
    "bicycle": (255, 255, 0),
    "car": (0, 255, 0),
    "motorcycle": (255, 0, 255),
    "bus": (0, 128, 255),
    "truck": (255, 128, 0),
}
COLOR_NEAREST_HIGHLIGHT = (0, 255, 255)


def draw_vehicle_detections(frame: Frame, results: dict[str, Any]) -> Frame:
    """Draw bounding boxes and class labels for vehicle detections.

    Expected ``results`` keys:
        - ``detections``: list of dicts with ``label``, ``confidence``, ``bbox``
        - ``count_by_label`` (optional)
        - ``nearest_object`` (optional)

    Args:
        frame: BGR input image.
        results: Vehicle prediction dictionary from ``VehicleDetectionModule``.

    Returns:
        Annotated copy of ``frame``.
    """
    output = _ensure_bgr_frame(frame)
    detections = results.get("detections", [])
    nearest_bbox = None
    nearest = results.get("nearest_object")
    if isinstance(nearest, dict):
        nearest_bbox = nearest.get("bbox")

    for det in detections:
        if not isinstance(det, dict):
            continue

        label = str(det.get("label", "unknown"))
        confidence = float(det.get("confidence", 0.0))
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        color = VEHICLE_COLORS.get(label, COLOR_PLACEHOLDER)
        thickness = 4 if bbox == nearest_bbox else 2

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        text = f"{label} {confidence:.2f}"
        text_y = y1 - 8 if y1 > 20 else y2 + 20
        cv2.putText(
            output,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    count_by_label = results.get("count_by_label", {})
    total = results.get("total_count", len(detections))
    if count_by_label:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(count_by_label.items()))
        hud = f"Objects: {total} ({summary})"
    else:
        hud = f"Objects: {total}"

    cv2.putText(
        output,
        hud,
        (output.shape[1] - 420, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_NEAREST_HIGHLIGHT,
        2,
        lineType=cv2.LINE_AA,
    )

    logger.debug("Vehicle detections drawn — count=%d", total)
    return output


def draw_lane_results(frame: Frame, results: LaneResults) -> Frame:
    """Draw all lane visualization overlays from a prediction dictionary.

    Expected ``results`` keys:
        - ``left_lane``
        - ``right_lane``
        - ``lane_center``
        - ``vehicle_offset``
        - ``lane_departure``

    Args:
        frame: BGR input image.
        results: Lane prediction dictionary from ``LaneDetectionModule.predict``.

    Returns:
        Fully annotated BGR image with lane overlays applied in order.
    """
    logger.debug("Drawing complete lane results overlay")

    output = draw_lane_lines(
        frame,
        results.get("left_lane"),
        results.get("right_lane"),
    )
    output = draw_lane_center(output, results.get("lane_center"))
    output = draw_vehicle_offset(output, results.get("vehicle_offset"))
    output = draw_lane_departure_warning(output, bool(results.get("lane_departure", False)))

    return output
