"""Lane geometry extraction from binary lane segmentation masks.

Provides the first geometry stage for the ADAS lane detection pipeline:
lane pixel extraction, lane center estimation, and vehicle offset calculation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("adas.modules.yolop.lane_geometry")

LaneMask = np.ndarray


@dataclass(frozen=True)
class LanePixels:
    """Foreground lane pixel coordinates extracted from a binary mask.

    Attributes:
        x_coords: Horizontal pixel coordinates of lane foreground pixels.
        y_coords: Vertical pixel coordinates of lane foreground pixels.
        count: Number of lane pixels.
    """

    x_coords: np.ndarray
    y_coords: np.ndarray
    count: int


@dataclass(frozen=True)
class VehicleOffsetResult:
    """Vehicle lateral offset relative to the lane center.

    Attributes:
        offset_pixels: Signed offset in pixels (lane_center - vehicle_center).
        lane_center_x: Mean x-coordinate of lane pixels.
        vehicle_center_x: Ego vehicle x-position proxy (image center).
    """

    offset_pixels: float
    lane_center_x: float
    vehicle_center_x: float


class LaneGeometryExtractor:
    """Extract lane geometry metrics from binary lane segmentation masks.

    Computes lane center as the mean x-coordinate of all lane pixels and
    vehicle offset relative to the image center.  Lane departure detection
    is handled separately.
    """

    def extract_lane_pixels(self, lane_mask: LaneMask) -> LanePixels:
        """Extract foreground lane pixel coordinates from a binary mask.

        Args:
            lane_mask: Binary lane mask with foreground values ``> 0``
                (typically ``255``).

        Returns:
            :class:`LanePixels` containing x/y coordinate arrays.

        Raises:
            ValueError: If ``lane_mask`` is invalid or empty-shaped.
        """
        self._validate_lane_mask(lane_mask)

        y_coords, x_coords = np.where(lane_mask > 0)
        pixel_count = int(x_coords.size)

        logger.debug("Extracted %d lane pixels from mask shape %s", pixel_count, lane_mask.shape)

        return LanePixels(
            x_coords=x_coords.astype(np.float64),
            y_coords=y_coords.astype(np.float64),
            count=pixel_count,
        )

    def compute_lane_center(self, lane_mask: LaneMask) -> float | None:
        """Compute lane center as the mean x-coordinate of lane pixels.

        Args:
            lane_mask: Binary lane segmentation mask.

        Returns:
            Mean x-coordinate in pixels, or ``None`` if no lane pixels exist.
        """
        lane_pixels = self.extract_lane_pixels(lane_mask)

        if lane_pixels.count == 0:
            logger.warning("Cannot compute lane center — no lane pixels found")
            return None

        lane_center_x = float(np.mean(lane_pixels.x_coords))

        logger.info(
            "Computed lane center — x=%.2f px from %d lane pixels",
            lane_center_x,
            lane_pixels.count,
        )
        return lane_center_x

    def compute_vehicle_offset(
        self,
        lane_center_x: float,
        image_width: int,
    ) -> VehicleOffsetResult:
        """Compute vehicle offset relative to the lane center.

        Uses image center as the ego vehicle position proxy:
        ``vehicle_center = image_width / 2``
        ``offset = lane_center_x - vehicle_center``

        Args:
            lane_center_x: Lane center horizontal position in pixels.
            image_width: Frame width in pixels.

        Returns:
            :class:`VehicleOffsetResult` with offset and center positions.

        Raises:
            ValueError: If ``image_width`` is not positive.
        """
        if image_width <= 0:
            raise ValueError(f"image_width must be positive, got {image_width}")

        vehicle_center_x = image_width / 2.0
        offset_pixels = lane_center_x - vehicle_center_x

        logger.info(
            "Computed vehicle offset — lane_center=%.2f, vehicle_center=%.2f, "
            "offset=%.2f px",
            lane_center_x,
            vehicle_center_x,
            offset_pixels,
        )

        return VehicleOffsetResult(
            offset_pixels=offset_pixels,
            lane_center_x=lane_center_x,
            vehicle_center_x=vehicle_center_x,
        )

    @staticmethod
    def _validate_lane_mask(lane_mask: LaneMask) -> None:
        """Validate lane mask format.

        Args:
            lane_mask: Candidate lane mask array.

        Raises:
            ValueError: If the mask is missing or has invalid dimensions.
        """
        if lane_mask is None:
            raise ValueError("lane_mask is None")

        if not isinstance(lane_mask, np.ndarray):
            raise ValueError(
                f"lane_mask must be numpy.ndarray, got {type(lane_mask).__name__}"
            )

        if lane_mask.ndim != 2:
            raise ValueError(
                f"lane_mask must be 2D with shape (H, W), got {lane_mask.shape}"
            )

        if lane_mask.size == 0:
            raise ValueError("lane_mask is empty")
