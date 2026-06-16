"""Lane-specific preprocessing — edge detection and region masking."""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

Frame = np.ndarray


class LanePreprocessor:
    """Preprocess road frames for lane detection pipelines.

    Applies a standard computer-vision pipeline: resize, grayscale conversion,
    Gaussian blur, Canny edge detection, and region-of-interest masking.

    This class performs preprocessing only — no lane detection or YOLOP logic.

    Attributes:
        target_size: Optional ``(width, height)`` to resize frames before processing.
        blur_kernel_size: Kernel size for Gaussian blur (must be odd and positive).
        canny_low_threshold: Lower hysteresis threshold for Canny edge detection.
        canny_high_threshold: Upper hysteresis threshold for Canny edge detection.
        roi_vertices: Optional custom ROI polygon vertices as ``(N, 2)`` array.
            When ``None``, a default trapezoid covering the lower road area is used.
    """

    def __init__(
        self,
        target_size: tuple[int, int] | None = (1280, 720),
        blur_kernel_size: int = 5,
        canny_low_threshold: int = 50,
        canny_high_threshold: int = 150,
        roi_vertices: np.ndarray | None = None,
    ) -> None:
        """Initialize the lane preprocessor with tunable parameters.

        Args:
            target_size: Target ``(width, height)`` for resizing. ``None`` keeps
                the original frame dimensions.
            blur_kernel_size: Gaussian blur kernel size (odd integer).
            canny_low_threshold: Canny lower threshold.
            canny_high_threshold: Canny upper threshold.
            roi_vertices: Optional ROI polygon vertices. Coordinates must match
                the frame size at the ROI step (after resize).

        Raises:
            ValueError: If blur kernel size is invalid or Canny thresholds are invalid.
        """
        if blur_kernel_size < 1 or blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be a positive odd integer")
        if canny_low_threshold < 0 or canny_high_threshold < 0:
            raise ValueError("Canny thresholds must be non-negative")
        if canny_low_threshold >= canny_high_threshold:
            raise ValueError("canny_low_threshold must be less than canny_high_threshold")

        self.target_size = target_size
        self.blur_kernel_size = blur_kernel_size
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.roi_vertices = roi_vertices

        self.logger = logging.getLogger("adas.preprocessing.lane")

    def resize_frame(self, frame: Frame) -> Frame:
        """Resize the input frame to the configured target size.

        Args:
            frame: BGR input image.

        Returns:
            Resized BGR image, or a copy of the original if ``target_size`` is ``None``.
        """
        if self.target_size is None:
            self.logger.debug("Skipping resize — target_size is None")
            return frame.copy()

        width, height = self.target_size
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        self.logger.debug("Resized frame to %dx%d", width, height)
        return resized

    def convert_to_grayscale(self, frame: Frame) -> Frame:
        """Convert a BGR frame to grayscale.

        Args:
            frame: BGR input image.

        Returns:
            Single-channel grayscale image.
        """
        if frame.ndim == 2:
            self.logger.debug("Frame already grayscale")
            return frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.logger.debug("Converted frame to grayscale")
        return gray

    def apply_gaussian_blur(self, frame: Frame) -> Frame:
        """Apply Gaussian blur to reduce noise before edge detection.

        Args:
            frame: Grayscale or BGR input image.

        Returns:
            Blurred image with the same channel count as the input.
        """
        blurred = cv2.GaussianBlur(
            frame,
            (self.blur_kernel_size, self.blur_kernel_size),
            0,
        )
        self.logger.debug(
            "Applied Gaussian blur with kernel size %d",
            self.blur_kernel_size,
        )
        return blurred

    def detect_edges(self, frame: Frame) -> Frame:
        """Detect edges using the Canny algorithm.

        Args:
            frame: Grayscale input image.

        Returns:
            Binary edge map (single channel, values 0 or 255).
        """
        edges = cv2.Canny(
            frame,
            self.canny_low_threshold,
            self.canny_high_threshold,
        )
        self.logger.debug(
            "Detected edges with Canny thresholds (%d, %d)",
            self.canny_low_threshold,
            self.canny_high_threshold,
        )
        return edges

    def _default_roi_vertices(self, frame: Frame) -> np.ndarray:
        """Build a default trapezoidal ROI for the lower road region.

        Args:
            frame: Reference image used to scale vertex coordinates.

        Returns:
            ROI polygon vertices with shape ``(4, 2)`` and dtype ``int32``.
        """
        height, width = frame.shape[:2]
        vertices = np.array(
            [
                [int(width * 0.10), height],
                [int(width * 0.40), int(height * 0.60)],
                [int(width * 0.60), int(height * 0.60)],
                [int(width * 0.90), height],
            ],
            dtype=np.int32,
        )
        return vertices

    def region_of_interest(self, frame: Frame) -> Frame:
        """Mask the frame to a road-focused region of interest.

        Args:
            frame: Edge map or grayscale image to mask.

        Returns:
            Image with pixels outside the ROI set to zero.
        """
        masked = np.zeros_like(frame)
        vertices = (
            self.roi_vertices
            if self.roi_vertices is not None
            else self._default_roi_vertices(frame)
        )
        cv2.fillPoly(masked, [vertices], 255)
        roi_frame = cv2.bitwise_and(frame, masked)
        self.logger.debug("Applied region-of-interest mask")
        return roi_frame

    def preprocess(self, frame: Frame) -> Frame:
        """Run the full lane preprocessing pipeline.

        Pipeline:
            Input Frame → Resize → Grayscale → Gaussian Blur →
            Canny Edge Detection → Region Of Interest Masking → Output

        Args:
            frame: Original BGR input frame.

        Returns:
            Preprocessed single-channel edge image masked to the road ROI.
        """
        self.logger.info("Starting lane preprocessing pipeline")

        resized = self.resize_frame(frame)
        gray = self.convert_to_grayscale(resized)
        blurred = self.apply_gaussian_blur(gray)
        edges = self.detect_edges(blurred)
        processed = self.region_of_interest(edges)

        self.logger.info("Lane preprocessing pipeline complete")
        return processed

    def set_roi_vertices(self, vertices: Sequence[Sequence[int]]) -> None:
        """Update the ROI polygon vertices.

        Args:
            vertices: Polygon points as an iterable of ``(x, y)`` pairs.
        """
        self.roi_vertices = np.array(vertices, dtype=np.int32)
        self.logger.debug("Updated custom ROI vertices")
