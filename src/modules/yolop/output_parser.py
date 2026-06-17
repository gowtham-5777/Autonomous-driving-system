"""YOLOP output parser — segmentation mask extraction and lane data parsing.

Converts raw YOLOP inference outputs into structured dataclasses defined in
``output_schema.py``.  Segmentation mask extraction is implemented; lane
geometry calculations remain stubbed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .lane_geometry import LaneGeometryExtractor
from .mask_resize import resize_mask_to_frame
from .output_schema import (
    DrivableAreaData,
    LaneCenterData,
    LaneDepartureData,
    LaneLineData,
    ParsedYOLOPOutput,
    VehicleOffsetData,
)

logger = logging.getLogger("adas.modules.yolop.output_parser")

YOLOPRawOutput = dict[str, Any] | Sequence[Any] | tuple[Any, ...]
FrameShape = tuple[int, ...]

# YOLOP raw output indices for multi-head segmentation tensors
DRIVABLE_AREA_OUTPUT_INDEX = 1
LANE_SEGMENTATION_OUTPUT_INDEX = 2


@dataclass(frozen=True)
class ParserConfig:
    """Configuration for YOLOP output parsing.

    Attributes:
        departure_threshold_px: Lateral offset threshold for lane departure.
        evaluation_row_ratio: Vertical frame position used for offset checks
            (0.0 = top, 1.0 = bottom).
        foreground_class_threshold: Class index above which pixels are treated
            as foreground when building binary masks.
    """

    departure_threshold_px: float = 50.0
    evaluation_row_ratio: float = 0.90
    foreground_class_threshold: int = 0


class YOLOPOutputParser:
    """Parse raw YOLOP outputs into structured lane detection data.

    Implements binary mask extraction from YOLOP segmentation heads:
    ``raw_outputs[1]`` → drivable area, ``raw_outputs[2]`` → lane lines.

    Lane center, vehicle offset, and departure detection remain stubbed.

    Attributes:
        config: Parser configuration parameters.
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        """Create a YOLOP output parser.

        Args:
            config: Optional parser configuration.  Uses defaults when ``None``.
        """
        self.config = config or ParserConfig()
        self.geometry = LaneGeometryExtractor()
        logger.info(
            "YOLOPOutputParser created — departure_threshold=%.1fpx",
            self.config.departure_threshold_px,
        )

    def parse(
        self,
        raw_outputs: YOLOPRawOutput,
        frame_shape: FrameShape | None = None,
    ) -> ParsedYOLOPOutput:
        """Parse full raw YOLOP outputs into a :class:`ParsedYOLOPOutput`.

        Orchestrates all extraction and computation steps in pipeline order.

        Args:
            raw_outputs: Raw YOLOP forward-pass outputs (sequence or mapping).
                Index ``1`` = drivable area, index ``2`` = lane segmentation.
            frame_shape: Optional original frame shape ``(H, W, C)``.  Falls
                back to ``raw_outputs['original_shape']`` when omitted.

        Returns:
            Parsed output with segmentation masks and stubbed geometry fields.
        """
        logger.info("Parsing YOLOP raw outputs")

        resolved_shape = self._resolve_frame_shape(raw_outputs, frame_shape)
        frame_height = int(resolved_shape[0]) if len(resolved_shape) >= 1 else 0
        frame_width = int(resolved_shape[1]) if len(resolved_shape) >= 2 else 0

        lane_lines = self.extract_lane_information(raw_outputs)
        drivable_area = self.extract_drivable_area(raw_outputs)
        lane_lines_for_geometry = self._align_lane_lines_to_frame(
            lane_lines,
            frame_height,
            frame_width,
        )

        masks = {
            "drivable_mask": drivable_area.mask,
            "lane_mask": lane_lines.lane_mask,
        }
        lane_center = self.compute_lane_center(lane_lines_for_geometry, resolved_shape)
        vehicle_offset = self.compute_vehicle_offset(
            lane_center,
            frame_width=int(resolved_shape[1]) if len(resolved_shape) >= 2 else 0,
        )
        lane_departure = self.detect_lane_departure(
            vehicle_offset,
            threshold_px=self.config.departure_threshold_px,
        )

        parsed = ParsedYOLOPOutput(
            lane_lines=lane_lines,
            drivable_area=drivable_area,
            lane_center=lane_center,
            vehicle_offset=vehicle_offset,
            lane_departure=lane_departure,
            raw_status=self._resolve_status(raw_outputs),
            metadata={
                "input_size": self._get_optional_field(raw_outputs, "input_size"),
                "confidence_threshold": self._get_optional_field(
                    raw_outputs, "confidence_threshold"
                ),
                "frame_shape": resolved_shape,
                "drivable_mask": masks.get("drivable_mask"),
                "lane_mask": masks.get("lane_mask"),
            },
        )

        logger.debug(
            "YOLOP parse complete — status=%s, departure=%s",
            parsed.raw_status,
            parsed.lane_departure.is_departing,
        )
        return parsed

    def extract_segmentation_masks(
        self,
        raw_outputs: YOLOPRawOutput,
    ) -> dict[str, np.ndarray | None]:
        """Extract binary drivable and lane masks from raw YOLOP outputs.

        Args:
            raw_outputs: Raw YOLOP outputs where index ``1`` is drivable area
                segmentation and index ``2`` is lane segmentation.

        Returns:
            Dictionary with keys ``drivable_mask`` and ``lane_mask``.
        """
        drivable_area = self.extract_drivable_area(raw_outputs)
        lane_lines = self.extract_lane_information(raw_outputs)

        return {
            "drivable_mask": drivable_area.mask,
            "lane_mask": lane_lines.lane_mask,
        }

    def extract_lane_information(self, raw_outputs: YOLOPRawOutput) -> LaneLineData:
        """Extract lane segmentation mask from raw YOLOP outputs.

        Uses ``raw_outputs[2]`` as the lane segmentation tensor, applies
        argmax across the class dimension, and converts to a binary mask.

        Left/right lane polylines are not computed yet.

        Args:
            raw_outputs: Raw YOLOP forward-pass outputs.

        Returns:
            :class:`LaneLineData` with ``lane_mask`` populated when tensor
            extraction succeeds.

        Raises:
            ValueError: If ``raw_outputs`` has an unsupported container type.
            IndexError: If the lane output index is missing.
        """
        self._validate_raw_outputs(raw_outputs)

        lane_tensor = self._get_output_at_index(
            raw_outputs,
            LANE_SEGMENTATION_OUTPUT_INDEX,
        )
        lane_mask = self._segmentation_to_binary_mask(lane_tensor)

        logger.info(
            "Extracted lane mask — shape=%s, foreground_pixels=%d",
            lane_mask.shape,
            int(np.count_nonzero(lane_mask)),
        )

        # TODO: Skeletonize lane_mask and cluster into left/right components.
        # TODO: Fit polylines or polynomials for each lane boundary.
        return LaneLineData(
            left_lane=None,
            right_lane=None,
            lane_mask=lane_mask,
        )

    def extract_drivable_area(self, raw_outputs: YOLOPRawOutput) -> DrivableAreaData:
        """Extract drivable area mask from raw YOLOP outputs.

        Uses ``raw_outputs[1]`` as the drivable-area segmentation tensor,
        applies argmax across the class dimension, and converts to a binary
        mask.

        Args:
            raw_outputs: Raw YOLOP forward-pass outputs.

        Returns:
            :class:`DrivableAreaData` with binary ``mask`` and ``coverage_ratio``.

        Raises:
            ValueError: If ``raw_outputs`` has an unsupported container type.
            IndexError: If the drivable output index is missing.
        """
        self._validate_raw_outputs(raw_outputs)

        drivable_tensor = self._get_output_at_index(
            raw_outputs,
            DRIVABLE_AREA_OUTPUT_INDEX,
        )
        drivable_mask = self._segmentation_to_binary_mask(drivable_tensor)
        coverage_ratio = self._compute_coverage_ratio(drivable_mask)

        logger.info(
            "Extracted drivable mask — shape=%s, coverage=%.2f%%",
            drivable_mask.shape,
            (coverage_ratio or 0.0) * 100.0,
        )

        return DrivableAreaData(mask=drivable_mask, coverage_ratio=coverage_ratio)

    def compute_lane_center(
        self,
        lane_lines: LaneLineData,
        frame_shape: FrameShape,
    ) -> LaneCenterData:
        """Compute lane center geometry from a lane segmentation mask.

        Uses :class:`LaneGeometryExtractor` to compute the mean x-coordinate
        of all lane foreground pixels.

        Args:
            lane_lines: Extracted lane data including ``lane_mask``.
            frame_shape: Original frame shape ``(H, W, C)``.

        Returns:
            :class:`LaneCenterData` with ``center_x_at_bottom`` set when
            lane pixels are available.
        """
        if lane_lines.lane_mask is None:
            logger.warning("compute_lane_center — lane_mask is missing")
            return LaneCenterData(center_line=None, center_x_at_bottom=None)

        lane_center_x = self.geometry.compute_lane_center(lane_lines.lane_mask)

        logger.debug(
            "compute_lane_center — lane_mask shape=%s, frame_shape=%s, center_x=%s",
            lane_lines.lane_mask.shape,
            frame_shape,
            lane_center_x,
        )

        return LaneCenterData(
            center_line=None,
            center_x_at_bottom=lane_center_x,
        )

    def compute_vehicle_offset(
        self,
        lane_center: LaneCenterData,
        frame_width: int,
        vehicle_x: int | None = None,
    ) -> VehicleOffsetData:
        """Compute lateral vehicle offset relative to the lane center.

        Uses :class:`LaneGeometryExtractor` with image center as the default
        ego position proxy.

        Args:
            lane_center: Computed lane center data.
            frame_width: Frame width in pixels.
            vehicle_x: Optional override for ego x-position.  When provided,
                offset is recomputed as ``vehicle_x - lane_center_x`` for
                compatibility with custom ego proxies.

        Returns:
            :class:`VehicleOffsetData` with signed pixel offset.
        """
        if lane_center.center_x_at_bottom is None or frame_width <= 0:
            logger.warning(
                "compute_vehicle_offset — missing lane center or invalid width"
            )
            return VehicleOffsetData(
                offset_pixels=None,
                vehicle_x=vehicle_x if vehicle_x is not None else frame_width // 2,
                lane_center_x=lane_center.center_x_at_bottom,
            )

        if vehicle_x is not None:
            vehicle_center_x = float(vehicle_x)
            offset_pixels = lane_center.center_x_at_bottom - vehicle_center_x
            logger.info(
                "Computed vehicle offset (custom ego) — offset=%.2f px",
                offset_pixels,
            )
            return VehicleOffsetData(
                offset_pixels=offset_pixels,
                vehicle_x=int(vehicle_center_x),
                lane_center_x=lane_center.center_x_at_bottom,
            )

        offset_result = self.geometry.compute_vehicle_offset(
            lane_center_x=lane_center.center_x_at_bottom,
            image_width=frame_width,
        )

        return VehicleOffsetData(
            offset_pixels=offset_result.offset_pixels,
            vehicle_x=int(offset_result.vehicle_center_x),
            lane_center_x=offset_result.lane_center_x,
        )

    def detect_lane_departure(
        self,
        vehicle_offset: VehicleOffsetData,
        threshold_px: float | None = None,
    ) -> LaneDepartureData:
        """Detect lane departure based on lateral vehicle offset.

        Args:
            vehicle_offset: Computed vehicle offset data.
            threshold_px: Departure threshold in pixels.  Uses parser config
                default when ``None``.

        Returns:
            :class:`LaneDepartureData` with departure flag and direction.
        """
        threshold = (
            threshold_px
            if threshold_px is not None
            else self.config.departure_threshold_px
        )

        logger.debug(
            "detect_lane_departure() TODO — offset=%s, threshold=%.1f",
            vehicle_offset.offset_pixels,
            threshold,
        )

        # TODO: Compare abs(offset_pixels) against threshold.
        # TODO: Set direction to "left" or "right" based on offset sign.
        return LaneDepartureData(is_departing=False, direction=None)

    # ------------------------------------------------------------------
    # Segmentation helpers
    # ------------------------------------------------------------------

    def _segmentation_to_binary_mask(self, segmentation: Any) -> np.ndarray:
        """Convert a multi-class segmentation tensor to a binary mask.

        Applies argmax across the class dimension for tensors shaped
        ``(C, H, W)`` or ``(B, C, H, W)``.  Foreground pixels are those
        with class index greater than ``foreground_class_threshold``.

        Args:
            segmentation: Raw segmentation logits or class map.

        Returns:
            Binary mask with values ``0`` or ``255`` and dtype ``uint8``.

        Raises:
            ValueError: If the segmentation array has an unsupported shape.
        """
        array = self._to_numpy(segmentation)

        if array.ndim == 4:
            array = array[0]

        if array.ndim == 3:
            class_map = np.argmax(array, axis=0)
        elif array.ndim == 2:
            class_map = array.astype(np.int64)
        else:
            raise ValueError(
                f"Unsupported segmentation tensor shape: {array.shape}. "
                "Expected (C, H, W), (B, C, H, W), or (H, W)."
            )

        binary_mask = (
            class_map > self.config.foreground_class_threshold
        ).astype(np.uint8) * 255

        logger.debug(
            "Converted segmentation %s -> binary mask %s",
            array.shape,
            binary_mask.shape,
        )
        return binary_mask

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Convert a torch tensor or array-like value to ``numpy.ndarray``."""
        if hasattr(value, "detach") and callable(value.detach):
            return value.detach().cpu().numpy()

        return np.asarray(value)

    @staticmethod
    def _get_output_at_index(raw_outputs: YOLOPRawOutput, index: int) -> Any:
        """Retrieve a YOLOP output tensor by standard head index.

        Supports sequence outputs (``raw_outputs[1]``, ``raw_outputs[2]``)
        and mapping outputs with integer or fallback string keys.

        Args:
            raw_outputs: Raw YOLOP container.
            index: Output index (``1`` drivable, ``2`` lane).

        Returns:
            Segmentation tensor for the requested head.

        Raises:
            IndexError: If the requested output index is unavailable.
        """
        if isinstance(raw_outputs, (list, tuple)):
            if index >= len(raw_outputs):
                raise IndexError(
                    f"raw_outputs index {index} out of range for sequence "
                    f"of length {len(raw_outputs)}"
                )
            return raw_outputs[index]

        if isinstance(raw_outputs, Mapping):
            if index in raw_outputs:
                return raw_outputs[index]

            fallback_keys = {
                DRIVABLE_AREA_OUTPUT_INDEX: ("drivable_mask", "drivable_head"),
                LANE_SEGMENTATION_OUTPUT_INDEX: ("lane_mask", "lane_head"),
            }
            for key in fallback_keys.get(index, ()):
                if key in raw_outputs and raw_outputs[key] is not None:
                    return raw_outputs[key]

            raise IndexError(
                f"raw_outputs missing index {index} and fallback keys "
                f"{fallback_keys.get(index, ())}"
            )

        raise ValueError(
            f"Unsupported raw_outputs container type: {type(raw_outputs).__name__}"
        )

    @staticmethod
    def _compute_coverage_ratio(mask: np.ndarray) -> float:
        """Compute foreground pixel ratio for a binary mask."""
        if mask.size == 0:
            return 0.0
        return float(np.count_nonzero(mask)) / float(mask.size)

    @staticmethod
    def _validate_raw_outputs(raw_outputs: YOLOPRawOutput) -> None:
        """Validate raw output container type.

        Args:
            raw_outputs: Candidate raw inference output.

        Raises:
            ValueError: If ``raw_outputs`` is not a sequence or mapping.
        """
        if not isinstance(raw_outputs, (Mapping, list, tuple)):
            raise ValueError(
                "raw_outputs must be a mapping or sequence, got "
                f"{type(raw_outputs).__name__}"
            )

    @staticmethod
    def _align_lane_lines_to_frame(
        lane_lines: LaneLineData,
        frame_height: int,
        frame_width: int,
    ) -> LaneLineData:
        """Resize lane mask to frame dimensions when model output differs."""
        if lane_lines.lane_mask is None or frame_height <= 0 or frame_width <= 0:
            return lane_lines

        return LaneLineData(
            left_lane=lane_lines.left_lane,
            right_lane=lane_lines.right_lane,
            lane_mask=resize_mask_to_frame(
                lane_lines.lane_mask,
                frame_height,
                frame_width,
            ),
        )

    @staticmethod
    def _resolve_frame_shape(
        raw_outputs: YOLOPRawOutput,
        frame_shape: FrameShape | None,
    ) -> FrameShape:
        """Resolve original frame shape from args or raw output metadata."""
        if frame_shape is not None:
            return frame_shape

        if isinstance(raw_outputs, Mapping):
            original_shape = raw_outputs.get("original_shape")
            if original_shape is not None:
                return tuple(original_shape)

        return (0, 0, 3)

    @staticmethod
    def _resolve_status(raw_outputs: YOLOPRawOutput) -> str:
        """Extract inference status string when available."""
        if isinstance(raw_outputs, Mapping):
            return str(raw_outputs.get("inference_status", "parsed"))
        return "parsed"

    @staticmethod
    def _get_optional_field(
        raw_outputs: YOLOPRawOutput,
        field_name: str,
    ) -> Any:
        """Read an optional metadata field from mapping-style outputs."""
        if isinstance(raw_outputs, Mapping):
            return raw_outputs.get(field_name)
        return None

    def __repr__(self) -> str:
        return (
            f"YOLOPOutputParser(departure_threshold="
            f"{self.config.departure_threshold_px})"
        )
