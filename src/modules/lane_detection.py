"""Lane Detection module — end-to-end YOLOP pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseModule, Frame
from .yolop.inference import (
    InferenceExecutionError,
    InferenceNotReadyError,
    InvalidFrameError,
    YOLOPInferenceEngine,
)
from .yolop.lane_geometry import LaneGeometryExtractor
from .yolop.model_loader import (
    CheckpointLoadError,
    CheckpointNotFoundError,
    CheckpointValidationError,
    YOLOPModelLoader,
)
from .yolop.output_parser import YOLOPOutputParser
from .yolop.output_schema import LaneDetectionResult
from .yolop.postprocess import postprocess_lane_mask, resize_mask_to_frame
from ..preprocessing.lane_preprocess import LanePreprocessor
from ..utils.model_paths import get_yolop_weights_path

LANE_OUTPUT_KEYS = (
    "left_lane",
    "right_lane",
    "lane_center",
    "vehicle_offset",
    "lane_departure",
)


class LaneDetectionModule(BaseModule):
    """Lane detection module using the full YOLOP inference pipeline.

    Connects preprocessing, model loading, inference, output parsing, and
    lane geometry extraction into a single end-to-end workflow:

        Input Frame → Preprocess → YOLOP Inference → Parse Outputs
        → Extract Lane Geometry → LaneDetectionResult

    Attributes:
        weights_path: Resolved filesystem path to YOLOP checkpoint.
        preprocessor: Lane-specific OpenCV preprocessing helper.
        model_loader: YOLOP checkpoint loader.
        inference_engine: YOLOP inference wrapper.
        output_parser: Raw output parser for segmentation masks.
        geometry_extractor: Lane center and offset calculator.
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        preprocessor: LanePreprocessor | None = None,
        model_loader: YOLOPModelLoader | None = None,
        inference_engine: YOLOPInferenceEngine | None = None,
        output_parser: YOLOPOutputParser | None = None,
        geometry_extractor: LaneGeometryExtractor | None = None,
        device: str = "cpu",
        apply_mask_postprocess: bool = True,
    ) -> None:
        """Create a lane detection module with injectable dependencies.

        Args:
            weights_path: Optional override for YOLOP weights location.
            preprocessor: Optional ``LanePreprocessor`` instance.
            model_loader: Optional ``YOLOPModelLoader`` instance.
            inference_engine: Optional ``YOLOPInferenceEngine`` instance.
            output_parser: Optional ``YOLOPOutputParser`` instance.
            geometry_extractor: Optional ``LaneGeometryExtractor`` instance.
            device: Target device string for model loading.
            apply_mask_postprocess: Whether to run morphological/connected
                component refinement on lane masks before geometry extraction.
        """
        super().__init__(module_name="lane_detection")

        resolved_weights = Path(weights_path) if weights_path else get_yolop_weights_path()

        self.weights_path = resolved_weights
        self.preprocessor = preprocessor or LanePreprocessor()
        self.model_loader = model_loader or YOLOPModelLoader(
            weights_path=resolved_weights,
            device=device,
        )
        self.inference_engine = inference_engine or YOLOPInferenceEngine()
        self.output_parser = output_parser or YOLOPOutputParser()
        self.geometry_extractor = geometry_extractor or LaneGeometryExtractor()
        self.apply_mask_postprocess = apply_mask_postprocess

        self._model_package: dict[str, Any] | None = None

        self._log_info(
            "LaneDetectionModule created — weights=%s, device=%s",
            self.weights_path,
            device,
        )

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load YOLOP weights and attach them to the inference engine.

        Raises:
            CheckpointNotFoundError: If the checkpoint file is missing.
            CheckpointValidationError: If the checkpoint structure is invalid.
            CheckpointLoadError: If PyTorch cannot load the checkpoint.
        """
        self._log_info("Initializing lane detection pipeline")

        try:
            metadata = self.model_loader.load_model()
            self._model_package = self.model_loader.get_model()
            self.inference_engine.attach_model(self._model_package)
            self._initialized = True

            self._log_info(
                "Lane detection initialized — checkpoint format=%s, tensors=%d",
                metadata.checkpoint_format,
                metadata.num_tensor_keys,
            )
        except (
            CheckpointNotFoundError,
            CheckpointValidationError,
            CheckpointLoadError,
        ) as exc:
            self._log_error("Failed to initialize lane detection: %s", exc)
            self._model_package = None
            self._initialized = False
            raise

    def predict(self, frame: Frame) -> LaneDetectionResult:
        """Run the end-to-end lane detection pipeline on a single frame.

        Pipeline:
            1. Validate input frame
            2. Preprocess with ``LanePreprocessor``
            3. Run ``YOLOPInferenceEngine``
            4. Parse outputs with ``YOLOPOutputParser``
            5. Extract geometry with ``LaneGeometryExtractor``
            6. Return ``LaneDetectionResult``

        Args:
            frame: BGR input image with shape ``(H, W, 3)``.

        Returns:
            Standardized :class:`LaneDetectionResult`.

        Raises:
            ValueError: If the input frame is invalid.
            RuntimeError: If the module has not been initialized.
        """
        if not self._initialized:
            self._log_warning("predict() called before initialize() — auto-initializing")
            try:
                self.initialize()
            except (
                CheckpointNotFoundError,
                CheckpointValidationError,
                CheckpointLoadError,
            ) as exc:
                self._log_error("Auto-initialize failed: %s", exc)
                return LaneDetectionResult.empty(raw_status="init_failed")

        self._validate_input(frame)
        self._log_info("Running lane detection pipeline on frame %s", frame.shape)

        try:
            return self._run_pipeline(frame)
        except (InvalidFrameError, InferenceExecutionError) as exc:
            self._log_error("Lane detection pipeline failed: %s", exc)
            return LaneDetectionResult.empty(raw_status="pipeline_error")
        except Exception as exc:
            self._log_error("Unexpected lane detection error: %s", exc)
            raise

    def visualize(self, frame: Frame, results: LaneDetectionResult | dict[str, Any]) -> Frame:
        """Return a copy of the frame (visualization not implemented)."""
        return frame.copy()

    def cleanup(self) -> None:
        """Release model resources and reset module state."""
        self._log_info("Cleaning up lane detection module")
        self.model_loader.unload()
        self.inference_engine.detach_model()
        self._model_package = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self, frame: Frame) -> LaneDetectionResult:
        """Execute the connected lane detection workflow."""
        # Step 1: Classical preprocessing (edges + ROI)
        preprocessed_edges = self.preprocessor.preprocess(frame)
        self._log_debug("Preprocessed edges shape: %s", preprocessed_edges.shape)

        # Step 2: YOLOP inference
        if not self.inference_engine.is_ready:
            self._log_warning("Inference engine not ready — returning empty result")
            return LaneDetectionResult.empty(
                raw_status="inference_not_ready",
                preprocessed_edges=preprocessed_edges,
            )

        try:
            raw_outputs = self.inference_engine.run(frame)
        except InferenceNotReadyError:
            return LaneDetectionResult.empty(
                raw_status="inference_not_ready",
                preprocessed_edges=preprocessed_edges,
            )

        self._log_debug("YOLOP inference complete — status=%s", raw_outputs.get("inference_status"))

        # Step 3: Parse segmentation outputs
        parsed = self.output_parser.parse(raw_outputs, frame_shape=frame.shape)
        lane_mask = parsed.lane_lines.lane_mask
        drivable_mask = parsed.drivable_area.mask

        # Step 4: Optional mask post-processing (model resolution)
        if lane_mask is not None and self.apply_mask_postprocess:
            lane_mask = postprocess_lane_mask(lane_mask)
            self._log_debug("Lane mask post-processing applied")

        # Step 5: Resize masks to original frame dimensions before geometry.
        # Must use frame.shape — not YOLOP input_size (640x640).
        frame_height = int(frame.shape[0])
        frame_width = int(frame.shape[1])
        target_mask_shape = (frame_height, frame_width)

        self._log_debug(
            "Resizing masks from %s to frame shape %s",
            lane_mask.shape if lane_mask is not None else None,
            target_mask_shape,
        )
        lane_mask = resize_mask_to_frame(
            lane_mask,
            frame_height=frame_height,
            frame_width=frame_width,
        )
        drivable_mask = resize_mask_to_frame(
            drivable_mask,
            frame_height=frame_height,
            frame_width=frame_width,
        )
        if lane_mask is not None and lane_mask.shape[:2] != target_mask_shape:
            raise RuntimeError(
                f"Lane mask resize failed: got {lane_mask.shape[:2]}, "
                f"expected {target_mask_shape}"
            )
        if drivable_mask is not None and drivable_mask.shape[:2] != target_mask_shape:
            raise RuntimeError(
                f"Drivable mask resize failed: got {drivable_mask.shape[:2]}, "
                f"expected {target_mask_shape}"
            )

        # Step 6: Lane geometry extraction (frame-space coordinates)
        lane_center_x: float | None = None
        vehicle_center_x: float | None = None
        vehicle_offset: float | None = None
        lane_departure = False

        if lane_mask is not None:
            lane_center_x = self.geometry_extractor.compute_lane_center(lane_mask)
            if lane_center_x is not None:
                offset_result = self.geometry_extractor.compute_vehicle_offset(
                    lane_center_x=lane_center_x,
                    image_width=frame_width,
                )
                vehicle_center_x = offset_result.vehicle_center_x
                vehicle_offset = offset_result.offset_pixels
                lane_departure = (
                    abs(vehicle_offset)
                    > self.output_parser.config.departure_threshold_px
                )

        result = LaneDetectionResult(
            left_lane=parsed.lane_lines.left_lane,
            right_lane=parsed.lane_lines.right_lane,
            lane_center_x=lane_center_x,
            vehicle_center_x=vehicle_center_x,
            vehicle_offset=vehicle_offset,
            lane_mask=lane_mask,
            drivable_mask=drivable_mask,
            lane_departure=lane_departure,
            preprocessed_edges=preprocessed_edges,
            raw_status=parsed.raw_status,
        )

        self._log_info(
            "Lane detection complete — center_x=%s, offset=%s px",
            result.lane_center_x,
            result.vehicle_offset,
        )
        return result

    def _validate_input(self, frame: Frame) -> None:
        """Validate input frame format."""
        if frame is None:
            raise ValueError("Input frame is None")

        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Input frame must be numpy.ndarray, got {type(frame)}")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Input frame must have shape (H, W, 3), got {frame.shape}")

        if frame.size == 0:
            raise ValueError("Input frame is empty")

        if frame.dtype != np.uint8:
            self._log_warning("Input frame dtype is %s — expected uint8", frame.dtype)

    @staticmethod
    def empty_prediction() -> LaneDetectionResult:
        """Return an empty lane detection result."""
        return LaneDetectionResult.empty()
