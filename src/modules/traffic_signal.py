"""Traffic Signal Detection module — YOLOv8 fine-tuned wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseModule, Frame
from .yolov8_signal.inference import (
    InferenceExecutionError,
    InferenceNotReadyError,
    InvalidFrameError,
    YOLOv8SignalInferenceConfig,
    YOLOv8SignalInferenceEngine,
)
from .yolov8_signal.model_loader import (
    WeightsLoadError,
    WeightsNotFoundError,
    WeightsValidationError,
    YOLOv8SignalModelLoader,
    resolve_variant_name,
)
from .yolov8_signal.output_parser import ParserConfig, YOLOv8SignalOutputParser
from .yolov8_signal.output_schema import (
    TRAFFIC_SIGNAL_OUTPUT_KEYS,
    TrafficSignalDetectionResult,
)
from ..utils.model_paths import get_traffic_signal_weights_path, get_yolov8_signal_config

__all__ = [
    "TRAFFIC_SIGNAL_OUTPUT_KEYS",
    "TrafficSignalModule",
]


class TrafficSignalModule(BaseModule):
    """Traffic signal state detection using Ultralytics YOLOv8 fine-tuned.

    Detects three ADAS signal states: red_light, yellow_light, green_light.

    Pipeline:
        Input Frame → YOLOv8 Signal Inference → Parse & Filter → TrafficSignalDetectionResult
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        model_loader: YOLOv8SignalModelLoader | None = None,
        inference_engine: YOLOv8SignalInferenceEngine | None = None,
        output_parser: YOLOv8SignalOutputParser | None = None,
        device: str | None = None,
        model_variant: str | None = None,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        imgsz: int | None = None,
    ) -> None:
        """Create a traffic signal module with injectable dependencies."""
        super().__init__(module_name="traffic_signal")

        signal_config = get_yolov8_signal_config()
        resolved_variant = resolve_variant_name(
            model_variant or signal_config.get("model_variant", "n")
        )
        resolved_confidence = float(
            confidence_threshold
            if confidence_threshold is not None
            else signal_config.get("confidence_threshold", 0.5)
        )
        resolved_iou = float(
            iou_threshold
            if iou_threshold is not None
            else signal_config.get("iou_threshold", 0.45)
        )
        resolved_imgsz = int(
            imgsz if imgsz is not None else signal_config.get("imgsz", 640)
        )
        resolved_device = device or signal_config.get("device", "cpu")

        resolved_weights = (
            Path(weights_path)
            if weights_path is not None
            else get_traffic_signal_weights_path()
        )

        self.weights_path = resolved_weights
        self.model_variant = resolved_variant
        self.confidence_threshold = resolved_confidence
        self.device = resolved_device

        self.model_loader = model_loader or YOLOv8SignalModelLoader(
            weights_path=resolved_weights,
            model_variant=resolved_variant,
            device=resolved_device,
        )
        self.inference_engine = inference_engine or YOLOv8SignalInferenceEngine(
            config=YOLOv8SignalInferenceConfig(
                imgsz=resolved_imgsz,
                confidence_threshold=resolved_confidence,
                iou_threshold=resolved_iou,
                device=resolved_device,
                max_det=int(signal_config.get("max_detections", 20)),
            )
        )
        self.output_parser = output_parser or YOLOv8SignalOutputParser(
            config=ParserConfig(confidence_threshold=resolved_confidence)
        )

        self._model_package: dict[str, Any] | None = None

        self._log_info(
            "TrafficSignalModule created — weights=%s, variant=yolov8%s, device=%s",
            self.weights_path,
            self.model_variant,
            self.device,
        )

    def initialize(self) -> None:
        """Load YOLOv8 signal weights and attach them to the inference engine."""
        self._log_info("Initializing traffic signal detection pipeline")

        try:
            metadata = self.model_loader.load_model()
            self._model_package = self.model_loader.get_model()
            self.inference_engine.attach_model(self._model_package)
            self._initialized = True

            self._log_info(
                "Traffic signal detection initialized — variant=yolov8%s, source=%s",
                metadata.model_variant,
                metadata.weights_path,
            )
        except (WeightsValidationError, WeightsLoadError, WeightsNotFoundError) as exc:
            self._log_error("Failed to initialize traffic signal detection: %s", exc)
            self._model_package = None
            self._initialized = False
            raise

    def predict(self, frame: Frame) -> TrafficSignalDetectionResult:
        """Run YOLOv8 traffic signal detection on a single frame."""
        if not self._initialized:
            self._log_warning("predict() called before initialize() — auto-initializing")
            try:
                self.initialize()
            except (WeightsValidationError, WeightsLoadError, WeightsNotFoundError) as exc:
                self._log_error("Auto-initialize failed: %s", exc)
                return TrafficSignalDetectionResult.empty(raw_status="init_failed")

        self._validate_input(frame)
        self._log_info("Running traffic signal detection pipeline on frame %s", frame.shape)

        try:
            return self._run_pipeline(frame)
        except (InvalidFrameError, InferenceExecutionError) as exc:
            self._log_error("Traffic signal detection pipeline failed: %s", exc)
            return TrafficSignalDetectionResult.empty(raw_status="pipeline_error")
        except Exception as exc:
            self._log_error("Unexpected traffic signal detection error: %s", exc)
            raise

    def visualize(
        self,
        frame: Frame,
        results: TrafficSignalDetectionResult | dict[str, Any],
    ) -> Frame:
        """Draw traffic signal bounding boxes and state labels on the frame."""
        from ..visualization.overlays import draw_traffic_signals

        if isinstance(results, TrafficSignalDetectionResult):
            payload = results.to_prediction_dict()
        else:
            payload = results

        return draw_traffic_signals(frame, payload)

    def cleanup(self) -> None:
        """Release model resources and reset module state."""
        self._log_info("Cleaning up traffic signal detection module")
        self.model_loader.unload()
        self.inference_engine.detach_model()
        self._model_package = None
        self._initialized = False

    def _run_pipeline(self, frame: Frame) -> TrafficSignalDetectionResult:
        """Execute the connected traffic signal detection workflow."""
        if not self.inference_engine.is_ready:
            self._log_warning("Inference engine not ready — returning empty result")
            return TrafficSignalDetectionResult.empty(raw_status="inference_not_ready")

        try:
            raw_outputs = self.inference_engine.run(frame)
        except InferenceNotReadyError:
            return TrafficSignalDetectionResult.empty(raw_status="inference_not_ready")

        parsed = self.output_parser.parse(raw_outputs, frame_shape=frame.shape)
        summary = self.output_parser.build_summary(
            parsed.detections,
            frame_shape=frame.shape,
        )

        result = TrafficSignalDetectionResult(
            detections=parsed.detections,
            summary=summary,
            frame_shape=(int(frame.shape[0]), int(frame.shape[1])),
            inference_time_ms=raw_outputs.get("inference_time_ms"),
            model_variant=self.model_variant,
            confidence_threshold=self.confidence_threshold,
            raw_status=parsed.raw_status,
        )

        self._assert_detections_within_frame(result, frame)
        self._log_info(
            "Traffic signal detection complete — count=%d, dominant=%s, status=%s",
            result.summary.total_count,
            result.summary.dominant_state,
            result.raw_status,
        )
        return result

    @staticmethod
    def _assert_detections_within_frame(
        result: TrafficSignalDetectionResult,
        frame: Frame,
    ) -> None:
        """Fail fast if any bounding box lies outside the input frame."""
        frame_height = int(frame.shape[0])
        frame_width = int(frame.shape[1])

        for det in result.detections:
            bbox = det.bbox
            if not (0 <= bbox.x1 < bbox.x2 <= frame_width):
                raise RuntimeError(
                    f"Detection bbox x out of frame bounds: {bbox.to_list()} "
                    f"for frame width {frame_width}"
                )
            if not (0 <= bbox.y1 < bbox.y2 <= frame_height):
                raise RuntimeError(
                    f"Detection bbox y out of frame bounds: {bbox.to_list()} "
                    f"for frame height {frame_height}"
                )

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
    def empty_prediction() -> TrafficSignalDetectionResult:
        """Return an empty traffic signal detection result."""
        return TrafficSignalDetectionResult.empty()
