"""YOLOv8 sign inference wrapper — forward-pass execution layer."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("adas.modules.yolov8_sign.inference")

Frame = np.ndarray
ModelPackage = dict[str, Any]
YOLOv8SignRawOutput = dict[str, Any]

DEFAULT_IMGSZ = 640


class InferenceNotReadyError(RuntimeError):
    """Raised when inference is attempted without an attached model."""


class InvalidFrameError(ValueError):
    """Raised when an input frame fails validation."""


class InferenceExecutionError(RuntimeError):
    """Raised when the YOLOv8 sign forward pass fails unexpectedly."""


@dataclass(frozen=True)
class YOLOv8SignInferenceConfig:
    """Configuration for YOLOv8 sign inference."""

    imgsz: int = DEFAULT_IMGSZ
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cpu"
    max_det: int = 50
    half: bool = False
    verbose: bool = False


class YOLOv8SignInferenceEngine:
    """Execute YOLOv8 traffic sign detection on video frames."""

    def __init__(
        self,
        model_package: ModelPackage | None = None,
        config: YOLOv8SignInferenceConfig | None = None,
    ) -> None:
        self.config = config or YOLOv8SignInferenceConfig()
        self.model_package: ModelPackage | None = None
        self._model: Any | None = None

        if model_package is not None:
            self.attach_model(model_package)

        logger.info(
            "YOLOv8SignInferenceEngine created — ready=%s, imgsz=%d",
            self.is_ready,
            self.config.imgsz,
        )

    @property
    def is_ready(self) -> bool:
        """Return whether a model is attached and ready for inference."""
        return self._model is not None

    def attach_model(self, model_package: ModelPackage) -> None:
        """Attach a loaded YOLOv8 sign model package."""
        required_keys = {"model", "metadata"}
        missing = required_keys - set(model_package.keys())
        if missing:
            raise ValueError(
                f"Invalid YOLOv8 sign model package — missing keys: {sorted(missing)}"
            )

        model = model_package.get("model")
        if model is None:
            raise ValueError("Invalid YOLOv8 sign model package — model is None")

        self.model_package = model_package
        self._model = model
        logger.info("YOLOv8 sign model attached to inference engine")

    def detach_model(self) -> None:
        """Detach the current model package."""
        logger.info("Detaching YOLOv8 sign model from inference engine")
        self.model_package = None
        self._model = None

    def run(self, frame: Frame) -> YOLOv8SignRawOutput:
        """Run YOLOv8 sign detection on a single BGR frame."""
        if not self.is_ready:
            raise InferenceNotReadyError(
                "YOLOv8 sign inference engine is not ready — attach a model package first"
            )

        self._validate_frame(frame)
        logger.info("Running YOLOv8 sign inference on frame shape %s", frame.shape)

        start = time.perf_counter()
        try:
            results = self._model.predict(
                source=frame,
                imgsz=self.config.imgsz,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                device=self.config.device,
                max_det=self.config.max_det,
                half=self.config.half,
                verbose=self.config.verbose,
            )
        except Exception as exc:
            logger.exception("YOLOv8 sign forward pass failed")
            raise InferenceExecutionError(
                f"YOLOv8 sign forward pass failed: {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return self._build_raw_output(
            results=results,
            frame=frame,
            inference_time_ms=elapsed_ms,
        )

    def _build_raw_output(
        self,
        results: Any,
        frame: Frame,
        inference_time_ms: float,
    ) -> YOLOv8SignRawOutput:
        """Convert Ultralytics results into a parser-friendly raw dictionary."""
        if not results:
            return self.empty_output(
                original_shape=frame.shape,
                inference_time_ms=inference_time_ms,
            )

        result = results[0]
        boxes_xyxy = np.empty((0, 4), dtype=np.float32)
        confidences = np.empty((0,), dtype=np.float32)
        class_ids = np.empty((0,), dtype=np.int32)

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
            class_ids = result.boxes.cls.cpu().numpy().astype(np.int32)

        return {
            "boxes_xyxy": boxes_xyxy,
            "confidences": confidences,
            "class_ids": class_ids,
            "inference_status": "ok",
            "original_shape": frame.shape,
            "inference_time_ms": inference_time_ms,
            "imgsz": self.config.imgsz,
        }

    @staticmethod
    def _validate_frame(frame: Frame) -> None:
        """Validate input frame format."""
        if frame is None:
            raise InvalidFrameError("Input frame is None")

        if not isinstance(frame, np.ndarray):
            raise InvalidFrameError(
                f"Input frame must be numpy.ndarray, got {type(frame)}"
            )

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise InvalidFrameError(
                f"Input frame must have shape (H, W, 3), got {frame.shape}"
            )

        if frame.size == 0:
            raise InvalidFrameError("Input frame is empty")

    @staticmethod
    def empty_output(
        original_shape: tuple[int, ...] | None = None,
        inference_time_ms: float | None = None,
    ) -> YOLOv8SignRawOutput:
        """Return an empty standardized raw output structure."""
        return {
            "boxes_xyxy": np.empty((0, 4), dtype=np.float32),
            "confidences": np.empty((0,), dtype=np.float32),
            "class_ids": np.empty((0,), dtype=np.int32),
            "inference_status": "empty",
            "original_shape": original_shape,
            "inference_time_ms": inference_time_ms,
            "imgsz": DEFAULT_IMGSZ,
        }
