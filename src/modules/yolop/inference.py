"""YOLOP inference wrapper — integration layer for forward-pass execution.

Provides a production-ready interface for preprocessing frames, executing
YOLOP inference via the vendored MCnet architecture, and post-processing raw
multi-head outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.modules.yolop.vendor import get_net

logger = logging.getLogger("adas.modules.yolop.inference")

Frame = np.ndarray
ModelPackage = dict[str, Any]
PreprocessedInput = dict[str, Any]
RawForwardOutput = dict[str, Any]
YOLOPRawOutput = dict[str, Any]

# YOLOP default input resolution (width, height)
DEFAULT_INPUT_SIZE = (640, 640)


class InferenceNotReadyError(RuntimeError):
    """Raised when inference is attempted without an attached model package."""


class InvalidFrameError(ValueError):
    """Raised when an input frame fails validation."""


class InferenceExecutionError(RuntimeError):
    """Raised when the forward pass fails unexpectedly."""


class StateDictMissingError(ValueError):
    """Raised when a model package does not contain weight tensors."""


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for YOLOP inference preprocessing and postprocessing.

    Attributes:
        input_size: Model input resolution as ``(width, height)``.
        confidence_threshold: Detection confidence cutoff for object head.
        device: Target inference device string.
    """

    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE
    confidence_threshold: float = 0.5
    device: str = "cpu"


class YOLOPInferenceEngine:
    """Execute YOLOP inference on video frames.

    Accepts a loaded model package from ``YOLOPModelLoader.get_model()``,
    validates and preprocesses BGR frames, runs the vendored MCnet forward
    pass, and returns standardized raw outputs for downstream parsing.

    Attributes:
        config: Inference configuration parameters.
        model_package: Loaded checkpoint package (or ``None``).
        _model: Instantiated MCnet module when architecture is ready.
    """

    def __init__(
        self,
        model_package: ModelPackage | None = None,
        config: InferenceConfig | None = None,
    ) -> None:
        """Create a YOLOP inference engine.

        Args:
            model_package: Optional model package from ``YOLOPModelLoader``.
            config: Optional inference configuration.  Uses defaults when
                ``None``.
        """
        self.config = config or InferenceConfig()
        self.model_package: ModelPackage | None = None
        self._model: nn.Module | None = None
        self._architecture_ready = False

        if model_package is not None:
            self.attach_model(model_package)

        logger.info(
            "YOLOPInferenceEngine created — model=%s, input_size=%s",
            "attached" if self.is_ready else "none",
            self.config.input_size,
        )

    # ------------------------------------------------------------------
    # Model attachment
    # ------------------------------------------------------------------

    def attach_model(self, model_package: ModelPackage) -> None:
        """Attach a loaded YOLOP model package for inference.

        Instantiates the vendored MCnet architecture, loads checkpoint weights,
        and moves the network to the configured inference device.

        Args:
            model_package: Dictionary from ``YOLOPModelLoader.get_model()``
                containing ``checkpoint``, ``state_dict``, and ``metadata``.

        Raises:
            ValueError: If the package is missing required keys.
            StateDictMissingError: If ``state_dict`` is missing or empty.
            InferenceExecutionError: If model construction or weight loading fails.
        """
        required_keys = {"checkpoint", "state_dict", "metadata"}
        missing = required_keys - set(model_package.keys())
        if missing:
            raise ValueError(
                f"Invalid model package — missing keys: {sorted(missing)}"
            )

        state_dict = model_package.get("state_dict")
        if not state_dict:
            raise StateDictMissingError(
                "Invalid model package — state_dict is missing or empty"
            )

        self.model_package = model_package
        self._model = None
        self._architecture_ready = False

        device = self._resolve_device()
        logger.info("Instantiating vendored MCnet via get_net()")
        try:
            model = get_net(cfg=None)
        except Exception as exc:
            logger.exception("YOLOP MCnet instantiation failed")
            raise InferenceExecutionError(
                f"YOLOP MCnet instantiation failed: {exc}"
            ) from exc

        logger.info(
            "Loading YOLOP state_dict — %d tensor(s)",
            len(state_dict),
        )
        try:
            load_result = model.load_state_dict(state_dict, strict=True)
        except RuntimeError as strict_exc:
            logger.warning(
                "Strict state_dict load failed (%s) — retrying with strict=False",
                strict_exc,
            )
            load_result = model.load_state_dict(state_dict, strict=False)

        if load_result.missing_keys or load_result.unexpected_keys:
            logger.warning(
                "state_dict load — missing=%s unexpected=%s",
                load_result.missing_keys,
                load_result.unexpected_keys,
            )
        else:
            logger.info("state_dict loaded successfully with strict=True")

        logger.info("Moving MCnet to device %s", device)
        model = model.to(device)
        model.eval()

        self._model = model
        self._architecture_ready = True
        logger.info(
            "YOLOP model attached — architecture_ready=True, device=%s",
            device,
        )

    def detach_model(self) -> None:
        """Detach the current model package and release references."""
        logger.info("Detaching YOLOP model package from inference engine")
        self.model_package = None
        self._model = None
        self._architecture_ready = False

    @property
    def is_ready(self) -> bool:
        """Return whether a model package is attached and architecture is loaded."""
        return (
            self.model_package is not None
            and self._model is not None
            and self._architecture_ready
        )

    # ------------------------------------------------------------------
    # Public inference pipeline
    # ------------------------------------------------------------------

    def preprocess(self, frame: Frame) -> PreprocessedInput:
        """Convert a BGR frame into model-ready format.

        Performs validation, resizing, color conversion, normalization, and
        channel reordering to NCHW layout.  Returns a dictionary that can be
        passed to the forward pass once the YOLOP architecture is integrated.

        Args:
            frame: BGR input image with shape ``(H, W, 3)``.

        Returns:
            Dictionary containing:
                - ``input_tensor``: ``float32`` array shaped ``(1, 3, H, W)``
                - ``original_shape``: ``(height, width, channels)``
                - ``input_size``: configured model input size
                - ``scale_x``, ``scale_y``: resize scale factors

        Raises:
            InvalidFrameError: If the frame is invalid.
        """
        self._validate_frame(frame)
        original_shape = frame.shape
        target_width, target_height = self.config.input_size

        resized = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0)

        scale_x = target_width / frame.shape[1]
        scale_y = target_height / frame.shape[0]

        logger.debug(
            "Preprocessed frame %s -> tensor %s",
            original_shape,
            input_tensor.shape,
        )

        return {
            "input_tensor": input_tensor,
            "original_shape": original_shape,
            "input_size": self.config.input_size,
            "scale_x": scale_x,
            "scale_y": scale_y,
        }

    def run(self, frame: Frame) -> YOLOPRawOutput:
        """Run the full inference pipeline on a single frame.

        Orchestrates ``preprocess`` → forward pass → ``postprocess``.

        Args:
            frame: BGR input image.

        Returns:
            Standardized raw YOLOP output dictionary.

        Raises:
            InferenceNotReadyError: If no model package is attached.
            InvalidFrameError: If the frame fails validation.
            InferenceExecutionError: If the forward pass fails.
        """
        if not self.is_ready:
            raise InferenceNotReadyError(
                "YOLOP inference engine is not ready — attach a model package first"
            )

        self._validate_frame(frame)
        logger.info("Running YOLOP inference pipeline")

        preprocessed = self.preprocess(frame)
        raw_forward = self._execute_forward(preprocessed)
        results = self.postprocess(raw_forward, preprocessed)

        logger.info("YOLOP inference pipeline complete")
        return results

    def postprocess(
        self,
        outputs: RawForwardOutput,
        preprocessed: PreprocessedInput,
    ) -> YOLOPRawOutput:
        """Convert raw forward-pass outputs into standardized YOLOP results.

        Placeholder: maps stub keys to the expected output schema.  Full
        decoding of lane/drivable/detection heads will be added when YOLOP
        architecture integration is complete.

        Args:
            outputs: Raw tensors from the forward pass.
            preprocessed: Preprocessing metadata from ``preprocess()``.

        Returns:
            Dictionary with ``lane_mask``, ``drivable_mask``, ``detections``,
            and diagnostic metadata.
        """
        logger.debug("Postprocessing YOLOP raw outputs")

        # TODO: Decode lane segmentation head to binary lane mask.
        # TODO: Decode drivable area head to binary mask.
        # TODO: Decode detection head with confidence_threshold filtering.
        # TODO: Resize masks back to original_shape using preprocessed scale factors.

        return {
            "lane_mask": outputs.get("lane_head"),
            "drivable_mask": outputs.get("drivable_head"),
            "detections": outputs.get("detection_head"),
            "confidence_threshold": self.config.confidence_threshold,
            "original_shape": preprocessed.get("original_shape"),
            "input_size": preprocessed.get("input_size"),
            "inference_status": outputs.get("status", "unknown"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_frame(frame: Frame) -> None:
        """Validate input frame format.

        Args:
            frame: Candidate BGR frame.

        Raises:
            InvalidFrameError: If validation fails.
        """
        if frame is None:
            raise InvalidFrameError("Input frame is None")

        if not isinstance(frame, np.ndarray):
            raise InvalidFrameError(
                f"Input frame must be numpy.ndarray, got {type(frame).__name__}"
            )

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise InvalidFrameError(
                f"Input frame must have shape (H, W, 3), got {frame.shape}"
            )

        if frame.size == 0:
            raise InvalidFrameError("Input frame is empty")

    def _execute_forward(self, preprocessed: PreprocessedInput) -> RawForwardOutput:
        """Execute the YOLOP forward pass on preprocessed input.

        Args:
            preprocessed: Model-ready input from ``preprocess()``.

        Returns:
            Raw forward output dictionary with lane, drivable, and detection heads.

        Raises:
            InferenceNotReadyError: If the MCnet model is not attached.
            InferenceExecutionError: If forward execution fails.
        """
        if self._model is None or not self._architecture_ready:
            raise InferenceNotReadyError(
                "YOLOP MCnet model is not attached — call attach_model() first"
            )

        device = self._resolve_device()
        input_array = preprocessed["input_tensor"]
        logger.info(
            "Executing YOLOP forward pass — input_shape=%s, device=%s",
            tuple(input_array.shape),
            device,
        )

        try:
            input_tensor = torch.as_tensor(
                input_array,
                dtype=torch.float32,
                device=device,
            )
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)

            with torch.no_grad():
                outputs = self._model(input_tensor)

            det_out, drivable_head, lane_head = outputs
            logger.info(
                "YOLOP forward pass complete — det=%s, drivable=%s, lane=%s",
                type(det_out).__name__,
                tuple(drivable_head.shape),
                tuple(lane_head.shape),
            )

            return {
                "lane_head": lane_head,
                "drivable_head": drivable_head,
                "detection_head": det_out,
                "status": "ok",
                "raw_outputs": outputs,
            }
        except InferenceNotReadyError:
            raise
        except Exception as exc:
            logger.exception("YOLOP forward pass failed")
            raise InferenceExecutionError(f"YOLOP forward pass failed: {exc}") from exc

    def _resolve_device(self) -> torch.device:
        """Resolve the configured inference device string to ``torch.device``."""
        device_name = self.config.device
        if device_name == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable — falling back to CPU")
            return torch.device("cpu")
        return torch.device(device_name)

    @staticmethod
    def empty_output() -> YOLOPRawOutput:
        """Return an empty standardized YOLOP output structure."""
        return {
            "lane_mask": None,
            "drivable_mask": None,
            "detections": None,
            "confidence_threshold": None,
            "original_shape": None,
            "input_size": None,
            "inference_status": "empty",
        }

    def __repr__(self) -> str:
        return (
            f"YOLOPInferenceEngine(ready={self.is_ready}, "
            f"input_size={self.config.input_size}, "
            f"device={self.config.device!r})"
        )


class YOLOPInference(YOLOPInferenceEngine):
    """Backward-compatible alias for :class:`YOLOPInferenceEngine`.

    Provides ``predict()`` and ``set_model()`` method names used by earlier
    integration scaffolding.
    """

    def predict(self, frame: Frame) -> YOLOPRawOutput:
        """Run inference — alias for :meth:`run`."""
        return self.run(frame)

    def set_model(self, model_package: ModelPackage) -> None:
        """Attach model package — alias for :meth:`attach_model`."""
        self.attach_model(model_package)
