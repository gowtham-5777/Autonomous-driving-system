"""Abstract base class for all ADAS perception modules.

Defines the shared interface that every detection/recognition module must
implement: initialization, inference, visualization, and cleanup.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# Type alias for a single video frame (OpenCV / NumPy BGR image).
Frame = np.ndarray

# Type alias for structured module output returned by ``predict``.
PredictionResult = dict[str, Any]


class BaseModule(ABC):
    """Abstract base class for ADAS perception modules.

    Every module in the pipeline (lane detection, object detection, traffic
    signs, traffic signals, segmentation) must inherit from this class and
    implement the four lifecycle methods.

    Attributes:
        module_name: Human-readable identifier for this module.
        logger: Module-scoped logger for structured output.
        _initialized: Whether ``initialize`` has completed successfully.
    """

    def __init__(self, module_name: str) -> None:
        """Create a module instance.

        Args:
            module_name: Unique name used for logging and diagnostics
                (e.g. ``"lane_detection"``).
        """
        self.module_name: str = module_name
        self.logger: logging.Logger = logging.getLogger(
            f"adas.modules.{module_name}"
        )
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Logging hooks (available to all subclasses)
    # ------------------------------------------------------------------

    def _log_info(self, message: str, *args: Any) -> None:
        """Emit an INFO-level log message prefixed with the module name."""
        self.logger.info("[%s] " + message, self.module_name, *args)

    def _log_debug(self, message: str, *args: Any) -> None:
        """Emit a DEBUG-level log message prefixed with the module name."""
        self.logger.debug("[%s] " + message, self.module_name, *args)

    def _log_warning(self, message: str, *args: Any) -> None:
        """Emit a WARNING-level log message prefixed with the module name."""
        self.logger.warning("[%s] " + message, self.module_name, *args)

    def _log_error(self, message: str, *args: Any) -> None:
        """Emit an ERROR-level log message prefixed with the module name."""
        self.logger.error("[%s] " + message, self.module_name, *args)

    def _on_before_initialize(self) -> None:
        """Hook called immediately before ``initialize`` body runs."""
        self._log_debug("Initializing module")

    def _on_after_initialize(self) -> None:
        """Hook called immediately after ``initialize`` completes."""
        self._initialized = True
        self._log_info("Module initialized successfully")

    def _on_before_predict(self, frame: Frame) -> None:
        """Hook called immediately before ``predict`` body runs.

        Args:
            frame: Input frame about to be processed.
        """
        self._log_debug(
            "Starting prediction on frame shape %s",
            getattr(frame, "shape", "unknown"),
        )

    def _on_after_predict(self, results: PredictionResult) -> None:
        """Hook called immediately after ``predict`` completes.

        Args:
            results: Dictionary returned by ``predict``.
        """
        self._log_debug(
            "Prediction complete — %d result key(s)",
            len(results),
        )

    def _on_before_cleanup(self) -> None:
        """Hook called immediately before ``cleanup`` body runs."""
        self._log_debug("Cleaning up module resources")

    def _on_after_cleanup(self) -> None:
        """Hook called immediately after ``cleanup`` completes."""
        self._initialized = False
        self._log_info("Module cleanup complete")

    @property
    def is_initialized(self) -> bool:
        """Return whether the module has been initialized."""
        return self._initialized

    # ------------------------------------------------------------------
    # Abstract interface (must be implemented by every subclass)
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """Load model weights and prepare the module for inference.

        Called once before the first ``predict`` call.  Subclasses should
        load checkpoints, warm up the model, and validate resources here.
        """

    @abstractmethod
    def predict(self, frame: Frame) -> PredictionResult:
        """Run inference on a single frame.

        Args:
            frame: BGR image as a NumPy array with shape ``(H, W, 3)``.

        Returns:
            Dictionary containing structured detection/recognition outputs.
            Keys and values are module-specific but must be JSON-serializable
            where possible (e.g. bounding boxes as lists, labels as strings).
        """

    @abstractmethod
    def visualize(self, frame: Frame, results: PredictionResult) -> Frame:
        """Draw module-specific overlays on the frame.

        Args:
            frame: Original BGR frame (will not be mutated in-place).
            results: Output dictionary from ``predict``.

        Returns:
            Annotated copy of ``frame`` with visual overlays applied.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release model weights, GPU memory, and other resources.

        Called when the module is no longer needed (e.g. app shutdown).
        """

    # ------------------------------------------------------------------
    # Convenience wrappers that invoke hooks around abstract methods
    # ------------------------------------------------------------------

    def run_initialize(self) -> None:
        """Execute ``initialize`` with lifecycle logging hooks."""
        self._on_before_initialize()
        self.initialize()
        self._on_after_initialize()

    def run_predict(self, frame: Frame) -> PredictionResult:
        """Execute ``predict`` with lifecycle logging hooks.

        Args:
            frame: BGR input frame.

        Returns:
            Prediction result dictionary from ``predict``.
        """
        if not self._initialized:
            self._log_warning("predict() called before initialize() — auto-initializing")
            self.run_initialize()

        self._on_before_predict(frame)
        results = self.predict(frame)
        self._on_after_predict(results)
        return results

    def run_cleanup(self) -> None:
        """Execute ``cleanup`` with lifecycle logging hooks."""
        self._on_before_cleanup()
        self.cleanup()
        self._on_after_cleanup()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"{self.__class__.__name__}(module_name={self.module_name!r}, {status})"
