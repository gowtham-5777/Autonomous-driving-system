"""YOLOP inference wrapper — architecture placeholder.

This module will provide a clean inference interface around the YOLOP
multi-task model (lane detection, drivable area, object detection heads).
No inference logic is implemented yet.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("adas.modules.yolop.inference")

Frame = np.ndarray
YOLOPRawOutput = dict[str, Any]


class YOLOPInference:
    """Run forward-pass inference with a loaded YOLOP model.

    Placeholder class defining the inference API.  Will accept a loaded
    model from ``YOLOPModelLoader`` and return raw multi-head outputs.

    Attributes:
        model: Reference to the loaded YOLOP model (may be ``None``).
        confidence_threshold: Minimum confidence for detections.
    """

    def __init__(
        self,
        model: Any | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Create a YOLOP inference wrapper.

        Args:
            model: Loaded YOLOP model from ``YOLOPModelLoader.load()``.
            confidence_threshold: Detection confidence cutoff.
        """
        self.model = model
        self.confidence_threshold = confidence_threshold

        logger.info(
            "YOLOPInference created — model=%s, threshold=%.2f",
            "attached" if model is not None else "none",
            confidence_threshold,
        )

    def predict(self, frame: Frame) -> YOLOPRawOutput:
        """Run YOLOP inference on a single BGR frame.

        Placeholder: returns an empty output dictionary.

        Args:
            frame: BGR input image with shape ``(H, W, 3)``.

        Returns:
            Raw YOLOP output dictionary.  Expected keys once implemented:
            ``lane_mask``, ``drivable_mask``, ``detections``.

        Raises:
            RuntimeError: If no model is attached.
        """
        if self.model is None:
            logger.warning("YOLOP predict() called without a loaded model — returning empty output")
            return self.empty_output()

        logger.debug(
            "YOLOP predict() placeholder — frame shape %s",
            frame.shape,
        )

        # TODO: Preprocess frame to YOLOP input tensor (resize, normalize).
        # TODO: Run model forward pass.
        # TODO: Post-process lane segmentation head output.
        # TODO: Post-process drivable area head output.
        # TODO: Post-process object detection head output (optional).
        return self.empty_output()

    def set_model(self, model: Any) -> None:
        """Attach a loaded YOLOP model for inference.

        Args:
            model: Loaded model instance from ``YOLOPModelLoader``.
        """
        self.model = model
        logger.debug("YOLOP model attached to inference wrapper")

    @staticmethod
    def empty_output() -> YOLOPRawOutput:
        """Return an empty YOLOP raw output structure.

        Returns:
            Dictionary with placeholder keys set to ``None``.
        """
        return {
            "lane_mask": None,
            "drivable_mask": None,
            "detections": None,
        }

    def __repr__(self) -> str:
        return (
            f"YOLOPInference(model={'attached' if self.model else 'none'}, "
            f"threshold={self.confidence_threshold})"
        )
