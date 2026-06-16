"""YOLOP model loading — architecture placeholder.

This module will host YOLOP checkpoint loading, device placement, and
model lifecycle management.  No YOLOP dependencies are imported yet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("adas.modules.yolop.model_loader")


class YOLOPModelLoader:
    """Load and manage YOLOP model weights for lane detection.

    Placeholder class that defines the interface for future YOLOP integration.
    Actual weight loading will be implemented when the YOLOP dependency is
    added to the project.

    Attributes:
        weights_path: Filesystem path to the YOLOP checkpoint.
        device: Target inference device (e.g. ``"cpu"`` or ``"cuda"``).
        model: Loaded YOLOP model instance (``None`` until implemented).
    """

    def __init__(
        self,
        weights_path: Path,
        device: str = "cpu",
    ) -> None:
        """Create a YOLOP model loader.

        Args:
            weights_path: Path to the YOLOP ``.pt`` checkpoint file.
            device: Device string for model placement.
        """
        self.weights_path = Path(weights_path)
        self.device = device
        self.model: Any | None = None

        logger.info(
            "YOLOPModelLoader created — weights=%s, device=%s",
            self.weights_path,
            self.device,
        )

    def load(self) -> Any | None:
        """Load YOLOP weights and return the model instance.

        Placeholder: validates the weight path exists and logs intent.
        Does not import or instantiate YOLOP yet.

        Returns:
            Loaded model instance, or ``None`` until implemented.

        Raises:
            FileNotFoundError: If the weights file does not exist.
        """
        logger.info("YOLOP load() placeholder — checking weights at %s", self.weights_path)

        if not self.weights_path.exists():
            logger.warning("YOLOP weights not found: %s", self.weights_path)
            raise FileNotFoundError(f"YOLOP weights not found: {self.weights_path}")

        # TODO: Import YOLOP model class from the upstream repository.
        # TODO: Load state dict from self.weights_path.
        # TODO: Move model to self.device and set eval mode.
        self.model = None
        logger.debug("YOLOP model loading stub complete — model is None")
        return self.model

    def unload(self) -> None:
        """Release the loaded model and free resources."""
        logger.info("YOLOP unload() placeholder — releasing model reference")
        self.model = None
        # TODO: Clear CUDA cache when running on GPU.

    @property
    def is_loaded(self) -> bool:
        """Return whether a model instance is currently held."""
        return self.model is not None

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"YOLOPModelLoader(weights_path={self.weights_path!s}, "
            f"device={self.device!r}, {status})"
        )
