"""YOLOv8 sign model loading and weight management."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("adas.modules.yolov8_sign.model_loader")

DEFAULT_MODEL_VARIANT = "n"
ALLOWED_VARIANTS = frozenset({"n", "s", "m"})


class WeightsNotFoundError(FileNotFoundError):
    """Raised when fine-tuned sign weights cannot be resolved."""


class WeightsValidationError(ValueError):
    """Raised when weight file validation fails."""


class WeightsLoadError(RuntimeError):
    """Raised when Ultralytics cannot load the model."""


@dataclass(frozen=True)
class WeightsMetadata:
    """Metadata describing a successfully loaded YOLOv8 sign model."""

    weights_path: str
    model_variant: str
    file_size_bytes: int | None
    device: str
    loaded_at: str
    ultralytics_version: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "weights_path": self.weights_path,
            "model_variant": self.model_variant,
            "file_size_bytes": self.file_size_bytes,
            "device": self.device,
            "loaded_at": self.loaded_at,
            "ultralytics_version": self.ultralytics_version,
            "extra": self.extra,
        }


def resolve_variant_name(variant: str) -> str:
    """Normalize and validate a YOLOv8 variant letter (n, s, m)."""
    normalized = variant.strip().lower().lstrip("yolov8")
    if normalized not in ALLOWED_VARIANTS:
        raise WeightsValidationError(
            f"Invalid YOLOv8 sign variant '{variant}'. "
            f"Expected one of: {sorted(ALLOWED_VARIANTS)}"
        )
    return normalized


class YOLOv8SignModelLoader:
    """Load and manage fine-tuned YOLOv8 sign checkpoints via Ultralytics.

    Unlike the vehicle detector, there is **no** public COCO fallback —
    fine-tuned sign weights must exist at the configured path.
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        model_variant: str = DEFAULT_MODEL_VARIANT,
        device: str = "cpu",
    ) -> None:
        self.model_variant = resolve_variant_name(model_variant)
        self.device = device
        self.weights_path = (
            Path(weights_path) if weights_path is not None else None
        )

        self._model: Any | None = None
        self._metadata: WeightsMetadata | None = None
        self._resolved_source: str | None = None

        logger.info(
            "YOLOv8SignModelLoader created — variant=yolov8%s, device=%s",
            self.model_variant,
            device,
        )

    @property
    def is_loaded(self) -> bool:
        """Return whether a model has been loaded."""
        return self._model is not None

    def resolve_weights_source(self) -> str:
        """Resolve the weight source path for fine-tuned sign weights."""
        if self.weights_path is None:
            raise WeightsNotFoundError(
                "Traffic sign weights path is not configured. "
                "Set weights_path or update config/default.yaml."
            )

        if self.weights_path.is_file():
            return str(self.weights_path)

        if self.weights_path.exists() and not self.weights_path.is_file():
            raise WeightsValidationError(
                f"YOLOv8 sign weights path is not a file: {self.weights_path}"
            )

        raise WeightsNotFoundError(
            f"Fine-tuned traffic sign weights not found at {self.weights_path}. "
            "Train or place traffic_signs_yolov8n.pt before running real inference."
        )

    def load_model(self) -> WeightsMetadata:
        """Load fine-tuned YOLOv8 sign weights via Ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise WeightsLoadError(
                "ultralytics is required for YOLOv8 traffic sign detection. "
                "Install with: pip install ultralytics"
            ) from exc

        import ultralytics

        source = self.resolve_weights_source()
        self._resolved_source = source

        logger.info("Loading YOLOv8 sign model from %s", source)

        try:
            self._model = YOLO(source)
        except Exception as exc:
            logger.exception("YOLOv8 sign model load failed")
            raise WeightsLoadError(
                f"Failed to load YOLOv8 sign model from {source}: {exc}"
            ) from exc

        file_size: int | None = None
        if Path(source).is_file():
            file_size = Path(source).stat().st_size

        self._metadata = WeightsMetadata(
            weights_path=source,
            model_variant=self.model_variant,
            file_size_bytes=file_size,
            device=self.device,
            loaded_at=datetime.now(timezone.utc).isoformat(),
            ultralytics_version=getattr(ultralytics, "__version__", "unknown"),
        )

        logger.info(
            "YOLOv8 sign model loaded — source=%s, variant=yolov8%s",
            source,
            self.model_variant,
        )
        return self._metadata

    def get_model(self) -> dict[str, Any]:
        """Return the model package for the inference engine."""
        if self._model is None or self._metadata is None:
            raise RuntimeError(
                "YOLOv8 sign model is not loaded. Call load_model() first."
            )

        return {
            "model": self._model,
            "metadata": self._metadata,
            "weights_path": self._resolved_source or self._metadata.weights_path,
            "model_variant": self.model_variant,
            "device": self.device,
        }

    def unload(self) -> None:
        """Release the loaded model and clear GPU cache when available."""
        logger.info("Unloading YOLOv8 sign model")
        self._model = None
        self._metadata = None
        self._resolved_source = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
