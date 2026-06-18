"""YOLOv8 model loading and weight management."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("adas.modules.yolov8.model_loader")

DEFAULT_MODEL_VARIANT = "s"
ALLOWED_VARIANTS = frozenset({"n", "s", "m"})


class WeightsNotFoundError(FileNotFoundError):
    """Raised when YOLOv8 weights cannot be resolved."""


class WeightsValidationError(ValueError):
    """Raised when weight file validation fails."""


class WeightsLoadError(RuntimeError):
    """Raised when Ultralytics cannot load the model."""


@dataclass(frozen=True)
class WeightsMetadata:
    """Metadata describing a successfully loaded YOLOv8 model."""

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
            f"Invalid YOLOv8 variant '{variant}'. Expected one of: {sorted(ALLOWED_VARIANTS)}"
        )
    return normalized


def variant_to_filename(variant: str) -> str:
    """Return the standard Ultralytics weight filename for a variant."""
    return f"yolov8{resolve_variant_name(variant)}.pt"


class YOLOv8ModelLoader:
    """Load and manage YOLOv8 checkpoints via Ultralytics.

    If the configured filesystem path does not exist, falls back to the
    variant filename (e.g. ``yolov8s.pt``) so Ultralytics can download
    weights on first use (Colab-friendly).
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
            "YOLOv8ModelLoader created — variant=yolov8%s, device=%s",
            self.model_variant,
            device,
        )

    @property
    def is_loaded(self) -> bool:
        """Return whether a model has been loaded."""
        return self._model is not None

    def resolve_weights_source(self) -> str:
        """Resolve the weight source path or Ultralytics model name."""
        if self.weights_path is not None and self.weights_path.is_file():
            return str(self.weights_path)

        if self.weights_path is not None and self.weights_path.exists():
            if not self.weights_path.is_file():
                raise WeightsValidationError(
                    f"YOLOv8 weights path is not a file: {self.weights_path}"
                )

        # Colab / first-run: Ultralytics downloads yolov8{s,n,m}.pt automatically.
        fallback = variant_to_filename(self.model_variant)
        logger.warning(
            "Configured YOLOv8 weights not found at %s — using %s (auto-download if needed)",
            self.weights_path,
            fallback,
        )
        return fallback

    def load_model(self) -> WeightsMetadata:
        """Load YOLOv8 weights via Ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise WeightsLoadError(
                "ultralytics is required for YOLOv8 vehicle detection. "
                "Install with: pip install ultralytics"
            ) from exc

        import ultralytics

        source = self.resolve_weights_source()
        self._resolved_source = source

        logger.info("Loading YOLOv8 model from %s", source)

        try:
            self._model = YOLO(source)
        except Exception as exc:
            logger.exception("YOLOv8 model load failed")
            raise WeightsLoadError(f"Failed to load YOLOv8 model from {source}: {exc}") from exc

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
            "YOLOv8 model loaded — source=%s, variant=yolov8%s",
            source,
            self.model_variant,
        )
        return self._metadata

    def get_model(self) -> dict[str, Any]:
        """Return the model package for the inference engine."""
        if self._model is None or self._metadata is None:
            raise RuntimeError(
                "YOLOv8 model is not loaded. Call load_model() first."
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
        logger.info("Unloading YOLOv8 model")
        self._model = None
        self._metadata = None
        self._resolved_source = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
