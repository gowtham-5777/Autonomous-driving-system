"""YOLOP model loading and checkpoint management.

Loads and validates YOLOP weight checkpoints without instantiating the
YOLOP network architecture.  Inference is handled separately in
``inference.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("adas.modules.yolop.model_loader")

# Default checkpoint location on Google Drive (Colab Pro)
DEFAULT_YOLOP_CHECKPOINT = Path(
    "/content/drive/MyDrive/adas-project/models/pretrained/yolop/End-to-end.pth"
)

ALLOWED_CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when the YOLOP checkpoint file does not exist."""


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint file fails structural validation."""


class CheckpointLoadError(RuntimeError):
    """Raised when PyTorch cannot deserialize the checkpoint file."""


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata describing a successfully loaded YOLOP checkpoint.

    Attributes:
        weights_path: Absolute path to the checkpoint file.
        file_size_bytes: Size of the checkpoint on disk.
        checkpoint_format: Detected layout (``state_dict``, ``nested``, etc.).
        num_tensor_keys: Number of tensor entries in the state dictionary.
        top_level_keys: Keys at the root of the loaded checkpoint object.
        sample_tensor_keys: First few parameter names for diagnostics.
        device: Target device used during ``torch.load`` map_location.
        loaded_at: UTC timestamp when the checkpoint was loaded.
        pytorch_version: PyTorch version string used for loading.
        extra: Additional non-sensitive fields extracted from the checkpoint.
    """

    weights_path: Path
    file_size_bytes: int
    checkpoint_format: str
    num_tensor_keys: int
    top_level_keys: list[str]
    sample_tensor_keys: list[str]
    device: str
    loaded_at: str
    pytorch_version: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the metadata."""
        return {
            "weights_path": str(self.weights_path),
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": round(self.file_size_bytes / (1024 * 1024), 2),
            "checkpoint_format": self.checkpoint_format,
            "num_tensor_keys": self.num_tensor_keys,
            "top_level_keys": self.top_level_keys,
            "sample_tensor_keys": self.sample_tensor_keys,
            "device": self.device,
            "loaded_at": self.loaded_at,
            "pytorch_version": self.pytorch_version,
            "extra": self.extra,
        }


class YOLOPModelLoader:
    """Load and manage YOLOP checkpoint files for lane detection.

    This loader deserializes the checkpoint with PyTorch, validates its
    structure, and exposes metadata.  It does **not** construct the YOLOP
    network or run inference.

    Attributes:
        weights_path: Filesystem path to the YOLOP checkpoint.
        device: Target device string for ``map_location`` during load.
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        """Create a YOLOP model loader.

        Args:
            weights_path: Path to the YOLOP checkpoint.  Defaults to
                ``DEFAULT_YOLOP_CHECKPOINT`` on Google Drive.
            device: Device for ``torch.load(map_location=...)``.
        """
        self.weights_path = Path(weights_path or DEFAULT_YOLOP_CHECKPOINT)
        self.device = device

        self._checkpoint: dict[str, Any] | None = None
        self._state_dict: dict[str, Any] | None = None
        self._metadata: CheckpointMetadata | None = None

        logger.info(
            "YOLOPModelLoader created — weights=%s, device=%s",
            self.weights_path,
            self.device,
        )

    def load_model(self) -> CheckpointMetadata:
        """Load and validate the YOLOP checkpoint from disk.

        Performs existence checks, file-type validation, PyTorch
        deserialization, and structural validation.  The raw checkpoint and
        extracted state dictionary are cached for ``get_model()``.

        Returns:
            ``CheckpointMetadata`` describing the loaded checkpoint.

        Raises:
            CheckpointNotFoundError: If the checkpoint file is missing.
            CheckpointValidationError: If the checkpoint structure is invalid.
            CheckpointLoadError: If PyTorch fails to read the file.
        """
        logger.info("Loading YOLOP checkpoint from %s", self.weights_path)

        self._validate_checkpoint_path()
        checkpoint = self._deserialize_checkpoint()
        state_dict, checkpoint_format = self._extract_state_dict(checkpoint)
        self._validate_state_dict(state_dict)

        self._checkpoint = checkpoint if isinstance(checkpoint, dict) else {"model": checkpoint}
        self._state_dict = state_dict
        self._metadata = self._build_metadata(checkpoint, state_dict, checkpoint_format)

        logger.info(
            "YOLOP checkpoint loaded — format=%s, tensor_keys=%d, size=%.2f MB",
            self._metadata.checkpoint_format,
            self._metadata.num_tensor_keys,
            self._metadata.file_size_bytes / (1024 * 1024),
        )
        return self._metadata

    def get_model(self) -> dict[str, Any]:
        """Return the loaded checkpoint package.

        The returned dictionary contains the raw checkpoint, extracted
        state dictionary, and metadata.  No YOLOP architecture is attached.

        Returns:
            Dictionary with keys:
                - ``checkpoint``: Raw object returned by ``torch.load``
                - ``state_dict``: Extracted weight tensors
                - ``metadata``: :class:`CheckpointMetadata` as a dict

        Raises:
            RuntimeError: If ``load_model()`` has not been called successfully.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "YOLOP checkpoint is not loaded. Call load_model() first."
            )

        assert self._checkpoint is not None
        assert self._state_dict is not None
        assert self._metadata is not None

        return {
            "checkpoint": self._checkpoint,
            "state_dict": self._state_dict,
            "metadata": self._metadata.to_dict(),
        }

    @property
    def is_loaded(self) -> bool:
        """Return whether a checkpoint has been successfully loaded."""
        return self._checkpoint is not None and self._state_dict is not None

    @property
    def metadata(self) -> CheckpointMetadata | None:
        """Return metadata from the last successful load, or ``None``."""
        return self._metadata

    def unload(self) -> None:
        """Release cached checkpoint data from memory."""
        logger.info("Unloading YOLOP checkpoint from memory")
        self._checkpoint = None
        self._state_dict = None
        self._metadata = None

        try:
            import torch

            if torch.cuda.is_available() and self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared after unload")
        except ImportError:
            logger.debug("PyTorch not available — skipped CUDA cache clear")

    def load(self) -> dict[str, Any] | None:
        """Backward-compatible alias for :meth:`load_model` + :meth:`get_model`.

        Returns:
            Checkpoint package from ``get_model()``, or ``None`` on failure.
        """
        try:
            self.load_model()
            return self.get_model()
        except (CheckpointNotFoundError, CheckpointValidationError, CheckpointLoadError) as exc:
            logger.error("YOLOP checkpoint load failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal validation and deserialization
    # ------------------------------------------------------------------

    def _validate_checkpoint_path(self) -> None:
        """Validate that the checkpoint path exists and looks like a weights file."""
        if not self.weights_path.exists():
            logger.error("YOLOP checkpoint not found: %s", self.weights_path)
            raise CheckpointNotFoundError(
                f"YOLOP checkpoint not found: {self.weights_path}"
            )

        if not self.weights_path.is_file():
            raise CheckpointValidationError(
                f"YOLOP checkpoint path is not a file: {self.weights_path}"
            )

        suffix = self.weights_path.suffix.lower()
        if suffix not in ALLOWED_CHECKPOINT_SUFFIXES:
            raise CheckpointValidationError(
                f"Unsupported checkpoint extension '{suffix}'. "
                f"Expected one of: {sorted(ALLOWED_CHECKPOINT_SUFFIXES)}"
            )

        file_size = self.weights_path.stat().st_size
        if file_size == 0:
            raise CheckpointValidationError(
                f"YOLOP checkpoint file is empty: {self.weights_path}"
            )

        logger.debug(
            "Checkpoint path validated — size=%d bytes, suffix=%s",
            file_size,
            suffix,
        )

    def _deserialize_checkpoint(self) -> Any:
        """Deserialize the checkpoint using PyTorch."""
        try:
            import torch
        except ImportError as exc:
            raise CheckpointLoadError(
                "PyTorch is required to load YOLOP checkpoints. "
                "Install it with: pip install torch"
            ) from exc

        try:
            checkpoint = torch.load(
                self.weights_path,
                map_location=self.device,
                weights_only=False,
            )
            logger.debug("Checkpoint deserialized with torch %s", torch.__version__)
            return checkpoint
        except Exception as exc:
            logger.exception("Failed to deserialize YOLOP checkpoint")
            raise CheckpointLoadError(
                f"Failed to load YOLOP checkpoint from {self.weights_path}: {exc}"
            ) from exc

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> tuple[dict[str, Any], str]:
        """Extract a state dictionary and identify the checkpoint layout."""
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    nested = checkpoint[key]
                    if any(hasattr(v, "shape") for v in nested.values()):
                        return nested, f"nested:{key}"

            if any(hasattr(v, "shape") for v in checkpoint.values()):
                return checkpoint, "state_dict"

            raise CheckpointValidationError(
                "Checkpoint dictionary does not contain a recognizable state_dict. "
                f"Top-level keys: {list(checkpoint.keys())}"
            )

        if hasattr(checkpoint, "state_dict") and callable(checkpoint.state_dict):
            return checkpoint.state_dict(), "torch_module"

        raise CheckpointValidationError(
            f"Unsupported checkpoint type: {type(checkpoint).__name__}. "
            "Expected a dict or torch.nn.Module."
        )

    @staticmethod
    def _validate_state_dict(state_dict: dict[str, Any]) -> None:
        """Validate that the state dictionary contains tensor weights."""
        tensor_keys = [key for key, value in state_dict.items() if hasattr(value, "shape")]

        if not tensor_keys:
            raise CheckpointValidationError(
                "Checkpoint state_dict contains no tensor parameters."
            )

        logger.debug(
            "State dict validated — %d tensor key(s), sample=%s",
            len(tensor_keys),
            tensor_keys[:3],
        )

    def _build_metadata(
        self,
        checkpoint: Any,
        state_dict: dict[str, Any],
        checkpoint_format: str,
    ) -> CheckpointMetadata:
        """Build metadata describing the loaded checkpoint."""
        import torch

        tensor_keys = [key for key, value in state_dict.items() if hasattr(value, "shape")]
        top_level_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else ["model"]

        extra: dict[str, Any] = {}
        if isinstance(checkpoint, dict):
            for key in ("epoch", "best_fitness", "optimizer", "version"):
                if key in checkpoint:
                    extra[key] = checkpoint[key]

        return CheckpointMetadata(
            weights_path=self.weights_path.resolve(),
            file_size_bytes=self.weights_path.stat().st_size,
            checkpoint_format=checkpoint_format,
            num_tensor_keys=len(tensor_keys),
            top_level_keys=top_level_keys,
            sample_tensor_keys=tensor_keys[:5],
            device=self.device,
            loaded_at=datetime.now(timezone.utc).isoformat(),
            pytorch_version=torch.__version__,
            extra=extra,
        )

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"YOLOPModelLoader(weights_path={self.weights_path!s}, "
            f"device={self.device!r}, {status})"
        )
