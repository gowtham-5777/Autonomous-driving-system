"""Dataclass schemas for YOLOv8 traffic signal detection outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .class_map import load_traffic_signal_classes

TRAFFIC_SIGNAL_OUTPUT_KEYS = (
    "detections",
    "count_by_label",
    "total_count",
    "nearest_signal",
    "controlling_signal",
    "dominant_state",
    "has_stop_state",
    "has_proceed_state",
    "raw_status",
)

ADAS_SIGNAL_LABELS = frozenset(load_traffic_signal_classes())


@dataclass(frozen=True)
class SignalBoundingBoxData:
    """Axis-aligned detection box in original frame coordinates (inclusive corners)."""

    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    center_x: float
    center_y: float
    area: int

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> SignalBoundingBoxData:
        """Build a bounding box from inclusive corner coordinates."""
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return cls(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width=width,
            height=height,
            center_x=(x1 + x2) / 2.0,
            center_y=(y1 + y2) / 2.0,
            area=width * height,
        )

    def to_list(self) -> list[int]:
        """Return ``[x1, y1, x2, y2]`` for JSON serialization."""
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class DetectedSignal:
    """Single filtered traffic signal detection."""

    signal_label: str
    class_id: int
    confidence: float
    bbox: SignalBoundingBoxData
    is_stop_state: bool = False
    is_caution_state: bool = False
    is_proceed_state: bool = False
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "signal_label": self.signal_label,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_list(),
            "is_stop_state": self.is_stop_state,
            "is_caution_state": self.is_caution_state,
            "is_proceed_state": self.is_proceed_state,
            "track_id": self.track_id,
        }


@dataclass
class TrafficSignalSummary:
    """Aggregate statistics for downstream decision logic."""

    count_by_label: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    nearest_signal: DetectedSignal | None = None
    highest_confidence: DetectedSignal | None = None
    controlling_signal: DetectedSignal | None = None
    dominant_state: str | None = None
    has_stop_state: bool = False
    has_proceed_state: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "count_by_label": dict(self.count_by_label),
            "total_count": self.total_count,
            "nearest_signal": (
                self.nearest_signal.to_dict() if self.nearest_signal else None
            ),
            "highest_confidence": (
                self.highest_confidence.to_dict()
                if self.highest_confidence
                else None
            ),
            "controlling_signal": (
                self.controlling_signal.to_dict()
                if self.controlling_signal
                else None
            ),
            "dominant_state": self.dominant_state,
            "has_stop_state": self.has_stop_state,
            "has_proceed_state": self.has_proceed_state,
        }


@dataclass
class ParsedYOLOv8SignalOutput:
    """Intermediate parser output before module-level result assembly."""

    detections: list[DetectedSignal] = field(default_factory=list)
    raw_status: str = "unparsed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrafficSignalDetectionResult:
    """Standardized result from :class:`TrafficSignalModule`."""

    detections: list[DetectedSignal] = field(default_factory=list)
    summary: TrafficSignalSummary = field(default_factory=TrafficSignalSummary)
    frame_shape: tuple[int, int] | None = None
    inference_time_ms: float | None = None
    model_variant: str = "n"
    confidence_threshold: float = 0.5
    raw_status: str = "empty"

    def to_prediction_dict(self) -> dict[str, Any]:
        """Convert to the orchestrator-facing prediction dictionary."""
        return {
            "detections": [det.to_dict() for det in self.detections],
            "count_by_label": dict(self.summary.count_by_label),
            "total_count": self.summary.total_count,
            "nearest_signal": (
                self.summary.nearest_signal.to_dict()
                if self.summary.nearest_signal
                else None
            ),
            "highest_confidence": (
                self.summary.highest_confidence.to_dict()
                if self.summary.highest_confidence
                else None
            ),
            "controlling_signal": (
                self.summary.controlling_signal.to_dict()
                if self.summary.controlling_signal
                else None
            ),
            "dominant_state": self.summary.dominant_state,
            "has_stop_state": self.summary.has_stop_state,
            "has_proceed_state": self.summary.has_proceed_state,
            "frame_shape": self.frame_shape,
            "inference_time_ms": self.inference_time_ms,
            "model_variant": self.model_variant,
            "confidence_threshold": self.confidence_threshold,
            "raw_status": self.raw_status,
        }

    @classmethod
    def empty(cls, raw_status: str = "empty") -> TrafficSignalDetectionResult:
        """Return an empty traffic signal detection result."""
        return cls(raw_status=raw_status)
