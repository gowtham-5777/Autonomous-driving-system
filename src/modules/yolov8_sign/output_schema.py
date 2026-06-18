"""Dataclass schemas for YOLOv8 traffic sign detection outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .class_map import load_traffic_sign_classes

TRAFFIC_SIGN_OUTPUT_KEYS = (
    "detections",
    "count_by_label",
    "total_count",
    "nearest_sign",
    "active_speed_limit_kmh",
    "raw_status",
)

ADAS_SIGN_LABELS = frozenset(load_traffic_sign_classes())


@dataclass(frozen=True)
class SignBoundingBoxData:
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
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> SignBoundingBoxData:
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
class DetectedSign:
    """Single filtered traffic sign detection."""

    sign_label: str
    class_id: int
    confidence: float
    bbox: SignBoundingBoxData
    speed_limit_kmh: int | None = None
    is_regulatory: bool = False
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "sign_label": self.sign_label,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_list(),
            "speed_limit_kmh": self.speed_limit_kmh,
            "is_regulatory": self.is_regulatory,
            "track_id": self.track_id,
        }


@dataclass
class TrafficSignDetectionSummary:
    """Aggregate statistics for downstream decision logic."""

    count_by_label: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    nearest_sign: DetectedSign | None = None
    highest_confidence: DetectedSign | None = None
    active_speed_limit_kmh: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "count_by_label": dict(self.count_by_label),
            "total_count": self.total_count,
            "nearest_sign": (
                self.nearest_sign.to_dict() if self.nearest_sign else None
            ),
            "highest_confidence": (
                self.highest_confidence.to_dict()
                if self.highest_confidence
                else None
            ),
            "active_speed_limit_kmh": self.active_speed_limit_kmh,
        }


@dataclass
class ParsedYOLOv8SignOutput:
    """Intermediate parser output before module-level result assembly."""

    detections: list[DetectedSign] = field(default_factory=list)
    raw_status: str = "unparsed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrafficSignDetectionResult:
    """Standardized result from :class:`TrafficSignModule`."""

    detections: list[DetectedSign] = field(default_factory=list)
    summary: TrafficSignDetectionSummary = field(default_factory=TrafficSignDetectionSummary)
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
            "nearest_sign": (
                self.summary.nearest_sign.to_dict()
                if self.summary.nearest_sign
                else None
            ),
            "highest_confidence": (
                self.summary.highest_confidence.to_dict()
                if self.summary.highest_confidence
                else None
            ),
            "active_speed_limit_kmh": self.summary.active_speed_limit_kmh,
            "frame_shape": self.frame_shape,
            "inference_time_ms": self.inference_time_ms,
            "model_variant": self.model_variant,
            "confidence_threshold": self.confidence_threshold,
            "raw_status": self.raw_status,
        }

    @classmethod
    def empty(cls, raw_status: str = "empty") -> TrafficSignDetectionResult:
        """Return an empty traffic sign detection result."""
        return cls(raw_status=raw_status)
