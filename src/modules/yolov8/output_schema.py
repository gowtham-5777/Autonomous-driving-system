"""Dataclass schemas for YOLOv8 vehicle detection outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ADAS road-user labels detected by this module.
ADAS_VEHICLE_LABELS = frozenset(
    {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
)


@dataclass(frozen=True)
class BoundingBoxData:
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
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> BoundingBoxData:
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
class DetectedObject:
    """Single filtered road-user detection."""

    label: str
    coco_class_id: int
    confidence: float
    bbox: BoundingBoxData
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "label": self.label,
            "coco_class_id": self.coco_class_id,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_list(),
            "track_id": self.track_id,
        }


@dataclass
class VehicleDetectionSummary:
    """Aggregate statistics for downstream decision logic."""

    count_by_label: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    nearest_object: DetectedObject | None = None
    highest_confidence: DetectedObject | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "count_by_label": dict(self.count_by_label),
            "total_count": self.total_count,
            "nearest_object": (
                self.nearest_object.to_dict() if self.nearest_object else None
            ),
            "highest_confidence": (
                self.highest_confidence.to_dict()
                if self.highest_confidence
                else None
            ),
        }


@dataclass
class ParsedYOLOv8Output:
    """Intermediate parser output before module-level result assembly."""

    detections: list[DetectedObject] = field(default_factory=list)
    raw_status: str = "unparsed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VehicleDetectionResult:
    """Standardized result from :class:`VehicleDetectionModule`."""

    detections: list[DetectedObject] = field(default_factory=list)
    summary: VehicleDetectionSummary = field(default_factory=VehicleDetectionSummary)
    frame_shape: tuple[int, int] | None = None
    inference_time_ms: float | None = None
    model_variant: str = "s"
    confidence_threshold: float = 0.5
    raw_status: str = "empty"

    def to_prediction_dict(self) -> dict[str, Any]:
        """Convert to the orchestrator-facing prediction dictionary."""
        return {
            "detections": [det.to_dict() for det in self.detections],
            "count_by_label": dict(self.summary.count_by_label),
            "total_count": self.summary.total_count,
            "nearest_object": (
                self.summary.nearest_object.to_dict()
                if self.summary.nearest_object
                else None
            ),
            "highest_confidence": (
                self.summary.highest_confidence.to_dict()
                if self.summary.highest_confidence
                else None
            ),
            "frame_shape": self.frame_shape,
            "inference_time_ms": self.inference_time_ms,
            "model_variant": self.model_variant,
            "confidence_threshold": self.confidence_threshold,
            "raw_status": self.raw_status,
        }

    @classmethod
    def empty(cls, raw_status: str = "empty") -> VehicleDetectionResult:
        """Return an empty vehicle detection result."""
        return cls(raw_status=raw_status)
