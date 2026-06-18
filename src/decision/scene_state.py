"""Unified scene state aggregating perception module outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..modules.yolop.output_schema import LaneDetectionResult
from ..modules.yolov8.output_schema import VehicleDetectionResult
from ..modules.yolov8_sign.output_schema import TrafficSignDetectionResult
from ..modules.yolov8_signal.output_schema import TrafficSignalDetectionResult

_LANE_OK_STATUSES = frozenset({"parsed", "stub_segmentation", "stub"})
_MODULE_OK_STATUSES = frozenset({"parsed", "stub"})


@dataclass
class ModuleStatus:
    """Per-module health for a single frame."""

    module_name: str
    raw_status: str
    ok: bool
    inference_time_ms: float | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "module_name": self.module_name,
            "raw_status": self.raw_status,
            "ok": self.ok,
            "inference_time_ms": self.inference_time_ms,
            "error_message": self.error_message,
        }


@dataclass
class SceneState:
    """Unified scene snapshot aggregating all perception outputs for one frame."""

    frame_index: int = 0
    frame_shape: tuple[int, int] | None = None
    timestamp_ms: float | None = None

    lane: LaneDetectionResult | None = None
    vehicles: VehicleDetectionResult | None = None
    signs: TrafficSignDetectionResult | None = None
    signals: TrafficSignalDetectionResult | None = None
    segmentation: Any | None = None

    module_statuses: list[ModuleStatus] = field(default_factory=list)

    lane_ok: bool = False
    vehicles_ok: bool = False
    signs_ok: bool = False
    signals_ok: bool = False

    @classmethod
    def from_perception(
        cls,
        *,
        frame_index: int,
        frame_shape: tuple[int, int],
        timestamp_ms: float | None,
        lane: LaneDetectionResult | None,
        vehicles: VehicleDetectionResult | None,
        signs: TrafficSignDetectionResult | None,
        signals: TrafficSignalDetectionResult | None,
        segmentation: Any | None = None,
    ) -> SceneState:
        """Build SceneState and derive module health flags."""
        lane_ok = _lane_module_ok(lane)
        vehicles_ok = _module_ok(vehicles)
        signs_ok = _module_ok(signs)
        signals_ok = _module_ok(signals)

        statuses: list[ModuleStatus] = []
        if lane is not None:
            statuses.append(
                ModuleStatus(
                    module_name="lane_detection",
                    raw_status=lane.raw_status,
                    ok=lane_ok,
                )
            )
        if vehicles is not None:
            statuses.append(
                ModuleStatus(
                    module_name="vehicle_detection",
                    raw_status=vehicles.raw_status,
                    ok=vehicles_ok,
                    inference_time_ms=vehicles.inference_time_ms,
                )
            )
        if signs is not None:
            statuses.append(
                ModuleStatus(
                    module_name="traffic_sign",
                    raw_status=signs.raw_status,
                    ok=signs_ok,
                    inference_time_ms=signs.inference_time_ms,
                )
            )
        if signals is not None:
            statuses.append(
                ModuleStatus(
                    module_name="traffic_signal",
                    raw_status=signals.raw_status,
                    ok=signals_ok,
                    inference_time_ms=signals.inference_time_ms,
                )
            )

        return cls(
            frame_index=frame_index,
            frame_shape=frame_shape,
            timestamp_ms=timestamp_ms,
            lane=lane,
            vehicles=vehicles,
            signs=signs,
            signals=signals,
            segmentation=segmentation,
            module_statuses=statuses,
            lane_ok=lane_ok,
            vehicles_ok=vehicles_ok,
            signs_ok=signs_ok,
            signals_ok=signals_ok,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot (mask arrays replaced with shape metadata)."""
        return {
            "frame_index": self.frame_index,
            "frame_shape": list(self.frame_shape) if self.frame_shape else None,
            "timestamp_ms": self.timestamp_ms,
            "lane": _lane_to_dict(self.lane),
            "vehicles": self.vehicles.to_prediction_dict() if self.vehicles else None,
            "signs": self.signs.to_prediction_dict() if self.signs else None,
            "signals": self.signals.to_prediction_dict() if self.signals else None,
            "module_statuses": [status.to_dict() for status in self.module_statuses],
            "lane_ok": self.lane_ok,
            "vehicles_ok": self.vehicles_ok,
            "signs_ok": self.signs_ok,
            "signals_ok": self.signals_ok,
        }

    def perception_dict(self) -> dict[str, Any]:
        """Orchestrator-facing dict using each module's to_prediction_dict()."""
        return {
            "frame_index": self.frame_index,
            "frame_shape": list(self.frame_shape) if self.frame_shape else None,
            "timestamp_ms": self.timestamp_ms,
            "lane": self.lane.to_prediction_dict() if self.lane else None,
            "vehicles": self.vehicles.to_prediction_dict() if self.vehicles else None,
            "signs": self.signs.to_prediction_dict() if self.signs else None,
            "signals": self.signals.to_prediction_dict() if self.signals else None,
            "module_statuses": [status.to_dict() for status in self.module_statuses],
        }


def _lane_module_ok(lane: LaneDetectionResult | None) -> bool:
    if lane is None:
        return False
    if lane.raw_status not in _LANE_OK_STATUSES:
        return False
    return lane.lane_center_x is not None or lane.lane_mask is not None


def _module_ok(result: Any | None) -> bool:
    if result is None:
        return False
    return getattr(result, "raw_status", "empty") in _MODULE_OK_STATUSES


def _lane_to_dict(lane: LaneDetectionResult | None) -> dict[str, Any] | None:
    if lane is None:
        return None

    payload = lane.to_prediction_dict()
    if lane.lane_mask is not None:
        payload["lane_mask"] = {
            "present": True,
            "shape": list(lane.lane_mask.shape),
        }
    else:
        payload["lane_mask"] = None

    if lane.drivable_mask is not None:
        payload["drivable_mask"] = {
            "present": True,
            "shape": list(lane.drivable_mask.shape),
        }
    else:
        payload["drivable_mask"] = None

    payload.pop("preprocessed_edges", None)
    return payload
