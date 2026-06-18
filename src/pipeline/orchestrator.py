"""Pipeline orchestrator — runs perception modules and evaluates decisions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..decision.decision_engine import DecisionEngine
from ..decision.scene_state import SceneState
from ..decision.types import DecisionResult
from ..modules.base import Frame
from ..modules.lane_detection import LaneDetectionModule
from ..modules.traffic_sign import TrafficSignModule
from ..modules.traffic_signal import TrafficSignalModule
from ..modules.vehicle_detection import VehicleDetectionModule
from ..utils.model_paths import get_pipeline_config
from ..visualization.hud import draw_decision_hud
from ..visualization.overlays import draw_scene_overlays

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Orchestrator runtime options."""

    run_lane: bool = True
    run_vehicles: bool = True
    run_signs: bool = True
    run_signals: bool = True
    run_segmentation: bool = False
    auto_initialize: bool = True
    collect_timing: bool = True


@dataclass
class PipelineResult:
    """Return value of a single orchestrated frame pass."""

    scene_state: SceneState
    decision: DecisionResult
    total_time_ms: float | None = None


class PipelineOrchestrator:
    """Runs perception modules in reference order and evaluates decisions."""

    REFERENCE_ORDER = (
        "lane_detection",
        "vehicle_detection",
        "traffic_sign",
        "traffic_signal",
    )

    def __init__(
        self,
        *,
        lane_module: LaneDetectionModule | None = None,
        vehicle_module: VehicleDetectionModule | None = None,
        sign_module: TrafficSignModule | None = None,
        signal_module: TrafficSignalModule | None = None,
        decision_engine: DecisionEngine | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.lane_module = lane_module or LaneDetectionModule()
        self.vehicle_module = vehicle_module or VehicleDetectionModule()
        self.sign_module = sign_module or TrafficSignModule()
        self.signal_module = signal_module or TrafficSignalModule()
        self.decision_engine = decision_engine or DecisionEngine()
        self.config = config or _pipeline_config_from_yaml()

    def initialize(self) -> None:
        """Initialize all enabled modules."""
        if self.config.run_lane:
            self.lane_module.initialize()
        if self.config.run_vehicles:
            self.vehicle_module.initialize()
        if self.config.run_signs:
            self.sign_module.initialize()
        if self.config.run_signals:
            self.signal_module.initialize()

    def run_frame(
        self,
        frame: Frame,
        *,
        frame_index: int = 0,
        timestamp_ms: float | None = None,
    ) -> PipelineResult:
        """Execute full perception + decision pipeline on one frame."""
        self._validate_frame(frame)

        if self.config.auto_initialize:
            self._ensure_initialized()

        t0 = time.perf_counter() if self.config.collect_timing else None

        lane_result = (
            self.lane_module.predict(frame) if self.config.run_lane else None
        )
        vehicle_result = (
            self.vehicle_module.predict(frame) if self.config.run_vehicles else None
        )
        sign_result = (
            self.sign_module.predict(frame) if self.config.run_signs else None
        )
        signal_result = (
            self.signal_module.predict(frame) if self.config.run_signals else None
        )

        scene_state = SceneState.from_perception(
            frame_index=frame_index,
            frame_shape=(int(frame.shape[0]), int(frame.shape[1])),
            timestamp_ms=timestamp_ms,
            lane=lane_result,
            vehicles=vehicle_result,
            signs=sign_result,
            signals=signal_result,
        )

        decision = self.decision_engine.evaluate(scene_state)

        total_time_ms = None
        if t0 is not None:
            total_time_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "Pipeline frame=%d recommendation=%s modules_ok=(lane=%s veh=%s sign=%s sig=%s)",
            frame_index,
            decision.recommendation.value,
            scene_state.lane_ok,
            scene_state.vehicles_ok,
            scene_state.signs_ok,
            scene_state.signals_ok,
        )

        return PipelineResult(
            scene_state=scene_state,
            decision=decision,
            total_time_ms=total_time_ms,
        )

    def visualize(
        self,
        frame: Frame,
        result: PipelineResult,
        *,
        show_hud: bool = True,
        show_lane: bool = True,
        show_vehicles: bool = True,
        show_signs: bool = True,
        show_signals: bool = True,
    ) -> Frame:
        """Composite visualization using overlays and decision HUD."""
        annotated = draw_scene_overlays(
            frame,
            result.scene_state,
            show_lane=show_lane,
            show_vehicles=show_vehicles,
            show_signs=show_signs,
            show_signals=show_signals,
        )
        if show_hud:
            annotated = draw_decision_hud(annotated, result.decision, scene_state=result.scene_state)
        return annotated

    def cleanup(self) -> None:
        """Release all module resources."""
        if self.config.run_lane:
            self.lane_module.cleanup()
        if self.config.run_vehicles:
            self.vehicle_module.cleanup()
        if self.config.run_signs:
            self.sign_module.cleanup()
        if self.config.run_signals:
            self.signal_module.cleanup()

    def _ensure_initialized(self) -> None:
        modules: list[tuple[bool, object]] = [
            (self.config.run_lane, self.lane_module),
            (self.config.run_vehicles, self.vehicle_module),
            (self.config.run_signs, self.sign_module),
            (self.config.run_signals, self.signal_module),
        ]
        for enabled, module in modules:
            if enabled and not module.is_initialized:
                module.initialize()

    @staticmethod
    def _validate_frame(frame: Frame) -> None:
        if frame is None:
            raise ValueError("Input frame is None")
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Input frame must be numpy.ndarray, got {type(frame)}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Input frame must have shape (H, W, 3), got {frame.shape}")
        if frame.size == 0:
            raise ValueError("Input frame is empty")


def _pipeline_config_from_yaml() -> PipelineConfig:
    cfg = get_pipeline_config()
    return PipelineConfig(
        run_lane=bool(cfg.get("run_lane", True)),
        run_vehicles=bool(cfg.get("run_vehicles", True)),
        run_signs=bool(cfg.get("run_signs", True)),
        run_signals=bool(cfg.get("run_signals", True)),
        run_segmentation=bool(cfg.get("run_segmentation", False)),
        auto_initialize=bool(cfg.get("auto_initialize", True)),
        collect_timing=bool(cfg.get("collect_timing", True)),
    )


def create_default_orchestrator(
    device: str = "cpu",
    config: PipelineConfig | None = None,
) -> PipelineOrchestrator:
    """Construct orchestrator with default module instances."""
    return PipelineOrchestrator(
        lane_module=LaneDetectionModule(device=device),
        vehicle_module=VehicleDetectionModule(device=device),
        sign_module=TrafficSignModule(device=device),
        signal_module=TrafficSignalModule(device=device),
        config=config,
    )
