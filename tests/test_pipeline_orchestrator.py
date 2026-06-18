"""End-to-end integration tests for PipelineOrchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from src.decision.types import ADASRecommendation
from src.pipeline.orchestrator import PipelineConfig, PipelineOrchestrator


class TestPipelineOrchestrator:
    def test_initialize_all_modules(self, pipeline_orchestrator: PipelineOrchestrator) -> None:
        orch = pipeline_orchestrator
        assert orch.lane_module.is_initialized
        assert orch.vehicle_module.is_initialized
        assert orch.sign_module.is_initialized
        assert orch.signal_module.is_initialized

    def test_run_frame_returns_pipeline_result(
        self,
        road_frame: np.ndarray,
        pipeline_orchestrator: PipelineOrchestrator,
    ) -> None:
        result = pipeline_orchestrator.run_frame(road_frame, frame_index=0)
        assert result.scene_state.frame_shape == (road_frame.shape[0], road_frame.shape[1])
        assert result.decision.recommendation in ADASRecommendation
        assert result.total_time_ms is not None

    def test_stub_pipeline_stop_on_red_light(
        self,
        road_frame: np.ndarray,
        pipeline_orchestrator: PipelineOrchestrator,
    ) -> None:
        result = pipeline_orchestrator.run_frame(road_frame)
        assert result.scene_state.signals_ok is True
        assert result.decision.recommendation == ADASRecommendation.STOP

    def test_module_status_order_matches_reference(
        self,
        road_frame: np.ndarray,
        pipeline_orchestrator: PipelineOrchestrator,
    ) -> None:
        result = pipeline_orchestrator.run_frame(road_frame)
        names = [status.module_name for status in result.scene_state.module_statuses]
        assert names == [
            "lane_detection",
            "vehicle_detection",
            "traffic_sign",
            "traffic_signal",
        ]

    def test_visualize_returns_bgr_frame(
        self,
        road_frame: np.ndarray,
        pipeline_orchestrator: PipelineOrchestrator,
    ) -> None:
        result = pipeline_orchestrator.run_frame(road_frame)
        annotated = pipeline_orchestrator.visualize(road_frame, result)
        assert annotated.shape == road_frame.shape
        assert annotated.dtype == np.uint8
        assert not np.array_equal(annotated, road_frame)

    def test_failed_lane_skips_lane_rules_but_completes(
        self,
        road_frame: np.ndarray,
        vehicle_detection_module,
        traffic_sign_module,
        traffic_signal_module,
    ) -> None:
        from unittest.mock import MagicMock

        from src.modules.lane_detection import LaneDetectionModule
        from src.modules.yolop.output_schema import LaneDetectionResult
        from src.pipeline.orchestrator import PipelineConfig, PipelineOrchestrator

        lane_module = MagicMock(spec=LaneDetectionModule)
        lane_module.is_initialized = True
        lane_module.predict.return_value = LaneDetectionResult.empty(
            raw_status="init_failed"
        )
        lane_module.cleanup.return_value = None

        orch = PipelineOrchestrator(
            lane_module=lane_module,
            vehicle_module=vehicle_detection_module,
            sign_module=traffic_sign_module,
            signal_module=traffic_signal_module,
            config=PipelineConfig(auto_initialize=False, collect_timing=True),
        )

        result = orch.run_frame(road_frame)
        assert result.scene_state.lane_ok is False
        assert result.decision.recommendation == ADASRecommendation.STOP

    def test_cleanup_releases_modules(self, pipeline_orchestrator: PipelineOrchestrator) -> None:
        pipeline_orchestrator.cleanup()
        assert pipeline_orchestrator.lane_module.is_initialized is False
        assert pipeline_orchestrator.vehicle_module.is_initialized is False

    def test_invalid_frame_raises(self, pipeline_orchestrator: PipelineOrchestrator) -> None:
        with pytest.raises(ValueError):
            pipeline_orchestrator.run_frame(np.zeros((10, 10), dtype=np.uint8))
