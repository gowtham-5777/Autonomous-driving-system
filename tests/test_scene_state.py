"""Unit tests for SceneState aggregation."""

from __future__ import annotations

import numpy as np

from src.decision.scene_state import SceneState
from src.modules.yolop.output_schema import LaneDetectionResult
from src.modules.yolov8.output_schema import VehicleDetectionResult
from src.modules.yolov8_sign.output_schema import TrafficSignDetectionResult
from src.modules.yolov8_signal.output_schema import TrafficSignalDetectionResult


class TestSceneState:
    def test_from_perception_sets_lane_ok_with_mask(self) -> None:
        lane = LaneDetectionResult(
            lane_center_x=320.0,
            lane_mask=np.zeros((480, 640), dtype=np.uint8),
            raw_status="stub_segmentation",
        )
        state = SceneState.from_perception(
            frame_index=1,
            frame_shape=(480, 640),
            timestamp_ms=100.0,
            lane=lane,
            vehicles=VehicleDetectionResult.empty(raw_status="stub"),
            signs=TrafficSignDetectionResult.empty(raw_status="stub"),
            signals=TrafficSignalDetectionResult.empty(raw_status="stub"),
        )

        assert state.lane_ok is True
        assert state.vehicles_ok is True
        assert len(state.module_statuses) == 4

    def test_lane_not_ok_on_init_failed(self) -> None:
        lane = LaneDetectionResult.empty(raw_status="init_failed")
        state = SceneState.from_perception(
            frame_index=0,
            frame_shape=(480, 640),
            timestamp_ms=None,
            lane=lane,
            vehicles=None,
            signs=None,
            signals=None,
        )
        assert state.lane_ok is False

    def test_to_dict_omits_mask_arrays(self) -> None:
        lane = LaneDetectionResult(
            lane_center_x=100.0,
            lane_mask=np.ones((10, 10), dtype=np.uint8),
            drivable_mask=np.ones((10, 10), dtype=np.uint8),
            raw_status="parsed",
        )
        state = SceneState.from_perception(
            frame_index=0,
            frame_shape=(10, 10),
            timestamp_ms=None,
            lane=lane,
            vehicles=None,
            signs=None,
            signals=None,
        )
        payload = state.to_dict()
        assert payload["lane"]["lane_mask"]["present"] is True
        assert payload["lane"]["lane_mask"]["shape"] == [10, 10]
        assert "preprocessed_edges" not in payload["lane"]

    def test_perception_dict_includes_module_outputs(self) -> None:
        vehicles = VehicleDetectionResult.empty(raw_status="stub")
        state = SceneState.from_perception(
            frame_index=5,
            frame_shape=(720, 1280),
            timestamp_ms=50.0,
            lane=None,
            vehicles=vehicles,
            signs=None,
            signals=None,
        )
        perception = state.perception_dict()
        assert perception["frame_index"] == 5
        assert perception["vehicles"]["raw_status"] == "stub"

    def test_module_ok_false_for_pipeline_error(self) -> None:
        state = SceneState.from_perception(
            frame_index=0,
            frame_shape=(480, 640),
            timestamp_ms=None,
            lane=None,
            vehicles=VehicleDetectionResult.empty(raw_status="pipeline_error"),
            signs=None,
            signals=None,
        )
        assert state.vehicles_ok is False
