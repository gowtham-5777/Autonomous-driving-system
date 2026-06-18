"""Unit tests for ADAS decision rules."""

from __future__ import annotations

import numpy as np
import pytest

from src.decision.rules import (
    DecisionConfig,
    bbox_area_ratio,
    overlaps_drivable_mask,
    rule_r01_red_light_stop,
    rule_r02_stop_sign,
    rule_r04_yellow_light_caution,
    rule_r09_lane_departure,
    rule_r11_green_proceed,
    rule_r12_default_proceed,
)
from src.decision.scene_state import SceneState
from src.decision.types import ADASRecommendation
from src.modules.yolop.output_schema import LaneDetectionResult
from src.modules.yolov8_sign.output_schema import (
    DetectedSign,
    SignBoundingBoxData,
    TrafficSignDetectionResult,
    TrafficSignDetectionSummary,
)
from src.modules.yolov8_signal.output_schema import (
    DetectedSignal,
    SignalBoundingBoxData,
    TrafficSignalDetectionResult,
    TrafficSignalSummary,
)


def _bbox(y2: int = 400, x_center: int = 320) -> SignBoundingBoxData:
    return SignBoundingBoxData.from_xyxy(x_center - 40, 200, x_center + 40, y2)


def _signal_bbox(y2: int = 150) -> SignalBoundingBoxData:
    return SignalBoundingBoxData.from_xyxy(300, 80, 340, y2)


def _scene(
  *,
    lane: LaneDetectionResult | None = None,
    vehicles: VehicleDetectionResult | None = None,
    signs: TrafficSignDetectionResult | None = None,
    signals: TrafficSignalDetectionResult | None = None,
) -> SceneState:
    return SceneState.from_perception(
        frame_index=0,
        frame_shape=(720, 1280),
        timestamp_ms=None,
        lane=lane,
        vehicles=vehicles,
        signs=signs,
        signals=signals,
    )


class TestSpatialHelpers:
    def test_bbox_area_ratio(self) -> None:
        ratio = bbox_area_ratio(100 * 100, (100, 100))
        assert ratio == pytest.approx(1.0)

    def test_overlaps_drivable_mask(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50:80, 40:60] = 1
        assert overlaps_drivable_mask(40, 50, 60, 80, mask, min_overlap=0.5)


class TestDecisionRules:
    def test_r01_red_light_stop(self) -> None:
        controlling = DetectedSignal(
            signal_label="red_light",
            class_id=0,
            confidence=0.9,
            bbox=_signal_bbox(),
            is_stop_state=True,
        )
        signals = TrafficSignalDetectionResult(
            summary=TrafficSignalSummary(
                dominant_state="red_light",
                controlling_signal=controlling,
                has_stop_state=True,
            ),
            raw_status="stub",
        )
        hit = rule_r01_red_light_stop(_scene(signals=signals), DecisionConfig())
        assert hit is not None
        assert hit.recommendation == ADASRecommendation.STOP

    def test_r01_below_confidence(self) -> None:
        controlling = DetectedSignal(
            signal_label="red_light",
            class_id=0,
            confidence=0.5,
            bbox=_signal_bbox(),
            is_stop_state=True,
        )
        signals = TrafficSignalDetectionResult(
            summary=TrafficSignalSummary(
                dominant_state="red_light",
                controlling_signal=controlling,
            ),
            raw_status="stub",
        )
        hit = rule_r01_red_light_stop(_scene(signals=signals), DecisionConfig())
        assert hit is None

    def test_r02_stop_sign_lower_frame(self) -> None:
        stop_sign = DetectedSign(
            sign_label="stop",
            class_id=0,
            confidence=0.85,
            bbox=_bbox(y2=600),
            is_regulatory=True,
        )
        signs = TrafficSignDetectionResult(
            detections=[stop_sign],
            summary=TrafficSignDetectionSummary(
                nearest_sign=stop_sign,
                total_count=1,
            ),
            raw_status="stub",
        )
        hit = rule_r02_stop_sign(_scene(signs=signs), DecisionConfig())
        assert hit is not None
        assert hit.rule_id == "R02_stop_sign"

    def test_r04_yellow_slow_down(self) -> None:
        signals = TrafficSignalDetectionResult(
            summary=TrafficSignalSummary(dominant_state="yellow_light"),
            raw_status="stub",
        )
        hit = rule_r04_yellow_light_caution(_scene(signals=signals), DecisionConfig())
        assert hit is not None
        assert hit.recommendation == ADASRecommendation.SLOW_DOWN

    def test_r09_lane_departure_warning(self) -> None:
        lane = LaneDetectionResult(
            lane_center_x=400.0,
            lane_mask=np.zeros((720, 1280), dtype=np.uint8),
            lane_departure=True,
            raw_status="parsed",
        )
        hit = rule_r09_lane_departure(_scene(lane=lane), DecisionConfig())
        assert hit is not None
        assert hit.recommendation == ADASRecommendation.WARNING

    def test_r11_green_proceed(self) -> None:
        signals = TrafficSignalDetectionResult(
            summary=TrafficSignalSummary(
                dominant_state="green_light",
                has_proceed_state=True,
            ),
            raw_status="stub",
        )
        hit = rule_r11_green_proceed(_scene(signals=signals), DecisionConfig())
        assert hit is not None
        assert hit.recommendation == ADASRecommendation.PROCEED

    def test_r12_default_proceed(self) -> None:
        hit = rule_r12_default_proceed(_scene(), DecisionConfig())
        assert hit is not None
        assert hit.recommendation == ADASRecommendation.PROCEED

    def test_conflict_red_beats_green_via_engine(self) -> None:
        from src.decision.decision_engine import DecisionEngine

        red = DetectedSignal(
            signal_label="red_light",
            class_id=0,
            confidence=0.9,
            bbox=_signal_bbox(),
            is_stop_state=True,
        )
        green = DetectedSignal(
            signal_label="green_light",
            class_id=2,
            confidence=0.85,
            bbox=_signal_bbox(y2=200),
            is_proceed_state=True,
        )
        signals = TrafficSignalDetectionResult(
            detections=[red, green],
            summary=TrafficSignalSummary(
                dominant_state="red_light",
                controlling_signal=red,
                has_stop_state=True,
                has_proceed_state=True,
            ),
            raw_status="stub",
        )
        engine = DecisionEngine()
        result = engine.evaluate(_scene(signals=signals))
        assert result.recommendation == ADASRecommendation.STOP
        assert result.rule_hits[0].rule_id == "R01_red_light_stop"
