"""Unit tests for DecisionEngine arbitration."""

from __future__ import annotations

from src.decision.decision_engine import DecisionEngine
from src.decision.rules import DecisionConfig
from src.decision.scene_state import SceneState
from src.decision.types import ADASRecommendation, RuleHit
from src.modules.yolov8_signal.output_schema import (
    DetectedSignal,
    SignalBoundingBoxData,
    TrafficSignalDetectionResult,
    TrafficSignalSummary,
)


def _signal_bbox() -> SignalBoundingBoxData:
    return SignalBoundingBoxData.from_xyxy(300, 80, 340, 150)


class TestDecisionEngine:
    def test_arbitrate_picks_highest_priority(self) -> None:
        hits = [
            RuleHit(
                rule_id="R12",
                recommendation=ADASRecommendation.PROCEED,
                priority=1,
                message="default",
                source_module="decision_engine",
            ),
            RuleHit(
                rule_id="R01",
                recommendation=ADASRecommendation.STOP,
                priority=100,
                message="stop",
                source_module="traffic_signal",
            ),
        ]
        result = DecisionEngine.arbitrate(hits)
        assert result.recommendation == ADASRecommendation.STOP
        assert result.primary_message == "stop"

    def test_evaluate_excludes_default_when_other_rules_fire(self) -> None:
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
            ),
            raw_status="stub",
        )
        scene = SceneState.from_perception(
            frame_index=0,
            frame_shape=(720, 1280),
            timestamp_ms=None,
            lane=None,
            vehicles=None,
            signs=None,
            signals=signals,
        )
        engine = DecisionEngine()
        result = engine.evaluate(scene)
        assert result.recommendation == ADASRecommendation.STOP
        rule_ids = {hit.rule_id for hit in result.rule_hits}
        assert "R12_default_proceed" not in rule_ids

    def test_evaluate_default_when_no_perception(self) -> None:
        scene = SceneState.from_perception(
            frame_index=0,
            frame_shape=(720, 1280),
            timestamp_ms=None,
            lane=None,
            vehicles=None,
            signs=None,
            signals=None,
        )
        engine = DecisionEngine()
        result = engine.evaluate(scene)
        assert result.recommendation == ADASRecommendation.PROCEED

    def test_empty_arbitration_returns_proceed(self) -> None:
        result = DecisionEngine.arbitrate([])
        assert result.recommendation == ADASRecommendation.PROCEED

    def test_decision_result_to_dict(self) -> None:
        engine = DecisionEngine()
        scene = SceneState.from_perception(
            frame_index=0,
            frame_shape=(100, 100),
            timestamp_ms=None,
            lane=None,
            vehicles=None,
            signs=None,
            signals=None,
        )
        result = engine.evaluate(scene)
        payload = result.to_dict()
        assert payload["recommendation"] == "PROCEED"
        assert isinstance(payload["rule_hits"], list)

    def test_custom_config_threshold(self) -> None:
        config = DecisionConfig(red_light_confidence=0.95)
        controlling = DetectedSignal(
            signal_label="red_light",
            class_id=0,
            confidence=0.80,
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
        scene = SceneState.from_perception(
            frame_index=0,
            frame_shape=(720, 1280),
            timestamp_ms=None,
            lane=None,
            vehicles=None,
            signs=None,
            signals=signals,
        )
        result = DecisionEngine(config=config).evaluate(scene)
        assert result.recommendation == ADASRecommendation.PROCEED
