"""Rule-based ADAS decision functions mapping SceneState to rule hits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .scene_state import SceneState
from .types import ADASRecommendation, RuleHit


@dataclass
class DecisionConfig:
    """Thresholds for decision rule evaluation."""

    red_light_confidence: float = 0.70
    stop_sign_confidence: float = 0.70
    stop_sign_lower_frame_fraction: float = 0.40
    vulnerable_user_confidence: float = 0.60
    large_vehicle_area_ratio: float = 0.08
    lane_offset_warn_px: float = 35.0
    drivable_overlap_threshold: float = 0.15


RuleEvaluator = Callable[[SceneState, DecisionConfig], RuleHit | None]

RULE_REGISTRY: list[RuleEvaluator] = []


def _register(rule_fn: RuleEvaluator) -> RuleEvaluator:
    RULE_REGISTRY.append(rule_fn)
    return rule_fn


def evaluate_all(scene: SceneState, config: DecisionConfig) -> list[RuleHit]:
    """Run all registered rules; return only non-None hits."""
    hits: list[RuleHit] = []
    for rule_fn in RULE_REGISTRY:
        hit = rule_fn(scene, config)
        if hit is not None:
            hits.append(hit)
    return hits


def bbox_lower_fraction(center_y: float, frame_height: int) -> float:
    """Return normalized vertical position (0=top, 1=bottom)."""
    if frame_height <= 0:
        return 0.0
    return center_y / frame_height


def bbox_area_ratio(area: int, frame_shape: tuple[int, int]) -> float:
    """Detection area divided by frame area."""
    frame_height, frame_width = frame_shape
    frame_area = frame_height * frame_width
    if frame_area <= 0:
        return 0.0
    return area / frame_area


def overlaps_drivable_mask(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    drivable_mask: np.ndarray,
    min_overlap: float = 0.15,
) -> bool:
    """True if >= min_overlap fraction of bbox interior lies on drivable pixels."""
    if drivable_mask is None or drivable_mask.size == 0:
        return False

    height, width = drivable_mask.shape[:2]
    x1_clipped = max(0, min(x1, width - 1))
    y1_clipped = max(0, min(y1, height - 1))
    x2_clipped = max(0, min(x2, width))
    y2_clipped = max(0, min(y2, height))

    if x2_clipped <= x1_clipped or y2_clipped <= y1_clipped:
        return False

    region = drivable_mask[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
    if region.size == 0:
        return False

    binary = (region > 0).astype(np.uint8)
    overlap_ratio = float(binary.sum()) / float(region.size)
    return overlap_ratio >= min_overlap


def _frame_shape(scene: SceneState) -> tuple[int, int] | None:
    return scene.frame_shape


@_register
def rule_r01_red_light_stop(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    if not scene.signals_ok or scene.signals is None:
        return None

    summary = scene.signals.summary
    controlling = summary.controlling_signal
    if summary.dominant_state != "red_light":
        return None

    confidence = controlling.confidence if controlling is not None else 0.0
    if confidence < config.red_light_confidence:
        return None

    return RuleHit(
        rule_id="R01_red_light_stop",
        recommendation=ADASRecommendation.STOP,
        priority=100,
        message="Red traffic light detected — stop required",
        source_module="traffic_signal",
        confidence=confidence,
    )


@_register
def rule_r02_stop_sign(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    if not scene.signs_ok or scene.signs is None:
        return None

    frame_shape = _frame_shape(scene)
    if frame_shape is None:
        return None

    frame_height, _ = frame_shape
    lower_bound = frame_height * config.stop_sign_lower_frame_fraction

    best: Any | None = None
    for det in scene.signs.detections:
        if det.sign_label != "stop":
            continue
        if det.confidence < config.stop_sign_confidence:
            continue
        if det.bbox.center_y < lower_bound:
            continue
        if best is None or det.confidence > best.confidence:
            best = det

    if best is None:
        nearest = scene.signs.summary.nearest_sign
        if (
            nearest is not None
            and nearest.sign_label == "stop"
            and nearest.confidence >= config.stop_sign_confidence
            and nearest.bbox.center_y >= lower_bound
        ):
            best = nearest

    if best is None:
        return None

    return RuleHit(
        rule_id="R02_stop_sign",
        recommendation=ADASRecommendation.STOP,
        priority=95,
        message="Stop sign detected — stop required",
        source_module="traffic_sign",
        confidence=best.confidence,
    )


@_register
def rule_r03_pedestrian_on_drivable(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    if not scene.lane_ok or not scene.vehicles_ok:
        return None
    if scene.lane is None or scene.vehicles is None:
        return None

    drivable = scene.lane.drivable_mask
    if drivable is None:
        return None

    for det in scene.vehicles.detections:
        if det.label != "person":
            continue
        bbox = det.bbox
        if overlaps_drivable_mask(
            bbox.x1,
            bbox.y1,
            bbox.x2,
            bbox.y2,
            drivable,
            min_overlap=config.drivable_overlap_threshold,
        ):
            return RuleHit(
                rule_id="R03_pedestrian_on_drivable",
                recommendation=ADASRecommendation.STOP,
                priority=90,
                message="Pedestrian detected on drivable area — stop required",
                source_module="vehicle_detection",
                confidence=det.confidence,
            )
    return None


@_register
def rule_r04_yellow_light_caution(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    _ = config
    if not scene.signals_ok or scene.signals is None:
        return None

    if scene.signals.summary.dominant_state != "yellow_light":
        return None

    controlling = scene.signals.summary.controlling_signal
    confidence = controlling.confidence if controlling is not None else 1.0

    return RuleHit(
        rule_id="R04_yellow_light_caution",
        recommendation=ADASRecommendation.SLOW_DOWN,
        priority=70,
        message="Yellow traffic light — slow down",
        source_module="traffic_signal",
        confidence=confidence,
    )


@_register
def rule_r05_active_speed_limit(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    _ = config
    if not scene.signs_ok or scene.signs is None:
        return None

    speed_limit = scene.signs.summary.active_speed_limit_kmh
    if speed_limit is None:
        return None

    nearest = scene.signs.summary.nearest_sign
    confidence = nearest.confidence if nearest is not None else 1.0

    return RuleHit(
        rule_id="R05_active_speed_limit",
        recommendation=ADASRecommendation.SLOW_DOWN,
        priority=65,
        message=f"Speed limit {speed_limit} km/h detected — slow down",
        source_module="traffic_sign",
        confidence=confidence,
    )


@_register
def rule_r06_vulnerable_road_user(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    if not scene.vehicles_ok or scene.vehicles is None:
        return None

    frame_shape = _frame_shape(scene)
    if frame_shape is None:
        return None

    frame_height, _ = frame_shape
    lower_third = frame_height * (2.0 / 3.0)

    best_confidence = 0.0
    best_label = ""
    for det in scene.vehicles.detections:
        if det.label not in {"person", "bicycle"}:
            continue
        if det.confidence < config.vulnerable_user_confidence:
            continue
        if det.bbox.center_y < lower_third:
            continue
        if det.confidence > best_confidence:
            best_confidence = det.confidence
            best_label = det.label

    if best_confidence <= 0.0:
        return None

    return RuleHit(
        rule_id="R06_vulnerable_road_user",
        recommendation=ADASRecommendation.SLOW_DOWN,
        priority=60,
        message=f"Vulnerable road user ({best_label}) ahead — slow down",
        source_module="vehicle_detection",
        confidence=best_confidence,
    )


@_register
def rule_r07_pedestrian_crossing_sign(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    _ = config
    if not scene.signs_ok or scene.signs is None:
        return None

    for det in scene.signs.detections:
        if det.sign_label == "pedestrian_crossing":
            return RuleHit(
                rule_id="R07_pedestrian_crossing_sign",
                recommendation=ADASRecommendation.WARNING,
                priority=55,
                message="Pedestrian crossing sign detected — increase caution",
                source_module="traffic_sign",
                confidence=det.confidence,
            )
    return None


@_register
def rule_r08_large_vehicle_proximity(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    if not scene.vehicles_ok or scene.vehicles is None:
        return None

    frame_shape = _frame_shape(scene)
    if frame_shape is None:
        return None

    nearest = scene.vehicles.summary.nearest_object
    if nearest is None:
        return None
    if nearest.label not in {"truck", "bus"}:
        return None

    ratio = bbox_area_ratio(nearest.bbox.area, frame_shape)
    if ratio < config.large_vehicle_area_ratio:
        return None

    return RuleHit(
        rule_id="R08_large_vehicle_proximity",
        recommendation=ADASRecommendation.WARNING,
        priority=50,
        message=f"Large vehicle ({nearest.label}) in proximity — warning",
        source_module="vehicle_detection",
        confidence=nearest.confidence,
    )


@_register
def rule_r09_lane_departure(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    _ = config
    if not scene.lane_ok or scene.lane is None:
        return None

    if not scene.lane.lane_departure:
        return None

    return RuleHit(
        rule_id="R09_lane_departure",
        recommendation=ADASRecommendation.WARNING,
        priority=45,
        message="Lane departure detected — warning",
        source_module="lane_detection",
        confidence=1.0,
    )


@_register
def rule_r10_lane_offset_correct(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    if not scene.lane_ok or scene.lane is None:
        return None

    if scene.lane.lane_departure:
        return None

    offset = scene.lane.vehicle_offset
    if offset is None:
        return None
    if abs(offset) <= config.lane_offset_warn_px:
        return None

    return RuleHit(
        rule_id="R10_lane_offset_correct",
        recommendation=ADASRecommendation.KEEP_LANE,
        priority=40,
        message="Vehicle offset from lane center — keep lane",
        source_module="lane_detection",
        confidence=1.0,
    )


@_register
def rule_r11_green_proceed(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    _ = config
    if not scene.signals_ok or scene.signals is None:
        return None

    summary = scene.signals.summary
    if summary.dominant_state != "green_light":
        return None
    if summary.has_stop_state:
        return None

    controlling = summary.controlling_signal
    confidence = controlling.confidence if controlling is not None else 1.0

    return RuleHit(
        rule_id="R11_green_proceed",
        recommendation=ADASRecommendation.PROCEED,
        priority=10,
        message="Green traffic light — proceed when clear",
        source_module="traffic_signal",
        confidence=confidence,
    )


@_register
def rule_r12_default_proceed(scene: SceneState, config: DecisionConfig) -> RuleHit | None:
    _ = scene
    _ = config
    return RuleHit(
        rule_id="R12_default_proceed",
        recommendation=ADASRecommendation.PROCEED,
        priority=1,
        message="No higher-priority rule triggered — proceed with caution",
        source_module="decision_engine",
        confidence=1.0,
    )
