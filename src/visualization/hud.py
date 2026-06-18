"""Decision HUD panel — recommendation text and rule explanations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from ..decision.types import ADASRecommendation, DecisionResult

if TYPE_CHECKING:
    from ..decision.scene_state import SceneState

Frame = np.ndarray

_RECOMMENDATION_COLORS_BGR: dict[ADASRecommendation, tuple[int, int, int]] = {
    ADASRecommendation.PROCEED: (0, 200, 0),
    ADASRecommendation.STOP: (0, 0, 220),
    ADASRecommendation.SLOW_DOWN: (0, 165, 255),
    ADASRecommendation.KEEP_LANE: (255, 200, 0),
    ADASRecommendation.WARNING: (0, 140, 255),
}

_RECOMMENDATION_DISPLAY: dict[ADASRecommendation, str] = {
    ADASRecommendation.PROCEED: "PROCEED",
    ADASRecommendation.STOP: "STOP",
    ADASRecommendation.SLOW_DOWN: "SLOW DOWN",
    ADASRecommendation.KEEP_LANE: "KEEP LANE",
    ADASRecommendation.WARNING: "WARNING",
}


def draw_decision_hud(
    frame: Frame,
    decision: DecisionResult,
    *,
    scene_state: SceneState | None = None,
    show_rule_list: bool = True,
    panel_position: str = "top-left",
) -> Frame:
    """Render recommendation banner, primary message, and optional rule list."""
    output = frame.copy()
    height, width = output.shape[:2]

    banner_color = _RECOMMENDATION_COLORS_BGR.get(
        decision.recommendation,
        (80, 80, 80),
    )
    banner_text = _RECOMMENDATION_DISPLAY.get(
        decision.recommendation,
        decision.recommendation.value,
    )

    banner_height = 48
    cv2.rectangle(output, (0, 0), (width, banner_height), banner_color, thickness=-1)
    cv2.putText(
        output,
        banner_text,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    if decision.primary_message:
        cv2.putText(
            output,
            _truncate(decision.primary_message, 70),
            (16, banner_height + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            lineType=cv2.LINE_AA,
        )

    if show_rule_list and decision.rule_hits:
        y_offset = height - 16
        for hit in decision.rule_hits[:3]:
            line = f"{hit.rule_id}: {hit.message}"
            cv2.putText(
                output,
                _truncate(line, 60),
                (16, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                lineType=cv2.LINE_AA,
            )
            y_offset -= 20

    if scene_state is not None:
        _draw_module_status_strip(output, scene_state, width, height)

    _ = panel_position
    return output


def _draw_module_status_strip(
    frame: Frame,
    scene_state: SceneState,
    width: int,
    height: int,
) -> None:
    """Draw module ok/fail indicators at bottom-right."""
    labels = [
        ("L", scene_state.lane_ok),
        ("V", scene_state.vehicles_ok),
        ("S", scene_state.signs_ok),
        ("T", scene_state.signals_ok),
    ]
    x = width - 120
    y = height - 12
    for label, ok in labels:
        color = (0, 200, 0) if ok else (0, 0, 200)
        cv2.circle(frame, (x, y), 8, color, thickness=-1)
        cv2.putText(
            frame,
            label,
            (x - 5, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
        x += 28


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
