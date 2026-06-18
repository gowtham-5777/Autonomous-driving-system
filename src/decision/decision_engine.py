"""Rule-based decision engine — evaluates rules and arbitrates recommendations."""

from __future__ import annotations

import logging

from ..utils.model_paths import get_decision_config
from .rules import DecisionConfig, evaluate_all
from .scene_state import SceneState
from .types import ADASRecommendation, DecisionResult, RuleHit

logger = logging.getLogger(__name__)

_STOP_CLASS_RECOMMENDATIONS = frozenset({ADASRecommendation.STOP})


class DecisionEngine:
    """Evaluates ADAS rules and returns a single arbitrated recommendation."""

    def __init__(self, config: DecisionConfig | None = None) -> None:
        if config is None:
            config_dict = get_decision_config()
            self.config = DecisionConfig(**config_dict)
        else:
            self.config = config

    def evaluate(self, scene: SceneState) -> DecisionResult:
        """Evaluate all rules and arbitrate a final recommendation."""
        hits = evaluate_all(scene, self.config)
        non_default = [hit for hit in hits if hit.rule_id != "R12_default_proceed"]
        arbitration_hits = non_default if non_default else hits
        return self.arbitrate(arbitration_hits)

    @staticmethod
    def arbitrate(hits: list[RuleHit]) -> DecisionResult:
        """Select the highest-priority rule; tie-break by confidence."""
        if not hits:
            default = RuleHit(
                rule_id="R12_default_proceed",
                recommendation=ADASRecommendation.PROCEED,
                priority=1,
                message="No rules evaluated — proceed with caution",
                source_module="decision_engine",
            )
            hits = [default]

        winning = max(hits, key=lambda hit: (hit.priority, hit.confidence))
        sorted_hits = sorted(hits, key=lambda hit: (-hit.priority, -hit.confidence))

        explanation_lines = [hit.message for hit in sorted_hits]
        explanation = "\n".join(explanation_lines)

        if winning.recommendation in _STOP_CLASS_RECOMMENDATIONS:
            logger.info(
                "Decision STOP — rule=%s priority=%d",
                winning.rule_id,
                winning.priority,
            )

        return DecisionResult(
            recommendation=winning.recommendation,
            priority=winning.priority,
            rule_hits=sorted_hits,
            primary_message=winning.message,
            explanation=explanation,
        )
