"""ADAS decision types — recommendations, rule hits, and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ADASRecommendation(str, Enum):
    """High-level driving recommendation emitted by the decision engine."""

    PROCEED = "PROCEED"
    STOP = "STOP"
    SLOW_DOWN = "SLOW_DOWN"
    KEEP_LANE = "KEEP_LANE"
    WARNING = "WARNING"


@dataclass
class RuleHit:
    """Single fired rule contributing to the final decision."""

    rule_id: str
    recommendation: ADASRecommendation
    priority: int
    message: str
    source_module: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "rule_id": self.rule_id,
            "recommendation": self.recommendation.value,
            "priority": self.priority,
            "message": self.message,
            "source_module": self.source_module,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class DecisionResult:
    """Final ADAS recommendation for one frame."""

    recommendation: ADASRecommendation
    priority: int
    rule_hits: list[RuleHit] = field(default_factory=list)
    primary_message: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "recommendation": self.recommendation.value,
            "priority": self.priority,
            "primary_message": self.primary_message,
            "explanation": self.explanation,
            "rule_hits": [hit.to_dict() for hit in self.rule_hits],
        }
