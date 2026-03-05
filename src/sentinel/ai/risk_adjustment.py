from __future__ import annotations

from dataclasses import dataclass

from sentinel.ai.market_context import MarketContext


@dataclass(frozen=True)
class AIRiskAdjustment:
    delta_score: int
    rationale: str


def compute_ai_adjustment(context: MarketContext) -> AIRiskAdjustment:
    """
    Transparent rules (NOT black box):
    - Negative sentiment increases risk.
    - Higher confidence strengthens the adjustment.
    """
    # Base adjustment from sentiment
    base = {2: -6, 1: -3, 0: 0, -1: 4, -2: 8}[context.sentiment]
    scaled = int(round(base * context.confidence))

    rationale = (
        f"AI signal: sentiment={context.sentiment}, "
        f"confidence={context.confidence:.2f}, classification={context.classification}"
    )

    return AIRiskAdjustment(delta_score=scaled, rationale=rationale)
