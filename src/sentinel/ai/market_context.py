from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MarketContext:
    bullets: list[str]
    classification: str
    sentiment: int  # -2..+2
    confidence: float  # 0..1


def build_market_context_prompt(headlines: str, portfolio_tickers: list[str]) -> str:
    tickers = ", ".join(portfolio_tickers)
    return f"""
You are an institutional risk analyst.

Given the following market headlines, produce a compact market risk brief relevant to this portfolio: {tickers}.

Return STRICT JSON with the following schema:
{{
  "bullets": ["...","...","...","...","..."],
  "classification": "Macro|Sector|Company|Geopolitical|Mixed",
  "sentiment": -2|-1|0|1|2,
  "confidence": 0.0-1.0
}}

Headlines:
{headlines}
""".strip()


def parse_market_context(text: str) -> MarketContext:
    """
    Parse the STRICT JSON response from the LLM.
    If parsing fails, raise ValueError so caller can fall back.
    """
    try:
        obj: dict[str, Any] = json.loads(text)
        bullets = list(obj["bullets"])
        classification = str(obj["classification"])
        sentiment = int(obj["sentiment"])
        confidence = float(obj["confidence"])

        if not (-2 <= sentiment <= 2):
            raise ValueError("Sentiment out of range.")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence out of range.")
        if len(bullets) < 3:
            raise ValueError("Not enough bullets.")

        return MarketContext(
            bullets=bullets[:5],
            classification=classification,
            sentiment=sentiment,
            confidence=confidence,
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse market context: {exc}") from exc
