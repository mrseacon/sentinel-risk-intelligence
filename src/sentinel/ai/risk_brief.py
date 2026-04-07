from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskBrief:
    summary: str
    key_risks: list[str]
    interpretation: str


def build_risk_brief_prompt(
    portfolio: dict,
    volatility: float,
    var: float,
    cvar: float,
    context_text: str,
) -> str:
    return f"""
You are a senior portfolio risk analyst.

Portfolio:
{portfolio}

Risk Metrics:
- Volatility: {volatility:.2%}
- VaR: {var:.2%}
- CVaR: {cvar:.2%}

Market Context:
{context_text}

Task:
Generate a concise executive risk brief.

Return STRICT JSON:
{{
  "summary": "...",
  "key_risks": ["...", "..."],
  "interpretation": "..."
}}
""".strip()


def parse_risk_brief(text: str) -> RiskBrief:
    import json

    try:
        data = json.loads(text)
        return RiskBrief(
            summary=data.get("summary", ""),
            key_risks=data.get("key_risks", []),
            interpretation=data.get("interpretation", ""),
        )
    except Exception:
        return RiskBrief(
            summary="Failed to parse AI response.",
            key_risks=[],
            interpretation=text[:500],
        )
