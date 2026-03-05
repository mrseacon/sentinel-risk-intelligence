from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskScore:
    score: int
    label: str
    drivers: list[str]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_risk_score(
    vol_annual: float,
    max_dd: float,
    var_95: float,
    cvar_95: float,
    concentration_hhi: float | None = None,
) -> RiskScore:
    """
    Inputs are return-based metrics (negative values for loss) except vol.
    Output: 0 (best) .. 100 (worst)
    """

    # Normalize to 0..1 risk factors (heuristic, explainable)
    vol_factor = _clamp(vol_annual / 0.40, 0, 1)  # 40% annual vol = high
    dd_factor = _clamp(abs(max_dd) / 0.50, 0, 1)  # 50% drawdown = extreme
    var_factor = _clamp(abs(var_95) / 0.05, 0, 1)  # 5% daily VaR = high
    cvar_factor = _clamp(abs(cvar_95) / 0.08, 0, 1)  # 8% daily ES = high

    conc_factor = 0.0
    if concentration_hhi is not None:
        # 0.25 = concentrated (e.g., 2-4 assets heavy)
        conc_factor = _clamp((concentration_hhi - 0.10) / 0.20, 0, 1)

    # Weighted sum (transparent)
    raw = (
        0.30 * vol_factor
        + 0.30 * dd_factor
        + 0.20 * var_factor
        + 0.15 * cvar_factor
        + 0.05 * conc_factor
    )
    score = int(round(100 * raw))

    # Labeling
    if score <= 25:
        label = "Low"
    elif score <= 50:
        label = "Moderate"
    elif score <= 75:
        label = "High"
    else:
        label = "Severe"

    # Explain drivers: top contributors
    contributions = {
        "Volatility": 0.30 * vol_factor,
        "Max drawdown": 0.30 * dd_factor,
        "VaR (95%)": 0.20 * var_factor,
        "CVaR (95%)": 0.15 * cvar_factor,
    }
    if concentration_hhi is not None:
        contributions["Concentration"] = 0.05 * conc_factor

    top = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
    drivers = [f"{k} ({v:.2f})" for k, v in top if v > 0]

    return RiskScore(score=score, label=label, drivers=drivers)
