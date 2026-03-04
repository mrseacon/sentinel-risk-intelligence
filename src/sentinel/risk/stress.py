from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class StressResult:
    scenario: str
    shocked_last_price: float
    shocked_return: float
    pnl_pct: float


def apply_market_shock(price_series: pd.Series, shock_pct: float) -> StressResult:
    """
    Apply an instantaneous market shock to the last available price.
    shock_pct: e.g. -0.10 for -10%
    """
    last_price = float(price_series.dropna().iloc[-1])
    shocked_price = last_price * (1 + shock_pct)
    shocked_return = (shocked_price / last_price) - 1
    return StressResult(
        scenario=f"Market shock {shock_pct:.0%}",
        shocked_last_price=shocked_price,
        shocked_return=float(shocked_return),
        pnl_pct=float(shocked_return),
    )


def apply_single_asset_crash(price_series: pd.Series, crash_pct: float) -> StressResult:
    """
    Crash a single asset by crash_pct at the last price point.
    """
    last_price = float(price_series.dropna().iloc[-1])
    shocked_price = last_price * (1 + crash_pct)
    shocked_return = (shocked_price / last_price) - 1
    return StressResult(
        scenario=f"Single-asset crash {crash_pct:.0%}",
        shocked_last_price=shocked_price,
        shocked_return=float(shocked_return),
        pnl_pct=float(shocked_return),
    )