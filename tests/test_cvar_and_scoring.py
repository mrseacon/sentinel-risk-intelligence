import numpy as np
import pandas as pd
import pytest

from sentinel.risk.scoring import compute_risk_score
from sentinel.risk.stress import apply_market_shock
from sentinel.risk.var import historical_cvar, historical_var


def test_historical_cvar_is_worse_than_var_for_normal_returns():
    returns = pd.Series(np.random.normal(0, 0.01, 2000))
    var95 = historical_var(returns, 0.95)
    cvar95 = historical_cvar(returns, 0.95)

    # CVaR should be <= VaR (more negative / worse tail)
    assert cvar95 <= var95


def test_risk_score_range():
    score = compute_risk_score(
        vol_annual=0.20, max_dd=-0.20, var_95=-0.02, cvar_95=-0.03
    )
    assert 0 <= score.score <= 100
    assert score.label in {"Low", "Moderate", "High", "Severe"}


def test_stress_market_shock():
    prices = pd.Series([100, 101, 99, 105])
    res = apply_market_shock(prices, -0.10)
    assert res.scenario.startswith("Market shock")
    assert res.pnl_pct == pytest.approx(-0.10, abs=1e-12)
