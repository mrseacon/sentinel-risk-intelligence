from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def _portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    return float(
        np.sqrt(weights.T @ cov_matrix.values @ weights) * np.sqrt(TRADING_DAYS)
    )


def _portfolio_return(weights: np.ndarray, mean_returns: pd.Series) -> float:
    return float(np.sum(mean_returns.values * weights) * TRADING_DAYS)


def _negative_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> float:
    port_return = _portfolio_return(weights, mean_returns)
    port_vol = _portfolio_volatility(weights, cov_matrix)

    if port_vol == 0:
        return 1e6

    sharpe = (port_return - risk_free_rate) / port_vol
    return -float(sharpe)


def optimize_min_variance(
    returns: pd.DataFrame,
    max_weight: float = 0.6,
) -> pd.Series:
    n_assets = returns.shape[1]
    cov_matrix = returns.cov()

    initial_weights = np.ones(n_assets) / n_assets
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        _portfolio_volatility,
        initial_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    return pd.Series(result.x, index=returns.columns, name="min_variance_weights")


def optimize_max_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.6,
) -> pd.Series:
    n_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    initial_weights = np.ones(n_assets) / n_assets
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        _negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    return pd.Series(result.x, index=returns.columns, name="max_sharpe_weights")
