import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_volatility(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> float:

    cov_matrix = returns.cov()

    w = np.array(list(weights.values()))

    port_var = w.T @ cov_matrix @ w

    return float(np.sqrt(port_var) * np.sqrt(TRADING_DAYS))


def portfolio_var(
    portfolio_returns,
    confidence=0.95,
):

    percentile = 1 - confidence

    return np.percentile(portfolio_returns, percentile * 100)
