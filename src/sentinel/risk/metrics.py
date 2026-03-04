import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_volatility(returns: pd.Series) -> float:
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(price_series: pd.Series) -> float:
    prices = price_series.dropna()
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())


def herfindahl_index(weights: dict[str, float]) -> float:
    """
    Concentration measure: sum(w_i^2). Range: (0, 1].
    """
    w = np.array(list(weights.values()), dtype=float)
    if w.sum() <= 0:
        raise ValueError("Weights must sum to a positive value.")
    w = w / w.sum()
    return float(np.sum(w**2))