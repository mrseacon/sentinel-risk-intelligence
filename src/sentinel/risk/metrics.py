import numpy as np
import pandas as pd


TRADING_DAYS = 252


def annualized_volatility(returns: pd.Series) -> float:
    """
    Calculate annualized volatility.

    Parameters
    ----------
    returns : pd.Series

    Returns
    -------
    float
    """

    return np.std(returns) * np.sqrt(TRADING_DAYS)

def max_drawdown(price_series: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Parameters
    ----------
    price_series : pd.Series

    Returns
    -------
    float
    """

    cumulative_max = price_series.cummax()

    drawdown = (price_series - cumulative_max) / cumulative_max

    return drawdown.min()