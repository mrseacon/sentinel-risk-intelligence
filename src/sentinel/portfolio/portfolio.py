import pandas as pd


def calculate_returns(price_series: pd.Series) -> pd.Series:
    """
    Calculate daily log returns.

    Parameters
    ----------
    price_series : pd.Series

    Returns
    -------
    pd.Series
    """

    returns = price_series.pct_change().dropna()

    return returns