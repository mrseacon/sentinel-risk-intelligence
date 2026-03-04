import numpy as np
import pandas as pd
from scipy.stats import norm

def historical_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate historical Value-at-Risk.

    Parameters
    ----------
    returns : pd.Series
    confidence_level : float

    Returns
    -------
    float
    """

    percentile = 1 - confidence_level

    return np.percentile(returns, percentile * 100)

def parametric_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Parametric (Gaussian) VaR.
    """

    mean = returns.mean()

    std = returns.std()

    z = norm.ppf(1 - confidence_level)

    return mean + z * std