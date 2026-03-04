import numpy as np
import pandas as pd
from scipy.stats import norm


def historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    percentile = 1 - confidence_level
    return float(np.percentile(returns, percentile * 100))


def historical_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Historical CVaR (Expected Shortfall): mean of returns worse than (<=) VaR.
    Returns a negative number (loss) in most cases.
    """
    var_value = historical_var(returns, confidence_level=confidence_level)
    tail_losses = returns[returns <= var_value]
    if tail_losses.empty:
        return float(var_value)
    return float(tail_losses.mean())


def parametric_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    z = float(norm.ppf(1 - confidence_level))
    return mean + z * std


def parametric_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Gaussian CVaR (Expected Shortfall).
    """
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    alpha = 1 - confidence_level
    z = float(norm.ppf(alpha))
    # ES = mu - sigma * (phi(z)/alpha)  for left tail; here returns convention
    es = mean - std * (norm.pdf(z) / alpha)
    return float(es)