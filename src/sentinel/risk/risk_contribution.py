import numpy as np
import pandas as pd


def portfolio_risk_contribution(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:

    cov = returns.cov()

    w = np.array(list(weights.values()))

    portfolio_var = w.T @ cov @ w

    marginal_risk = cov @ w

    contribution = w * marginal_risk / portfolio_var

    return pd.Series(contribution, index=returns.columns)
