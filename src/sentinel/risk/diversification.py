import numpy as np
import pandas as pd


def diversification_ratio(
    returns: pd.DataFrame,
    weights: dict[str, float],
):

    vol_assets = returns.std()

    cov = returns.cov()

    w = np.array(list(weights.values()))

    portfolio_vol = np.sqrt(w.T @ cov @ w)

    weighted_vol = np.sum(w * vol_assets)

    return weighted_vol / portfolio_vol