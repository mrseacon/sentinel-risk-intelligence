import pandas as pd
from dataclasses import dataclass

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

@dataclass
class Portfolio:
    weights: dict[str, float]

    def normalize_weights(self) -> dict[str, float]:
        total = sum(self.weights.values())
        return {k: v / total for k, v in self.weights.items()}