import pandas as pd


def calculate_asset_returns(price_df: pd.DataFrame) -> pd.DataFrame:

    return price_df.pct_change().dropna()


def portfolio_returns(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:

    weights_series = pd.Series(weights)

    weights_series = weights_series / weights_series.sum()

    port_returns = returns @ weights_series

    return port_returns
