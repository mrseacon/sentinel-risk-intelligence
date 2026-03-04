import pandas as pd
import yfinance as yf


def load_price_data(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load historical price data for a given ticker.

    Parameters
    ----------
    ticker : str
        Asset ticker symbol (e.g., 'AAPL', 'SPY').
    start : str
        Start date.
    end : str | None
        End date.

    Returns
    -------
    pd.DataFrame
        DataFrame containing historical price data.
    """

    data = yf.download(ticker, start=start, end=end)

    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    return data