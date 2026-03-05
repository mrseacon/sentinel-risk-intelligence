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


def load_multiple_assets(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load adjusted prices for multiple tickers.
    Robust to yfinance variations (Adj Close vs Close; MultiIndex columns).
    Returns a DataFrame with columns = tickers.
    """
    if not tickers:
        raise ValueError("Tickers list must not be empty.")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,  # ensures Adj Close is available when provided by Yahoo
        progress=False,
        group_by="column",
    )

    if data is None or data.empty:
        raise ValueError("No data returned from yfinance.")

    # yfinance can return:
    # 1) columns: ["Open","High","Low","Close","Adj Close","Volume"] (single ticker)
    # 2) MultiIndex columns: ("Adj Close","AAPL"), ("Adj Close","MSFT"), ...
    # 3) sometimes only "Close" exists (Adj Close missing)
    price_key = None
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        if "Adj Close" in level0:
            price_key = "Adj Close"
        elif "Close" in level0:
            price_key = "Close"
        else:
            raise KeyError(
                f"Neither 'Adj Close' nor 'Close' found in columns: {sorted(level0)}"
            )

        prices = data[price_key]
    else:
        if "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif "Close" in data.columns:
            prices = data["Close"]
        else:
            raise KeyError(
            "Neither 'Adj Close' nor 'Close' found in columns: "
            f"{list(data.columns)}"
            )

        # If single ticker, prices may be a Series -> convert to DataFrame
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

    # Ensure columns exactly match tickers order if possible
    # (yfinance sometimes returns sorted columns)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Missing tickers in returned price data: {missing}")

    return prices[tickers].dropna(how="all")
