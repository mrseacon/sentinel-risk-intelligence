from __future__ import annotations

from io import StringIO

import pandas as pd

REQUIRED_COLUMNS = {"ticker", "weight"}


def parse_portfolio_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse an uploaded CSV file containing at least:
    - ticker
    - weight
    """
    text = file_bytes.decode("utf-8")
    df = pd.read_csv(StringIO(text))

    normalized_columns = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=normalized_columns)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="raise")

    if df["ticker"].eq("").any():
        raise ValueError("Ticker column contains empty values.")

    if (df["weight"] < 0).any():
        raise ValueError("Weights must be non-negative.")

    if df["weight"].sum() <= 0:
        raise ValueError("Sum of weights must be positive.")

    return df[["ticker", "weight"]]


def normalize_portfolio_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize weights so that they sum to 1.
    """
    result = df.copy()
    total_weight = result["weight"].sum()

    if total_weight <= 0:
        raise ValueError("Sum of weights must be positive.")

    result["weight"] = result["weight"] / total_weight
    return result


def portfolio_dict_from_dataframe(df: pd.DataFrame) -> dict[str, float]:
    """
    Convert validated portfolio dataframe into a dictionary.
    Duplicate tickers are aggregated.
    """
    grouped = df.groupby("ticker", as_index=False)["weight"].sum()
    return dict(
        zip(
            grouped["ticker"],
            grouped["weight"],
            strict=False,
        )
    )
