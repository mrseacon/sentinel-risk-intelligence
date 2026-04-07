import pandas as pd
import pytest

from sentinel.portfolio.upload import (
    normalize_portfolio_weights,
    parse_portfolio_csv,
    portfolio_dict_from_dataframe,
)


def test_parse_portfolio_csv_valid():
    content = b"ticker,weight\nAAPL,0.4\nMSFT,0.3\nSPY,0.3\n"
    df = parse_portfolio_csv(content)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["ticker", "weight"]
    assert len(df) == 3


def test_parse_portfolio_csv_missing_column():
    content = b"ticker,value\nAAPL,0.4\n"
    with pytest.raises(ValueError):
        parse_portfolio_csv(content)


def test_normalize_portfolio_weights():
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "weight": [4.0, 6.0],
        }
    )
    normalized = normalize_portfolio_weights(df)

    assert abs(normalized["weight"].sum() - 1.0) < 1e-9


def test_portfolio_dict_from_dataframe_aggregates_duplicates():
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "weight": [0.2, 0.3, 0.5],
        }
    )
    portfolio = portfolio_dict_from_dataframe(df)

    assert portfolio["AAPL"] == 0.5
    assert portfolio["MSFT"] == 0.5