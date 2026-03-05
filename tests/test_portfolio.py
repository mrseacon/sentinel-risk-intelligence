import numpy as np
import pandas as pd

from sentinel.portfolio.returns import portfolio_returns


def test_portfolio_returns():

    data = pd.DataFrame(
        {
            "A": np.random.normal(0, 0.01, 100),
            "B": np.random.normal(0, 0.01, 100),
        }
    )

    weights = {"A": 0.5, "B": 0.5}

    port = portfolio_returns(data, weights)

    assert isinstance(port, pd.Series)
