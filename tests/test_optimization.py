import numpy as np
import pandas as pd

from sentinel.portfolio.optimization import (
    optimize_max_sharpe,
    optimize_min_variance,
)


def test_optimize_min_variance_returns_valid_weights():
    returns = pd.DataFrame(
        {
            "AAPL": np.random.normal(0.001, 0.02, 300),
            "MSFT": np.random.normal(0.001, 0.015, 300),
            "SPY": np.random.normal(0.0008, 0.01, 300),
        }
    )

    weights = optimize_min_variance(returns)

    assert isinstance(weights, pd.Series)
    assert len(weights) == 3
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= 0).all()


def test_optimize_max_sharpe_returns_valid_weights():
    returns = pd.DataFrame(
        {
            "AAPL": np.random.normal(0.0012, 0.02, 300),
            "MSFT": np.random.normal(0.0011, 0.015, 300),
            "SPY": np.random.normal(0.0007, 0.01, 300),
        }
    )

    weights = optimize_max_sharpe(returns)

    assert isinstance(weights, pd.Series)
    assert len(weights) == 3
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= 0).all()
