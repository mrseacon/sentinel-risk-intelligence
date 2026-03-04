import pandas as pd
import numpy as np

from sentinel.risk.risk_contribution import portfolio_risk_contribution


def test_risk_contribution():

    data = pd.DataFrame(
        {
            "A": np.random.normal(0, 0.01, 200),
            "B": np.random.normal(0, 0.02, 200),
        }
    )

    weights = {"A": 0.5, "B": 0.5}

    rc = portfolio_risk_contribution(data, weights)

    assert isinstance(rc, pd.Series)
    assert len(rc) == 2