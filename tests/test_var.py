import pandas as pd
import numpy as np

from sentinel.risk.var import historical_var


def test_historical_var():
    returns = pd.Series(np.random.normal(0, 0.01, 1000))

    var = historical_var(returns)

    assert isinstance(var, float)