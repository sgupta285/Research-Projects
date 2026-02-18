import pandas as pd
from src.engine.data import DataHandler


def test_history_asof_enforces_causality():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"open": range(10), "high": range(10), "low": range(10), "close": range(10), "volume": 100},
        index=idx,
    )
    dh = DataHandler({"SPY": df})

    t = idx[4]
    hist = dh.get_history_asof("SPY", t)
    assert hist.index.max() == t
    assert len(hist) == 5
    assert (hist.index <= t).all()
