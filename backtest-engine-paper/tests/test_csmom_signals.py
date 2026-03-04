import pandas as pd

from src.engine.data import DataHandler
from src.engine.events import MarketEvent
from src.engine.strategy import CrossSectionalMomentum


def test_csmom_emits_once_per_timestamp_with_ranked_signals():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    data = {
        "A": pd.DataFrame({"open": [10, 11, 12, 13], "high": [10, 11, 12, 13], "low": [10, 11, 12, 13], "close": [10, 11, 12, 13], "volume": 1000}, index=idx),
        "B": pd.DataFrame({"open": [10, 10, 10, 10.5], "high": [10, 10, 10, 10.5], "low": [10, 10, 10, 10.5], "close": [10, 10, 10, 10.5], "volume": 1000}, index=idx),
        "C": pd.DataFrame({"open": [10, 9, 8, 7], "high": [10, 9, 8, 7], "low": [10, 9, 8, 7], "close": [10, 9, 8, 7], "volume": 1000}, index=idx),
    }
    dh = DataHandler(data)
    strat = CrossSectionalMomentum(lookback=2, top_k=1)

    t = idx[3]
    evt_not_last = MarketEvent(t=t, symbol="A", bar=dh.get_bar("A", t))
    assert strat.on_market(evt_not_last, dh) is None

    evt_last = MarketEvent(t=t, symbol="C", bar=dh.get_bar("C", t))
    sigs = strat.on_market(evt_last, dh)
    assert sigs is not None
    by_sym = {s.symbol: s.side for s in sigs}
    assert by_sym["A"] == "BUY"
    assert by_sym["B"] == "SELL"
    assert by_sym["C"] == "SELL"
