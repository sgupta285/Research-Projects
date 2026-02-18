import pandas as pd
from src.engine.portfolio import Portfolio, PortfolioConfig
from src.engine.data import DataHandler


def test_accounting_identity_holds_simple():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1_000_000}, index=idx)
    dh = DataHandler({"SPY": df})

    p = Portfolio(PortfolioConfig(initial_cash=1000, target_weight=1.0), symbols=["SPY"])

    while dh.has_next():
        t = dh.next_time()
        p.mark_to_market(t, dh)
        eq = p.state.equity()
        cash = p.state.cash
        pos = p.state.positions["SPY"]
        px = p.state.last_price.get("SPY", 0.0)
        assert abs(eq - (cash + pos * px)) < 1e-6
