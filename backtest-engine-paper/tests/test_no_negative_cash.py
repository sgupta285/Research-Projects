import pandas as pd
from src.engine.backtest import Backtester
from src.engine.strategy import TimeSeriesMomentum
from src.engine.portfolio import PortfolioConfig
from src.engine.execution import ExecConfig


def test_cash_never_negative_long_only():
    idx = pd.date_range("2020-01-01", periods=260, freq="D")
    df = pd.DataFrame({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000}, index=idx)
    data = {"SPY": df}

    bt = Backtester(
        data=data,
        strategy=TimeSeriesMomentum(lookback=5),
        portfolio_cfg=PortfolioConfig(initial_cash=1000, target_weight=1.0, max_weight=1.0),
        exec_cfg=ExecConfig(fee_bps=10.0, half_spread_bps=5.0, delay_days=1),
    )
    res = bt.run()
    assert (res.ledger["cash"] >= -1e-6).all()
