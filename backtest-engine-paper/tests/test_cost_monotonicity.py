import pandas as pd
from src.engine.backtest import Backtester
from src.engine.strategy import TimeSeriesMomentum
from src.engine.portfolio import PortfolioConfig
from src.engine.execution import ExecConfig


def test_adding_costs_does_not_improve_sharpe_on_constant_price():
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    df = pd.DataFrame({"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1_000_000}, index=idx)
    data = {"SPY": df}

    strat = TimeSeriesMomentum(lookback=5)
    port = PortfolioConfig(initial_cash=10000, target_weight=1.0, max_weight=1.0)

    r0 = Backtester(data=data, strategy=strat, portfolio_cfg=port, exec_cfg=ExecConfig(fee_bps=0.0, half_spread_bps=0.0, delay_days=1)).run()
    r1 = Backtester(data=data, strategy=strat, portfolio_cfg=port, exec_cfg=ExecConfig(fee_bps=20.0, half_spread_bps=10.0, delay_days=1)).run()

    assert r1.metrics["sharpe"] <= r0.metrics["sharpe"] + 1e-6
