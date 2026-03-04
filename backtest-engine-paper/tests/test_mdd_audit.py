from pathlib import Path

import pandas as pd

from src.engine.backtest import Backtester
from src.engine.events import MarketEvent, SignalEvent
from src.engine.execution import ExecConfig
from src.engine.portfolio import PortfolioConfig
from src.engine.strategy import Strategy


class BuyOnce(Strategy):
    def __init__(self):
        self.done = False

    def on_market(self, evt: MarketEvent, data):
        if self.done:
            return None
        self.done = True
        return SignalEvent(t=evt.t, symbol=evt.symbol, side="BUY")


def test_mdd_audit_files_written_when_threshold_breached(tmp_path: Path):
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [100, 100, 1, 1, 1],
            "high": [100, 100, 1, 1, 1],
            "low": [100, 100, 1, 1, 1],
            "close": [100, 100, 1, 1, 1],
            "volume": [1000, 1000, 1000, 1000, 1000],
        },
        index=idx,
    )
    bt = Backtester(
        data={"SPY": df},
        strategy=BuyOnce(),
        portfolio_cfg=PortfolioConfig(initial_cash=1000, target_weight=1.0, max_weight=1.0),
        exec_cfg=ExecConfig(delay_days=0),
        mdd_audit_threshold=-0.90,
        mdd_audit_dir=str(tmp_path),
        run_label="unit_test",
    )
    res = bt.run()
    assert res.metrics["max_drawdown"] <= -0.90
    assert (tmp_path / "mdd_audit_fills_unit_test.csv").exists()
    assert (tmp_path / "mdd_audit_summary_unit_test.json").exists()
