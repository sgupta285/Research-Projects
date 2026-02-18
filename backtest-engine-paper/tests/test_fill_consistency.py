import pandas as pd
from src.engine.execution import ExecutionHandler, ExecConfig
from src.engine.events import OrderEvent
from src.engine.data import DataHandler


def test_fill_qty_leq_order_qty_with_participation_cap():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    df = pd.DataFrame({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}, index=idx)
    dh = DataHandler({"SPY": df})

    cfg = ExecConfig(delay_days=1, participation_rate=0.05, adv_lookback=20)
    ex = ExecutionHandler(cfg)

    order = OrderEvent(t=idx[0], symbol="SPY", side="BUY", qty=10_000, order_type="MKT")
    fill = ex.execute(order, dh)
    assert fill is not None
    assert fill.qty <= order.qty
    assert fill.qty >= 1
