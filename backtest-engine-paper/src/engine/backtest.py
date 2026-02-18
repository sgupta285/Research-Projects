from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from .events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from .data import DataHandler
from .strategy import Strategy
from .portfolio import Portfolio, PortfolioConfig
from .execution import ExecutionHandler, ExecConfig
from .metrics import compute_metrics
from .event_queue import EventQueue
from .logger import EventLogger


@dataclass
class BacktestResult:
    equity: pd.Series
    metrics: Dict[str, float]
    ledger: pd.DataFrame
    returns: pd.Series


class Backtester:
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: Strategy,
        portfolio_cfg: PortfolioConfig,
        exec_cfg: ExecConfig,
        logger: Optional[EventLogger] = None,
        period: Optional[Tuple[str, str]] = None,
    ):
        if period is not None:
            start, end = pd.to_datetime(period[0]), pd.to_datetime(period[1])
            sliced = {}
            for sym, df in data.items():
                sliced[sym] = df.loc[(df.index >= start) & (df.index <= end)].copy()
            data = sliced

        self.data_handler = DataHandler(data)
        self.strategy = strategy
        self.portfolio = Portfolio(portfolio_cfg, symbols=self.data_handler.symbols)
        self.exec_handler = ExecutionHandler(exec_cfg)
        self.logger = logger or EventLogger(enabled=False)

        self.q = EventQueue()
        self._turnover_rows: List[Dict[str, Any]] = []

    def run(self) -> BacktestResult:
        self.data_handler.reset()

        while self.data_handler.has_next():
            t = self.data_handler.next_time()

            # enqueue market events
            for sym in self.data_handler.symbols:
                bar = self.data_handler.get_bar(sym, t)
                if bar is None:
                    continue
                self.q.put(MarketEvent(t=t, symbol=sym, bar=bar))

            # mark once per timestep (close(t))
            self.portfolio.mark_to_market(t, self.data_handler)

            # drain queue
            while not self.q.empty():
                evt = self.q.get()
                self.logger.log(evt)

                if isinstance(evt, MarketEvent):
                    sig = self.strategy.on_market(evt, self.data_handler)
                    if sig is not None:
                        self.q.put(sig)

                elif isinstance(evt, SignalEvent):
                    order = self.portfolio.on_signal(evt)
                    if order is not None:
                        self.q.put(order)

                elif isinstance(evt, OrderEvent):
                    fill = self.exec_handler.execute(evt, self.data_handler)
                    if fill is not None:
                        self.q.put(fill)

                elif isinstance(evt, FillEvent):
                    base_price = float(evt.meta.get("base_price", evt.price)) if evt.meta else float(evt.price)
                    self._turnover_rows.append({"t": evt.t, "turnover": abs(base_price * float(evt.qty))})
                    self.portfolio.on_fill(evt)

        ledger = pd.DataFrame(self.portfolio.history).set_index("t").sort_index()
        eq = ledger["equity"].astype(float)
        rets = eq.pct_change().dropna()

        if len(self._turnover_rows) == 0:
            turnover = pd.Series(0.0, index=eq.index)
        else:
            td = pd.DataFrame(self._turnover_rows)
            td["t"] = pd.to_datetime(td["t"])
            turnover = td.groupby("t")["turnover"].sum().reindex(eq.index).fillna(0.0)

        m = compute_metrics(eq, turnover)
        metrics = {
            "cagr": m.cagr,
            "sharpe": m.sharpe,
            "max_drawdown": m.max_drawdown,
            "vol_ann": m.vol_ann,
            "turnover": m.turnover,
        }

        return BacktestResult(equity=eq, metrics=metrics, ledger=ledger, returns=rets)
