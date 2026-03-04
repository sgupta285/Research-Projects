from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import json
import os
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
        mdd_audit_threshold: Optional[float] = -0.90,
        mdd_audit_dir: Optional[str] = None,
        run_label: Optional[str] = None,
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
        self.mdd_audit_threshold = mdd_audit_threshold
        self.mdd_audit_dir = mdd_audit_dir
        self.run_label = run_label or "run"

        self.q = EventQueue()
        self._turnover_rows: List[Dict[str, Any]] = []
        self._fill_rows: List[Dict[str, Any]] = []

    def _maybe_write_mdd_audit(self, equity: pd.Series, metrics: Dict[str, float]) -> None:
        if self.mdd_audit_threshold is None:
            return
        if float(metrics.get("max_drawdown", 0.0)) > float(self.mdd_audit_threshold):
            return
        if self.mdd_audit_dir is None:
            return
        if len(self._fill_rows) == 0 or len(equity) == 0:
            return

        eq = equity.astype(float)
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        trough_t = dd.idxmin()
        pre = eq.loc[:trough_t]
        peak_t = pre.idxmax()

        fills = pd.DataFrame(self._fill_rows)
        fills["t"] = pd.to_datetime(fills["t"])
        fills_win = fills.loc[(fills["t"] >= peak_t) & (fills["t"] <= trough_t)].copy()

        os.makedirs(self.mdd_audit_dir, exist_ok=True)
        safe_label = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in self.run_label)
        fills_path = os.path.join(self.mdd_audit_dir, f"mdd_audit_fills_{safe_label}.csv")
        summary_path = os.path.join(self.mdd_audit_dir, f"mdd_audit_summary_{safe_label}.json")

        fills_win.to_csv(fills_path, index=False)

        summary = {
            "run_label": self.run_label,
            "max_drawdown": float(metrics["max_drawdown"]),
            "peak_time": str(pd.Timestamp(peak_t)),
            "trough_time": str(pd.Timestamp(trough_t)),
            "peak_equity": float(eq.loc[peak_t]),
            "trough_equity": float(eq.loc[trough_t]),
            "equity_loss": float(eq.loc[trough_t] - eq.loc[peak_t]),
            "fills_in_window": int(len(fills_win)),
            "total_fee": float(fills_win["fee"].sum()) if len(fills_win) else 0.0,
            "total_slippage": float(fills_win["slippage"].sum()) if len(fills_win) else 0.0,
            "total_base_notional": float((fills_win["base_price"] * fills_win["qty"]).abs().sum()) if len(fills_win) else 0.0,
            "estimated_execution_cost": float((fills_win["fee"] + fills_win["slippage"]).sum()) if len(fills_win) else 0.0,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

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
                    if isinstance(sig, list):
                        for s in sig:
                            self.q.put(s)
                    elif sig is not None:
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
                    self._fill_rows.append({
                        "t": pd.Timestamp(evt.t),
                        "symbol": evt.symbol,
                        "side": evt.side,
                        "qty": int(evt.qty),
                        "price": float(evt.price),
                        "base_price": float(base_price),
                        "fee": float(evt.fee),
                        "slippage": float(evt.slippage),
                    })
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
        self._maybe_write_mdd_audit(eq, metrics)

        return BacktestResult(equity=eq, metrics=metrics, ledger=ledger, returns=rets)
