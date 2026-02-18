from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd

from .events import SignalEvent, OrderEvent, FillEvent
from .data import DataHandler


@dataclass
class PortfolioConfig:
    initial_cash: float = 100_000.0
    target_weight: float = 1.0
    allow_short: bool = False
    min_qty: int = 1
    max_weight: float = 1.0  # <=1.0 for no leverage


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)
    last_price: Dict[str, float] = field(default_factory=dict)

    def equity(self) -> float:
        eq = float(self.cash)
        for sym, qty in self.positions.items():
            px = self.last_price.get(sym)
            if px is not None:
                eq += float(qty) * float(px)
        return float(eq)


class Portfolio:
    def __init__(self, cfg: PortfolioConfig, symbols: list[str]):
        self.cfg = cfg
        self.state = PortfolioState(
            cash=float(cfg.initial_cash),
            positions={s: 0 for s in symbols},
            last_price={},
        )
        self.history = []

    def mark_to_market(self, t: pd.Timestamp, data: DataHandler) -> None:
        for sym in data.symbols:
            bar = data.get_bar(sym, t)
            if bar is not None:
                self.state.last_price[sym] = float(bar["close"])

        self.history.append({
            "t": pd.Timestamp(t),
            "cash": float(self.state.cash),
            "equity": float(self.state.equity()),
            **{f"pos_{s}": int(self.state.positions.get(s, 0)) for s in data.symbols},
        })

    def _cash_constrained_target_qty(self, symbol: str, target_value: float) -> int:
        px = self.state.last_price.get(symbol)
        if px is None or px <= 0:
            return 0

        eq = self.state.equity()
        target_value = min(target_value, float(self.cfg.max_weight) * eq)

        current_qty = int(self.state.positions.get(symbol, 0))
        current_value = current_qty * px
        spendable = max(0.0, float(self.state.cash) + float(current_value))
        target_value = min(target_value, spendable)

        return max(0, int(target_value // px))

    def _order_to_target_long(self, t: pd.Timestamp, symbol: str) -> Optional[OrderEvent]:
        eq = self.state.equity()
        target_value = float(self.cfg.target_weight) * eq
        target_qty = self._cash_constrained_target_qty(symbol, target_value)

        current_qty = int(self.state.positions.get(symbol, 0))
        delta = target_qty - current_qty
        if abs(delta) < int(self.cfg.min_qty):
            return None

        side = "BUY" if delta > 0 else "SELL"
        return OrderEvent(t=pd.Timestamp(t), symbol=symbol, side=side, qty=abs(int(delta)), order_type="MKT")

    def on_signal(self, sig: SignalEvent) -> Optional[OrderEvent]:
        sym = sig.symbol
        if sig.side == "BUY":
            return self._order_to_target_long(sig.t, sym)

        current_qty = int(self.state.positions.get(sym, 0))
        if current_qty <= 0:
            return None
        return OrderEvent(t=pd.Timestamp(sig.t), symbol=sym, side="SELL", qty=current_qty, order_type="MKT")

    def on_fill(self, fill: FillEvent) -> None:
        sym = fill.symbol
        qty = int(fill.qty)
        px = float(fill.price)
        fee = float(fill.fee)

        if fill.side == "BUY":
            cost = px * qty + fee
            if cost > self.state.cash and px > 0:
                affordable_qty = int(max(0, (self.state.cash - fee) // px))
                qty = min(qty, affordable_qty)
                if qty <= 0:
                    return
                cost = px * qty + fee
            self.state.cash -= cost
            self.state.positions[sym] = int(self.state.positions.get(sym, 0)) + qty
        else:
            sell_qty = min(qty, int(self.state.positions.get(sym, 0)))
            if sell_qty <= 0:
                return
            self.state.cash += (px * sell_qty - fee)
            self.state.positions[sym] = int(self.state.positions.get(sym, 0)) - sell_qty

        self.state.last_price[sym] = px
