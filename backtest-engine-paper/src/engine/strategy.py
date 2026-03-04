from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, List
import pandas as pd

from .events import MarketEvent, SignalEvent
from .data import DataHandler


class Strategy:
    def on_market(self, evt: MarketEvent, data: DataHandler) -> Optional[Union[SignalEvent, List[SignalEvent]]]:
        raise NotImplementedError


@dataclass
class TimeSeriesMomentum(Strategy):
    lookback: int = 60

    def on_market(self, evt: MarketEvent, data: DataHandler) -> Optional[SignalEvent]:
        hist = data.get_history_asof(evt.symbol, evt.t)
        if len(hist) < self.lookback + 1:
            return None
        closes = hist["close"].astype(float)
        ret = closes.iloc[-1] / closes.iloc[-1 - self.lookback] - 1.0
        side = "BUY" if ret > 0 else "SELL"
        return SignalEvent(t=evt.t, symbol=evt.symbol, side=side, strength=1.0)


@dataclass
class MeanReversionZ(Strategy):
    window: int = 20
    z_enter: float = 1.0

    def on_market(self, evt: MarketEvent, data: DataHandler) -> Optional[SignalEvent]:
        hist = data.get_history_asof(evt.symbol, evt.t)
        if len(hist) < self.window + 2:
            return None
        rets = hist["close"].astype(float).pct_change().dropna()
        if len(rets) < self.window:
            return None
        w = rets.iloc[-self.window:]
        mu = float(w.mean())
        sd = float(w.std(ddof=1)) if float(w.std(ddof=1)) > 0 else 1e-12
        z = (float(rets.iloc[-1]) - mu) / sd
        side = "BUY" if z < -self.z_enter else "SELL"
        return SignalEvent(t=evt.t, symbol=evt.symbol, side=side, strength=1.0)


@dataclass
class CrossSectionalMomentum(Strategy):
    lookback: int = 60
    top_k: int = 3

    def on_market(self, evt: MarketEvent, data: DataHandler) -> Optional[List[SignalEvent]]:
        # Emit once per timestamp (on the final symbol event in the daily queue)
        # to avoid duplicate cross-sectional rebalances.
        if len(data.symbols) == 0 or evt.symbol != data.symbols[-1]:
            return None

        rets = []
        for sym in data.symbols:
            hist = data.get_history_asof(sym, evt.t)
            if len(hist) < self.lookback + 1:
                continue
            closes = hist["close"].astype(float)
            ret = closes.iloc[-1] / closes.iloc[-1 - self.lookback] - 1.0
            prev_ret = closes.iloc[-1] / closes.iloc[-2] - 1.0 if len(closes) >= 2 else 0.0
            rets.append((sym, float(ret), float(prev_ret)))

        if len(rets) == 0:
            return None

        # Primary rank by lookback return; tie-break by prior-day return.
        rets = sorted(rets, key=lambda x: (x[1], x[2]), reverse=True)
        k = max(1, min(int(self.top_k), len(rets)))
        winners = {sym for sym, _, _ in rets[:k]}
        winner_weight = 1.0 / float(k)

        # Emit in deterministic universe order so downstream FIFO allocation
        # is reproducible and independent of rank ordering.
        ranked_syms = {sym for sym, _, _ in rets}
        out: List[SignalEvent] = []
        for sym in data.symbols:
            if sym not in ranked_syms:
                continue
            side = "BUY" if sym in winners else "SELL"
            strength = winner_weight if side == "BUY" else 1.0
            out.append(SignalEvent(t=evt.t, symbol=sym, side=side, strength=float(strength)))
        return out
