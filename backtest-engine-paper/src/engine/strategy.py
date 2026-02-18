from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from .events import MarketEvent, SignalEvent
from .data import DataHandler


class Strategy:
    def on_market(self, evt: MarketEvent, data: DataHandler) -> Optional[SignalEvent]:
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
