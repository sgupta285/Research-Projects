from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from .events import OrderEvent, FillEvent
from .data import DataHandler


@dataclass
class ExecConfig:
    fee_bps: float = 0.0
    half_spread_bps: float = 0.0
    vol_k: float = 0.0
    impact_k: float = 0.0
    delay_days: int = 1
    vol_lookback: int = 20
    adv_lookback: int = 20
    participation_rate: float = 1.0  # <=1.0 caps fills as fraction of ADV shares


class ExecutionHandler:
    """Execution simulator for market orders with a realism ladder + partial fills."""

    def __init__(self, cfg: ExecConfig):
        self.cfg = cfg

    def _effective_price(self, side: str, base_price: float, half_spread_bps: float, slip_bps: float, impact_bps: float) -> float:
        spread_adj = (half_spread_bps / 1e4)
        if side == "BUY":
            base_price *= (1.0 + spread_adj)
        else:
            base_price *= (1.0 - spread_adj)

        total_bps = (slip_bps + impact_bps) / 1e4
        if side == "BUY":
            return base_price * (1.0 + total_bps)
        else:
            return base_price * (1.0 - total_bps)

    def _rolling_vol_annualized(self, hist: pd.DataFrame) -> float:
        rets = hist["close"].astype(float).pct_change().dropna()
        if len(rets) < max(2, self.cfg.vol_lookback):
            return 0.0
        w = rets.iloc[-self.cfg.vol_lookback:]
        sd = float(w.std(ddof=1))
        return sd * (252.0 ** 0.5)

    def _adv_dollar(self, hist: pd.DataFrame) -> float:
        if len(hist) < max(2, self.cfg.adv_lookback):
            return 0.0
        w = hist.iloc[-self.cfg.adv_lookback:].copy()
        return float((w["close"].astype(float) * w["volume"].astype(float)).mean())

    def _adv_shares(self, hist: pd.DataFrame) -> float:
        if len(hist) < max(2, self.cfg.adv_lookback):
            return 0.0
        w = hist.iloc[-self.cfg.adv_lookback:].copy()
        return float(w["volume"].astype(float).mean())

    def _cap_partial_fill_qty(self, desired_qty: int, hist_asof: pd.DataFrame) -> int:
        pr = float(self.cfg.participation_rate)
        if pr >= 1.0:
            return int(desired_qty)
        adv_sh = self._adv_shares(hist_asof)
        if adv_sh <= 0:
            return int(desired_qty)
        cap = int(max(1, adv_sh * pr))
        return int(min(desired_qty, cap))

    def execute(self, order: OrderEvent, data: DataHandler) -> Optional[FillEvent]:
        sym = order.symbol
        df = data.data[sym]

        if order.t not in df.index:
            future = df.index[df.index > order.t]
            if len(future) == 0:
                return None
            t0 = future[0]
        else:
            t0 = order.t

        pos = df.index.get_loc(t0)
        exec_pos = pos + max(0, int(self.cfg.delay_days))
        if exec_pos >= len(df.index):
            return None
        t_exec = df.index[exec_pos]

        bar = data.get_bar(sym, t_exec)
        if bar is None:
            return None
        base_price = float(bar["open"])

        hist_asof = df.loc[:t_exec]
        qty = self._cap_partial_fill_qty(int(order.qty), hist_asof)
        if qty <= 0:
            return None

        vol_ann = self._rolling_vol_annualized(hist_asof)
        slip_bps = float(self.cfg.vol_k) * vol_ann

        adv = self._adv_dollar(hist_asof)
        trade_value = base_price * float(qty)
        impact_bps = 0.0
        if adv > 0 and float(self.cfg.impact_k) > 0:
            impact_bps = float(self.cfg.impact_k) * ((trade_value / adv) ** 0.5) * 1e4

        px = self._effective_price(order.side, base_price, float(self.cfg.half_spread_bps), slip_bps, impact_bps)
        notional = px * float(qty)
        fee = (float(self.cfg.fee_bps) / 1e4) * notional
        slippage = abs(px - base_price) * float(qty)

        return FillEvent(
            t=pd.Timestamp(t_exec),
            symbol=sym,
            side=order.side,
            qty=int(qty),
            price=float(px),
            fee=float(fee),
            slippage=float(slippage),
            meta={
                "base_price": float(base_price),
                "slip_bps": float(slip_bps),
                "impact_bps": float(impact_bps),
                "half_spread_bps": float(self.cfg.half_spread_bps),
                "desired_qty": int(order.qty),
                "filled_qty": int(qty),
            },
        )
