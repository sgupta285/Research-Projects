"""
TIER 2-1 Ablation: Monthly rebalancing and vol-targeted sizing.

Compares three TSMOM-60 variants at M0 (naive) and M4 (impact):
  1. canonical    - daily rebalance, constant weight (baseline)
  2. monthly      - rebalance only on last trading day of each month
  3. vol_targeted - daily rebalance, weight scaled by target_vol / realized_vol
                   (target_vol=10%, cap at 1.5x; reduces position when vol is high)

Results saved to outputs/tables/ablation_frequencies.csv
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.engine.backtest import Backtester
from src.engine.events import MarketEvent, SignalEvent
from src.engine.data import DataHandler
from src.engine.portfolio import PortfolioConfig
from src.engine.execution import ExecConfig
from src.engine.strategy import Strategy, TimeSeriesMomentum
from src.utils.io import ensure_dir, load_processed_symbols

CONFIG_PATH = "src/experiments/configs/default.yaml"
TARGET_VOL = 0.10       # 10% annualised target vol for vol-targeted variant
MAX_LEVERAGE = 1.5      # cap position multiplier at 1.5× equity
TRAD_DAYS = 252


# ── Strategy variants ──────────────────────────────────────────────────────

@dataclass
class MonthlyTSMOM(TimeSeriesMomentum):
    """TSMOM that emits signals only on the last observed trading day of each month."""

    lookback: int = 60

    def on_market(self, evt: MarketEvent, data: DataHandler) -> Optional[SignalEvent]:
        t: pd.Timestamp = evt.t
        # Determine if t is the last trading day of its calendar month.
        # Use data._timeline (global list of all trading timestamps) to find
        # the last date in the same (year, month).
        timeline = data._timeline
        candidates = [d for d in timeline if d.year == t.year and d.month == t.month]
        if not candidates:
            return None
        last_trading_day = max(candidates)
        if t < last_trading_day:
            return None  # not the month-end trading day for this symbol
        return super().on_market(evt, data)


@dataclass
class VolTargetedTSMOM(TimeSeriesMomentum):
    """
    TSMOM with volatility-scaled position sizing.

    BUY signal strength = min(TARGET_VOL / σ̂_t, MAX_LEVERAGE) where
    σ̂_t is the 20-day trailing annualised volatility. SELL signals are
    unchanged (always exit). When vol data is unavailable, falls back to
    strength=1.0 (canonical sizing).
    """

    lookback: int = 60
    vol_window: int = 20

    def on_market(self, evt: MarketEvent, data: DataHandler) -> Optional[SignalEvent]:
        sig = super().on_market(evt, data)
        if sig is None or sig.side != "BUY":
            return sig

        # Compute trailing realized vol for this symbol
        hist = data.get_history_asof(evt.symbol, evt.t)
        if len(hist) < self.vol_window + 2:
            return sig  # not enough history; keep strength=1.0

        prices = hist["close"].astype(float)
        log_rets = np.log(prices / prices.shift(1)).dropna()
        if len(log_rets) < self.vol_window:
            return sig

        realized_vol = float(log_rets.iloc[-self.vol_window:].std(ddof=1)) * np.sqrt(TRAD_DAYS)
        if realized_vol <= 0.0:
            return sig

        scaled_strength = min(TARGET_VOL / realized_vol, MAX_LEVERAGE)
        # Return new signal with scaled strength (immutable pattern: new object)
        return SignalEvent(
            t=sig.t,
            symbol=sig.symbol,
            side=sig.side,
            strength=scaled_strength,
        )


# ── Runner ─────────────────────────────────────────────────────────────────

def _sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if len(rets) < 2 or float(rets.std()) == 0.0:
        return float("nan")
    return float(rets.mean() / rets.std() * np.sqrt(TRAD_DAYS))


def _max_dd(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or float(equity.iloc[0]) == 0.0:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def run_ablation():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    symbols = cfg["universe"]["symbols"]
    data = load_processed_symbols(cfg["data"]["processed_dir"], symbols)

    full_period_cfg = next(
        (p for p in cfg.get("periods", []) if p.get("name") == "full"), None
    )
    period = (str(full_period_cfg["start"]), str(full_period_cfg["end"])) if full_period_cfg else None

    port_cfg_canonical = PortfolioConfig(
        initial_cash=float(cfg["portfolio"]["initial_cash"]),
        target_weight=float(cfg["portfolio"]["target_weight"]),
        max_weight=1.0,
        allow_short=False, min_qty=1,
    )
    port_cfg_vt = PortfolioConfig(
        initial_cash=float(cfg["portfolio"]["initial_cash"]),
        target_weight=float(cfg["portfolio"]["target_weight"]),
        max_weight=MAX_LEVERAGE,  # allow up to 1.5× for vol-targeting
        allow_short=False, min_qty=1,
    )

    # Execution tiers
    exec_m0 = ExecConfig(fee_bps=0.0, half_spread_bps=0.0, vol_k=0.0,
                         impact_k=0.0, delay_days=1, participation_rate=1.0)
    exec_m4 = ExecConfig(fee_bps=5.0, half_spread_bps=5.0, vol_k=10.0,
                         impact_k=0.5, delay_days=1, participation_rate=0.05)

    strategies: Dict[str, tuple] = {
        "canonical":    (TimeSeriesMomentum(lookback=60),  port_cfg_canonical),
        "monthly":      (MonthlyTSMOM(lookback=60),        port_cfg_canonical),
        "vol_targeted": (VolTargetedTSMOM(lookback=60),    port_cfg_vt),
    }

    rows: List[dict] = []
    for variant, (strat, pcfg) in strategies.items():
        for tier_name, exec_cfg in [("M0", exec_m0), ("M4", exec_m4)]:
            print(f"  [{variant:12s} | {tier_name}] ...", end=" ", flush=True)
            bt = Backtester(
                data=data,
                strategy=strat,
                portfolio_cfg=pcfg,
                exec_cfg=exec_cfg,
                period=period,
            )
            result = bt.run()
            eq = result.equity
            s = _sharpe(eq)
            rows.append({
                "variant": variant,
                "tier": tier_name,
                "sharpe": round(s, 3),
                "cagr": round(_cagr(eq), 3),
                "max_drawdown": round(_max_dd(eq), 3),
            })
            print(f"Sharpe={s:+.3f}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(cfg["outputs"]["out_dir"], "tables", "ablation_frequencies.csv")
    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Saved {out_path}")

    # Print summary table
    print("\n" + "─" * 68)
    print(f"{'Variant':14s} {'Tier':4s} {'Sharpe':>8s} {'CAGR':>8s} {'MDD':>8s}")
    print("─" * 68)
    for _, row in df.iterrows():
        print(f"{row['variant']:14s} {row['tier']:4s} {row['sharpe']:+8.3f} "
              f"{row['cagr']:+8.3f} {row['max_drawdown']:+8.3f}")
    print("─" * 68)
    return df


if __name__ == "__main__":
    import sys
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    print(f"Working dir: {os.getcwd()}")
    print("Running TIER 2-1 frequency/sizing ablation...")
    run_ablation()
