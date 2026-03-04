"""
Power-law impact model ablation.

Compares the square-root impact model (baseline, Lillo 2003) against the
Almgren et al. (2005) 3/5 power law across the full execution-model ladder
(M0-M4) for TSMOM-60 and CSMOM-60 over the canonical 2005-2025 period.

Execution model tiers (matching default.yaml):
  M0  naive:        fee=0,   spread=0,  vol_k=0,   impact_k=0
  M1  fees_5bps:    fee=5,   spread=0,  vol_k=0,   impact_k=0
  M2  spread_10bps: fee=5,   spread=5,  vol_k=0,   impact_k=0
  M3  vol_slip:     fee=5,   spread=5,  vol_k=10,  impact_k=0
  M4  impact_proxy: fee=5,   spread=5,  vol_k=10,  impact_k=0.50

Only M4 differs between impact models; M0-M3 are identical (impact_k=0).

Impact formulas:
  sqrt      (baseline): impact_bps = k_imp * (Q/V)^0.5  * 10000
  power_3_5 (Almgren):  impact_bps = k_imp * (Q/V)^0.6  * 10000

Saves: outputs/tables/power_law_comparison.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SQRT_EXPONENT: float = 0.5
POWER_LAW_EXPONENT: float = 0.6  # Almgren et al. (2005) 3/5 power law

PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "tables", "power_law_comparison.csv")

PERIOD_START = "2005-01-01"
PERIOD_END = "2025-12-31"

SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLP",
]

# Execution-model ladder tiers: (tier_label, exec_name, params_dict)
EXEC_TIERS = [
    ("M0", "naive",          {"fee_bps": 0.0,  "spread_bps": 0.0, "vol_k": 0.0,  "impact_k": 0.0,  "participation": 1.0}),
    ("M1", "fees_5bps",      {"fee_bps": 5.0,  "spread_bps": 0.0, "vol_k": 0.0,  "impact_k": 0.0,  "participation": 1.0}),
    ("M2", "spread_10bps",   {"fee_bps": 5.0,  "spread_bps": 5.0, "vol_k": 0.0,  "impact_k": 0.0,  "participation": 1.0}),
    ("M3", "vol_slip",       {"fee_bps": 5.0,  "spread_bps": 5.0, "vol_k": 10.0, "impact_k": 0.0,  "participation": 1.0}),
    ("M4", "impact_proxy",   {"fee_bps": 5.0,  "spread_bps": 5.0, "vol_k": 10.0, "impact_k": 0.50, "participation": 0.05}),
]


# ---------------------------------------------------------------------------
# Standalone backtester (self-contained, no engine imports needed)
# ---------------------------------------------------------------------------

def _load_data(processed_dir: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load processed CSVs for each symbol."""
    data = {}
    for sym in symbols:
        path = os.path.join(processed_dir, f"{sym}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing processed data: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        data[sym] = df
    return data


def _rolling_vol_ann(prices: pd.Series, window: int = 20) -> float:
    """Annualised volatility from log-returns over last `window` bars."""
    if len(prices) < 2:
        return 0.0
    rets = np.log(prices / prices.shift(1)).dropna()
    rets = rets.iloc[-window:] if len(rets) > window else rets
    return float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0


def _adv_dollar(prices: pd.Series, volumes: pd.Series, window: int = 20) -> float:
    """20-day rolling average dollar volume."""
    if len(prices) < 1 or len(volumes) < 1:
        return 0.0
    dv = (prices * volumes).iloc[-window:]
    return float(dv.mean()) if len(dv) > 0 else 0.0


def _tsmom_signal(closes: dict[str, pd.Series], lookback: int, date: pd.Timestamp) -> dict[str, float]:
    """Time-series momentum: sign of trailing return over lookback days."""
    signals = {}
    for sym, s in closes.items():
        hist = s.loc[:date]
        if len(hist) <= lookback:
            continue
        ret = float(hist.iloc[-1] / hist.iloc[-lookback - 1] - 1)
        signals[sym] = 1.0 if ret > 0 else -1.0
    return signals


def _csmom_signal(closes: dict[str, pd.Series], lookback: int, top_k: int,
                  date: pd.Timestamp) -> dict[str, float]:
    """Cross-sectional momentum: long top-k, short bottom-k."""
    rets = {}
    for sym, s in closes.items():
        hist = s.loc[:date]
        if len(hist) <= lookback:
            continue
        rets[sym] = float(hist.iloc[-1] / hist.iloc[-lookback - 1] - 1)
    if not rets:
        return {}
    ranked = sorted(rets.keys(), key=lambda x: rets[x], reverse=True)
    signals = {}
    for i, sym in enumerate(ranked):
        if i < top_k:
            signals[sym] = 1.0
        elif i >= len(ranked) - top_k:
            signals[sym] = -1.0
        else:
            signals[sym] = 0.0
    return signals


def _run_backtest(
    data: dict[str, pd.DataFrame],
    strategy: str,
    params: dict,
    impact_exponent: float,
    period: tuple[str, str],
) -> dict[str, float]:
    """Run a single backtest and return Sharpe, CAGR, MaxDD."""
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])

    # Align all data to common dates in period
    all_dates: set[pd.Timestamp] = set()
    for sym in SYMBOLS:
        df = data[sym]
        mask = (df.index >= start) & (df.index <= end)
        all_dates.update(df.index[mask].tolist())
    dates = sorted(all_dates)
    if not dates:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

    initial_cash = 100_000.0
    cash = initial_cash
    holdings: dict[str, int] = {sym: 0 for sym in SYMBOLS}
    port_values: list[float] = []

    lookback = 60
    top_k = 3

    for t_idx, date in enumerate(dates):
        # Portfolio value at current open prices
        port_val = cash
        for sym in SYMBOLS:
            df = data[sym]
            if date in df.index and holdings[sym] != 0:
                port_val += holdings[sym] * float(df.loc[date, "open"])

        port_values.append(port_val)

        # Generate signals
        closes = {sym: data[sym]["close"].loc[:date] for sym in SYMBOLS if date >= data[sym].index[lookback] if len(data[sym].loc[:date]) > lookback}
        if not closes:
            continue

        if strategy == "tsmom":
            signals = _tsmom_signal(closes, lookback, date)
        else:  # csmom
            signals = _csmom_signal(closes, lookback, top_k, date)

        # Target weights (equal-weight among signal != 0, long-only)
        active = {sym: sig for sym, sig in signals.items() if sig > 0}
        if not active:
            active = {}

        n_active = len(active)
        target_weights = {sym: (1.0 / n_active if n_active > 0 else 0.0)
                          for sym in active}

        # Execute trades at next open (t+1 delay approximated as same bar open)
        total_val = max(port_val, 1.0)
        for sym in SYMBOLS:
            df = data[sym]
            if date not in df.index:
                continue

            target_w = target_weights.get(sym, 0.0)
            target_val = total_val * target_w
            px_open = float(df.loc[date, "open"])
            if px_open <= 0:
                continue
            target_qty = int(target_val / px_open)

            # Participation cap
            hist_asof = df.loc[:date]
            vol_col = "volume" if "volume" in df.columns else "Volume"
            if vol_col in df.columns:
                adv = _adv_dollar(hist_asof["close"], hist_asof[vol_col])
            else:
                adv = 0.0

            current_qty = holdings[sym]
            delta_qty = target_qty - current_qty
            if delta_qty == 0:
                continue

            # Participation cap: limit trade to rho * ADV / price
            rho = params["participation"]
            if adv > 0 and rho < 1.0:
                max_trade_val = rho * adv
                max_qty = int(max_trade_val / px_open)
                if abs(delta_qty) > max_qty:
                    delta_qty = int(np.sign(delta_qty)) * max_qty

            if delta_qty == 0:
                continue

            trade_val = abs(delta_qty) * px_open
            side = "BUY" if delta_qty > 0 else "SELL"

            # Compute costs
            fee_bps = params["fee_bps"]
            spread_bps = params["spread_bps"]
            vol_k = params["vol_k"]
            impact_k = params["impact_k"]

            vol_ann = _rolling_vol_ann(hist_asof["close"])
            slip_bps = vol_k * vol_ann

            impact_bps = 0.0
            if adv > 0 and impact_k > 0:
                impact_bps = impact_k * ((trade_val / adv) ** impact_exponent) * 1e4

            total_cost_bps = fee_bps + spread_bps + slip_bps + impact_bps
            if side == "BUY":
                px_fill = px_open * (1 + total_cost_bps / 1e4)
                cost = delta_qty * px_fill
                cash -= cost
                holdings[sym] += delta_qty
            else:
                px_fill = px_open * (1 - total_cost_bps / 1e4)
                proceeds = abs(delta_qty) * px_fill
                cash += proceeds
                holdings[sym] += delta_qty  # delta_qty is negative

    if len(port_values) < 2:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

    pv = np.array(port_values, dtype=float)
    daily_rets = np.diff(pv) / pv[:-1]
    daily_rets = daily_rets[~np.isnan(daily_rets) & ~np.isinf(daily_rets)]

    if len(daily_rets) < 10:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

    sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if np.std(daily_rets) > 0 else 0.0
    total_return = float(pv[-1] / pv[0] - 1)
    years = len(dates) / 252.0
    cagr = float((1 + total_return) ** (1 / max(years, 0.01)) - 1)
    roll_max = np.maximum.accumulate(pv)
    drawdowns = (pv - roll_max) / np.maximum(roll_max, 1e-8)
    max_dd = float(np.min(drawdowns))

    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": max_dd}


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run M0-M4 x strategy x impact-model grid. Returns tidy DataFrame."""
    impact_models = [
        ("sqrt", SQRT_EXPONENT),
        ("power_3_5", POWER_LAW_EXPONENT),
    ]
    strategies = [
        ("tsmom_60", "tsmom"),
        ("csmom_60", "csmom"),
    ]

    total = len(impact_models) * len(strategies) * len(EXEC_TIERS)
    done = 0
    rows = []

    for impact_label, exponent in impact_models:
        for strat_name, strat_key in strategies:
            for tier_label, exec_name, params in EXEC_TIERS:
                metrics = _run_backtest(
                    data=data,
                    strategy=strat_key,
                    params=params,
                    impact_exponent=exponent,
                    period=(PERIOD_START, PERIOD_END),
                )
                done += 1
                row = {
                    "impact_model": impact_label,
                    "strategy": strat_name,
                    "exec_model": tier_label,
                    "exec_name": exec_name,
                    "sharpe": round(metrics["sharpe"], 4),
                    "cagr": round(metrics["cagr"], 4),
                    "max_drawdown": round(metrics["max_drawdown"], 4),
                }
                rows.append(row)
                print(
                    f"  [{done:2d}/{total}] {impact_label:10s} | {strat_name:10s}"
                    f" | {tier_label} ({exec_name:14s})"
                    f"  Sharpe={row['sharpe']:+.3f}"
                )

    return pd.DataFrame(rows)


def _print_comparison(df: pd.DataFrame) -> None:
    """Print side-by-side qualitative comparison."""
    sep = "-" * 80
    print()
    print(sep)
    print("POWER-LAW ABLATION: Almgren 3/5 vs Square-Root Baseline")
    print(sep)
    print("  k_imp=0.50, participation=5% (only M4 differs between models)")
    print()

    for strat in ["tsmom_60", "csmom_60"]:
        sub = df[df["strategy"] == strat]
        print(f"  {strat}")
        print(f"  {'Tier':<5} {'sqrt Sharpe':>12} {'3/5 Sharpe':>12} {'Delta':>8}")
        print("  " + "-" * 40)
        for tier_label, _, _ in EXEC_TIERS:
            s_sqrt = float(sub[(sub["exec_model"] == tier_label) & (sub["impact_model"] == "sqrt")]["sharpe"].iloc[0])
            s_pl   = float(sub[(sub["exec_model"] == tier_label) & (sub["impact_model"] == "power_3_5")]["sharpe"].iloc[0])
            flag = " [SIGN FLIP]" if (s_sqrt > 0) != (s_pl > 0) else ""
            print(f"  {tier_label:<5} {s_sqrt:+12.3f} {s_pl:+12.3f} {s_pl-s_sqrt:+8.3f}{flag}")
        print()

    # Qualitative conclusion check
    print(sep)
    print("QUALITATIVE CONCLUSIONS PRESERVED?")
    for strat in ["tsmom_60", "csmom_60"]:
        sub = df[df["strategy"] == strat]
        for lbl in ["sqrt", "power_3_5"]:
            m0 = float(sub[(sub["exec_model"] == "M0") & (sub["impact_model"] == lbl)]["sharpe"].iloc[0])
            m4 = float(sub[(sub["exec_model"] == "M4") & (sub["impact_model"] == lbl)]["sharpe"].iloc[0])
            verdict = "unviable at M4" if m4 < 0.05 else "viable at M4"
            print(f"  {strat} / {lbl:<12}: M0={m0:+.3f} M4={m4:+.3f} -> {verdict}")
    print(sep)


def main() -> None:
    print(f"Loading processed data from {PROCESSED_DIR} ...")
    data = _load_data(PROCESSED_DIR, SYMBOLS)

    print("Running power-law ablation: sqrt vs Almgren 3/5 | M0-M4 | TSMOM-60 + CSMOM-60")
    print(f"Period: {PERIOD_START} -- {PERIOD_END}\n")

    df = run_ablation(data)

    os.makedirs(os.path.join(OUTPUT_DIR, "tables"), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Saved {OUTPUT_CSV}")
    _print_comparison(df)


if __name__ == "__main__":
    main()
