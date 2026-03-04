"""
Tasks 13 & 14 robustness additions:
  Task 13: Bootstrap seed sensitivity (seeds 0, 1, 42, 100) for headline tiers.
  Task 14: Zero-commission (fee_bps=0) re-run for all strategies M1-M5.
"""
from __future__ import annotations

import os
import pandas as pd
import yaml

from src.engine.backtest import Backtester
from src.engine.portfolio import PortfolioConfig
from src.engine.execution import ExecConfig
from src.engine.strategy import TimeSeriesMomentum, MeanReversionZ, CrossSectionalMomentum
from src.experiments.bootstrap import block_bootstrap_sharpe
from src.utils.io import ensure_dir, load_processed_symbols

CONFIG_PATH = "src/experiments/configs/default.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    symbols = cfg["universe"]["symbols"]
    data = load_processed_symbols(cfg["data"]["processed_dir"], symbols)
    out_dir = cfg["outputs"]["out_dir"]
    ensure_dir(os.path.join(out_dir, "tables"))

    port_cfg = PortfolioConfig(
        initial_cash=float(cfg["portfolio"]["initial_cash"]),
        target_weight=float(cfg["portfolio"]["target_weight"]),
        max_weight=float(cfg["portfolio"].get("max_weight", 1.0)),
        allow_short=False, min_qty=1,
    )
    full_period_cfg = next(
        (p for p in cfg.get("periods", []) if p.get("name") == "full"), None
    )
    full_period = (str(full_period_cfg["start"]), str(full_period_cfg["end"])) if full_period_cfg else None

    bs_n = int(cfg["bootstrap"]["n_samples"])
    bs_b = int(cfg["bootstrap"]["primary_block_size"])

    # ── TASK 13: Bootstrap seed sensitivity ───────────────────────────────────
    print("\n=== TASK 13: Bootstrap Seed Sensitivity ===")
    seeds = [0, 1, 42, 100]

    strategies_map = {
        "tsmom_60": TimeSeriesMomentum(lookback=60),
        "csmom_60": CrossSectionalMomentum(lookback=60, top_k=3),
        "meanrev_z1": MeanReversionZ(window=20, z_enter=1.0),
    }
    exec_models = {
        "naive":         ExecConfig(fee_bps=0.0,  half_spread_bps=0.0, vol_k=0.0,  impact_k=0.0, delay_days=1, participation_rate=1.0),
        "fees_5bps":     ExecConfig(fee_bps=5.0,  half_spread_bps=0.0, vol_k=0.0,  impact_k=0.0, delay_days=1, participation_rate=1.0),
        "impact_proxy":  ExecConfig(fee_bps=5.0,  half_spread_bps=5.0, vol_k=10.0, impact_k=0.5, delay_days=1, participation_rate=0.05),
    }

    # Headline tiers: TSMOM/CSMOM M0+M4, MeanRev M0+M1
    headline_tiers = {
        "tsmom_60":   ["naive", "impact_proxy"],
        "csmom_60":   ["naive", "impact_proxy"],
        "meanrev_z1": ["naive", "fees_5bps"],
    }

    # Compute returns once per (strategy, exec_model) pair
    returns_cache = {}
    for strat_name, tiers in headline_tiers.items():
        for tier in tiers:
            key = (strat_name, tier)
            if key not in returns_cache:
                strat = strategies_map[strat_name]
                ex = exec_models[tier]
                res = Backtester(data=data, strategy=strat, portfolio_cfg=port_cfg,
                                 exec_cfg=ex, period=full_period).run()
                returns_cache[key] = res.returns
                print(f"  Ran {strat_name}/{tier}: Sharpe={res.metrics['sharpe']:.4f}")

    seed_rows = []
    for seed in seeds:
        for strat_name, tiers in headline_tiers.items():
            for tier in tiers:
                rets = returns_cache[(strat_name, tier)]
                lo, hi = block_bootstrap_sharpe(rets, n_samples=bs_n, block_size=bs_b, seed=seed)
                seed_rows.append({
                    "strategy": strat_name,
                    "exec_model": tier,
                    "seed": seed,
                    "ci_lo": round(lo, 2),
                    "ci_hi": round(hi, 2),
                })
                print(f"  seed={seed:3d}  {strat_name}/{tier}: [{lo:.2f}, {hi:.2f}]")

    seed_df = pd.DataFrame(seed_rows)
    seed_csv = os.path.join(out_dir, "tables", "bootstrap_seed_sensitivity.csv")
    seed_df.to_csv(seed_csv, index=False)
    print(f"[OK] Saved {seed_csv}")

    # ── TASK 14: Zero-commission sensitivity ──────────────────────────────────
    print("\n=== TASK 14: Zero-Commission Sensitivity ===")
    zero_fee_models = {
        "fees_0bps":    ExecConfig(fee_bps=0.0, half_spread_bps=0.0, vol_k=0.0,  impact_k=0.0, delay_days=1, participation_rate=1.0),
        "spread_10bps": ExecConfig(fee_bps=0.0, half_spread_bps=5.0, vol_k=0.0,  impact_k=0.0, delay_days=1, participation_rate=1.0),
        "vol_slip":     ExecConfig(fee_bps=0.0, half_spread_bps=5.0, vol_k=10.0, impact_k=0.0, delay_days=1, participation_rate=1.0),
        "impact_proxy": ExecConfig(fee_bps=0.0, half_spread_bps=5.0, vol_k=10.0, impact_k=0.5, delay_days=1, participation_rate=0.05),
        "delay_2d":     ExecConfig(fee_bps=0.0, half_spread_bps=5.0, vol_k=10.0, impact_k=0.5, delay_days=2, participation_rate=0.05),
    }
    zfee_rows = []
    for strat_name, strat in strategies_map.items():
        for model_name, ex in zero_fee_models.items():
            res = Backtester(data=data, strategy=strat, portfolio_cfg=port_cfg,
                             exec_cfg=ex, period=full_period).run()
            zfee_rows.append({
                "strategy": strat_name,
                "exec_model": model_name,
                "sharpe": res.metrics["sharpe"],
                "cagr": res.metrics["cagr"],
                "max_drawdown": res.metrics["max_drawdown"],
            })
            print(f"  {strat_name}/{model_name}: Sharpe={res.metrics['sharpe']:.4f}")

    zfee_df = pd.DataFrame(zfee_rows)
    zfee_csv = os.path.join(out_dir, "tables", "zero_commission_sensitivity.csv")
    zfee_df.to_csv(zfee_csv, index=False)
    print(f"[OK] Saved {zfee_csv}")

    # ── Summary table comparison: fee vs zero-fee at M4 ──────────────────────
    canonical_m4 = {
        "tsmom_60":   0.020,
        "csmom_60":   -0.242,
        "meanrev_z1": -2.248,
    }
    print("\n=== Fee Impact Isolation (M4 Sharpe: with 5bps fee vs zero-fee) ===")
    for strat_name in ["tsmom_60", "csmom_60", "meanrev_z1"]:
        zfee_m4 = zfee_df[
            (zfee_df["strategy"] == strat_name) & (zfee_df["exec_model"] == "impact_proxy")
        ]["sharpe"].values[0]
        canonical = canonical_m4[strat_name]
        delta = zfee_m4 - canonical
        print(f"  {strat_name}: canonical M4={canonical:.3f}, zero-fee M4={zfee_m4:.3f}, delta={delta:+.3f}")


if __name__ == "__main__":
    main()
