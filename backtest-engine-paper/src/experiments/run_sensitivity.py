"""
Sensitivity sweep for k_imp and commission parameters.
Run after run_grid.py to produce sensitivity_kimp.csv.
"""
from __future__ import annotations

import argparse
import os
import pandas as pd
import yaml

from src.engine.backtest import Backtester
from src.engine.portfolio import PortfolioConfig
from src.engine.execution import ExecConfig
from src.engine.strategy import TimeSeriesMomentum, MeanReversionZ
from src.utils.io import ensure_dir, load_processed_symbols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    symbols = cfg["universe"]["symbols"]
    data = load_processed_symbols(cfg["data"]["processed_dir"], symbols)
    out_dir = cfg["outputs"]["out_dir"]
    ensure_dir(out_dir + "/tables")

    port_cfg = PortfolioConfig(
        initial_cash=float(cfg["portfolio"]["initial_cash"]),
        target_weight=float(cfg["portfolio"]["target_weight"]),
        max_weight=float(cfg["portfolio"].get("max_weight", 1.0)),
        allow_short=False, min_qty=1,
    )

    strategies = {
        "tsmom_60": TimeSeriesMomentum(lookback=60),
        "meanrev_z1": MeanReversionZ(window=20, z_enter=1.0),
    }

    # k_imp sweep (all other params fixed at their ladder values)
    kimp_values = [0.0, 0.1, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]
    rows = []
    for strat_name, strat in strategies.items():
        for k in kimp_values:
            exec_cfg = ExecConfig(
                fee_bps=5.0, half_spread_bps=5.0, vol_k=10.0,
                impact_k=k, delay_days=1, participation_rate=0.05
            )
            res = Backtester(data=data, strategy=strat,
                             portfolio_cfg=port_cfg, exec_cfg=exec_cfg).run()
            rows.append({
                "strategy": strat_name,
                "impact_k": k,
                "sharpe": res.metrics["sharpe"],
                "cagr": res.metrics["cagr"],
                "max_drawdown": res.metrics["max_drawdown"],
            })
            print(f"  k_imp={k:.2f}  {strat_name}: Sharpe={res.metrics['sharpe']:.3f}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "tables", "sensitivity_kimp.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()
