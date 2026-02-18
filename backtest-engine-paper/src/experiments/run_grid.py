from __future__ import annotations

import argparse
import os
from typing import Any, Dict
import pandas as pd
import yaml

from src.engine.backtest import Backtester
from src.engine.portfolio import PortfolioConfig
from src.engine.execution import ExecConfig
from src.engine.strategy import TimeSeriesMomentum, MeanReversionZ
from src.engine.logger import EventLogger
from src.utils.io import ensure_dir, load_processed_symbols
from src.experiments.make_figures import make_all_figures
from src.experiments.bootstrap import block_bootstrap_sharpe


STRATEGY_REGISTRY = {
    "TimeSeriesMomentum": TimeSeriesMomentum,
    "MeanReversionZ": MeanReversionZ,
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    symbols = cfg["universe"]["symbols"]
    data = load_processed_symbols(cfg["data"]["processed_dir"], symbols)

    port_cfg = PortfolioConfig(
        initial_cash=float(cfg["portfolio"]["initial_cash"]),
        target_weight=float(cfg["portfolio"]["target_weight"]),
        max_weight=float(cfg["portfolio"].get("max_weight", 1.0)),
        allow_short=False,
        min_qty=1,
    )

    out_dir = cfg["outputs"]["out_dir"]
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "tables"))
    ensure_dir(os.path.join(out_dir, "figures"))
    ensure_dir(os.path.join(out_dir, "events"))

    log_enabled = bool(cfg.get("logging", {}).get("enable_event_log", False))

    bs_cfg = cfg.get("bootstrap", {})
    n_samples = int(bs_cfg.get("n_samples", 500))
    block_size = int(bs_cfg.get("block_size", 10))

    periods = cfg.get("periods", [{"name": "full", "start": "1900-01-01", "end": "2100-01-01"}])

    rows = []
    by_period_rows = []
    ci_rows = []

    for s_cfg in cfg["strategies"]:
        strat = STRATEGY_REGISTRY[s_cfg["type"]](**s_cfg.get("params", {}))
        s_name = s_cfg["name"]

        for e_cfg in cfg["execution_models"]:
            e_name = e_cfg["name"]
            exec_cfg = ExecConfig(**e_cfg.get("params", {}))

            logger = EventLogger(enabled=log_enabled)
            bt = Backtester(data=data, strategy=strat, portfolio_cfg=port_cfg, exec_cfg=exec_cfg, logger=logger)
            res = bt.run()

            rows.append({
                "strategy": s_name,
                "exec_model": e_name,
                **res.metrics,
                "start": str(res.equity.index.min().date()) if len(res.equity) else "",
                "end": str(res.equity.index.max().date()) if len(res.equity) else "",
                "n_days": int(len(res.equity)),
            })

            eq_path = os.path.join(out_dir, "tables", f"equity_{s_name}__{e_name}.csv")
            res.equity.rename("equity").to_csv(eq_path, index=True)

            lo, hi = block_bootstrap_sharpe(res.returns, n_samples=n_samples, block_size=block_size, seed=0)
            ci_rows.append({
                "strategy": s_name,
                "exec_model": e_name,
                "sharpe_ci_lo": lo,
                "sharpe_ci_hi": hi,
                "bootstrap_n": n_samples,
                "block_size": block_size,
            })

            if log_enabled:
                logger.flush_csv(os.path.join(out_dir, "events", f"events_{s_name}__{e_name}.csv"))

            print(f"[OK] FULL {s_name} x {e_name}: Sharpe={res.metrics['sharpe']:.3f} CAGR={res.metrics['cagr']:.3f} MDD={res.metrics['max_drawdown']:.3f}")

            for p in periods:
                pname = p["name"]
                period = (p["start"], p["end"])
                pres = Backtester(data=data, strategy=strat, portfolio_cfg=port_cfg, exec_cfg=exec_cfg, period=period).run()
                by_period_rows.append({
                    "period": pname,
                    "start": p["start"],
                    "end": p["end"],
                    "strategy": s_name,
                    "exec_model": e_name,
                    **pres.metrics,
                    "n_days": int(len(pres.equity)),
                })

    metrics_df = pd.DataFrame(rows).sort_values(["strategy", "exec_model"])
    metrics_path = os.path.join(out_dir, "tables", "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    byp_df = pd.DataFrame(by_period_rows).sort_values(["period", "strategy", "exec_model"])
    byp_path = os.path.join(out_dir, "tables", "metrics_by_period.csv")
    byp_df.to_csv(byp_path, index=False)

    ci_df = pd.DataFrame(ci_rows).sort_values(["strategy", "exec_model"])
    ci_path = os.path.join(out_dir, "tables", "bootstrap_ci.csv")
    ci_df.to_csv(ci_path, index=False)

    infl_rows = []
    for strat in metrics_df["strategy"].unique():
        sub = metrics_df[metrics_df["strategy"] == strat].copy()
        base = sub[sub["exec_model"] == "naive"]
        if len(base) != 1:
            continue
        base_sh = float(base.iloc[0]["sharpe"])
        base_cagr = float(base.iloc[0]["cagr"])
        for _, r in sub.iterrows():
            if r["exec_model"] == "naive":
                continue
            infl_rows.append({
                "strategy": strat,
                "exec_model": r["exec_model"],
                "sharpe_inflation_ratio": (base_sh / (float(r["sharpe"]) + 1e-12)),
                "cagr_inflation_ratio": (base_cagr / (float(r["cagr"]) + 1e-12)),
            })
    infl_df = pd.DataFrame(infl_rows)
    infl_path = os.path.join(out_dir, "tables", "inflation_ratios.csv")
    infl_df.to_csv(infl_path, index=False)

    make_all_figures(metrics_path, out_dir)

    print("\nSaved:", metrics_path)
    print("Saved:", byp_path)
    print("Saved:", ci_path)
    print("Saved:", infl_path)
    print("Figures:", os.path.join(out_dir, "figures"))


if __name__ == "__main__":
    main()
