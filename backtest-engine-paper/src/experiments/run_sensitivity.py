"""
Sensitivity sweep for k_imp and commission parameters.
Run after run_grid.py to produce sensitivity tables.
"""
from __future__ import annotations

import argparse
import os
import pandas as pd
import yaml

from src.engine.backtest import Backtester
from src.engine.portfolio import PortfolioConfig
from src.engine.execution import ExecConfig
from src.engine.strategy import TimeSeriesMomentum, MeanReversionZ, CrossSectionalMomentum
from src.utils.io import ensure_dir, load_processed_symbols


def _write_lookback_table_tex(df: pd.DataFrame, out_path: str) -> None:
    rows = []
    rows.append(r"\begin{tabular}{lrrrr}")
    rows.append(r"\toprule")
    rows.append(r"Strategy / Model & J=20 & J=40 & J=60 & J=120 \\")
    rows.append(r"\midrule")
    for strat in ["tsmom", "csmom"]:
        for model in ["naive", "impact_proxy"]:
            sub = df[(df["strategy"] == strat) & (df["exec_model"] == model)].copy()
            if sub.empty:
                continue
            pivot = sub.set_index("lookback")["sharpe"].to_dict()
            vals = [pivot.get(j, float("nan")) for j in [20, 40, 60, 120]]
            model_disp = "Na\\\"{i}ve (M0)" if model == "naive" else "+Impact (M4)"
            label = f"{strat.upper()} {model_disp}"
            rows.append(
                f"  {label} & {vals[0]:.3f} & {vals[1]:.3f} & {vals[2]:.3f} & {vals[3]:.3f} \\\\"
            )
        rows.append(r"\midrule")
    if rows[-1] == r"\midrule":
        rows.pop()
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"[OK] Saved {out_path}")


def _write_k_table_tex(df: pd.DataFrame, out_path: str) -> None:
    rows = []
    rows.append(r"\begin{tabular}{lrrrrr}")
    rows.append(r"\toprule")
    rows.append(r"CSMOM model & K=1 & K=2 & K=3 & K=4 & K=5 \\")
    rows.append(r"\midrule")
    for model in ["naive", "impact_proxy"]:
        sub = df[df["exec_model"] == model].copy()
        if sub.empty:
            continue
        pivot = sub.set_index("top_k")["sharpe"].to_dict()
        vals = [pivot.get(k, float("nan")) for k in [1, 2, 3, 4, 5]]
        model_disp = "Na\\\"{i}ve (M0)" if model == "naive" else "+Impact (M4)"
        rows.append(
            f"  {model_disp} & {vals[0]:.3f} & {vals[1]:.3f} & {vals[2]:.3f} & {vals[3]:.3f} & {vals[4]:.3f} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"[OK] Saved {out_path}")


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

    # Extract canonical period to match run_grid.py exactly
    full_period_cfg = next(
        (p for p in cfg.get("periods", []) if p.get("name") == "full"), None
    )
    full_period = (str(full_period_cfg["start"]), str(full_period_cfg["end"])) if full_period_cfg else None

    strategies = {
        "tsmom_60": TimeSeriesMomentum(lookback=60),
        "csmom_60": CrossSectionalMomentum(lookback=60, top_k=3),
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
                             portfolio_cfg=port_cfg, exec_cfg=exec_cfg,
                             period=full_period).run()
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

    # Lookback sweep: J in {20, 40, 60, 120}, models M0 and M4.
    lookback_rows = []
    for lookback in [20, 40, 60, 120]:
        for strat_key, strat in [
            ("tsmom", TimeSeriesMomentum(lookback=lookback)),
            ("csmom", CrossSectionalMomentum(lookback=lookback, top_k=3)),
        ]:
            for model, ex in [
                ("naive", ExecConfig(fee_bps=0.0, half_spread_bps=0.0, vol_k=0.0, impact_k=0.0, delay_days=1, participation_rate=1.0)),
                ("impact_proxy", ExecConfig(fee_bps=5.0, half_spread_bps=5.0, vol_k=10.0, impact_k=0.5, delay_days=1, participation_rate=0.05)),
            ]:
                res = Backtester(data=data, strategy=strat, portfolio_cfg=port_cfg, exec_cfg=ex,
                                period=full_period).run()
                lookback_rows.append(
                    {
                        "strategy": strat_key,
                        "lookback": lookback,
                        "exec_model": model,
                        "sharpe": res.metrics["sharpe"],
                        "cagr": res.metrics["cagr"],
                    }
                )
                print(f"  lookback={lookback:3d} {strat_key:5s} {model:12s}: Sharpe={res.metrics['sharpe']:.3f}")

    look_df = pd.DataFrame(lookback_rows).sort_values(["strategy", "exec_model", "lookback"])
    look_csv = os.path.join(out_dir, "tables", "lookback_sensitivity.csv")
    look_df.to_csv(look_csv, index=False)
    print(f"[OK] Saved {look_csv}")
    _write_lookback_table_tex(look_df, os.path.join(out_dir, "tables", "table_lookback_sensitivity.tex"))

    # CSMOM top-k sweep: K in {1,2,3,4,5}, models M0 and M4.
    k_rows = []
    for top_k in [1, 2, 3, 4, 5]:
        strat = CrossSectionalMomentum(lookback=60, top_k=top_k)
        for model, ex in [
            ("naive", ExecConfig(fee_bps=0.0, half_spread_bps=0.0, vol_k=0.0, impact_k=0.0, delay_days=1, participation_rate=1.0)),
            ("impact_proxy", ExecConfig(fee_bps=5.0, half_spread_bps=5.0, vol_k=10.0, impact_k=0.5, delay_days=1, participation_rate=0.05)),
        ]:
            res = Backtester(data=data, strategy=strat, portfolio_cfg=port_cfg, exec_cfg=ex,
                            period=full_period).run()
            k_rows.append(
                {
                    "top_k": top_k,
                    "exec_model": model,
                    "sharpe": res.metrics["sharpe"],
                    "cagr": res.metrics["cagr"],
                }
            )
            print(f"  top_k={top_k} {model:12s}: Sharpe={res.metrics['sharpe']:.3f}")
    k_df = pd.DataFrame(k_rows).sort_values(["exec_model", "top_k"])
    k_csv = os.path.join(out_dir, "tables", "csmom_k_sensitivity.csv")
    k_df.to_csv(k_csv, index=False)
    print(f"[OK] Saved {k_csv}")
    _write_k_table_tex(k_df, os.path.join(out_dir, "tables", "table_csmom_k_sensitivity.tex"))


if __name__ == "__main__":
    main()
