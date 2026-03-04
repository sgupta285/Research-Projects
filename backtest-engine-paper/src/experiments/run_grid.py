from __future__ import annotations

import argparse
import os
import signal
from typing import Any, Dict
import pandas as pd
import yaml

from src.engine.backtest import Backtester
from src.engine.portfolio import PortfolioConfig
from src.engine.execution import ExecConfig
from src.engine.strategy import TimeSeriesMomentum, MeanReversionZ, CrossSectionalMomentum
from src.engine.logger import EventLogger
from src.utils.io import ensure_dir, load_processed_symbols
from src.experiments.make_figures import make_all_figures, export_paper_figures
from src.experiments.bootstrap import block_bootstrap_sharpe
from src.experiments.validate_cross_source import validate_against_stooq


STRATEGY_REGISTRY = {
    "TimeSeriesMomentum": TimeSeriesMomentum,
    "MeanReversionZ": MeanReversionZ,
    "CrossSectionalMomentum": CrossSectionalMomentum,
}


def _timeout_handler(signum, frame):
    raise TimeoutError("Cross-source validation timed out")


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
    ensure_dir(os.path.join(out_dir, "audits"))

    log_enabled = bool(cfg.get("logging", {}).get("enable_event_log", False))

    bs_cfg = cfg.get("bootstrap", {})
    n_samples = int(bs_cfg.get("n_samples", 500))
    bootstrap_seed = int(bs_cfg.get("seed", 42))
    primary_block_size = int(bs_cfg.get("primary_block_size", bs_cfg.get("block_size", 10)))
    robustness_block_sizes = [int(b) for b in bs_cfg.get("robustness_block_sizes", [])]

    periods = cfg.get("periods", [{"name": "full", "start": "1900-01-01", "end": "2100-01-01"}])
    full_period_cfg = next((p for p in periods if p.get("name") == "full"), None)
    full_period = None if full_period_cfg is None else (full_period_cfg["start"], full_period_cfg["end"])
    full_start = None if full_period is None else full_period[0]
    full_end = None if full_period is None else full_period[1]

    xsrc_cfg = cfg.get("cross_source_validation", {})
    if bool(xsrc_cfg.get("enable", False)):
        xsrc_out = os.path.join(out_dir, "tables", "cross_source_validation.csv")
        xsrc_window = int(xsrc_cfg.get("window", 60))
        xsrc_strict = bool(xsrc_cfg.get("strict", False))
        xsrc_timeout = int(xsrc_cfg.get("timeout_seconds", 90))
        try:
            if xsrc_timeout > 0 and hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"):
                prev_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, float(xsrc_timeout))
                try:
                    xsrc_df = validate_against_stooq(
                        processed_data=data,
                        symbols=symbols,
                        window=xsrc_window,
                        start=full_start,
                        end=full_end,
                    )
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    signal.signal(signal.SIGALRM, prev_handler)
            else:
                xsrc_df = validate_against_stooq(
                    processed_data=data,
                    symbols=symbols,
                    window=xsrc_window,
                    start=full_start,
                    end=full_end,
                )
            xsrc_df.to_csv(xsrc_out, index=False)
            print(f"[OK] Saved {xsrc_out}")
        except Exception as exc:
            msg = f"Cross-source validation failed: {exc}"
            if isinstance(exc, TimeoutError):
                msg = f"Cross-source validation timed out after {xsrc_timeout}s"
            if xsrc_strict:
                raise RuntimeError(msg) from exc
            print(f"[WARN] {msg}")

    rows = []
    by_period_rows = []
    ci_rows = []
    ci_robust_rows = []

    for s_cfg in cfg["strategies"]:
        strat = STRATEGY_REGISTRY[s_cfg["type"]](**s_cfg.get("params", {}))
        s_name = s_cfg["name"]

        for e_cfg in cfg["execution_models"]:
            e_name = e_cfg["name"]
            exec_cfg = ExecConfig(**e_cfg.get("params", {}))

            logger = EventLogger(enabled=log_enabled)
            bt = Backtester(
                data=data,
                strategy=strat,
                portfolio_cfg=port_cfg,
                exec_cfg=exec_cfg,
                logger=logger,
                period=full_period,
                mdd_audit_threshold=float(cfg.get("logging", {}).get("mdd_audit_threshold", -0.90)),
                mdd_audit_dir=os.path.join(out_dir, "audits"),
                run_label=f"{s_name}__{e_name}__full",
            )
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

            lo, hi = block_bootstrap_sharpe(
                res.returns, n_samples=n_samples, block_size=primary_block_size, seed=bootstrap_seed
            )
            ci_rows.append({
                "strategy": s_name,
                "exec_model": e_name,
                "sharpe_ci_lo": lo,
                "sharpe_ci_hi": hi,
                "bootstrap_n": n_samples,
                "block_size": primary_block_size,
            })

            bs_all = []
            for b in [primary_block_size] + robustness_block_sizes:
                if b > 0 and b not in bs_all:
                    bs_all.append(b)
            for b in bs_all:
                lo_b, hi_b = block_bootstrap_sharpe(
                    res.returns, n_samples=n_samples, block_size=b, seed=bootstrap_seed
                )
                ci_robust_rows.append({
                    "strategy": s_name,
                    "exec_model": e_name,
                    "sharpe_ci_lo": lo_b,
                    "sharpe_ci_hi": hi_b,
                    "bootstrap_n": n_samples,
                    "block_size": int(b),
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
    ci_robust_df = pd.DataFrame(ci_robust_rows).sort_values(["strategy", "exec_model", "block_size"])
    ci_robust_path = os.path.join(out_dir, "tables", "bootstrap_ci_robustness.csv")
    ci_robust_df.to_csv(ci_robust_path, index=False)

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
    export_paper_figures(out_dir=out_dir, paper_fig_dir="paper/figs")

    print("\nSaved:", metrics_path)
    print("Saved:", byp_path)
    print("Saved:", ci_path)
    print("Saved:", ci_robust_path)
    print("Saved:", infl_path)
    print("Figures:", os.path.join(out_dir, "figures"))


if __name__ == "__main__":
    main()
