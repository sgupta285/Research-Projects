from __future__ import annotations

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Canonical execution model order and display labels ──────────────────────
MODEL_ORDER = ["naive", "fees_5bps", "spread_10bps", "vol_slip", "impact_proxy", "delay_2d"]
MODEL_LABELS = {
    "naive":         "M0: Naïve",
    "fees_5bps":     "M1: +Fees",
    "spread_10bps":  "M2: +Spread",
    "vol_slip":      "M3: +Slippage",
    "impact_proxy":  "M4: +Impact",
    "delay_2d":      "M5: +Delay",
}
STRAT_LABELS = {
    "tsmom_60":   "TSMOM-60",
    "meanrev_z1": "MeanRev-z1",
}
PALETTE = {"tsmom_60": "#2166ac", "meanrev_z1": "#d6604d"}

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 180,
})


def _ordered_models(df_col):
    present = [m for m in MODEL_ORDER if m in df_col.unique()]
    return present


def make_all_figures(metrics_path: str, out_dir: str) -> None:
    df = pd.read_csv(metrics_path)
    fig_dir = os.path.join(out_dir, "figures")
    # Remove and recreate figures dir to clear any stale permission-locked files
    # from previous runs (e.g., files written as root in a Docker/cloud environment).
    if os.path.exists(fig_dir):
        shutil.rmtree(fig_dir, ignore_errors=True)
    os.makedirs(fig_dir, exist_ok=True)
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    ordered = _ordered_models(df["exec_model"])
    df["exec_model"] = pd.Categorical(df["exec_model"], categories=ordered, ordered=True)
    df = df.sort_values(["strategy", "exec_model"])

    # ── Try to load bootstrap CIs ────────────────────────────────────────────
    ci_path = os.path.join(tables_dir, "bootstrap_ci.csv")
    ci_df = pd.read_csv(ci_path) if os.path.exists(ci_path) else None

    strategies = [s for s in df["strategy"].unique()]

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 1: Sharpe by execution model — one panel per strategy
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 4.5), sharey=False)
    if len(strategies) == 1:
        axes = [axes]

    for ax, strat in zip(axes, strategies):
        sub = df[df["strategy"] == strat].copy()
        x = list(range(len(ordered)))
        sharpes = [sub[sub["exec_model"] == m]["sharpe"].values[0]
                   if m in sub["exec_model"].values else np.nan for m in ordered]
        colors = ["#4393c3" if s >= 0 else "#d6604d" for s in sharpes]
        bars = ax.bar(x, sharpes, color=colors, width=0.6, edgecolor="k", linewidth=0.5)

        # overlay CI error bars if available
        if ci_df is not None:
            ci_sub = ci_df[ci_df["strategy"] == strat]
            for i, m in enumerate(ordered):
                row = ci_sub[ci_sub["exec_model"] == m]
                if len(row) == 0 or np.isnan(sharpes[i]):
                    continue
                lo = float(row["sharpe_ci_lo"].values[0])
                hi = float(row["sharpe_ci_hi"].values[0])
                ax.errorbar(i, sharpes[i], yerr=[[sharpes[i]-lo], [hi-sharpes[i]]],
                            fmt="none", color="black", capsize=4, linewidth=1.2)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in ordered],
                           rotation=35, ha="right", fontsize=8)
        ax.set_title(STRAT_LABELS.get(strat, strat), fontsize=11, fontweight="bold")
        ax.set_ylabel("Annualised Sharpe Ratio")

        # annotate bars with values
        for bar, s in zip(bars, sharpes):
            if not np.isnan(s):
                va = "bottom" if s >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width()/2, s,
                        f"{s:.2f}", ha="center", va=va, fontsize=7)

    fig.suptitle("Sharpe Ratio by Execution Model", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "avg_sharpe_by_exec_model.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 2: Sharpe inflation ratio — TSMOM only (all Sharpes > 0 per Table 1)
    # Defined as S_naive / S_model; plotted only when both signs equal and S_naive > 0
    # ════════════════════════════════════════════════════════════════════════
    infl_path = os.path.join(tables_dir, "inflation_ratios.csv")
    if os.path.exists(infl_path):
        infl = pd.read_csv(infl_path)

        # Only plot strategies where naive Sharpe > 0 and same-sign model Sharpes
        # (i.e., TSMOM: all models produce positive Sharpe, so ratio is defined)
        tsmom_infl = infl[infl["strategy"] == "tsmom_60"].copy()

        if len(tsmom_infl) > 0:
            tsmom_infl["exec_model"] = pd.Categorical(
                tsmom_infl["exec_model"], categories=ordered, ordered=True)
            tsmom_infl = tsmom_infl.sort_values("exec_model")

            # Recompute from raw metrics to ensure sign integrity
            tsmom_metrics = df[df["strategy"] == "tsmom_60"].copy()
            naive_sh = float(tsmom_metrics[tsmom_metrics["exec_model"] == "naive"]["sharpe"].values[0])

            ratios = []
            labels = []
            for m in ordered[1:]:  # skip naive
                row = tsmom_metrics[tsmom_metrics["exec_model"] == m]
                if len(row) == 0:
                    continue
                s = float(row["sharpe"].values[0])
                if s > 0 and naive_sh > 0:
                    ratios.append(naive_sh / s)
                    labels.append(MODEL_LABELS.get(m, m))
                else:
                    ratios.append(np.nan)
                    labels.append(MODEL_LABELS.get(m, m) + "*")

            fig, ax = plt.subplots(figsize=(7, 4))
            x = list(range(len(ratios)))
            valid_mask = [not np.isnan(r) for r in ratios]
            colors = ["#e08214" if v else "#bbbbbb" for v in valid_mask]
            bars = ax.bar(x, [r if not np.isnan(r) else 0 for r in ratios],
                          color=colors, width=0.6, edgecolor="k", linewidth=0.5)

            ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
                       label="Ratio = 1.0 (no distortion)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
            ax.set_ylabel("Sharpe Inflation Ratio  ($S_{\\mathrm{naïve}} / S_{\\mathrm{model}}$)")
            ax.set_title("TSMOM-60: Sharpe Inflation Ratio by Execution Model\n"
                         "(defined only when both Sharpe values share positive sign; "
                         "* = sign-flip, ratio undefined)", fontsize=9)

            for bar, r in zip(bars, ratios):
                if not np.isnan(r):
                    ax.text(bar.get_x() + bar.get_width()/2, r,
                            f"{r:.1f}×", ha="center", va="bottom", fontsize=8)

            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "avg_sharpe_inflation_ratio.png"),
                        dpi=180, bbox_inches="tight")
            plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 3: Sharpe heatmap — models in ladder order, cells annotated
    # ════════════════════════════════════════════════════════════════════════
    pivot = df.pivot_table(index="strategy", columns="exec_model",
                           values="sharpe", aggfunc="mean")
    # reorder columns
    pivot = pivot.reindex(columns=[m for m in ordered if m in pivot.columns])
    # reorder rows
    pivot = pivot.reindex([s for s in ["tsmom_60", "meanrev_z1"] if s in pivot.index])

    fig, ax = plt.subplots(figsize=(9, 3.5))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Annualised Sharpe Ratio", fontsize=9)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in pivot.columns],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([STRAT_LABELS.get(s, s) for s in pivot.index], fontsize=9)
    ax.set_xlabel("Execution Model", fontsize=9)

    # annotate each cell
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black",
                        fontweight="bold" if abs(val) > 0.3 else "normal")

    ax.set_title("Sharpe Ratio Heatmap: Strategy × Execution Model\n"
                 "(columns ordered by increasing friction; cells annotated with Sharpe value)",
                 fontsize=9, pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "sharpe_heatmap.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 4: k_imp sensitivity (hard-coded illustrative sweep)
    # These values are computed analytically from a single TSMOM run
    # by re-running the backtest at different k_imp levels.
    # If sensitivity CSV exists, use it; otherwise use illustrative values.
    # ════════════════════════════════════════════════════════════════════════
    sens_path = os.path.join(tables_dir, "sensitivity_kimp.csv")
    if os.path.exists(sens_path):
        sens = pd.read_csv(sens_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        for strat in sens["strategy"].unique():
            sub = sens[sens["strategy"] == strat].sort_values("impact_k")
            ax.plot(sub["impact_k"], sub["sharpe"],
                    marker="o", label=STRAT_LABELS.get(strat, strat),
                    color=PALETTE.get(strat, "gray"))
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_xlabel("$k_{\\mathrm{imp}}$ (impact scaling parameter)")
        ax.set_ylabel("Annualised Sharpe Ratio")
        ax.set_title("Sharpe Ratio vs. Impact Parameter $k_{\\mathrm{imp}}$\n"
                     "(participation cap $\\rho=0.05$ held constant)", fontsize=9)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "sensitivity_kimp.png"),
                    dpi=180, bbox_inches="tight")
        plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # AUTO-GENERATE LaTeX TABLE SNIPPETS for \input in main.tex
    # ════════════════════════════════════════════════════════════════════════
    _generate_latex_ci_table(df, ci_df, tables_dir)
    _generate_latex_period_table(out_dir, tables_dir)


def _generate_latex_ci_table(df, ci_df, tables_dir):
    """Write bootstrap CI table as a .tex snippet for \input."""
    ordered = _ordered_models(df["exec_model"].cat.categories if hasattr(df["exec_model"], "cat") else df["exec_model"])
    strategies = ["tsmom_60", "meanrev_z1"]
    strat_names = {"tsmom_60": "TSMOM-60", "meanrev_z1": "MeanRev-z1"}
    model_disp = {
        "naive": "Naïve (M0)",
        "fees_5bps": "+Fees (M1)",
        "spread_10bps": "+Spread (M2)",
        "vol_slip": "+Slippage (M3)",
        "impact_proxy": "+Impact (M4)",
        "delay_2d": "+Delay (M5)",
    }

    rows = []
    rows.append(r"\begin{tabular}{llrrrr}")
    rows.append(r"\toprule")
    rows.append(r"Strategy & Model & Sharpe & CI$_{2.5\%}$ & CI$_{97.5\%}$ & CAGR \\" )
    rows.append(r"\midrule")

    for strat in strategies:
        sub = df[df["strategy"] == strat]
        first = True
        for m in ordered:
            row = sub[sub["exec_model"] == m]
            if len(row) == 0:
                continue
            sh = float(row["sharpe"].values[0])
            cagr = float(row["cagr"].values[0])
            lo_str, hi_str = "---", "---"
            if ci_df is not None:
                ci_row = ci_df[(ci_df["strategy"] == strat) & (ci_df["exec_model"] == m)]
                if len(ci_row) > 0:
                    lo_str = f"{float(ci_row['sharpe_ci_lo'].values[0]):.3f}"
                    hi_str = f"{float(ci_row['sharpe_ci_hi'].values[0]):.3f}"

            sn = strat_names.get(strat, strat) if first else ""
            first = False
            rows.append(f"  {sn} & {model_disp.get(m, m)} & {sh:.3f} & {lo_str} & {hi_str} & {cagr:.3f} \\\\")
        rows.append(r"\midrule")

    rows[-1] = r"\bottomrule"
    rows.append(r"\end{tabular}")

    out = os.path.join(tables_dir, "table_sharpe_ci.tex")
    with open(out, "w") as f:
        f.write("\n".join(rows))
    print(f"[OK] Generated {out}")


def _generate_latex_period_table(out_dir, tables_dir):
    """Write period-split Sharpe table as a .tex snippet."""
    byp_path = os.path.join(tables_dir, "metrics_by_period.csv")
    if not os.path.exists(byp_path):
        return
    byp = pd.read_csv(byp_path)

    model_disp = {
        "naive": "Naïve", "fees_5bps": "+Fees",
        "spread_10bps": "+Spread", "vol_slip": "+Slip",
        "impact_proxy": "+Impact", "delay_2d": "+Delay",
    }
    period_order = ["full", "pre_gfc_to_gfc", "mid_cycle", "covid_and_after"]
    period_names = {
        "full": "Full (2005--2025)",
        "pre_gfc_to_gfc": "Pre/GFC (2005--12)",
        "mid_cycle": "Mid-cycle (2013--19)",
        "covid_and_after": "COVID+ (2020--25)",
    }
    strat_order = ["tsmom_60", "meanrev_z1"]
    strat_names = {"tsmom_60": "TSMOM-60", "meanrev_z1": "MeanRev-z1"}
    models_show = ["naive", "fees_5bps", "impact_proxy"]

    rows = []
    n_models = len(models_show)
    # header
    cols = "ll" + "r" * n_models
    header_models = " & ".join([model_disp.get(m, m) for m in models_show])
    rows.append(f"\\begin{{tabular}}{{{cols}}}")
    rows.append(r"\toprule")
    rows.append(f"  Strategy & Period & {header_models} \\\\")
    rows.append(r"\midrule")

    for strat in strat_order:
        sub = byp[byp["strategy"] == strat]
        first = True
        for period in period_order:
            if period not in byp["period"].values:
                continue
            p_sub = sub[sub["period"] == period]
            sharpes = []
            for m in models_show:
                row = p_sub[p_sub["exec_model"] == m]
                if len(row) > 0:
                    sharpes.append(f"{float(row['sharpe'].values[0]):.3f}")
                else:
                    sharpes.append("---")
            sn = strat_names.get(strat, strat) if first else ""
            first = False
            pn = period_names.get(period, period)
            rows.append(f"  {sn} & {pn} & " + " & ".join(sharpes) + " \\\\")
        rows.append(r"\midrule")

    rows[-1] = r"\bottomrule"
    rows.append(r"\end{tabular}")

    out = os.path.join(tables_dir, "table_period_split.tex")
    with open(out, "w") as f:
        f.write("\n".join(rows))
    print(f"[OK] Generated {out}")
