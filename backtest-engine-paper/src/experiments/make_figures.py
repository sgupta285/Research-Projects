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
    "csmom_60":   "CSMOM-60",
    "meanrev_z1": "MeanRev-z1",
}
PALETTE = {"tsmom_60": "#2166ac", "csmom_60": "#1b7837", "meanrev_z1": "#d6604d"}

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

    strategies = [s for s in ["tsmom_60", "csmom_60", "meanrev_z1"] if s in df["strategy"].unique()]
    fig3_rows = []

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
        for m, s in zip(ordered, sharpes):
            if not np.isnan(s):
                fig3_rows.append({"strategy": strat, "exec_model": m, "sharpe": float(s)})
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
    pd.DataFrame(fig3_rows).to_csv(
        os.path.join(tables_dir, "fig_sharpe_by_model_values.csv"), index=False
    )

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 2: Sharpe inflation ratio — TSMOM + CSMOM
    # Defined as S_naive / S_model; plotted only when both signs equal and S_naive > 0
    # ════════════════════════════════════════════════════════════════════════
    infl_path = os.path.join(tables_dir, "inflation_ratios.csv")
    fig4_rows = []
    if os.path.exists(infl_path):
        _ = pd.read_csv(infl_path)  # keep file dependency explicit for reproducibility
        fig, ax = plt.subplots(figsize=(8, 4))
        x_labels = [MODEL_LABELS.get(m, m) for m in ordered[1:]]
        x = np.arange(len(x_labels))
        width = 0.36

        plotted_any = False
        for offset, strat in [(-width / 2, "tsmom_60"), (width / 2, "csmom_60")]:
            sub = df[df["strategy"] == strat].copy()
            if len(sub) == 0:
                continue
            naive_row = sub[sub["exec_model"] == "naive"]
            if len(naive_row) == 0:
                continue
            naive_sh = float(naive_row["sharpe"].values[0])
            ratios = []
            for m in ordered[1:]:
                row = sub[sub["exec_model"] == m]
                if len(row) == 0:
                    fig4_rows.append({
                        "strategy": strat, "exec_model": m, "ratio": np.nan, "plotted": False
                    })
                    ratios.append(np.nan)
                    continue
                s = float(row["sharpe"].values[0])
                if naive_sh > 0 and s > 0:
                    r = naive_sh / s
                    ratios.append(r)
                    fig4_rows.append({
                        "strategy": strat, "exec_model": m, "ratio": float(r), "plotted": True
                    })
                else:
                    fig4_rows.append({
                        "strategy": strat, "exec_model": m, "ratio": np.nan, "plotted": False
                    })
                    ratios.append(np.nan)

            vals = np.array([0.0 if np.isnan(v) else v for v in ratios], dtype=float)
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                label=STRAT_LABELS.get(strat, strat),
                color=PALETTE.get(strat, "gray"),
                edgecolor="k",
                linewidth=0.5,
                alpha=0.9,
            )
            for bar, r in zip(bars, ratios):
                if np.isnan(r):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    r,
                    f"{r:.1f}×",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            plotted_any = True

        if plotted_any:
            ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="Ratio = 1.0")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Sharpe Inflation Ratio  ($S_{\\mathrm{naïve}} / S_{\\mathrm{model}}$)")
            ax.set_title(
                "Sharpe Inflation Ratios by Execution Model (TSMOM-60 and CSMOM-60)\n"
                "Ratios shown only when both Sharpe values are positive",
                fontsize=9,
            )
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(
                os.path.join(fig_dir, "avg_sharpe_inflation_ratio.png"),
                dpi=180,
                bbox_inches="tight",
            )
        plt.close(fig)
    if len(fig4_rows):
        pd.DataFrame(fig4_rows).to_csv(
            os.path.join(tables_dir, "fig_inflation_ratio_values.csv"), index=False
        )

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 3: Sharpe heatmap — models in ladder order, cells annotated
    # ════════════════════════════════════════════════════════════════════════
    pivot = df.pivot_table(index="strategy", columns="exec_model",
                           values="sharpe", aggfunc="mean")
    # reorder columns
    pivot = pivot.reindex(columns=[m for m in ordered if m in pivot.columns])
    # reorder rows
    pivot = pivot.reindex([s for s in ["tsmom_60", "csmom_60", "meanrev_z1"] if s in pivot.index])

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
    heat_rows = []
    for strat in pivot.index:
        for model in pivot.columns:
            val = pivot.loc[strat, model]
            if not np.isnan(val):
                heat_rows.append({
                    "strategy": str(strat),
                    "exec_model": str(model),
                    "sharpe": float(val),
                })
    pd.DataFrame(heat_rows).to_csv(
        os.path.join(tables_dir, "fig_heatmap_values.csv"), index=False
    )

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 4: k_imp sensitivity with error bars at anchor points
    # Bootstrap CIs from Table~\ref{tab:master} (M3 and M4 rows) are
    # used as error bars at the two canonical anchor points:
    #   kimp=0.00  -> M3 (fees+spread+vol, no impact)
    #   kimp=0.50  -> M4 canonical run
    # Anchor CIs (95% moving-block bootstrap, b=20, N=1000, seed=42):
    ANCHOR_CI = {
        "tsmom_60": {
            0.00: (-0.02, 0.86),   # M3 CI  [lo, hi]
            0.50: (-0.46, 0.48),   # M4 CI
        },
        "csmom_60": {
            0.00: (-0.07, 0.72),   # M3 CI
            0.50: (-0.62, 0.18),   # M4 CI
        },
    }
    # ════════════════════════════════════════════════════════════════════════
    sens_path = os.path.join(tables_dir, "sensitivity_kimp.csv")
    if os.path.exists(sens_path):
        sens = pd.read_csv(sens_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        for strat in [s for s in ["tsmom_60", "csmom_60"] if s in sens["strategy"].unique()]:
            sub = sens[sens["strategy"] == strat].sort_values("impact_k")
            ax.plot(sub["impact_k"], sub["sharpe"],
                    marker="o", label=STRAT_LABELS.get(strat, strat),
                    color=PALETTE.get(strat, "gray"), zorder=3)
            # Add error bars at anchor points using main-table bootstrap CIs
            anchor_data = ANCHOR_CI.get(strat, {})
            for kimp_anchor, (ci_lo, ci_hi) in anchor_data.items():
                anchor_row = sub[sub["impact_k"] == kimp_anchor]
                if len(anchor_row) == 0:
                    continue
                s_point = float(anchor_row["sharpe"].iloc[0])
                lo_err = s_point - ci_lo   # downward error bar
                hi_err = ci_hi - s_point   # upward error bar
                ax.errorbar(kimp_anchor, s_point,
                            yerr=[[lo_err], [hi_err]],
                            fmt="none",
                            color=PALETTE.get(strat, "gray"),
                            capsize=5, capthick=1.2, elinewidth=1.2,
                            zorder=4, alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        # Vertical line at kimp=1.0 (upper end of empirically calibrated range)
        ax.axvline(1.0, color="dimgray", linewidth=0.9, linestyle=":",
                   label="$k_{\\mathrm{imp}}=1.0$ (empirical upper bound)")
        ax.annotate("emp. upper bound",
                    xy=(1.0, ax.get_ylim()[0] if ax.get_ylim()[0] < -0.3 else -0.4),
                    xytext=(1.05, -0.55),
                    fontsize=6.5, color="dimgray",
                    arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.7))
        ax.set_xlabel("$k_{\\mathrm{imp}}$ (impact scaling parameter)")
        ax.set_ylabel("Annualised Sharpe Ratio")
        ax.set_title("Sharpe Ratio vs. Impact Parameter $k_{\\mathrm{imp}}$\n"
                     "(participation cap $\\rho=0.05$ held constant)", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "sensitivity_kimp.png"),
                    dpi=180, bbox_inches="tight")
        plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # AUTO-GENERATE LaTeX TABLE SNIPPETS for \input in main.tex
    # ════════════════════════════════════════════════════════════════════════
    _generate_latex_ci_table(df, ci_df, tables_dir)
    _generate_latex_period_table(out_dir, tables_dir)
    _generate_latex_inflation_table(df, tables_dir)
    _generate_latex_bootstrap_sensitivity_table(df, tables_dir)


def _generate_latex_ci_table(df, ci_df, tables_dir):
    r"""Write bootstrap CI table as a .tex snippet for \input."""
    ordered = _ordered_models(df["exec_model"].cat.categories if hasattr(df["exec_model"], "cat") else df["exec_model"])
    strategies = ["tsmom_60", "csmom_60", "meanrev_z1"]
    strat_names = {"tsmom_60": "TSMOM-60", "csmom_60": "CSMOM-60", "meanrev_z1": "MeanRev-z1"}
    model_disp = {
        "naive": "Na\\\"{i}ve (M0)",
        "fees_5bps": "+Fees (M1)",
        "spread_10bps": "+Spread (M2)",
        "vol_slip": "+Slippage (M3)",
        "impact_proxy": "+Impact (M4)",
        "delay_2d": "+Delay (M5)",
    }
    rows = []
    rows.append(r"\begin{tabular}{llrrrrr}")
    rows.append(r"\toprule")
    rows.append(r"Strategy & Model & Sharpe & CI$_{2.5\%}$ & CI$_{97.5\%}$ & CAGR & MDD \\" )
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
            mdd = float(row["max_drawdown"].values[0]) if "max_drawdown" in row.columns else float("nan")
            rows.append(f"  {sn} & {model_disp.get(m, m)} & {sh:.3f} & {lo_str} & {hi_str} & {cagr:.3f} & {mdd:.3f} \\\\")
        rows.append(r"\midrule")

    rows[-1] = r"\bottomrule"
    rows.append(r"\end{tabular}")

    out = os.path.join(tables_dir, "table_sharpe_ci.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"[OK] Generated {out}")


def _generate_latex_period_table(out_dir, tables_dir):
    """Write period-split Sharpe table as a .tex snippet."""
    byp_path = os.path.join(tables_dir, "metrics_by_period.csv")
    if not os.path.exists(byp_path):
        return
    byp = pd.read_csv(byp_path)

    model_disp = {
        "naive": "Na\\\"{i}ve", "fees_5bps": "+Fees",
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
    strat_order = ["tsmom_60", "csmom_60", "meanrev_z1"]
    strat_names = {"tsmom_60": "TSMOM-60", "csmom_60": "CSMOM-60", "meanrev_z1": "MeanRev-z1"}
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
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"[OK] Generated {out}")


def _generate_latex_inflation_table(df, tables_dir):
    ordered = _ordered_models(df["exec_model"].cat.categories if hasattr(df["exec_model"], "cat") else df["exec_model"])
    strategies = ["tsmom_60", "csmom_60", "meanrev_z1"]
    strat_names = {"tsmom_60": "TSMOM-60", "csmom_60": "CSMOM-60", "meanrev_z1": "MeanRev-z1"}
    model_disp = {
        "fees_5bps": "+Fees (M1)",
        "spread_10bps": "+Spread (M2)",
        "vol_slip": "+Slippage (M3)",
        "impact_proxy": "+Impact (M4)",
        "delay_2d": "+Delay (M5)",
    }

    rows = []
    rows.append(r"\begin{tabular}{llrr}")
    rows.append(r"\toprule")
    rows.append(r"Strategy & Model & IR & $\hat S_{Mk}$ \\")
    rows.append(r"\midrule")

    for strat in strategies:
        sub = df[df["strategy"] == strat]
        if len(sub) == 0:
            continue
        nrow = sub[sub["exec_model"] == "naive"]
        if len(nrow) == 0:
            continue
        s0 = float(nrow["sharpe"].values[0])
        first = True
        for m in ordered:
            if m == "naive":
                continue
            row = sub[sub["exec_model"] == m]
            if len(row) == 0:
                continue
            sm = float(row["sharpe"].values[0])
            ir = f"{(s0 / sm):.1f}\\times" if (s0 > 0 and sm > 0) else "sign flip"
            sm_str = f"{sm:+.3f}"
            sn = strat_names.get(strat, strat) if first else ""
            first = False
            rows.append(f"  {sn} & {model_disp.get(m, m)} & ${ir}$ & ${sm_str}$ \\\\")
        rows.append(r"\midrule")

    rows[-1] = r"\bottomrule"
    rows.append(r"\end{tabular}")
    out = os.path.join(tables_dir, "table_inflation.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"[OK] Generated {out}")


def _generate_latex_bootstrap_sensitivity_table(df, tables_dir):
    path = os.path.join(tables_dir, "bootstrap_ci_robustness.csv")
    if not os.path.exists(path):
        return
    rob = pd.read_csv(path)
    keep = [
        ("tsmom_60", "naive", "TSMOM-60 M0"),
        ("tsmom_60", "vol_slip", "TSMOM-60 M3"),
        ("tsmom_60", "impact_proxy", "TSMOM-60 M4"),
        ("tsmom_60", "delay_2d", "TSMOM-60 M5"),
        ("csmom_60", "naive", "CSMOM-60 M0"),
        ("csmom_60", "fees_5bps", "CSMOM-60 M1"),
        ("csmom_60", "spread_10bps", "CSMOM-60 M2"),
        ("csmom_60", "impact_proxy", "CSMOM-60 M4"),
        ("meanrev_z1", "naive", "MeanRev-z1 M0"),
        ("meanrev_z1", "fees_5bps", "MeanRev-z1 M1"),
        ("meanrev_z1", "impact_proxy", "MeanRev-z1 M4"),
    ]

    mkey = (
        df[["strategy", "exec_model", "sharpe"]]
        .set_index(["strategy", "exec_model"])
        .to_dict()["sharpe"]
    )

    rows = []
    rows.append(r"\begin{tabular}{llrrr}")
    rows.append(r"\toprule")
    rows.append(r"Strategy / Model & Sharpe & $b=10$ CI & $b=20$ CI & $b=30$ CI \\")
    rows.append(r"\midrule")

    for strat, model, label in keep:
        if (strat, model) not in mkey:
            continue
        ci10 = rob[(rob["strategy"] == strat) & (rob["exec_model"] == model) & (rob["block_size"] == 10)]
        ci20 = rob[(rob["strategy"] == strat) & (rob["exec_model"] == model) & (rob["block_size"] == 20)]
        ci30 = rob[(rob["strategy"] == strat) & (rob["exec_model"] == model) & (rob["block_size"] == 30)]
        if len(ci10) == 0 or len(ci20) == 0 or len(ci30) == 0:
            continue
        sh = float(mkey[(strat, model)])
        c10 = f"[{float(ci10.iloc[0]['sharpe_ci_lo']):+.2f},\\;{float(ci10.iloc[0]['sharpe_ci_hi']):+.2f}]"
        c20 = f"[{float(ci20.iloc[0]['sharpe_ci_lo']):+.2f},\\;{float(ci20.iloc[0]['sharpe_ci_hi']):+.2f}]"
        c30 = f"[{float(ci30.iloc[0]['sharpe_ci_lo']):+.2f},\\;{float(ci30.iloc[0]['sharpe_ci_hi']):+.2f}]"
        rows.append(f"  {label} & ${sh:.3f}$ & ${c10}$ & ${c20}$ & ${c30}$ \\\\")
        if label in ("TSMOM-60 M5", "CSMOM-60 M4"):
            rows.append(r"\midrule")

    if rows[-1] == r"\midrule":
        rows.pop()
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")

    out = os.path.join(tables_dir, "table_bootstrap_sensitivity.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"[OK] Generated {out}")


def export_paper_figures(out_dir: str, paper_fig_dir: str = "paper/figs") -> None:
    """Copy canonical experiment figures into paper/figs with LaTeX-stable names."""
    fig_dir = os.path.join(out_dir, "figures")
    if not os.path.exists(fig_dir):
        return
    os.makedirs(paper_fig_dir, exist_ok=True)
    mapping = {
        "avg_sharpe_by_exec_model.png": "fig_sharpe_by_model.png",
        "avg_sharpe_inflation_ratio.png": "fig_inflation_ratio.png",
        "sharpe_heatmap.png": "fig_heatmap.png",
        "sensitivity_kimp.png": "fig_sensitivity_kimp.png",
    }
    for src_name, dst_name in mapping.items():
        src = os.path.join(fig_dir, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(paper_fig_dir, dst_name))
