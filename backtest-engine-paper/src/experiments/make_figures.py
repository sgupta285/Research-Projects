from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


def make_all_figures(metrics_path: str, out_dir: str) -> None:
    df = pd.read_csv(metrics_path)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    g = df.groupby("exec_model")["sharpe"].mean().sort_values(ascending=False)
    plt.figure()
    plt.bar(g.index, g.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Average Sharpe")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "avg_sharpe_by_exec_model.png"), dpi=200)
    plt.close()

    g2 = df.groupby("exec_model")["max_drawdown"].mean().sort_values()
    plt.figure()
    plt.bar(g2.index, g2.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Average Max Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "avg_mdd_by_exec_model.png"), dpi=200)
    plt.close()

    pivot = df.pivot_table(index="strategy", columns="exec_model", values="sharpe", aggfunc="mean")
    plt.figure()
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Sharpe")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "sharpe_heatmap.png"), dpi=200)
    plt.close()

    infl_path = os.path.join(out_dir, "tables", "inflation_ratios.csv")
    if os.path.exists(infl_path):
        infl = pd.read_csv(infl_path)
        if len(infl) > 0:
            g3 = infl.groupby("exec_model")["sharpe_inflation_ratio"].mean().sort_values(ascending=False)
            plt.figure()
            plt.bar(g3.index, g3.values)
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("Avg Sharpe Inflation (naive / model)")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "avg_sharpe_inflation_ratio.png"), dpi=200)
            plt.close()
