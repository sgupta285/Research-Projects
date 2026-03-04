"""
make_pareto.py — Generate publication-quality Pareto figures and LaTeX tables
from sweep results CSV.

Figures produced:
  pareto_recall_latency.png   — Quality vs. warm latency at fixed k=K_MAX
  pareto_recall_cost.png      — Quality vs. per-query cost at fixed k=K_MAX
  cold_vs_warm_latency.png    — Cold-start vs. warm-start (log y-scale, y=x line)
  recall_vs_k.png             — Recall@K vs. K ablation by retrieval type

Tables produced (LaTeX):
  pareto_table.tex           — Pareto-optimal configurations with 95% Wilson CIs
  full_results.tex           — All 36 configurations
  chunking_ablation.tex      — Chunk-size ablation at fixed k=K_MAX

Statistical notes:
  Recall@K 95% confidence intervals use the Wilson score interval
  (Wilson, 1927) for a binomial proportion with n=500 queries.
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------
PALETTE = {
    "bm25":   "#2166ac",   # blue
    "dense":  "#1a9641",   # green
    "hybrid": "#d73027",   # red
}
MARKERS = {
    "bm25":   "o",
    "dense":  "s",
    "hybrid": "^",
}
FIG_W = 5.5
FIG_H = 4.2
DPI = 300


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score confidence interval for proportion p over n trials."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def ci_half(p: float, n: int = 500) -> float:
    lo, hi = wilson_ci(p, n)
    return (hi - lo) / 2.0


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------

def _dominates_warm(b: dict, a: dict) -> bool:
    return (
        b["mean_retrieval_recall_k"] >= a["mean_retrieval_recall_k"]
        and b["end_to_end_warm_mean_ms"] <= a["end_to_end_warm_mean_ms"]
        and b["mean_cost_usd"] <= a["mean_cost_usd"]
        and (
            b["mean_retrieval_recall_k"] > a["mean_retrieval_recall_k"]
            or b["end_to_end_warm_mean_ms"] < a["end_to_end_warm_mean_ms"]
            or b["mean_cost_usd"] < a["mean_cost_usd"]
        )
    )


def _dominates_cold(b: dict, a: dict) -> bool:
    return (
        b["mean_retrieval_recall_k"] >= a["mean_retrieval_recall_k"]
        and b["total_cold_ms"] <= a["total_cold_ms"]
        and b["mean_cost_usd"] <= a["mean_cost_usd"]
        and (
            b["mean_retrieval_recall_k"] > a["mean_retrieval_recall_k"]
            or b["total_cold_ms"] < a["total_cold_ms"]
            or b["mean_cost_usd"] < a["mean_cost_usd"]
        )
    )


def pareto_front(df: pd.DataFrame, mode: str = "warm") -> pd.DataFrame:
    dom = _dominates_warm if mode == "warm" else _dominates_cold
    rows = df.to_dict("records")
    keep = [a for i, a in enumerate(rows)
            if not any(dom(b, a) for j, b in enumerate(rows) if i != j)]
    return pd.DataFrame(keep)


def _legend_handles(include_frontier: bool = False):
    handles = [
        mpatches.Patch(color=PALETTE[rt], label=rt.capitalize())
        for rt in ["bm25", "dense", "hybrid"]
    ]
    handles += [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=8, markeredgecolor="black", markeredgewidth=1.5,
                   label="Pareto-optimal"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   alpha=0.35, markersize=5, markeredgecolor="gray",
                   label="Dominated"),
    ]
    if include_frontier:
        handles.insert(3, plt.Line2D([0], [0], color="black", linewidth=1.2,
                                     linestyle="--", label="Pareto frontier"))
    return handles


def _scatter_row(ax, row: dict, pf_hashes: set, n: int = 500) -> None:
    rtype = row["retrieval.type"]
    is_par = row["config_hash"] in pf_hashes
    recall = row["mean_retrieval_recall_k"]
    lat = row["end_to_end_warm_mean_ms"]
    ax.errorbar(
        lat, recall, yerr=ci_half(recall, n),
        fmt=MARKERS[rtype],
        color=PALETTE[rtype],
        alpha=1.0 if is_par else 0.35,
        markersize=8 if is_par else 5,
        markeredgewidth=1.5 if is_par else 0.5,
        markeredgecolor="black" if is_par else PALETTE[rtype],
        capsize=3, elinewidth=1,
        zorder=4 if is_par else 2,
    )


def _scatter_row_cost(ax, row: dict, pf_hashes: set, n: int = 500) -> None:
    rtype = row["retrieval.type"]
    is_par = row["config_hash"] in pf_hashes
    recall = row["mean_retrieval_recall_k"]
    cost = row["mean_cost_usd"]
    ax.errorbar(
        cost, recall, yerr=ci_half(recall, n),
        fmt=MARKERS[rtype],
        color=PALETTE[rtype],
        alpha=1.0 if is_par else 0.35,
        markersize=8 if is_par else 5,
        markeredgewidth=1.5 if is_par else 0.5,
        markeredgecolor="black" if is_par else PALETTE[rtype],
        capsize=3, elinewidth=1,
        zorder=4 if is_par else 2,
    )


# ---------------------------------------------------------------------------
# Figure 1: Quality vs. Warm Latency at fixed k=K_MAX
# ---------------------------------------------------------------------------

def fig_recall_latency(df: pd.DataFrame, fig_dir: str, n: int = 500) -> None:
    k_val = int(df["retrieval.top_k"].max())
    sub = df[df["retrieval.top_k"] == k_val].copy()
    pf = pareto_front(sub, mode="warm")
    pf_hashes = set(pf["config_hash"].tolist())

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for _, row in sub.iterrows():
        _scatter_row(ax, row.to_dict(), pf_hashes, n)

    # Pareto step line
    pf_s = pf.sort_values("end_to_end_warm_mean_ms")
    ax.step(
        pf_s["end_to_end_warm_mean_ms"].tolist(),
        pf_s["mean_retrieval_recall_k"].tolist(),
        where="post", color="black", linewidth=1.2,
        linestyle="--", zorder=3, alpha=0.7,
    )

    ax.set_xlabel("Mean warm-start latency (ms)", fontsize=11)
    ax.set_ylabel(r"Recall@$K$ $\pm$ 95\% CI", fontsize=11)
    ax.set_title(f"Quality vs. Warm-Start Latency ($K={k_val}$)", fontsize=12)
    ax.legend(handles=_legend_handles(include_frontier=True), fontsize=8,
              loc="lower right")
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    out = os.path.join(fig_dir, "pareto_recall_latency.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Quality vs. Reranking Latency Overhead at fixed k=K_MAX
# ---------------------------------------------------------------------------

def fig_recall_rerank(df: pd.DataFrame, fig_dir: str, n: int = 500) -> None:
    k_val = int(df["retrieval.top_k"].max())
    sub = df[df["retrieval.top_k"] == k_val].copy()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    rerank_col = "mean_rerank_latency_ms" if "mean_rerank_latency_ms" in sub.columns else "mean_cost_usd"

    for _, row in sub.iterrows():
        rtype = row["retrieval.type"]
        is_reranked = bool(row["rerank.enabled"])
        recall = row["mean_retrieval_recall_k"]
        overhead = float(row[rerank_col])
        ax.errorbar(
            overhead, recall, yerr=ci_half(recall, n),
            fmt="^" if is_reranked else MARKERS[rtype],
            color=PALETTE[rtype],
            alpha=1.0 if is_reranked else 0.4,
            markersize=8 if is_reranked else 5,
            markeredgewidth=1.5 if is_reranked else 0.5,
            markeredgecolor="black" if is_reranked else PALETTE[rtype],
            capsize=3, elinewidth=1,
            zorder=4 if is_reranked else 2,
        )

    # Add vertical separator at 0 (rerank off)
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # Ensure x-axis shows all data points (max rerank overhead ~47ms at K=12)
    max_x = max(float(row[rerank_col]) for _, row in sub.iterrows())
    ax.set_xlim(left=-2, right=max(max_x * 1.1, 52))

    ax.set_xlabel("Mean reranking latency overhead (ms/query)", fontsize=11)
    ax.set_ylabel(r"Recall@$K$ $\pm$ 95\% CI", fontsize=11)
    ax.set_title(f"Quality vs. Reranking Overhead ($K={k_val}$)", fontsize=12)

    handles = [
        mpatches.Patch(color=PALETTE[rt], label=rt.capitalize())
        for rt in ["bm25", "dense", "hybrid"]
    ]
    handles += [
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                   markersize=8, markeredgecolor="black", markeredgewidth=1.5,
                   label="Reranked"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   alpha=0.4, markersize=5, markeredgecolor="gray",
                   label="No reranking"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    out = os.path.join(fig_dir, "pareto_recall_cost.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Cold-Start vs. Warm-Start Latency (log y-scale, y=x line)
# ---------------------------------------------------------------------------

def fig_cold_warm(df: pd.DataFrame, fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for _, row in df.iterrows():
        rtype = row["retrieval.type"]
        ax.scatter(
            row["end_to_end_warm_mean_ms"],
            row["total_cold_ms"],
            c=PALETTE[rtype],
            marker=MARKERS[rtype],
            s=55, alpha=0.7, edgecolors="none", zorder=3,
        )

    # y = x reference line
    xmin = df["end_to_end_warm_mean_ms"].min() * 0.8
    xmax = df["end_to_end_warm_mean_ms"].max() * 1.3
    ax.plot([xmin, xmax], [xmin, xmax], "k--", linewidth=1.0, alpha=0.5,
            label="$T_{\\mathrm{cold}} = T_{\\mathrm{warm}}$")

    # Annotate cluster centres
    bm_row = df[df["retrieval.type"] == "bm25"].iloc[0]
    den_row = df[df["retrieval.type"] == "dense"].iloc[0]
    ax.annotate(
        f"BM25: {bm_row['total_cold_ms']:.0f} ms",
        xy=(bm_row["end_to_end_warm_mean_ms"], bm_row["total_cold_ms"]),
        xytext=(bm_row["end_to_end_warm_mean_ms"] * 3,
                bm_row["total_cold_ms"] * 2.5),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.annotate(
        f"Dense: {den_row['total_cold_ms']/1000:.1f} s",
        xy=(den_row["end_to_end_warm_mean_ms"], den_row["total_cold_ms"]),
        xytext=(den_row["end_to_end_warm_mean_ms"] * 2.0,
                den_row["total_cold_ms"] * 0.55),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    ax.set_yscale("log")
    ax.set_xlabel("Warm-start mean latency (ms)", fontsize=11)
    ax.set_ylabel("Total cold-start latency (ms, log scale)", fontsize=11)
    ax.set_title("Cold-Start vs. Warm-Start Latency", fontsize=12)

    handles = [
        mpatches.Patch(color=PALETTE[rt], label=rt.capitalize())
        for rt in ["bm25", "dense", "hybrid"]
    ]
    handles.append(plt.Line2D([0], [0], color="black", linewidth=1.0,
                               linestyle="--", label="Cold = Warm"))
    ax.legend(handles=handles, fontsize=8, loc="upper left")

    ax.tick_params(labelsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    fig.tight_layout()
    out = os.path.join(fig_dir, "cold_vs_warm_latency.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: Recall@K vs. K ablation
# ---------------------------------------------------------------------------

def fig_recall_vs_k(df: pd.DataFrame, fig_dir: str, n: int = 500) -> None:
    sub = df[~df["rerank.enabled"]].copy()
    k_vals = sorted(sub["retrieval.top_k"].unique())

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for rtype in ["bm25", "dense", "hybrid"]:
        means, cis = [], []
        for k in k_vals:
            vals = sub[
                (sub["retrieval.type"] == rtype) & (sub["retrieval.top_k"] == k)
            ]["mean_retrieval_recall_k"].values
            m = float(vals.mean()) if len(vals) else 0.0
            means.append(m)
            cis.append(ci_half(m, n))
        ax.errorbar(
            k_vals, np.array(means), yerr=np.array(cis),
            marker=MARKERS[rtype],
            color=PALETTE[rtype],
            linewidth=2, markersize=7, capsize=4,
            label=rtype.capitalize(),
        )

    ax.set_xlabel("Top-$K$ retrieved passages", fontsize=11)
    ax.set_ylabel(r"Mean Recall@$K$ $\pm$ 95\% CI", fontsize=11)
    ax.set_title("Recall@$K$ vs. $K$ by Retrieval Type", fontsize=12)
    ax.set_xticks(k_vals)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    out = os.path.join(fig_dir, "recall_vs_k.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# LaTeX table helpers
# ---------------------------------------------------------------------------

def _type_tex(t: str) -> str:
    return {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}.get(t, t)


def _bool_tex(b) -> str:
    return "Yes" if bool(b) else "No"


def _ci_tex(p: float, n: int = 500) -> str:
    lo, hi = wilson_ci(p, n)
    return f"{p:.3f} ({lo:.3f}--{hi:.3f})"


def _cold_tex(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}\\,ms"
    return f"{ms/1000:.1f}\\,s"


def make_pareto_table(pf: pd.DataFrame, out_path: str, n: int = 500) -> None:
    pf = pf.sort_values("mean_retrieval_recall_k", ascending=False)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Pareto-optimal retrieval configurations on HotpotQA"
        r" (500 queries, dev\textsubscript{distractor}, $K\in\{5,8,12\}$,"
        r" overlap fixed at 30~words). Recall@$K$ with 95\,\% Wilson CI."
        r" Cost column is per-query API cost (zero for local inference).}",
        r"\label{tab:pareto}",
        r"\begin{tabular}{lccclccc}",
        r"\toprule",
        r"Type & $K$ & Chunk & Rerank & Recall@$K$ (95\% CI) & "
        r"$\bar{T}_\mathrm{warm}$ (ms) & $T_\mathrm{cold}$ & $C_q$ (USD) \\",
        r"\midrule",
    ]
    for _, row in pf.iterrows():
        lines.append(
            f"{_type_tex(row['retrieval.type'])} & "
            f"{int(row['retrieval.top_k'])} & "
            f"{int(row['chunking.chunk_size'])} & "
            f"{_bool_tex(row['rerank.enabled'])} & "
            f"{_ci_tex(row['mean_retrieval_recall_k'], n)} & "
            f"{row['end_to_end_warm_mean_ms']:.1f} & "
            f"{_cold_tex(row['total_cold_ms'])} & "
            f"{row['mean_cost_usd']:.2e} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


def make_full_table(df: pd.DataFrame, out_path: str, n: int = 500) -> None:
    df = df.sort_values(["retrieval.type", "retrieval.top_k",
                          "chunking.chunk_size", "rerank.enabled"])
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Complete results for all 36 fully factorial pipeline configurations"
        r" on HotpotQA (500 queries). Overlap fixed at 30~words."
        r" $T_\mathrm{cold}$ for BM25 = index build time only;"
        r" for dense/hybrid = model-load ($\approx$10--14\,s) + index + first query.}",
        r"\label{tab:full_results}",
        r"\begin{tabular}{lccclccc}",
        r"\toprule",
        r"Type & $K$ & Chunk & Rerank & Recall@$K$ (95\% CI) & "
        r"$\bar{T}_\mathrm{warm}$ (ms) & $T_\mathrm{cold}$ & $C_q$ (USD) \\",
        r"\midrule",
    ]
    prev_type = None
    for _, row in df.iterrows():
        if row["retrieval.type"] != prev_type and prev_type is not None:
            lines.append(r"\midrule")
        prev_type = row["retrieval.type"]
        lines.append(
            f"{_type_tex(row['retrieval.type'])} & "
            f"{int(row['retrieval.top_k'])} & "
            f"{int(row['chunking.chunk_size'])} & "
            f"{_bool_tex(row['rerank.enabled'])} & "
            f"{_ci_tex(row['mean_retrieval_recall_k'], n)} & "
            f"{row['end_to_end_warm_mean_ms']:.1f} & "
            f"{_cold_tex(row['total_cold_ms'])} & "
            f"{row['mean_cost_usd']:.2e} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


def make_chunking_table(df: pd.DataFrame, out_path: str, n: int = 500) -> None:
    k_val = int(df["retrieval.top_k"].max())
    sub = df[(df["retrieval.top_k"] == k_val) & (~df["rerank.enabled"])].copy()
    lines = [
        rf"\begin{{table}}[ht]",
        r"\centering",
        r"\small",
        rf"\caption{{Chunk-size ablation: Recall@{k_val} and warm latency"
        rf" at $K={k_val}$, no reranking, overlap fixed at 30~words.}}",
        rf"\label{{tab:chunking}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        rf"Type & Chunk (words) & Recall@{k_val} (95\% CI) & "
        r"$\bar{T}_\mathrm{warm}$ (ms) \\",
        r"\midrule",
    ]
    for rtype in ["bm25", "dense", "hybrid"]:
        for cs in sorted(sub["chunking.chunk_size"].unique()):
            rows = sub[(sub["retrieval.type"] == rtype) &
                        (sub["chunking.chunk_size"] == cs)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            lines.append(
                f"{_type_tex(rtype)} & {int(cs)} & "
                f"{_ci_tex(row['mean_retrieval_recall_k'], n)} & "
                f"{row['end_to_end_warm_mean_ms']:.1f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    n = int(df["n_queries"].iloc[0]) if "n_queries" in df.columns else 500

    out_dir = os.path.abspath(os.path.join(os.path.dirname(args.results), ".."))
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    k_max = int(df["retrieval.top_k"].max())
    sub_kmax = df[df["retrieval.top_k"] == k_max].copy()
    pf_warm = pareto_front(sub_kmax, mode="warm")
    pf_cold = pareto_front(df, mode="cold")

    pf_warm.to_csv(os.path.join(out_dir, "pareto_warm.csv"), index=False)
    pf_cold.to_csv(os.path.join(out_dir, "pareto_cold.csv"), index=False)

    print(f"Warm Pareto (k={k_max}): {len(pf_warm)} configurations")
    print(f"Cold Pareto (all k): {len(pf_cold)} configurations")
    for rtype in ["bm25", "dense", "hybrid"]:
        s = df[df["retrieval.type"] == rtype]
        print(
            f"  {rtype:7s}: Recall [{s['mean_retrieval_recall_k'].min():.3f},"
            f"{s['mean_retrieval_recall_k'].max():.3f}]  "
            f"warm [{s['end_to_end_warm_mean_ms'].min():.1f},"
            f"{s['end_to_end_warm_mean_ms'].max():.1f}] ms  "
            f"cold [{s['total_cold_ms'].min():.0f},"
            f"{s['total_cold_ms'].max():.0f}] ms"
        )

    if "model_load_ms" in df.columns:
        dense_rows = df[df["retrieval.type"] == "dense"]
        if len(dense_rows):
            ml = dense_rows["model_load_ms"].iloc[0]
            tc = dense_rows["total_cold_ms"].iloc[0]
            if tc > 0:
                print(f"  Dense model_load={ml:.0f}ms  total_cold={tc:.0f}ms"
                      f"  model_frac={ml/tc*100:.0f}%")

    fig_recall_latency(df, fig_dir, n)
    fig_recall_rerank(df, fig_dir, n)
    fig_cold_warm(df, fig_dir)
    fig_recall_vs_k(df, fig_dir, n)

    make_pareto_table(pf_warm, os.path.join(tab_dir, "pareto_table.tex"), n)
    make_full_table(df, os.path.join(tab_dir, "full_results.tex"), n)
    make_chunking_table(df, os.path.join(tab_dir, "chunking_ablation.tex"), n)

    print("\nAll outputs saved.")


if __name__ == "__main__":
    main()
