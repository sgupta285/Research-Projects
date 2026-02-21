"""
ROI Frontier and pre-specified plots for LLM-ROI Study.
Usage: python3 analysis/scripts/roi_frontier.py --data data/processed/ate_results.csv --sessions data/processed/sessions_synthetic.csv --output figures/
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
sns.set_style("whitegrid")
COLORS = {"T1": "#2563EB", "T2": "#16A34A", "control": "#9CA3AF"}
MARKERS = {"T1": "o", "T2": "s"}


def fig1_roi(ate, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    q = ate[(ate["outcome"] == "quality_score_final") & ate["contrast"].str.contains("control")]
    c = ate[(ate["outcome"] == "cost_usd_total") & ate["contrast"].str.contains("control")]
    for _, qr in q.iterrows():
        tr = qr["contrast"].split(" vs ")[0]
        cr = c[c["contrast"] == qr["contrast"]]
        if cr.empty or pd.isna(qr["ATE"]): continue
        x = cr["ATE"].values[0] or 0; y = qr["ATE"] or 0
        ax.scatter(x, y, s=200, color=COLORS[tr], marker=MARKERS[tr], zorder=5, label=f"{tr} vs Control")
        ax.annotate(tr, (x, y), xytext=(8, 4), textcoords="offset points", fontsize=11)
    ax.axhline(0, color="gray", ls="--", alpha=0.5); ax.axvline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Delta Cost per task (USD)"); ax.set_ylabel("Delta Quality (0-10)")
    ax.set_title("Figure 1: ROI Frontier — Quality Gain vs. Cost", fontweight="bold")
    ax.legend(); fig.tight_layout()
    p = os.path.join(out, "fig1_roi_frontier.png"); fig.savefig(p, dpi=150); plt.close(); print(f"Saved {p}")


def fig2_tlx(sessions, out):
    subs = ["nasa_tlx_mental", "nasa_tlx_physical", "nasa_tlx_temporal",
            "nasa_tlx_performance", "nasa_tlx_effort", "nasa_tlx_frustration"]
    labels = ["Mental", "Physical", "Temporal", "Performance", "Effort", "Frustration"]
    fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(len(subs)); width = 0.25
    for i, cond in enumerate(["control", "T1", "T2"]):
        sub = sessions[sessions["condition"] == cond]
        if sub.empty: continue
        ax.bar(x + i * width, [sub[s].mean() for s in subs], width,
               yerr=[sub[s].sem() for s in subs], color=COLORS.get(cond, "#888"),
               label=cond.upper(), alpha=0.85, capsize=3)
    ax.set_xticks(x + width); ax.set_xticklabels(labels)
    ax.set_ylim(0, 100); ax.set_ylabel("NASA-TLX Score (0-100)")
    ax.set_title("Figure 2: NASA-TLX Workload Profile by Condition", fontweight="bold")
    ax.legend(); fig.tight_layout()
    p = os.path.join(out, "fig2_nasa_tlx.png"); fig.savefig(p, dpi=150); plt.close(); print(f"Saved {p}")


def fig3_welfare(ate, out):
    """Welfare utility W for lambda=[0.5,1.0,1.5,2.0] by condition."""
    contrasts = ["T1 vs control", "T2 vs control"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(contrasts)); width = 0.2
    for j, lam in enumerate([0.5, 1.0, 1.5, 2.0]):
        vals = []
        for c in contrasts:
            q = ate[(ate["outcome"] == "quality_score_final") & (ate["contrast"] == c)]["ATE"].values
            t = ate[(ate["outcome"] == "nasa_tlx_composite") & (ate["contrast"] == c)]["ATE"].values
            if len(q) and len(t) and q[0] is not None and t[0] is not None:
                vals.append(q[0] / (1 + lam * max(0, t[0]) / 100))
            else:
                vals.append(0)
        ax.bar(x + j * width, vals, width, label=f"λ={lam}", alpha=0.85)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xticks(x + width * 1.5); ax.set_xticklabels(["T1 vs Control", "T2 vs Control"])
    ax.set_ylabel("Welfare Utility W"); ax.set_title("Figure 3: Welfare Utility by Condition and λ", fontweight="bold")
    ax.legend(title="Lambda"); fig.tight_layout()
    p = os.path.join(out, "fig3_welfare_utility.png"); fig.savefig(p, dpi=150); plt.close(); print(f"Saved {p}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--sessions", required=True)
    p.add_argument("--output", default="figures/")
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)
    ate = pd.read_csv(args.data)
    sessions = pd.read_csv(args.sessions)
    if "time_to_complete_min" not in sessions.columns:
        sessions["time_to_complete_min"] = sessions["time_to_complete_s"] / 60
    fig1_roi(ate, args.output)
    fig2_tlx(sessions, args.output)
    fig3_welfare(ate, args.output)
    print("\nAll figures saved to", args.output)


if __name__ == "__main__":
    main()
