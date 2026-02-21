"""
Primary ATE analysis for LLM-ROI Study.
Usage: python3 analysis/scripts/primary_analysis.py --data data/processed/sessions_synthetic.csv --output data/processed/
"""
import argparse, warnings, os
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

OUTCOMES = ["time_to_complete_min", "quality_score_final", "cost_usd_total",
            "hallucination_rate_human", "nasa_tlx_composite", "nasa_tlx_frustration"]
PRIMARY = ["time_to_complete_min", "quality_score_final", "cost_usd_total",
           "hallucination_rate_human", "nasa_tlx_composite", "nasa_tlx_frustration"]
CONTRASTS = [("T1", "control"), ("T2", "control"), ("T2", "T1")]


def load(path):
    df = pd.read_csv(path)
    if "time_to_complete_min" not in df.columns:
        df["time_to_complete_min"] = df["time_to_complete_s"] / 60
    df["low_skill"] = (df["participant_skill_score"] < df["participant_skill_score"].quantile(0.33)).astype(int)
    hard = df.groupby("task_id")["time_to_complete_min"].median().nlargest(
        max(1, int(df["task_id"].nunique() * 0.33))).index
    df["hard_task"] = df["task_id"].isin(hard).astype(int)
    return df


def ate(df, outcome, treat, control):
    sub = df[df["condition"].isin([treat, control])].copy().dropna(subset=[outcome])
    sub["treat"] = (sub["condition"] == treat).astype(int)
    if len(sub) < 10:
        return {"ATE": None, "SE": None, "CI_lo": None, "CI_hi": None, "t_stat": None, "p_value": None}
    formula = f"{outcome} ~ treat + C(task_id) + condition_order"
    covs = [c for c in ["participant_skill_score", "participant_prior_ai_use"] if c in sub.columns]
    if covs:
        formula += " + " + " + ".join(covs)
    try:
        m = ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["participant_id"]})
        ci = m.conf_int().loc["treat"]
        return {"ATE": round(m.params["treat"], 5), "SE": round(m.bse["treat"], 5),
                "CI_lo": round(ci[0], 5), "CI_hi": round(ci[1], 5),
                "t_stat": round(m.tvalues["treat"], 3), "p_value": round(m.pvalues["treat"], 4)}
    except Exception as e:
        print(f"  [WARN] {outcome} {treat} vs {control}: {e}")
        return {"ATE": None, "SE": None, "CI_lo": None, "CI_hi": None, "t_stat": None, "p_value": None}


def welfare_utility(res, lambda_=1.0):
    """Compute welfare utility W = deltaQ / (1 + lambda * max(0, deltaTLX) / 100)"""
    rows = []
    for contrast in ["T1 vs control", "T2 vs control", "T2 vs T1"]:
        q = res[(res["outcome"] == "quality_score_final") & (res["contrast"] == contrast)]["ATE"].values
        t = res[(res["outcome"] == "nasa_tlx_composite") & (res["contrast"] == contrast)]["ATE"].values
        if len(q) and len(t) and q[0] is not None and t[0] is not None:
            dQ = q[0]; dTLX = t[0]
            for lam in [0.5, 1.0, 1.5, 2.0]:
                W = dQ / (1 + lam * max(0, dTLX) / 100)
                rows.append({"contrast": contrast, "delta_quality": round(dQ, 4),
                             "delta_tlx": round(dTLX, 4), "lambda": lam, "welfare_W": round(W, 4)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="data/processed/")
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    df = load(args.data)
    print(f"Loaded {len(df)} rows | {df['participant_id'].nunique()} participants")
    print("\nDescriptive means by condition:")
    print(df.groupby("condition")[["time_to_complete_min", "quality_score_final",
                                    "cost_usd_total", "nasa_tlx_frustration"]].mean().round(3))

    rows = []
    for outcome in OUTCOMES:
        for treat, control in CONTRASTS:
            r = ate(df, outcome, treat, control)
            r["outcome"] = outcome; r["contrast"] = f"{treat} vs {control}"
            rows.append(r)

    res = pd.DataFrame(rows)
    mask = res["outcome"].isin(PRIMARY)
    _, qv, _, _ = multipletests(res.loc[mask, "p_value"].fillna(1.0).values, alpha=0.10, method="fdr_bh")
    res.loc[mask, "q_BH"] = qv.round(4)
    res.loc[mask, "reject_FDR"] = qv < 0.10

    ate_path = os.path.join(args.output, "ate_results.csv")
    res.to_csv(ate_path, index=False)
    print(f"\nATE results -> {ate_path}")
    print(res[["contrast", "outcome", "ATE", "SE", "p_value", "q_BH"]].to_string(index=False))

    wdf = welfare_utility(res)
    if not wdf.empty:
        wp = os.path.join(args.output, "welfare_utility.csv")
        wdf.to_csv(wp, index=False)
        print(f"\nWelfare utility -> {wp}")
        print(wdf.to_string(index=False))


if __name__ == "__main__":
    main()
