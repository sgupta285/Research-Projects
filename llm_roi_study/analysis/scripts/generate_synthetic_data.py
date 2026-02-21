"""
Generate synthetic dataset for testing analysis scripts BEFORE data collection.
Usage: python3 analysis/scripts/generate_synthetic_data.py --n 200 --output data/processed/sessions_synthetic.csv
WARNING: All rows flagged is_synthetic=True. Never report as real findings.
"""
import numpy as np, pandas as pd, argparse, os

RNG = np.random.default_rng(42)
TASKS = {"A": [("A01","easy"),("A02","medium"),("A03","hard"),("A04","easy"),("A05","medium"),("A06","hard")],
         "B": [("B01","easy"),("B02","medium"),("B03","hard"),("B04","easy"),("B05","medium"),("B06","hard")],
         "C": [("C01","easy"),("C02","medium"),("C03","hard"),("C04","easy"),("C05","medium"),("C06","hard")]}
LATIN = [["control","T1","T2"],["T1","T2","control"],["T2","control","T1"]]


def sim(pid, task_id, cond, diff, skill, prior_ai):
    da = {"easy": -1.0, "medium": 0.0, "hard": 2.0}[diff]
    sa = (1.0 - skill) * 3.0
    if cond == "control":
        t = 12 + da + sa + RNG.normal(0, 2.5)
        q = 5.5 - da * 0.3 - (1 - skill) + RNG.normal(0, 1.2)
        c = 0.0; h = float(np.clip(0.25 + (1-skill)*0.1 + RNG.normal(0,0.05), 0, 1)); rl = None
    elif cond == "T1":
        t = 9 + da * 0.5 + sa * 0.6 + RNG.normal(0, 2)
        q = 6.5 - da * 0.2 + (skill-0.5)*0.5 + RNG.normal(0, 1)
        c = max(0, 0.035 + RNG.normal(0, 0.008))
        h = float(np.clip(0.20 + (1-skill)*0.08 + RNG.normal(0,0.04), 0, 1)); rl = None
    else:
        t = 8 + da * 0.4 + sa * 0.5 + RNG.normal(0, 2.2)
        q = 7.0 - da * 0.15 + (skill-0.5)*0.6 + RNG.normal(0, 1)
        c = max(0, 0.055 + RNG.normal(0, 0.012))
        h = float(np.clip(0.10 + (1-skill)*0.05 + RNG.normal(0,0.03), 0, 1))
        rl = round(float(abs(RNG.normal(310, 60))), 1)
    nasa = {k: int(np.clip(RNG.integers(25,75), 0, 100))
            for k in ["mental","physical","temporal","performance","effort","frustration"]}
    if cond == "T1": nasa["frustration"] = int(np.clip(nasa["frustration"] + 3, 0, 100))
    if cond == "T2":
        nasa["frustration"] = int(np.clip(nasa["frustration"] + 12, 0, 100))
        nasa["mental"] = int(np.clip(nasa["mental"] + 5, 0, 100))
    return {"participant_id": pid, "task_id": task_id, "task_category": task_id[0],
            "condition": cond, "task_difficulty": diff,
            "participant_skill_score": round(skill,3), "participant_prior_ai_use": prior_ai,
            "time_to_complete_s": round(max(60, t*60), 1),
            "time_to_complete_min": round(max(1.0, t), 2),
            "quality_score_final": round(float(np.clip(q, 0, 10)), 2),
            "cost_usd_total": round(float(c), 5),
            "hallucination_rate_human": round(h, 3),
            "rework_count": int(RNG.poisson(1.5)) if cond != "control" else 0,
            "retrieval_enabled": cond == "T2", "retrieval_latency_ms": rl,
            "generation_latency_ms": round(float(abs(RNG.normal(3200, 400))), 1),
            "prompt_tokens_total": int(RNG.normal(4000,500)) if cond != "control" else 0,
            "completion_tokens_total": int(RNG.normal(900,150)) if cond != "control" else 0,
            **{f"nasa_tlx_{k}": v for k, v in nasa.items()},
            "nasa_tlx_composite": round(float(np.mean(list(nasa.values()))), 2),
            "ragas_faithfulness": round(float(RNG.beta(8,2)),3) if cond=="T2" else None,
            "ragas_answer_relevance": round(float(RNG.beta(9,2)),3) if cond=="T2" else None,
            "ragas_context_recall": round(float(RNG.beta(7,3)),3) if cond=="T2" else None,
            "ragas_context_precision": round(float(RNG.beta(7,3)),3) if cond=="T2" else None,
            "status": "completed", "condition_order": int(RNG.integers(1,4)), "is_synthetic": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--output", default="data/processed/sessions_synthetic.csv")
    args = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    rows = []
    for pid in range(1, args.n + 1):
        pstr = f"p{pid:03d}"; skill = float(RNG.beta(3,3)); prior_ai = bool(RNG.random() > 0.4)
        ls = LATIN[(pid-1) % 3]
        for cat, tlist in TASKS.items():
            shuf = RNG.permutation(len(tlist)).tolist()
            for i, cond in enumerate(ls):
                tid, diff = tlist[shuf[i]]
                rows.append(sim(pstr, tid, cond, diff, skill, prior_ai))
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} rows | {args.n} participants -> {args.output}")
    print(df.groupby("condition")[["time_to_complete_min","quality_score_final",
                                    "cost_usd_total","hallucination_rate_human",
                                    "nasa_tlx_composite","nasa_tlx_frustration"]].mean().round(3))
    print("\n*** WARNING: is_synthetic=True — do NOT report as study findings ***")


if __name__ == "__main__":
    main()
