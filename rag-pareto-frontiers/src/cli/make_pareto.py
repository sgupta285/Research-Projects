from __future__ import annotations
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def dominates(b,a):
    return (b["mean_retrieval_recall_k"]>=a["mean_retrieval_recall_k"]
        and b["end_to_end_warm_mean_ms"]<=a["end_to_end_warm_mean_ms"]
        and b["mean_cost_usd"]<=a["mean_cost_usd"]
        and (b["mean_retrieval_recall_k"]>a["mean_retrieval_recall_k"]
             or b["end_to_end_warm_mean_ms"]<a["end_to_end_warm_mean_ms"]
             or b["mean_cost_usd"]<a["mean_cost_usd"]))

def pareto(df):
    rows=df.to_dict("records")
    keep=[]
    for i,a in enumerate(rows):
        if any(dominates(b,a) for j,b in enumerate(rows) if i!=j): 
            continue
        keep.append(a)
    return pd.DataFrame(keep)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args=ap.parse_args()

    df=pd.read_csv(args.results)
    pf=pareto(df).sort_values(["mean_retrieval_recall_k","end_to_end_warm_mean_ms","mean_cost_usd"], ascending=[False,True,True])

    out_dir=os.path.abspath(os.path.join(os.path.dirname(args.results),".."))
    fig_dir=os.path.join(out_dir,"figures")
    os.makedirs(fig_dir, exist_ok=True)

    pf.to_csv(os.path.join(out_dir,"pareto.csv"), index=False)

    plt.figure()
    plt.scatter(df["total_warm_ms"], df["mean_retrieval_recall_k"])
    plt.scatter(pf["total_warm_ms"], pf["mean_retrieval_recall_k"])
    plt.xlabel("Warm mean end-to-end latency (ms)")
    plt.ylabel("Mean Retrieval Recall@k")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,"pareto_recall_latency.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(df["mean_cost_usd"], df["mean_retrieval_recall_k"])
    plt.scatter(pf["mean_cost_usd"], pf["mean_retrieval_recall_k"])
    plt.xlabel("Mean cost (USD)")
    plt.ylabel("Mean Retrieval Recall@k")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,"pareto_recall_cost.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(df["total_warm_ms"], df["end_to_end_cold_ms"])
    plt.xlabel("Warm mean end-to-end latency (ms)")
    plt.ylabel("Total cold-start latency (build + 1st query) (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,"cold_vs_warm_latency.png"), dpi=200)
    plt.close()

    print("Saved outputs/pareto.csv and outputs/figures/*.png")

if __name__=="__main__":
    main()
