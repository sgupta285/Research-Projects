from __future__ import annotations
import argparse, os
import pandas as pd
from src.utils.config import load_yaml, config_hash, ensure_dir
from src.pipeline.run import build_artifacts, run_eval

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()

    cfg=load_yaml(args.config)
    cfg["dataset"]=load_yaml(cfg["dataset"])
    cfg["pricing"]=load_yaml(cfg["pricing"])

    out_dir=cfg["outputs"]["out_dir"]
    ensure_dir(os.path.join(out_dir,"tables"))
    h=config_hash(cfg)

    art, timings = build_artifacts(cfg)
    rows, meta = run_eval(cfg, art)

    df=pd.DataFrame(rows)
    out=os.path.join(out_dir,"tables",f"results_{h}.csv")
    df.to_csv(out,index=False)

    agg=df[["quality_f1","retrieval_recall_k","retrieval_precision_k","retrieval_ms","latency_ms","cost_usd"]].mean().to_dict()
    print(f"[OK] Wrote: {out}")
    print("[AGG]", agg)
    print("[TIMINGS]", timings)
    print("[META]", meta)

if __name__=="__main__":
    main()
