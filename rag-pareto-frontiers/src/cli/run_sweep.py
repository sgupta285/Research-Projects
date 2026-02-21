from __future__ import annotations
import argparse, os, itertools, json
import pandas as pd
from src.utils.config import load_yaml, deep_set, config_hash, ensure_dir
from src.pipeline.run import build_artifacts, run_eval

def expand(grid):
    keys=list(grid.keys())
    vals=[grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    args=ap.parse_args()

    sweep=load_yaml(args.sweep)
    base={
      "dataset": load_yaml(sweep["dataset"]),
      "pricing": load_yaml(sweep["pricing"]),
      "dense_model_name":"sentence-transformers/all-MiniLM-L6-v2",
      "pipeline":{
        "chunking":{"mode":"words","chunk_size":200,"overlap":40},
        "retrieval":{"type":"bm25","top_k":8},
        "rerank":{"enabled":False,"type":"simple"},
        "caching":{"retrieval_cache":True},
        "seed":7,
      },
      "outputs":{"out_dir": sweep["outputs"]["out_dir"]},
    }

    out_dir=base["outputs"]["out_dir"]
    ensure_dir(os.path.join(out_dir,"tables"))

    rows_all=[]
    for params in expand(sweep["grid"]):
        cfg=json.loads(json.dumps(base))
        for k,v in params.items():
            deep_set(cfg["pipeline"], k, v)
        h=config_hash(cfg)

        art, timings = build_artifacts(cfg)
        rows, meta = run_eval(cfg, art)
        df=pd.DataFrame(rows)

        agg=df[["quality_f1","retrieval_recall_k","retrieval_precision_k","retrieval_ms","latency_ms","cost_usd"]].mean().to_dict()

        rows_all.append({
          "config_hash":h,
          **params,
          "mean_quality_f1":float(agg["quality_f1"]),
          "mean_retrieval_recall_k":float(agg["retrieval_recall_k"]),
          "mean_retrieval_precision_k":float(agg["retrieval_precision_k"]),
          "mean_retrieval_ms":float(agg["retrieval_ms"]),
          "mean_latency_ms":float(agg["latency_ms"]),
          "mean_cost_usd":float(agg["cost_usd"]),
          "model_load_ms":float(timings.get("model_load_ms",0.0)),
          "index_build_ms":float(timings.get("index_build_ms",0.0)),
          "artifact_total_ms":float(timings.get("artifact_total_ms",0.0)),
          "retrieval_cold_ms":float(meta["retrieval_cold_ms"]),
          "retrieval_warm_mean_ms":float(meta["retrieval_warm_mean_ms"]),
          "end_to_end_cold_ms":float(meta["end_to_end_cold_ms"]),
          "end_to_end_warm_mean_ms":float(meta["end_to_end_warm_mean_ms"]),
          "total_cold_ms":float(timings.get("model_load_ms",0.0)+timings.get("index_build_ms",0.0)+meta["end_to_end_cold_ms"]),
          "total_warm_ms":float(meta["end_to_end_warm_mean_ms"]),
          "n_queries":len(df),
        })
        print(f"[OK] {h} R@k={agg['retrieval_recall_k']:.3f} warm={meta['end_to_end_warm_mean_ms']:.1f}ms cold_query={meta['end_to_end_cold_ms']:.1f}ms total_cold={(timings.get('model_load_ms',0.0)+timings.get('index_build_ms',0.0)+meta['end_to_end_cold_ms']):.1f}ms model={timings.get('model_load_ms',0.0):.1f}ms index={timings.get('index_build_ms',0.0):.1f}ms mhit={int(timings.get('model_cache_hit',False))} ihit={int(timings.get('index_cache_hit',False))} model_obs={timings.get('model_load_ms_observed',0.0):.1f} index_obs={timings.get('index_build_ms_observed',0.0):.1f} C=${agg['cost_usd']:.6f}")

    out=os.path.join(out_dir,"tables","sweep_results.csv")
    pd.DataFrame(rows_all).to_csv(out,index=False)
    print(f"Saved: {out}")

if __name__=="__main__":
    main()
