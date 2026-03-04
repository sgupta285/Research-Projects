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
    ap.add_argument("--out", default=None, help="Override output CSV path")
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
      "generation": sweep.get("generation", {"enabled": False, "backend": "mock"}),
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

        base_cols=["quality_f1","retrieval_recall_k","retrieval_precision_k","retrieval_ms","rerank_latency_ms","latency_ms","cost_usd"]
        agg=df[base_cols].mean().to_dict()

        gen_enabled = cfg["generation"].get("enabled", False)
        gen_agg = {}
        if gen_enabled and "generation_f1" in df.columns:
            gen_agg = df[["generation_f1","generation_em","generation_faithfulness","generation_latency_ms","generation_cost_usd"]].mean().to_dict()

        row={
          "config_hash":h,
          **params,
          "mean_quality_f1":float(agg["quality_f1"]),
          "mean_retrieval_recall_k":float(agg["retrieval_recall_k"]),
          "mean_retrieval_precision_k":float(agg["retrieval_precision_k"]),
          "mean_retrieval_ms":float(agg["retrieval_ms"]),
          "mean_latency_ms":float(agg["latency_ms"]),
          "mean_cost_usd":float(agg["cost_usd"]),
          "mean_rerank_latency_ms":float(agg.get("rerank_latency_ms", 0.0)),
          "std_latency_ms":float(df["latency_ms"].std()),
          "model_load_ms":float(timings.get("model_load_ms",0.0)),
          "index_build_ms":float(timings.get("index_build_ms",0.0)),
          "artifact_total_ms":float(timings.get("artifact_total_ms",0.0)),
          "retrieval_cold_ms":float(meta["retrieval_cold_ms"]),
          "retrieval_warm_mean_ms":float(meta["retrieval_warm_mean_ms"]),
          "retrieval_warm_std_ms":float(meta.get("retrieval_warm_std_ms",0.0)),
          "end_to_end_cold_ms":float(meta["end_to_end_cold_ms"]),
          "end_to_end_warm_mean_ms":float(meta["end_to_end_warm_mean_ms"]),
          "end_to_end_warm_std_ms":float(meta.get("end_to_end_warm_std_ms",0.0)),
          "total_cold_ms":float(timings.get("model_load_ms",0.0)+timings.get("index_build_ms",0.0)+meta["end_to_end_cold_ms"]),
          "total_warm_ms":float(meta["end_to_end_warm_mean_ms"]),
          "gen_latency_mean_ms": float(meta.get("gen_latency_mean_ms", 0.0)),
          "gen_latency_std_ms": float(meta.get("gen_latency_std_ms", 0.0)),
          "gen_cost_total_usd": float(meta.get("gen_cost_total_usd", 0.0)),
          "mean_generation_f1": float(gen_agg.get("generation_f1", 0.0)),
          "mean_generation_em": float(gen_agg.get("generation_em", 0.0)),
          "mean_generation_faithfulness": float(gen_agg.get("generation_faithfulness", 0.0)),
          "n_queries":len(df),
        }
        rows_all.append(row)

        extra = ""
        if gen_enabled:
            extra = f" genF1={row['mean_generation_f1']:.3f} genEM={row['mean_generation_em']:.3f} faith={row['mean_generation_faithfulness']:.3f} genLat={row['gen_latency_mean_ms']:.0f}ms genC=${row['gen_cost_total_usd']:.4f}"
        print(f"[OK] {h} R@k={agg['retrieval_recall_k']:.3f} warm={meta['end_to_end_warm_mean_ms']:.1f}ms cold_query={meta['end_to_end_cold_ms']:.1f}ms total_cold={(timings.get('model_load_ms',0.0)+timings.get('index_build_ms',0.0)+meta['end_to_end_cold_ms']):.1f}ms C=${agg['cost_usd']:.6f}{extra}")

    csv_path = args.out or os.path.join(out_dir,"tables","sweep_results.csv")
    pd.DataFrame(rows_all).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

if __name__=="__main__":
    main()
