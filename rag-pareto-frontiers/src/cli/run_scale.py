"""Corpus-size scaling experiment.

Measures cold-start cost (model load + index build) for BM25, dense, and
hybrid retrieval as corpus size grows from 1K to 100K passages.

The base corpus is HotpotQA dev_distractor (~5K unique passages).  To reach
larger sizes the corpus is padded by repeating the base passages with minor
text perturbations so that the FAISS encoder sees distinct token sequences.
This correctly stresses corpus-encoding time—the dominant cold-start cost—
while keeping the experiment self-contained (no extra data download).

Outputs:
  outputs/tables/scale_results.csv   — raw timing data
  outputs/figures/fig_cold_scale.png — log-log plot of cold-start vs corpus size
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import List

import pandas as pd

from src.data.io import load_hotpotqa, Doc
from src.data.chunking import build_chunks
from src.retrieval.bm25 import build_bm25
from src.retrieval.dense import load_model, build_dense
from src.utils.config import ensure_dir


# --------------------------------------------------------------------------- #
# Corpus padding                                                               #
# --------------------------------------------------------------------------- #

def _pad_corpus(base_docs: List[Doc], target_n: int, rng: random.Random) -> List[Doc]:
    """Return a corpus of exactly `target_n` Doc objects.

    If target_n <= len(base_docs), returns a random sample.
    Otherwise pads by repeating docs with a numeric suffix to ensure distinct
    fingerprints (so caches don't short-circuit the build).
    """
    n = len(base_docs)
    if target_n <= n:
        return rng.sample(base_docs, target_n)

    result = list(base_docs)
    copy_idx = 0
    while len(result) < target_n:
        src = base_docs[copy_idx % n]
        suffix = f" [{len(result)}]"
        result.append(Doc(
            doc_id=f"{src.doc_id}_{len(result)}",
            title=src.title,
            text=src.text + suffix,
            source_path=src.source_path,
        ))
        copy_idx += 1
    return result


# --------------------------------------------------------------------------- #
# Timing helpers                                                               #
# --------------------------------------------------------------------------- #

def _time_bm25_cold(docs: List[Doc], chunk_size: int = 250, overlap: int = 30) -> dict:
    t0 = time.perf_counter()
    chunks = build_chunks(docs, mode="words", chunk_size=chunk_size, overlap=overlap)
    t1 = time.perf_counter()
    _ = build_bm25(chunks)
    t2 = time.perf_counter()
    return {
        "chunk_ms": (t1 - t0) * 1000.0,
        "index_ms": (t2 - t1) * 1000.0,
        "total_cold_ms": (t2 - t0) * 1000.0,
        "n_chunks": len(chunks),
    }


def _time_dense_cold(
    docs: List[Doc],
    model,
    model_load_ms: float,
    chunk_size: int = 250,
    overlap: int = 30,
) -> dict:
    t0 = time.perf_counter()
    chunks = build_chunks(docs, mode="words", chunk_size=chunk_size, overlap=overlap)
    t1 = time.perf_counter()
    _ = build_dense(chunks, model=model, model_name="sentence-transformers/all-MiniLM-L6-v2")
    t2 = time.perf_counter()
    encode_ms = (t2 - t1) * 1000.0
    return {
        "chunk_ms": (t1 - t0) * 1000.0,
        "encode_ms": encode_ms,
        "model_load_ms": model_load_ms,
        "total_cold_ms": model_load_ms + encode_ms,
        "n_chunks": len(chunks),
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

_CORPUS_SIZES = [1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000]
_CHUNK_SIZE = 250
_OVERLAP = 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hotpotqa")
    ap.add_argument("--split", default="dev_distractor")
    ap.add_argument("--max-base", type=int, default=500,
                    help="Number of HotpotQA examples used to build the base corpus")
    ap.add_argument("--sizes", nargs="+", type=int, default=_CORPUS_SIZES,
                    help="Corpus sizes to benchmark")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Timing repeats per (size, type) pair for variance estimation")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    ensure_dir(os.path.join(args.out_dir, "tables"))
    ensure_dir(os.path.join(args.out_dir, "figures"))

    # Load base corpus
    print(f"Loading HotpotQA ({args.split}, max {args.max_base} examples) ...")
    base_docs, _ = load_hotpotqa(args.data_dir, args.split, args.max_base)
    print(f"  Base corpus: {len(base_docs)} docs")

    # Pre-load dense model once (we record this separately)
    print("Loading sentence-transformer model ...")
    t_ml0 = time.perf_counter()
    model = load_model("sentence-transformers/all-MiniLM-L6-v2")
    model_load_ms = (time.perf_counter() - t_ml0) * 1000.0
    print(f"  Model loaded in {model_load_ms:.0f} ms")

    rng = random.Random(42)
    rows = []

    for size in sorted(set(args.sizes)):
        print(f"\n=== Corpus size: {size:,} ===")
        corpus = _pad_corpus(base_docs, size, rng)

        for rep in range(args.repeats):
            # BM25
            r_bm = _time_bm25_cold(corpus, _CHUNK_SIZE, _OVERLAP)
            rows.append({
                "corpus_size": size,
                "retrieval_type": "bm25",
                "repeat": rep,
                **r_bm,
                "model_load_ms": 0.0,
            })
            print(f"  BM25  rep={rep} total={r_bm['total_cold_ms']:.0f}ms  chunks={r_bm['n_chunks']}")

            # Dense (model already loaded; encode cost only)
            r_dn = _time_dense_cold(corpus, model, model_load_ms, _CHUNK_SIZE, _OVERLAP)
            rows.append({
                "corpus_size": size,
                "retrieval_type": "dense",
                "repeat": rep,
                **r_dn,
            })
            print(f"  Dense rep={rep} total={r_dn['total_cold_ms']:.0f}ms  encode={r_dn['encode_ms']:.0f}ms  chunks={r_dn['n_chunks']}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "tables", "scale_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # --- plot ---
    _make_scale_figure(df, os.path.join(args.out_dir, "figures", "fig_cold_scale.png"))


def _make_scale_figure(df: pd.DataFrame, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping figure generation")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = {"bm25": "#2166ac", "dense": "#d6604d"}
    markers = {"bm25": "s", "dense": "o"}
    labels  = {"bm25": "BM25", "dense": "Dense (all-MiniLM-L6-v2)"}

    agg = df.groupby(["corpus_size", "retrieval_type"])["total_cold_ms"].agg(["mean", "std"]).reset_index()

    for rtype in ["bm25", "dense"]:
        sub = agg[agg["retrieval_type"] == rtype].sort_values("corpus_size")
        ax.errorbar(
            sub["corpus_size"], sub["mean"] / 1000.0,
            yerr=sub["std"] / 1000.0,
            marker=markers[rtype], color=colors[rtype],
            label=labels[rtype], linewidth=1.5, capsize=3, markersize=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Corpus size (passages)", fontsize=11)
    ax.set_ylabel("Cold-start latency (s)", fontsize=11)
    ax.set_title("Cold-Start Latency vs Corpus Size", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # annotate crossover if it exists
    bm = agg[agg["retrieval_type"] == "bm25"].set_index("corpus_size")["mean"]
    dn = agg[agg["retrieval_type"] == "dense"].set_index("corpus_size")["mean"]
    common = bm.index.intersection(dn.index)
    for sz in sorted(common):
        if dn[sz] > bm[sz]:
            ax.axvline(sz, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(sz * 1.05, ax.get_ylim()[0] * 1.5,
                    f"dense > BM25\nat {sz//1000}K", fontsize=7, color="grey")
            break

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
