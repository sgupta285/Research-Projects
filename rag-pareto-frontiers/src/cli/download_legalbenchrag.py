"""Download and preprocess the LegalBench-RAG corpus.

LegalBench-RAG is available at:
  https://github.com/zeroentropy-ai/legalbenchrag

Usage:
  python -m src.cli.download_legalbenchrag --out data/legalbenchrag

This script:
  1. Clones / downloads the LegalBench-RAG repository
  2. Converts the benchmark JSON to the format expected by load_legalbenchrag()
  3. Copies corpus files to data/legalbenchrag/corpus/
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/zeroentropy-ai/legalbenchrag.git"


def clone_repo(target_dir: str) -> str:
    repo_dir = os.path.join(target_dir, "_legalbenchrag_repo")
    if os.path.exists(repo_dir):
        print(f"Repo already exists at {repo_dir}, skipping clone.")
        return repo_dir
    print(f"Cloning {REPO_URL} ...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, repo_dir], check=True)
    return repo_dir


def convert_benchmark(repo_dir: str, out_dir: str) -> None:
    """Locate benchmark JSON in the cloned repo and normalise field names."""
    candidates = []
    for root, dirs, files in os.walk(repo_dir):
        for fname in files:
            if fname.endswith(".json") and "benchmark" in fname.lower():
                candidates.append(os.path.join(root, fname))

    if not candidates:
        print("ERROR: Could not find benchmark JSON in cloned repo. "
              "Check the repo structure manually.")
        sys.exit(1)

    src_json = candidates[0]
    print(f"Using benchmark file: {src_json}")

    with open(src_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Normalise to list of dicts with keys: id, query, ground_truth
    if isinstance(raw, dict) and "test_cases" in raw:
        cases = raw["test_cases"]
    elif isinstance(raw, list):
        cases = raw
    else:
        cases = list(raw.values())

    normalised = []
    for i, c in enumerate(cases):
        qid = c.get("id") or c.get("_id") or str(i)
        query = c.get("query") or c.get("question") or ""
        gt = c.get("ground_truth") or c.get("ground_truth_snippets") or []
        spans = []
        for s in gt:
            path = s.get("path") or s.get("file_path") or s.get("corpus_path")
            start = s.get("start") or s.get("start_idx") or s.get("char_start")
            end = s.get("end") or s.get("end_idx") or s.get("char_end")
            if path is not None and start is not None and end is not None:
                spans.append({"path": str(path), "start": int(start), "end": int(end)})
        normalised.append({"id": qid, "query": query, "ground_truth": spans})

    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, "benchmark.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump({"test_cases": normalised}, f, indent=2)
    print(f"Benchmark written to {dest} ({len(normalised)} test cases)")


def copy_corpus(repo_dir: str, out_dir: str) -> None:
    """Copy corpus text files to out_dir/corpus/."""
    corpus_src = None
    for candidate in ["corpus", "data/corpus", "documents", "texts"]:
        p = os.path.join(repo_dir, candidate)
        if os.path.isdir(p):
            corpus_src = p
            break

    if corpus_src is None:
        print("WARNING: Could not find corpus directory in repo. "
              "You may need to copy corpus files manually to data/legalbenchrag/corpus/")
        return

    corpus_dst = os.path.join(out_dir, "corpus")
    if os.path.exists(corpus_dst):
        print(f"Corpus already at {corpus_dst}, skipping copy.")
        return

    print(f"Copying corpus from {corpus_src} to {corpus_dst} ...")
    shutil.copytree(corpus_src, corpus_dst)
    n = sum(1 for _, _, files in os.walk(corpus_dst) for _ in files)
    print(f"Copied {n} files.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download LegalBench-RAG corpus")
    ap.add_argument("--out", default="data/legalbenchrag", help="Output directory")
    ap.add_argument("--skip-clone", action="store_true", help="Skip git clone (repo already downloaded)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not args.skip_clone:
        repo_dir = clone_repo(args.out)
    else:
        repo_dir = os.path.join(args.out, "_legalbenchrag_repo")

    convert_benchmark(repo_dir, args.out)
    copy_corpus(repo_dir, args.out)
    print(f"\nDone. Run sweep with:\n  python -m src.cli.run_sweep --sweep configs/sweeps/legalbenchrag_sweep.yaml")


if __name__ == "__main__":
    main()
