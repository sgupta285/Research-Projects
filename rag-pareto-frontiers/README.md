# Latency–Cost–Quality Pareto Frontiers for RAG (v7, paper-grade)

This repo is a **reproducible RAG evaluation harness** that produces Pareto frontiers over:
- **Quality**: retrieval Recall@k / Precision@k (and optional answer F1 proxy)
- **Latency**: **cold-start** vs **warm-start** latency (separately reported)
- **Cost**: configurable pricing model for rerank/LLM calls

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Download dataset (HotpotQA)
```bash
python -m src.cli.download_dataset --dataset hotpotqa --split dev_distractor --out data/hotpotqa
```

## 2) Run a baseline
```bash
python -m src.cli.run_eval --config configs/baselines/hotpotqa_bm25.yaml
```

## 3) Run sweep + Pareto plots
```bash
python -m src.cli.run_sweep --sweep configs/sweeps/hotpotqa_sweep_small.yaml
python -m src.cli.make_pareto --results outputs/tables/sweep_results.csv
```

## 4) Compile the paper PDF

### Option A: Local LaTeX (latexmk)
```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
open main.pdf
```

### Option B: Overleaf (recommended for submission)
1. Upload the `paper/` folder to Overleaf at https://overleaf.com
2. Upload figures from `outputs/figures/` to the same project
3. Set compiler to **pdfLaTeX** and click **Recompile**

## Tokens / secrets
Do **NOT** hardcode tokens into config files. Set them as environment variables:
```bash
export HF_TOKEN="hf_...your_token..."
```

## LegalBench-RAG (optional second benchmark)
Download the dataset from the official project and place under `data/legalbenchrag/`
with `corpus/` and `benchmarks/` subdirectories. Then run:
```bash
python -m src.cli.run_eval --config configs/baselines/legalbenchrag_hybrid.yaml
```
