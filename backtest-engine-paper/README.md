# Backtest Engineering (Production-Ready Research Repo)

This repository is designed to be **submission-ready** (arXiv/workshop) and extensible for journal-level work.

It contains:
- **Strict event-driven engine** with an explicit event queue: `Market -> Signal -> Order -> Fill`
- **Execution realism ladder** (fees, spread, volatility slippage, impact proxy, delay)
- **Partial fill + no-leverage sizing** (long-only, cash-constrained)
- **Correctness invariants** + unit tests (accounting identity, causality, fill consistency, cost monotonicity)
- **Reproducible experiment harness**:
  - period splits (e.g., 2005–2012 / 2013–2019 / 2020–2025)
  - bootstrap confidence intervals for key metrics
  - figures and tables regenerated from config
- LaTeX paper scaffold under `paper/`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download & freeze data (Stooq, daily)
```bash
python -m src.experiments.download_data --symbols SPY QQQ IWM DIA XLF XLK XLE XLV XLY XLP --start 2005-01-01
```

### Run experiments (includes period splits + bootstrap CIs)
```bash
python -m src.experiments.run_grid --config src/experiments/configs/default.yaml
```

Outputs:
- `outputs/tables/metrics.csv`
- `outputs/tables/metrics_by_period.csv`
- `outputs/tables/inflation_ratios.csv`
- `outputs/tables/bootstrap_ci.csv`
- `outputs/figures/*.png`
- `outputs/events/*.csv` (optional event logs)

### Build paper
```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

### One-command rebuild (mac/linux)
```bash
bash scripts/rebuild_all.sh
```

## Disclaimer
This is a research backtester for methodological studies. It is **not** a brokerage OMS and does not model full limit order book microstructure.
