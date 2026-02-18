#!/usr/bin/env bash
set -euo pipefail

SYMS=${SYMS:-"SPY QQQ IWM DIA XLF XLK XLE XLV XLY XLP"}
START=${START:-"2005-01-01"}
CONFIG=${CONFIG:-"src/experiments/configs/default.yaml"}

python -m src.experiments.download_data --symbols ${SYMS} --start ${START}
python -m src.experiments.run_grid --config ${CONFIG}
cd paper && latexmk -pdf -interaction=nonstopmode main.tex
echo "Done. See outputs/ and paper/main.pdf"
