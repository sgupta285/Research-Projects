.PHONY: venv test data run paper clean verify_figures

PYTHON ?= python3

venv:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

test:
	PYTHONPATH=. pytest -q

data:
	$(PYTHON) -m src.experiments.download_data --symbols SPY QQQ IWM DIA XLF XLK XLE XLV XLY XLP --start 2005-01-01

run:
	$(PYTHON) -m src.experiments.run_grid --config src/experiments/configs/default.yaml

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	rm -rf outputs

sensitivity:
	$(PYTHON) -m src.experiments.run_sensitivity --config src/experiments/configs/default.yaml

verify_figures:
	$(PYTHON) -m src.experiments.verify_figures --out_dir outputs
