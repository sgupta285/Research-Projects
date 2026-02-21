.PHONY: venv test data run paper clean

venv:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

test:
	pytest -q

data:
	python -m src.experiments.download_data --symbols SPY QQQ IWM DIA XLF XLK XLE XLV XLY XLP --start 2005-01-01

run:
	python -m src.experiments.run_grid --config src/experiments/configs/default.yaml

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	rm -rf outputs

sensitivity:
	python -m src.experiments.run_sensitivity --config src/experiments/configs/default.yaml
