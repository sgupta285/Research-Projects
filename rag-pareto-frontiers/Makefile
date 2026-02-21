.PHONY: venv install hotpot_download eval sweep pareto test paper
venv:
	python3 -m venv .venv
install:
	. .venv/bin/activate && pip install -r requirements.txt
hotpot_download:
	. .venv/bin/activate && python -m src.cli.download_dataset --dataset hotpotqa --split dev_distractor --out data/hotpotqa
eval:
	. .venv/bin/activate && python -m src.cli.run_eval --config configs/baselines/hotpotqa_bm25.yaml
sweep:
	. .venv/bin/activate && python -m src.cli.run_sweep --sweep configs/sweeps/hotpotqa_sweep_small.yaml
pareto:
	. .venv/bin/activate && python -m src.cli.make_pareto --results outputs/tables/sweep_results.csv
test:
	. .venv/bin/activate && pytest -q
paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
