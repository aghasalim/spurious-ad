.PHONY: setup sweep mechanism test clean
PY := .venv/bin/python
setup:
	python3.12 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt
sweep:      ## AUROC vs faithfulness across confound-label correlation
	$(PY) -m experiments.sweep
mechanism:  ## label-shortcut vs training-set-absence -- the decisive ablation
	$(PY) -m experiments.mechanism
test:
	$(PY) -m pytest tests/ -q
clean:
	rm -rf reports/*.json reports/*.csv
