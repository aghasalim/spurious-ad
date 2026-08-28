.PHONY: setup sweep mechanism real real-sweep real-mechanism real-backbone figures test clean
PY := .venv/bin/python
# Real-data runs need MVTec AD on disk; set MVTEC_ROOT or pass --root.
# Tests never touch it, so CI stays download-free.
CATS := bottle carpet grid hazelnut tile
DETS := patchcore padim
setup:
	python3.12 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt
sweep:      ## AUROC vs faithfulness across confound-label correlation
	$(PY) -m experiments.sweep
mechanism:  ## label-shortcut vs training-set-absence -- the decisive ablation
	$(PY) -m experiments.mechanism
real: real-sweep real-mechanism real-backbone
real-sweep:      ## the sweep on real MVTec images, both detector families
	$(PY) -m experiments.real --mode sweep --categories $(CATS) --detectors $(DETS)
real-mechanism:  ## the decisive ablation, on real images
	$(PY) -m experiments.real --mode mechanism --categories $(CATS) --detectors $(DETS)
real-backbone:   ## same sweep, different feature extractor
	$(PY) -m experiments.real --mode backbone --arch resnet18 --categories $(CATS) --detectors $(DETS)
figures:    ## redraw reports/figures from the committed CSVs, no training
	$(PY) scripts/make_figures.py
test:
	$(PY) -m pytest tests/ -q
clean:
	rm -rf reports/*.json reports/*.csv
