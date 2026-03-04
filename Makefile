PYTHON ?= python3
PYTHONPATH := src

.PHONY: data features train validate submit all test

data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mm2026.cli data

features:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mm2026.cli features

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mm2026.cli train

validate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mm2026.cli validate

submit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mm2026.cli submit

all: data features train validate submit

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q
