PYTHON ?= python3
PYTHONPATH := src

.PHONY: data features train validate submit observe dashboard all test

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

observe:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mm2026.cli observe

dashboard:
	PYTHONPATH=$(PYTHONPATH) streamlit run src/mm2026/observability/app.py

all: data features train validate submit observe

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q
