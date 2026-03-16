PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHONPATH := src

.PHONY: data features train validate submit explain observe dashboard dashboard-bracket publish-bracket all test

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

explain:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mm2026.cli explain

observe:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mm2026.cli observe

dashboard:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m streamlit run src/mm2026/observability/app.py

dashboard-bracket:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m streamlit run streamlit_app.py

publish-bracket:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mm2026.observability.publishable_bracket --output deploy/bracket_center_payload.json

all: data features train validate submit observe

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q
