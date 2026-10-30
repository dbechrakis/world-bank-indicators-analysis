# Run the same checks as CI: `make check`. Override the interpreter with `make check PYTHON=python3.12`.
PYTHON ?= python

.PHONY: install check lint evidence test structure

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check wb_indicators scripts tests --select E4,E7,E9,F

evidence:
	$(PYTHON) ci/verify_evidence.py
	$(PYTHON) sql/build_database.py

test:
	$(PYTHON) -m unittest discover -s tests -v

structure:
	$(PYTHON) scripts/structure_analysis.py

check: lint evidence test
