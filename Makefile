# Developer shortcuts. Run `make check` before sending a change.
# Set PYTHON to pick an interpreter, e.g. `make PYTHON=.venv/bin/python check`.
PYTHON ?= python

.PHONY: install data test lint format typecheck check

install:  ## install closed-loop v0.1 first, then this package with its dev tools
	$(PYTHON) -m pip install "closed-loop @ git+https://github.com/darrshangovender/closed-loop@v0.1.0"
	$(PYTHON) -m pip install -e ".[dev]"

data:  ## download NASA's C-MAPSS files into data/cmapss and check them against data/MANIFEST.json
	$(PYTHON) scripts/fetch_cmapss.py

test:  ## tests marked "data" are skipped when the C-MAPSS files are not there
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m mypy

check: lint typecheck test
