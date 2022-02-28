OS := $(shell uname -s)

ifeq ($(OS), Linux)
  NPROCS := $(shell grep -c ^processor /proc/cpuinfo)
else ifeq ($(OS), Darwin)
  NPROCS := 2
else
  NPROCS := 0
endif # $(OS)

ifeq ($(NPROCS), 2)
        CONCURRENCY := 2
else ifeq ($(NPROCS), 1)
        CONCURRENCY := 1
else ifeq ($(NPROCS), 3)
        CONCURRENCY := 3
else ifeq ($(NPROCS), 0)
        CONCURRENCY := 0
else
        CONCURRENCY := $(shell echo "$(NPROCS) 2" | awk '{printf "%.0f", $$1 / $$2}')
endif

# define python (if in venv, use python)
ifeq (, $(shell which python ))
	PYTHON := python3
else
	PYTHON := python
endif

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install qlearnkit"
	@echo "  test               to run the test suite for all tested packages"
	@echo "  coverage           to generate a coverage report for all tested packages"
	@echo "  doc                to generate documentation for qlearnkit"
	@echo "  clean_doc          to delete all built documentation"
	@echo "  apidoc             to re-generate sphinx sources. Run ``make doc`` afterwards to build the documentation"
	@echo "  clean              to delete all temporary, cache, and build files"

.PHONY: install test coverage doc clean_doc apidoc clean
install:
	$(PYTHON) setup.py install

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=qlearnkit test/

doc:
	sphinx-build -M html docs docs/_build

clean_doc:
	$(MAKE) -C docs clean

apidoc:
	sphinx-apidoc --force -o  docs/apidoc . setup.py
	rm docs/apidoc/modules.rst

clean:
	rm -rf __pycache__
	rm -rf qlearnkit/__pycache__
	rm -rf test/__pycache__
	rm -rf dist
	rm -rf build
	rm -rf .pytest_cache
	rm -rf .coverage coverage_html_report/
