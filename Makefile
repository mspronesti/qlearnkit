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
	@echo "  all                to clean, check the style guidelines and run the test suite"
	@echo "  flake              to check overall code style guidelines "
	@echo "  install            to install qlearnkit"
	@echo "  test               to run the test suite for all tested packages"
	@echo "  test-parallel      to run the test suite for all tested packages in parallel"
	@echo "  coverage           to generate a coverage report for all tested packages"
	@echo "  coverage-parallel  to generate a coverage report for all tested packages in parallel"
	@echo "  doc                to generate documentation for qlearnkit"
	@echo "  clean-doc          to delete all built documentation"
	@echo "  apidoc             to re-generate sphinx sources. Run ``make doc`` afterwards to build the documentation"
	@echo "  clean              to delete all temporary, cache, and build files"

.PHONY: all flake install test test-parallel coverage coverage-parallel doc clean-doc apidoc clean
all: clean flake test-parallel

flake:
	flake8 qlearnkit | grep -v __init__ | grep -v external

install:
	$(PYTHON) setup.py install

test:
	$(PYTHON) -m pytest

test-parallel:
	echo "Detected $(NPROCS) CPUs running with $(CONCURRENCY) workers"
	$(PYTHON) -m pytest -n $(CONCURRENCY)

coverage:
	rm .coverage
	$(PYTHON) -m pytest --cov=qlearnkit test/

coverage-parallel:
	rm .coverage
	$(PYTHON) -m pytest -n $(CONCURRENCY) --cov=qlearnkit test/

doc:
	sphinx-build -M html docs docs/_build

clean-doc:
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
