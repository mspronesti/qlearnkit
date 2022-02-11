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


PYTHON ?= python

.PHONY: test coverage doc clean_doc
test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=qlearnkit test/

doc:
	sphinx-build -M html docs docs/_build

clean_doc:
	$(MAKE) -C docs clean

