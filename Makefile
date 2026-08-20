SHELL := /bin/bash
CONFIG ?= configs/3spark.env
CONTEXTS ?= 2048,8192,32768
CONCURRENCIES ?= 1,3,6
MAX_TOKENS ?= 256

.PHONY: preflight nccl-bootstrap nccl fabric baseline candidate compare test

preflight:
	bash scripts/preflight.sh $(CONFIG)

nccl-bootstrap:
	bash scripts/bootstrap_nccl.sh $(CONFIG)

nccl:
	bash scripts/run_nccl.sh $(CONFIG)

fabric:
	bash scripts/capture_fabric.sh $(CONFIG)

baseline:
	CONTEXTS=$(CONTEXTS) CONCURRENCIES=$(CONCURRENCIES) MAX_TOKENS=$(MAX_TOKENS) bash scripts/run_experiment.sh $(CONFIG)

candidate:
	CONTEXTS=$(CONTEXTS) CONCURRENCIES=$(CONCURRENCIES) MAX_TOKENS=$(MAX_TOKENS) bash scripts/run_experiment.sh $(CONFIG)

compare:
	python3 scripts/compare_results.py --results-dir results

test:
	python3 -m unittest discover -s tests -v
