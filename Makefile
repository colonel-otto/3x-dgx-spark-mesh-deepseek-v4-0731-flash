SHELL := /bin/bash
CONFIG ?= configs/3spark.env
CONTEXTS ?= 2048,8192,32768
CONCURRENCIES ?= 1,3,6
MAX_TOKENS ?= 256

.PHONY: preflight nccl-bootstrap nccl fabric baseline candidate compare test \n	check-sensitive install-hooks summary

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

# --- repo hygiene -----------------------------------------------------------

# Block serials, real emails/names, home directories and credentials from being
# committed. Run before publishing or opening a PR.
check-sensitive:
	python3 scripts/check_no_sensitive.py

# Install the pre-commit hook so the check runs automatically on every commit.
install-hooks:
	python3 scripts/check_no_sensitive.py --install-hook

# benchmarks/summary.csv is generated - never hand-edit it.
summary:
	python3 scripts/generate_summary.py
