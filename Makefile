SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

PYTHON ?= .venv/bin/python
UV ?= uv
PYTHON_VERSION ?= 3.12
PROFILE ?= dev
DEV_SAMPLE_MODULUS ?= 16
CONFIG := chimera_submission/code/business_entity_resolution/configs/$(PROFILE).yaml
MODULE := chimera_submission.code.business_entity_resolution.src
ARTIFACTS ?= artifacts
CANDIDATE_CAPS ?= 5 10 15 20 30
NEGATIVE_RATIO ?= 5
FROZEN ?=
SCORES ?=

ifeq ($(PROFILE),aws_cpu)
SAMPLE_MODULUS ?= 1
OUTPUT_DIR ?= chimera_submission/output
else
SAMPLE_MODULUS ?= 16
OUTPUT_DIR ?= $(ARTIFACTS)/output/sample_$(SAMPLE_MODULUS)
endif

SAMPLE := $(if $(filter 1,$(SAMPLE_MODULUS)),full,sample_$(SAMPLE_MODULUS))
SELECT := $(PYTHON) -m $(MODULE).output.select_manifest --config $(CONFIG) --sample-modulus $(SAMPLE_MODULUS)

.PHONY: help install test profile profile-train profile-test check-env \
	normalize-train exact-train rare-train address-train tfidf-train \
	train-retrieval cap-sweep blocking-report features-train training-pairs \
	pair-model entity-validation thresholds errors finalize train \
	infer submit validate dev aws-blocking aws-train aws-infer aws-submit

help:
	@printf '%s\n' \
	  'CPU entity resolution targets (run from repository root):' \
	  '  make install                         Install pinned CPU dependencies into .venv' \
	  '  make test                            Run unit tests' \
	  '  make dev                             Train, infer, write, and validate a 1/16 dev sample' \
	  '  make aws-blocking                    Build full retrieval and blocking report; inspect it' \
	  '  make aws-train                       Continue full training after blocking review' \
	  '  make aws-infer                       Infer with the full frozen model' \
	  '  make aws-submit                      Write and officially validate full submission' \
	  '  make PROFILE=dev blocking-report     Run an individual cached phase' \
	  '  make PROFILE=dev finalize            Run through frozen refit only' \
	  '  make PROFILE=dev validate            Infer/write/validate using existing refit' \
	  'Variables: PYTHON=.venv/bin/python PYTHON_VERSION=3.12 PROFILE=dev|aws_cpu' \
	  '           SAMPLE_MODULUS=16|1' \
	  '           DEV_SAMPLE_MODULUS=16 (for make dev)' \
	  '           NEGATIVE_RATIO=5 FROZEN=<manifest> SCORES=<manifest>' \
	  'No target deletes cached artifacts. AWS steps are intentionally separate.'

install:
	$(UV) python install $(PYTHON_VERSION)
	$(UV) venv --python $(PYTHON_VERSION) .venv
	$(UV) pip install --python .venv/bin/python -r chimera_submission/code/business_entity_resolution/requirements.txt

check-env:
	@test -x "$(PYTHON)" || { printf '%s\n' 'Missing $(PYTHON); run make install or set PYTHON.' >&2; exit 1; }
	@test -f "$(CONFIG)" || { printf '%s\n' 'Unknown PROFILE=$(PROFILE): $(CONFIG) not found.' >&2; exit 1; }
	@test "$(SAMPLE_MODULUS)" -ge 1 || { printf '%s\n' 'SAMPLE_MODULUS must be positive.' >&2; exit 1; }
	@if [ "$(PROFILE)" = aws_cpu ] && [ "$(SAMPLE_MODULUS)" != 1 ]; then \
	  printf '%s\n' 'aws_cpu requires SAMPLE_MODULUS=1; use PROFILE=dev for samples.' >&2; exit 1; \
	fi
	@config_cap="$$( $(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate_cap"])' "$(CONFIG)" )"; \
	if [ "$(lastword $(CANDIDATE_CAPS))" != "$$config_cap" ]; then \
	  printf '%s\n' "The last CANDIDATE_CAPS value must equal configured candidate_cap=$$config_cap." >&2; exit 1; \
	fi
	@config_artifacts="$$( $(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact_dir"])' "$(CONFIG)" )"; \
	if [ "$(ARTIFACTS)" != "$$config_artifacts" ]; then \
	  printf '%s\n' "ARTIFACTS=$(ARTIFACTS) differs from configured artifact_dir=$$config_artifacts." >&2; exit 1; \
	fi

test: check-env
	$(PYTHON) -m unittest discover -s chimera_submission/code/business_entity_resolution/tests -q

profile: profile-train profile-test

profile-train: check-env
	$(PYTHON) -m $(MODULE).data.profile --config $(CONFIG) --split train

profile-test: check-env
	$(PYTHON) -m $(MODULE).data.profile --config $(CONFIG) --split test

normalize-train: check-env
	@for source in 1 2 3; do \
	  $(PYTHON) -m $(MODULE).normalization.run --config $(CONFIG) --split train \
	    --source $$source --sample-modulus $(SAMPLE_MODULUS); \
	done

exact-train: normalize-train
	$(PYTHON) -m $(MODULE).blocking.run_exact --config $(CONFIG) --split train \
	  --sample-modulus $(SAMPLE_MODULUS)

rare-train: normalize-train
	$(PYTHON) -m $(MODULE).blocking.run_rare_numeric --config $(CONFIG) --split train \
	  --sample-modulus $(SAMPLE_MODULUS)

address-train: normalize-train
	$(PYTHON) -m $(MODULE).retrieval.run_exact_address --config $(CONFIG) --split train \
	  --sample-modulus $(SAMPLE_MODULUS)

tfidf-train: normalize-train
	$(PYTHON) -m $(MODULE).retrieval.run_tfidf --config $(CONFIG) --split train \
	  --sample-modulus $(SAMPLE_MODULUS)

train-retrieval: exact-train rare-train address-train tfidf-train

cap-sweep: train-retrieval
	@for cap in $(CANDIDATE_CAPS); do \
	  $(PYTHON) -m $(MODULE).retrieval.run_union --config $(CONFIG) --split train \
	    --sample-modulus $(SAMPLE_MODULUS) --cap $$cap; \
	done

blocking-report: cap-sweep
	$(PYTHON) -m $(MODULE).evaluation.blocking --sample-modulus $(SAMPLE_MODULUS) \
	  --artifact-dir $(ARTIFACTS) --caps $(CANDIDATE_CAPS)

features-train: blocking-report
	$(PYTHON) -m $(MODULE).features.run --config $(CONFIG) --split train \
	  --sample-modulus $(SAMPLE_MODULUS)

training-pairs: features-train
	$(PYTHON) -m $(MODULE).training.run_pairs --config $(CONFIG) \
	  --sample-modulus $(SAMPLE_MODULUS) --negative-ratio $(NEGATIVE_RATIO)

pair-model: training-pairs
	$(PYTHON) -m $(MODULE).scoring.run_lightgbm --config $(CONFIG) \
	  --sample-modulus $(SAMPLE_MODULUS) --negative-ratio $(NEGATIVE_RATIO)

entity-validation: pair-model
	$(PYTHON) -m $(MODULE).entity_decision.run --config $(CONFIG) \
	  --sample-modulus $(SAMPLE_MODULUS)

thresholds: entity-validation
	$(PYTHON) -m $(MODULE).entity_decision.run_optimize --config $(CONFIG) \
	  --sample-modulus $(SAMPLE_MODULUS)

errors: thresholds
	$(PYTHON) -m $(MODULE).evaluation.run_errors --config $(CONFIG) \
	  --sample-modulus $(SAMPLE_MODULUS)

finalize: errors
	$(PYTHON) -m $(MODULE).training.run_finalize --config $(CONFIG) \
	  --sample-modulus $(SAMPLE_MODULUS) --negative-ratio $(NEGATIVE_RATIO)

train: finalize

infer: check-env
	@frozen="$(FROZEN)"; \
	if [ -z "$$frozen" ]; then frozen="$$( $(SELECT) --kind frozen )"; fi; \
	$(PYTHON) -m $(MODULE).inference.run --config $(CONFIG) \
	  --sample-modulus $(SAMPLE_MODULUS) --frozen-manifest "$$frozen"

submit: infer
	@frozen="$(FROZEN)"; \
	if [ -z "$$frozen" ]; then frozen="$$( $(SELECT) --kind frozen )"; fi; \
	scores="$(SCORES)"; \
	if [ -z "$$scores" ]; then scores="$$( $(SELECT) --kind scores --frozen-manifest "$$frozen" )"; fi; \
	$(PYTHON) -m $(MODULE).output.run --frozen-manifest "$$frozen" \
	  --score-manifest "$$scores" --normalized-dir $(ARTIFACTS)/normalized/$(SAMPLE) \
	  --output-dir $(OUTPUT_DIR)

validate: submit
ifeq ($(PROFILE),aws_cpu)
	$(PYTHON) -m $(MODULE).output.validate \
	  --matching $(OUTPUT_DIR)/matching_results.tsv \
	  --candidate $(OUTPUT_DIR)/candidate_pairs.tsv \
	  --test-dir dataset/student_resource/dataset/test
else
	$(PYTHON) -m $(MODULE).output.validate \
	  --matching $(OUTPUT_DIR)/matching_results.tsv \
	  --candidate $(OUTPUT_DIR)/candidate_pairs.tsv \
	  --normalized-dir $(ARTIFACTS)/normalized/$(SAMPLE)
endif

dev:
	@$(MAKE) PROFILE=dev SAMPLE_MODULUS=$(DEV_SAMPLE_MODULUS) test
	@$(MAKE) PROFILE=dev SAMPLE_MODULUS=$(DEV_SAMPLE_MODULUS) finalize
	@$(MAKE) PROFILE=dev SAMPLE_MODULUS=$(DEV_SAMPLE_MODULUS) validate

aws-blocking:
	@$(MAKE) PROFILE=aws_cpu SAMPLE_MODULUS=1 blocking-report
	@printf '%s\n' 'Inspect full blocking recall, pair counts, runtime and RSS before make aws-train.'

aws-train:
	@$(MAKE) PROFILE=aws_cpu SAMPLE_MODULUS=1 finalize

aws-infer:
	@$(MAKE) PROFILE=aws_cpu SAMPLE_MODULUS=1 infer

aws-submit:
	@$(MAKE) PROFILE=aws_cpu SAMPLE_MODULUS=1 validate
