# Execution checklist

Unchecked means not yet verified. A phase is complete only after tests, benchmark/profile, required report, diff review, performance/RAM gate, and its own commit. Design documents have a separate `docs:` commit. Exact phase criteria, budgets, inputs and outputs are in `plan.md`.

## Design gate
- [x] Audit current repository and relevant history; record old costs and competition constraints
- [x] Finalize `architecture.md`, `plan.md`, `TODO.md`
- [x] Cross-check architecture components against plan and every plan action against this checklist
- [x] Check dependencies, open-set country, singleton/multi-match, candidate recall, exact macro F0.5, runtime/RAM, no GPU/SQL backbone, submission validation
- [x] Review documentation diff and commit `docs: redesign cpu-first entity resolution architecture`

## Phase 0 — Repository audit
- [x] Record current and historical file inventory, data paths and validator behavior
- [x] Identify reusable interfaces and costly/deprecated SQL, Python-object, KNN, CV and GPU choices
- [x] Write audit report with measured vs historical claims clearly distinguished
- [x] Verify audit against actual repository; review diff; commit `phase-00`

## Phase 1 — Data contract and profiling
- [x] Pin CPU dependencies and create dev/AWS configs with resource knobs and data paths
- [x] Implement streaming explicit-tab source and truth contract with string-preserving blanks
- [x] Validate source prefixes/unique IDs and truth S1/S2/S3 references
- [x] Compute row counts, null/empty rates, country, text length, common-name and GT-cardinality distributions
- [x] Build deterministic entity-level split; record seed and raw input fingerprints
- [x] Add TSV/missing/duplicate/reference/split unit tests and small-file integration test
- [x] Profile full or representative scan throughput, RAM, artifact bytes and projected AWS runtime
- [x] Review performance/memory gates and diff; write report; commit `phase-01`

## Phase 2 — CPU-efficient normalization
- [x] Implement raw/clean/core name views with NFKC, casefold, punctuation, whitespace and safe suffix rules
- [x] Implement raw/clean address, numbers and postal candidate views
- [x] Implement India, US and France/generic fallback parsers; preserve arbitrary country values
- [x] Write bounded normalized Parquet partitions with versioned manifests
- [x] Test Unicode, suffix, missing, idempotence, postal, numeric and open-set behavior
- [x] Benchmark realistic train/test sample; report collision rates, throughput/RSS/bytes and full projection
- [x] Resolve budget/recall issues; review diff; commit `phase-02`

## Phase 3 — Exact and frequency-aware blocking
- [x] Build compact clean/core inverted indexes and record unique keys/block-size distribution
- [x] Implement deterministic bounded oversize refinement and trim statistics
- [x] Emit exact/core candidates with provenance and block size
- [x] Test exact, empty, oversized, country fallback and deterministic ties; integrate on dev sample
- [x] Report pairs, unique GT, runtime/RSS/artifact size and full-scale estimate
- [x] Fix uncontrolled blocks or material GT loss; review diff; commit `phase-03`

## Phase 4 — Rare-token and numeric/postal blocking
- [x] Compute target token DF and bounded rare-token postings
- [x] Add name-evidence plus postal/number channels; prohibit number-only global joins
- [x] Emit channel provenance and posting/trim statistics
- [x] Test frequent tokens, number collisions, missing/country cases; integrate on dev sample
- [x] Report incremental unique GT, candidate volume, throughput/RSS/bytes and projection
- [x] Resolve budget/recall gate; review diff; commit `phase-04`

## Phase 5 — Sparse character TF-IDF retrieval
- [x] Define CPU retriever interface and frozen train-fitted 3–5 gram vectorizer
- [x] Build float32 CSR target/query matrices; estimate nnz and peak multiplication workspace
- [x] Implement sparse top-N batched multiplication with bounded memory and rank/score provenance
- [x] Implement cheap-stage uncertainty gate; audit gated vs all-query retrieval recall
- [x] Test tiny exact top-K fixture, score/rank stability, empty names and dev integration
- [x] Profile throughput/RSS/artifact bytes and project full train/test runtime
- [x] Redesign excessive stage or gate recall loss; review diff; commit `phase-05`

## Phase 6 — Candidate union, deduplication and cap
- [x] Merge channel outputs by canonical S1/target pair key and OR provenance
- [x] Implement frequency-aware evidence ranking and deterministic tie order
- [x] Evaluate K=5,10,15,20,30 and save exact final scored candidate Parquet
- [x] Test dedup, flags, cap, empty S1, stable ordering; integrate on dev sample
- [x] Report before/after volume, RSS/bytes/time and projected cost
- [x] Resolve major numeric/postal cap recall loss; review diff; commit `phase-06`

## Phase 7 — Blocking evaluation
- [x] Compute pair recall, GT hits, entity any-hit and complete recall per channel/union/cap
- [x] Compute candidate count mean/median/P90/P95/P99/max including zero-candidate S1
- [x] Compute unique GT recovered/channel and oversized-block statistics; slice country/cardinality
- [x] Test metrics on hand sets including singleton/multi-match; integrate with dev candidates
- [x] Publish blocking report with runtime/RSS/artifact bytes and oracle ceiling
- [x] Gate downstream work on recall and candidate budget; review diff; commit `phase-07`

## Phase 8 — Batched pair features
- [x] Define CPU feature-builder interface and stable feature schema
- [x] Add name, address, postal/numeric, country, missingness and retrieval-provenance features
- [x] Avoid redundant expensive features until ablated; batch 250k–1M pairs
- [x] Write bounded Parquet partitions; test exact feature values and finite/missing behavior
- [x] Benchmark pair throughput, per-feature cost, RSS/bytes and train/test projection
- [x] Resolve budget gate; review diff; commit `phase-08`

## Phase 9 — Training pairs
- [x] Label only final retrieved candidates; retain all retrieved positives
- [x] Sample deterministic top hard and small easy negatives at tested ratios
- [x] Keep validation candidate distribution whole and labels out of retrieval/features
- [x] Test positive retention, ratios, repeatability and split isolation; dev integration
- [x] Report hardness mix, rows, time/RSS/bytes and projection
- [x] Resolve class/sample issues; review diff; commit `phase-09`

## Phase 10 — LightGBM baseline
- [x] Implement CPU scorer interface, fixed seed, feature order, bounded training and early stopping
- [x] Train on train entities and score every validation candidate
- [x] Report pair PR-AUC/P/R, importance, iteration, time/RSS and model bytes
- [x] Test model save/load, score schema and no validation S1 in training; dev integration
- [x] Resolve training budget; review diff; commit `phase-10`

## Phase 11 — Entity-level decision
- [ ] Aggregate top scores/gaps/counts/channel agreement per S1 including zero candidates
- [ ] Implement deterministic singleton/one/many policy with configurable thresholds
- [ ] Implement official macro entity F0.5 and micro diagnostics
- [ ] Test empty/empty=1, empty/nonempty=0, multiple matches and published 0.714 example
- [ ] Report cardinality confusion, singleton accuracy/FP, time/RSS/bytes
- [ ] Review diff; commit `phase-11`

## Phase 12 — Threshold optimization
- [ ] Search singleton/strong/pair/gap/max-count parameters coarse-to-fine on cached validation scores
- [ ] Evaluate neighborhood stability and zero/one/many breakdown
- [ ] Report exact macro F0.5, micro P/R, singleton FP and threshold sensitivity
- [ ] Test synthetic optimum and no test-label/test-score tuning; integration on cached scores
- [ ] Profile search; lock policy; review diff; commit `phase-12`

## Phase 13 — Error analysis
- [ ] Label retrieval misses, wrong ranking, singleton FP, missed matches, wrong top and extra multi-match misses
- [ ] Identify normalization/address/common-name/country/postal-numeric subtypes
- [ ] Produce deterministic representative samples and counts by country/cardinality
- [ ] Test taxonomy on hand examples; profile time/RSS; identify highest-value target
- [ ] Review diff; commit `phase-13`

## Phase 14 — Evidence-driven improvement
- [ ] Specify one quantified error target and expected recall/F0.5/cost effect
- [ ] Implement minimal targeted change only after baseline completion
- [ ] Rerun invalidated stages and exact entity-level threshold optimization
- [ ] Append baseline/treatment row to `reports/ablation.csv` with recall, F0.5, time and RSS
- [ ] Keep only justified stable gain; test and profile; review diff; commit `phase-14`

## Phase 15 — Full training and frozen model
- [ ] Lock configs/schema/seed/manifests and train chosen scorer on all labeled train S1
- [ ] Preserve pre-refit validation report and note final refit is not independently scored
- [ ] Test frozen model reload and deterministic sample prediction
- [ ] Report full pipeline wall time, stage time, RSS and model size; meet/resolve budget
- [ ] Update reproduction commands; review diff; commit `phase-15`

## Phase 16 — Test inference
- [ ] Apply frozen train-fitted vectorizer/model/policy to all test S1, without test fitting or threshold changes
- [ ] Cache batched test normalization/retrieval/final candidates/features/scores
- [ ] Verify exact S1 coverage, score every final candidate, no missing batches
- [ ] Test dev smoke then run full test; profile actual stage time/RSS/artifact bytes
- [ ] Meet/resolve ~2h and <50GB targets; review diff; commit `phase-16`

## Phase 17 — Submission and validation
- [ ] Write exactly one row/test S1 to both required UTF-8 TSVs, blank singleton lists
- [ ] Enforce unique valid S2/S3 IDs, no duplicate S1, matched IDs subset of *scored* candidates
- [ ] Test malformed/duplicate/invalid/subset cases and sample full writer integration
- [ ] Run official validator with nested test dir and `--check-ids`; treat subset warning as internal failure
- [ ] Fill methodology, pin exact reproduction commands, report results/runtime/RSS/bottlenecks
- [ ] Reconcile architecture/plan/TODO against implementation; review diff; commit `phase-17`
- [ ] Confirm clean Git tree and final report with candidate recall, macro F0.5, P/R, singleton and 0/1/many results
