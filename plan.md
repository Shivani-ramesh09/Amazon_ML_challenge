# Phase plan: CPU-first entity resolution

Read `architecture.md` before each phase. Complete implementation, tests, realistic benchmark, profile, metrics, TODO update, diff review, and one commit before advancing. No checkbox or phase is complete on code alone. All runtime/RSS figures below are **planning allowances**, not observations. For a benchmark, report sample size, bytes, throughput, projected full-scale time with scaling assumptions, RSS, candidate recall where applicable, and resulting budget status. If a projected stage is materially over budget, stop downstream work and redesign it.

The 3-hour training target is a sum of cached stage work, roughly ingest/profile 10m, normalization 25m, cheap retrieval 25m, sparse retrieval 40m, union/evaluation 15m, feature build 35m, sampling/model 20m, decisions 5m, I/O/contingency 5m. Test target ~2h: ingestion/normalization 25m, retrieval 45m, features/scoring 35m, output/validation 15m. These are provisional allocations to challenge with measured timings. Peak practical RSS target 45–50 GB on AWS, 12 GB on dev. CPU threads <=8 AWS, <=10 dev, avoiding nested oversubscription. Dependencies are pinned in Phase 1 and adjusted only with a recorded reason.

## Phase 0 — Repository audit

- **Objective:** inventory current checkout, Git history and competition contract; decide reuse/removal.
- **Inputs / dependencies:** repository, `README.md`, nested dataset, validator, historical commits; none.
- **Outputs / files:** tracked `reports/repository_audit.md`; no production code. The planning docs are a prerequisite and have their own `docs:` commit.
- **Implementation / complexity / parallelism:** inspect tracked/ignored files and historical modules; O(repository bytes), one process. Do not execute historical pipeline.
- **Memory / runtime:** <1 GB, <10 min excluding human review.
- **Metrics / tests:** file inventory, old-cost evidence, current state, data paths; manual consistency check against statement.
- **Acceptance / rollback:** clear keep/replace decisions and clean diff; correct audit if a missed asset changes design.
- **Cache:** audit snapshot and commit SHA. **Commit:** `phase-00: audit repository and legacy costs`.

## Phase 1 — Data contract and profiling

- **Objective:** establish actual row counts, schema, missingness, cardinality, country and text-length/block distributions, deterministic S1 split.
- **Inputs / dependencies:** raw TSVs and Phase 0 audit. Truth is for profiling and split only.
- **Outputs / files:** `src/data/{contract,profile,split}.py`, `src/config`, `configs/{dev,aws_cpu}.yaml`, pinned `requirements.txt`, tests, `artifacts/profile/{train,test}.json`, split manifest.
- **Implementation / complexity / parallelism:** stream/scan TSV with explicit tab and string schema; validate unique IDs and truth references; profile source lengths/countries, truth 0/1/many, stable S1 hash. O(total input bytes + truth IDs), chunked scan, configurable 1–8 read threads; avoid full Python row dictionaries.
- **Memory / runtime:** target <8 GB AWS, <4 GB dev, ~10 min profile scan; report actual.
- **Metrics / tests:** counts vs stated sizes, null/empty/duplicate/invalid ID rates, length quantiles, country values, GT cardinality, scan MB/s; parsing/missing/duplicate/foreign-ID/split repeatability tests; small-file integration scan.
- **Acceptance / rollback:** full profile completes, counts reconciled, no silent malformed rows, deterministic split, projected scan budget credible; revise parser before proceeding otherwise.
- **Cache:** profile and split manifests keyed by raw-file fingerprints. **Commit:** `phase-01: add scalable data ingestion and profiling`.

## Phase 2 — CPU-efficient normalization

- **Objective:** deterministic multi-view name/address fields and open-set country parsing.
- **Inputs / dependencies:** Phase 1 contract and raw TSVs.
- **Outputs / files:** `src/normalization/{text,address}.py`, normalized train/test Parquet partitions, tests.
- **Implementation / complexity / parallelism:** batch NFKC/casefold/whitespace/punctuation, safe legal suffixes, country-dispatched India/US/generic postal and number parsing; O(total characters), partitioned batches, 1–8 CPU workers only if memory-safe.
- **Memory / runtime:** <16 GB AWS/<6 GB dev, ~25m train/~20m test target.
- **Metrics / tests:** throughput, blank/collision rates, output bytes/RSS; Unicode, suffix, PIN/ZIP/France generic, empty/idempotence tests; sampled raw-to-Parquet integration.
- **Acceptance / rollback:** raw retained, no empty-name collapse, same code path dev/AWS, projected budget credible; revert aggressive normalization that damages sampled truth identity.
- **Cache:** normalized partitions keyed by source + normalizer version/config. **Commit:** `phase-02: implement deterministic text normalization`.

## Phase 3 — Exact and frequency-aware blocking

- **Objective:** exact clean/core candidates without block explosion.
- **Inputs / dependencies:** Phase 2 normalized S1/S2/S3.
- **Outputs / files:** `src/blocking/{exact,index}.py`, compact index, per-channel candidate partitions, block report/tests.
- **Implementation / complexity / parallelism:** sorted key/posting arrays, per-key cardinality; oversize blocks refined by country, postal, number, extra token and bounded deterministic trim; O(target rows log rows + emitted bounded pairs), shard by key or S1 range.
- **Memory / runtime:** <25 GB AWS/<8 GB dev, ~15m train/~12m test target.
- **Metrics / tests:** block quantiles, oversize/trim stats, pairs/channel, GT recovered, throughput; exact/oversize/empty/country-fallback tests and sampled integration.
- **Acceptance / rollback:** no uncontrolled block, every trim counted, recall estimate and runtime documented; increase refinement quality if loss is material.
- **Cache:** exact indexes and channel candidates. **Commit:** `phase-03: add frequency-aware exact blocking`.

## Phase 4 — Rare-token and numeric/postal blocking

- **Objective:** recover matches beyond exact names with cheap high-information keys.
- **Inputs / dependencies:** Phase 2 fields, Phase 3 candidate schema/index design.
- **Outputs / files:** `src/blocking/{rare_token,numeric}.py`, token DF/postings, channel candidates/tests.
- **Implementation / complexity / parallelism:** bounded rare-token postings and name+postal/number combinations; no global number-only join; O(total tokens + scanned bounded postings + emitted pairs), S1 shard workers sharing read-only indexes.
- **Memory / runtime:** <30 GB AWS/<10 GB dev, ~10m train/~8m test target.
- **Metrics / tests:** token frequency/block distributions, unique GT recovered, scans/row, pairs/row, time/RSS; common-token and numeric collision tests; sampled integration.
- **Acceptance / rollback:** bounded postings, material incremental recall or justified low cost; adjust token limits if projected budget fails.
- **Cache:** DF, postings and channels. **Commit:** `phase-04: add rare-token and numeric blocking`.

## Phase 5 — Sparse character TF-IDF retrieval

- **Objective:** recover lexical variants for unresolved/uncertain S1 within bounded sparse work.
- **Inputs / dependencies:** Phase 2 names, Phase 3–4 cheap candidates/gate evidence.
- **Outputs / files:** `src/retrieval/{interface,char_tfidf}.py`, vectorizer/CSR indexes, gated top-K candidates, benchmark/tests.
- **Implementation / complexity / parallelism:** 3–5 gram TF-IDF float32 CSR; sparse-dot-topn query batches against S2/S3, top-N only; inspect nnz and workspace; compare gate vs all-query audit. Cost O(nonzero sparse products), not dense O(S1*S23); bounded batches and 8-thread library limit.
- **Memory / runtime:** <40 GB AWS/<12 GB dev, ~40m train/~35m test target.
- **Metrics / tests:** index nnz/bytes, throughput, top-K recall, gate loss, temporary RSS, score/rank determinism; exact tiny sparse fixture and realistic sample.
- **Acceptance / rollback:** no full similarity materialization; gate loss quantified and projected time/RAM plausible; redesign n-gram/gate/batch if over budget.
- **Cache:** vocabulary/IDF, CSR, query outputs keyed by relevant config. **Commit:** `phase-05: implement sparse tfidf candidate retrieval`.

## Phase 6 — Candidate union, deduplication, pruning

- **Objective:** one bounded truth-free candidate set with full provenance.
- **Inputs / dependencies:** Phase 3–5 channel outputs.
- **Outputs / files:** `src/retrieval/union.py`, final candidate Parquet/test candidate manifest, tests.
- **Implementation / complexity / parallelism:** merge sorted integer pair keys; OR channels, retain best score/rank; cheap evidence rank, stable tie, cap per S1; O(C log C) external/sharded sort for C emitted channel pairs, O(K*S1) output; shard by S1.
- **Memory / runtime:** <30 GB AWS/<10 GB dev, ~10m train/~8m test target.
- **Metrics / tests:** before/after duplicates, caps, zero-candidate count, bytes/RSS; dedup/provenance/tie/cap tests and batch integration.
- **Acceptance / rollback:** every pair unique and cap-respecting, no label rescue, candidate file equals scored set; fix recall harming rank if cap loss is large.
- **Cache:** candidate Parquet by retrieval hash. **Commit:** `phase-06: build bounded candidate union`.

## Phase 7 — Blocking evaluation

- **Objective:** verify recall ceiling and cap tradeoff before model work.
- **Inputs / dependencies:** Phase 1 GT, Phase 3–6 channel/final candidates.
- **Outputs / files:** `src/evaluation/blocking.py`, reports for each channel/union/K=5,10,15,20,30.
- **Implementation / complexity / parallelism:** sorted pair merge with GT; entity counters including zero candidates; O(C+GT), streaming partitions.
- **Memory / runtime:** <12 GB AWS/<6 GB dev, ~5m target.
- **Metrics / tests:** required candidate count quantiles, pair/any/complete recall, GT hits and unique per channel, oversize loss, runtime/RSS/bytes; exact hand-set and sampled integration.
- **Acceptance / rollback:** mathematically checked report and acceptable recall/cost frontier; return to Phase 3–6 if ceiling is poor.
- **Cache:** blocking reports. **Commit:** `phase-07: add candidate recall evaluation`.

## Phase 8 — Batched pair features

- **Objective:** compute useful lexical/address/provenance features without materializing all pairs.
- **Inputs / dependencies:** Phase 2 normalized rows, Phase 6 final candidates, Phase 7 accepted cap.
- **Outputs / files:** `src/features/{interface,cpu}.py`, feature Parquet partitions/schema, tests.
- **Implementation / complexity / parallelism:** integer-ID joins in 250k–1M pair batches; selective RapidFuzz on bounded strings; O(C*mean compared length), 1–8 workers with shared/mmap data, no duplicated GB tables.
- **Memory / runtime:** <35 GB AWS/<10 GB dev, ~35m train/~30m test target.
- **Metrics / tests:** pairs/s, feature null/finite rates, feature timing, RSS, bytes; numeric/postal/country/missing/provenance correctness and sampled batch integration.
- **Acceptance / rollback:** stable feature schema, finite values, projected budget credible; remove redundant expensive feature only after timing/ablation.
- **Cache:** features by candidate+feature hash. **Commit:** `phase-08: implement batched pair features`.

## Phase 9 — Training pair construction

- **Objective:** all retrieved positives plus representative hard/easy negatives.
- **Inputs / dependencies:** Phase 1 split/GT, Phase 8 features.
- **Outputs / files:** `src/training/{pairs,run_pairs}.py`, labeled train/validation Parquet and manifests, tests.
- **Implementation / complexity / parallelism:** GT membership and deterministic per-S1 top hard negatives plus small random/easy sample; streaming one feature shard at a time, O(C_shard log C_shard) sorting and O(C) total I/O.
- **Memory / runtime:** <12 GB AWS/<6 GB dev, ~5m target.
- **Metrics / tests:** positive recovery, 1:5/10/20 samples, hardness breakdown, no validation-label leakage; hand fixture and sample integration.
- **Acceptance / rollback:** all retrieved positives retained, validation untouched, repeatable sample; revise if imbalance/coverage fails.
- **Cache:** sampled pair indexes by seed/policy. **Commit:** `phase-09: add hard-negative training construction`.

## Phase 10 — LightGBM baseline

- **Objective:** CPU pair scorer and unbiased held-out entity scores.
- **Inputs / dependencies:** Phase 8 features, Phase 9 sample, Phase 1 split.
- **Outputs / files:** `src/scoring/{interface,lightgbm}.py`, model, validation scores/diagnostics/tests.
- **Implementation / complexity / parallelism:** bounded feature matrix loading, 8-thread LightGBM, early stopping, fixed seed; O(trees*sampled pairs*feature bins) measured.
- **Memory / runtime:** <40 GB AWS/<12 GB dev, ~20m target.
- **Metrics / tests:** pair PR-AUC/P/R, importance, iteration, time/RSS; fixture and train/predict schema integration.
- **Acceptance / rollback:** validation S1 unseen in training, scores for every validation candidate, projected budget credible; reduce sampled matrix/tree size if needed.
- **Cache:** model and validation predictions. **Commit:** `phase-10: train lightgbm pair scorer`.

## Phase 11 — Entity-level decision

- **Objective:** deterministic zero/one/many policy and exact metric.
- **Inputs / dependencies:** Phase 10 validation scores, Phase 1 truth.
- **Outputs / files:** `src/entity_decision/policy.py`, `src/evaluation/entity.py`, tests.
- **Implementation / complexity / parallelism:** group bounded sorted scores, compute top stats and F0.5 including empty/empty; O(C log K), streaming S1 groups.
- **Memory / runtime:** <8 GB AWS/<4 GB dev, ~3m target.
- **Metrics / tests:** macro F0.5, micro P/R, singleton accuracy, cardinality confusion; hand singleton/multi-match/example tests and validation integration.
- **Acceptance / rollback:** exact metric reproduces example, all S1 counted, multiple matches possible; correct policy before tuning.
- **Cache:** grouped validation score table. **Commit:** `phase-11: implement entity decision layer`.

## Phase 12 — Threshold optimization

- **Objective:** select stable policy by exact held-out macro F0.5.
- **Inputs / dependencies:** Phase 11 cached groups/truth.
- **Outputs / files:** `src/entity_decision/optimize.py`, policy JSON, sensitivity report/tests.
- **Implementation / complexity / parallelism:** coarse-to-fine bounded grid over singleton/pair/strong/gap/max-count; O(grid*C) on cached scores, vectorize where safe.
- **Memory / runtime:** <8 GB AWS/<4 GB dev, ~5m target.
- **Metrics / tests:** macro F0.5, micro P/R, singleton FP, zero/one/many, neighboring-threshold stability; synthetic optimum and holdout integration.
- **Acceptance / rollback:** stable measured gain over defaults, no test tuning; simplify grid/policy if unstable or slow.
- **Cache:** frozen policy + validation metrics. **Commit:** `phase-12: optimize macro f0.5 thresholds`.

## Phase 13 — Error analysis

- **Objective:** quantify remaining retrieval, rank, decision and country/address failures.
- **Inputs / dependencies:** Phase 6–7 candidates, Phase 10 scores, Phase 12 policy, truth.
- **Outputs / files:** `src/evaluation/errors.py`, sampled examples and category report.
- **Implementation / complexity / parallelism:** streaming joins and deterministic stratified sample; O(C+GT), one process.
- **Memory / runtime:** <10 GB AWS/<5 GB dev, ~5m target.
- **Metrics / tests:** category counts, slices by country/cardinality/common name, representative permitted-field samples; hand taxonomy fixture.
- **Acceptance / rollback:** prioritized measurable error target for Phase 14; fix classification if categories overlap ambiguously.
- **Cache:** error report. **Commit:** `phase-13: add entity error analysis`.

## Phase 14 — Evidence-driven improvement

- **Objective:** implement at most one justified improvement per ablation cycle.
- **Inputs / dependencies:** Phase 13 quantified error, baseline artifacts.
- **Outputs / files:** targeted module edit, `reports/ablation.csv`, updated metrics/tests.
- **Implementation / complexity / parallelism:** compare baseline vs treatment with same split; rerun only invalidated stages; optional word/address retrieval, calibration, extra mining or model only after evidence. Complexity and concurrency are measured for each proposal.
- **Memory / runtime:** no acceptance if resulting full pipeline exceeds 3h/2h or 50 GB without compensating redesign; experiment allowance ~15m cached plus changed stage.
- **Metrics / tests:** unique GT, candidate recall, entity macro F0.5, micro P/R, runtime/RSS; targeted regression and same-split integration.
- **Acceptance / rollback:** keep only reproducible entity gain with justified resource cost; otherwise record rejected ablation and restore baseline.
- **Cache:** versioned treatment artifacts and ablation row. **Commit:** `phase-14: implement validated model improvements` (or documented no-change outcome commit).

## Phase 15 — Full training and frozen model

- **Objective:** lock chosen configuration and train final model on all labeled training entities.
- **Inputs / dependencies:** Phase 12 policy, Phase 14 decision, all train features/GT.
- **Outputs / files:** `src/training/finalize.py`, final model, frozen manifest/config, reproduction docs.
- **Implementation / complexity / parallelism:** same sampler/features, bounded LightGBM CPU fit; no validation-based claim for refitted model; O(sampled training pairs*trees).
- **Memory / runtime:** <40 GB AWS, ~20m fitting within 3h pipeline.
- **Metrics / tests:** feature schema, hash, training count/time/RSS, reproducible load/predict; frozen artifact smoke test.
- **Acceptance / rollback:** model and policy reload exactly, train total measured, no later test tuning; revert to validated model if full refit fails.
- **Cache:** final model/config/manifests. **Commit:** `phase-15: freeze final training configuration`.

## Phase 16 — Test inference

- **Objective:** complete batched CPU inference for every test S1 in ~2h target.
- **Inputs / dependencies:** Phase 15 frozen artifacts, all test TSVs.
- **Outputs / files:** `src/inference/run.py`, test normalized/index/candidates/features/scores and runtime report.
- **Implementation / complexity / parallelism:** same stages, deterministic batching, no fit to test labels; O(test rows + bounded candidates), 8-vCPU stage limit.
- **Memory / runtime:** <50 GB AWS, ~2h entire inference target.
- **Metrics / tests:** all test S1 covered, scored candidate count, stage times/RSS, frozen schema; sample end-to-end smoke then full run.
- **Acceptance / rollback:** full inference completes within credible budget and no missing batches; profile/rework dominant stage if not.
- **Cache:** all deterministic test artifacts and scores. **Commit:** `phase-16: implement scalable test inference`.

## Phase 17 — Submission and validation

- **Objective:** produce both exact TSVs and pass internal plus official validation.
- **Inputs / dependencies:** Phase 16 scored candidates/policy, test IDs.
- **Outputs / files:** `src/output/write.py`, `chimera_submission/output/{matching_results,candidate_pairs}.tsv`, package README/methodology and validation report.
- **Implementation / complexity / parallelism:** streaming S1 order, unique target IDs, blank singleton, matching subset of final scored candidates; O(test S1 + scored pairs), single writer.
- **Memory / runtime:** <12 GB AWS, ~15m target including validation.
- **Metrics / tests:** exact row counts, schema, duplicate/invalid IDs, matched subset, bytes, official validator `--check-ids`; negative fixtures and full test validation.
- **Acceptance / rollback:** validator PASS, internal strict subset/ID check PASS, README exact commands, docs/TODO reconciled and clean Git; regenerate outputs if any failure.
- **Cache:** final TSVs and validation manifest. **Commit:** `phase-17: generate and validate submission`.
