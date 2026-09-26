# Business Entity Resolution Pipeline

Team: `chimera`

The CPU pipeline is implemented through development-sample inference and official sample validation. Full AWS training and test inference have **not** run, so the repository does not yet contain a valid full-test submission or a verified production score/runtime. Run stages using the commands below; `src/main.py` is not the entry point.

From the repository root, the Makefile runs the same cache-aware commands:

```bash
make help
make install
make dev             # complete 1/16 development run, including official sample validation
make aws-blocking    # full candidate generation and blocking report; inspect its metrics
make aws-train       # continue after reviewing candidate recall, runtime, and RSS
make aws-infer       # full test inference with the frozen full-universe model
make aws-submit      # canonical TSVs and official full-test validator
```

`make test` runs the unit suite. Individual phases are available, for example `make PROFILE=dev SAMPLE_MODULUS=16 blocking-report` or `make PROFILE=aws_cpu finalize`. `make dev` always uses the development profile; change its sample size with `DEV_SAMPLE_MODULUS=32`. `aws-*` targets require the full sample modulus. The inference and submission targets select a cached frozen model and its corresponding score manifest by config and lineage; set `FROZEN=path/to/manifest.json` and optionally `SCORES=path/to/manifest.json` to pin a specific run. AWS steps are separate so the measured blocking and validation reports can be reviewed before spending compute on later stages. Artifacts are never deleted by the Makefile.

From the repository root, run:

```bash
python3 -m unittest discover -s chimera_submission/code/business_entity_resolution/tests -v
python3 -m chimera_submission.code.business_entity_resolution.src.data.profile --split train --config chimera_submission/code/business_entity_resolution/configs/dev.yaml
python3 -m chimera_submission.code.business_entity_resolution.src.data.profile --split test --config chimera_submission/code/business_entity_resolution/configs/dev.yaml
```

The default data path is `dataset/student_resource/dataset`. Reports are written under `artifacts/profile/`. The config files use JSON syntax, which is valid YAML 1.2, so the Phase 1 profiler can load them without a third-party parser. CPU dependencies for later phases are pinned in `requirements.txt` and may be installed into a local environment with:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r chimera_submission/code/business_entity_resolution/requirements.txt
```

Phase 2 normalizes any source file into cached Parquet using bounded Arrow batches. For a realistic laptop sample:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.normalization.run --split train --source 1 --sample-modulus 16
```

Run the same command with `--source 2` and `--source 3`, and with `--split test`, for all six source files. Omit `--sample-modulus` to use the selected config's default (dev: 1/128 rows, AWS: full). For sampled **train** targets, every GT partner of a sampled S1 is included alongside background targets, so entity-level validation has complete truth sets. The reduced background candidate universe can still make the development score optimistic; full-universe evaluation remains required.

Phase 3 exact blocking, after the three normalized files for a split exist:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.blocking.run_exact --split train --sample-modulus 16
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.blocking.eval_exact --normalized-dir artifacts/normalized/sample_16 --candidate artifacts/blocking/sample_16/train_exact.parquet
```

The second command reports recall within the entity-complete development target universe. Full-universe recall remains a production evaluation gate.

Phase 4 adds rare-token and name-plus-postal/house channels:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.blocking.run_rare_numeric --split train --sample-modulus 16
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.blocking.eval_cheap --normalized-dir artifacts/normalized/sample_16 --exact artifacts/blocking/sample_16/train_exact.parquet --additional artifacts/blocking/sample_16/train_rare_numeric.parquet
```

Phase 5 builds a frozen train-fitted sparse name vectorizer and top-N fuzzy candidates. Run train first, then test so test uses the saved train vocabulary and IDF:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.retrieval.run_tfidf --split train --sample-modulus 16 --threads 4
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.retrieval.run_tfidf --split test --sample-modulus 16 --threads 4
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.blocking.eval_cheap --normalized-dir artifacts/normalized/sample_16 --exact artifacts/blocking/sample_16/train_exact.parquet --additional artifacts/blocking/sample_16/train_rare_numeric.parquet --additional artifacts/blocking/sample_16/train_tfidf_df0.001_f60000_all.parquet
```

Phase 6 unions the channel outputs, ORs provenance, and caps each S1 candidate set. Run cap experiments without repeating normalization or retrieval:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.retrieval.run_union --split train --sample-modulus 16 --cap 20
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.retrieval.run_union --split test --sample-modulus 16 --cap 20
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.retrieval.eval_union --sample-modulus 16
```

The cap audit expects cached K=5,10,15,20,30 training runs. The final scored candidates are the `part-*.parquet` files in the `final_dir` printed by `run_union`. Sample recall now covers all true partners of the sampled train S1, though distractor density remains lower than production.

Phase 7 evaluates each channel, union, and cap on both pair and entity measures, and profiles missed ground-truth pairs:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.evaluation.blocking --sample-modulus 16
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.evaluation.miss_diagnostics --sample-modulus 16
```

The evaluation JSON is saved in `artifacts/reports/blocking/`. The default candidate cap is now 30 because the measured recall gain over 20 justified the extra candidates on the development sample. Full-universe recall and final entity F0.5 still require the production run.

Phase 8 builds numeric pair features from the exact final candidate set using bounded Arrow gathers and 250,000-pair batches in dev. The Phase 14 exact-address provenance signal brings the current schema to 37 features:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.features.run --split train --sample-modulus 16 --cap 30
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.features.run --split test --sample-modulus 16 --cap 30
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.features.benchmark --sample-modulus 16 --pairs 50000
```

Feature Parquet batches and a manifest are cached in `artifacts/features/`. The train and test commands share the same code path and feature schema.

Phase 9 labels only the final retrieved pairs. It keeps every retrieved positive, selects hard and easy negatives for training, and leaves all validation candidates intact under a stable 85/15 S1 split:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.training.run_pairs --sample-modulus 16 --negative-ratio 5
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.training.run_pairs --sample-modulus 16 --negative-ratio 10
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.training.run_pairs --sample-modulus 16 --negative-ratio 20
```

Each policy creates a separate signed artifact under `artifacts/training/`. The requested ratio controls per-S1 negative quotas; the realized global ratio can differ, especially on the independently sampled dev target universe.

Phase 10 trains the CPU LightGBM pair scorer with early stopping and writes scores for **every** validation candidate:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.scoring.run_lightgbm --sample-modulus 16 --negative-ratio 5 --threads 8
```

The model and pair diagnostic metrics are cached in `artifacts/models/`; sharded validation scores are in `artifacts/predictions/validation/`. Pairwise precision at 0.5 is only a diagnostic; Phase 11–12 decide the actual zero/one/many sets against entity-level macro F0.5.

Phase 11 groups every held-out S1, including no-candidate rows, and applies an initial singleton-aware zero/one/many policy. It scores predictions with the exact per-entity F0.5 formula from the competition statement:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.entity_decision.run --sample-modulus 16
```

Grouped scores and entity metrics are cached in `artifacts/validation/`. Threshold tuning is a separate Phase 12 step over these cached groups.

Phase 12 tunes the zero/one/many thresholds with coarse-to-fine exact macro F0.5 search and records baseline and optimized ablations:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.entity_decision.run_optimize --sample-modulus 16
```

The provisional policy is saved under `artifacts/validation/policy/`; `reports/ablation.csv` compares it with the default. The optimizer reads train-validation groups only. A full-universe AWS run is required before the policy is frozen for test inference.

Phase 13 classifies held-out errors and saves bounded representative examples:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.evaluation.run_errors --sample-modulus 16
```

The signed error report is under `artifacts/reports/errors/`. Tags can overlap because one S1 entity can have both a retrieval miss and a wrong extra prediction.

Phase 14 adds an exact cleaned-address retrieval channel for the measured low-name/high-address misses. The current configs enable it; run this channel before union, then rerun the dependent stages with the same sample modulus and cap:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.retrieval.run_exact_address --split train --sample-modulus 16
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.retrieval.run_union --split train --sample-modulus 16 --cap 30
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.evaluation.blocking --sample-modulus 16 --caps 30
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.features.run --split train --sample-modulus 16 --cap 30
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.training.run_pairs --sample-modulus 16 --negative-ratio 5
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.scoring.run_lightgbm --sample-modulus 16 --negative-ratio 5 --threads 8
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.entity_decision.run --sample-modulus 16
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.entity_decision.run_optimize --sample-modulus 16
```

The ablation and resource measurements are in `reports/phase14_exact_address_ablation.md`. Development scores use a smaller distractor universe and are not production estimates.

Phase 15 has a frozen full-refit workflow. It uses the held-out model's chosen boosting-round count, the selected entity policy, the train-fitted TF-IDF vectorizer, and the 37-feature schema. It resamples negatives on **all** training S1 entities, then fits a final model without reusing validation labels for early stopping:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.training.run_finalize --sample-modulus 16
```

Use `--config chimera_submission/code/business_entity_resolution/configs/aws_cpu.yaml` on the production machine after all full-universe upstream stages have completed. The development command passed; the AWS full-data run and ~3-hour/<50-GB acceptance gate are still pending. The pre-refit validation macro F0.5 is recorded in the frozen manifest and is **not** an independent score for the final refit.

Phase 16 runs cache-aware test normalization, all frozen retrieval channels, capped candidate union, batched features, and scores every final candidate with the refitted model:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.inference.run \
  --config chimera_submission/code/business_entity_resolution/configs/dev.yaml \
  --sample-modulus 16 \
  --frozen-manifest artifacts/models/sample_16/final_ac77532ec2c6/manifest.json
```

For the full AWS run, use `configs/aws_cpu.yaml` and its corresponding **full-universe** frozen manifest, omitting `--sample-modulus`. The runner requires the inference config to match the frozen training config, checks the train-fitted vectorizer hash, and records per-stage timing and total scored-pair coverage in `artifacts/reports/inference/`. The development smoke result is in `reports/phase16_inference_development.md`; full inference remains pending.

Phase 17 writes both UTF-8 TSVs from the exact scored test candidates and applies the saved zero/one/many policy. For the validated development slice:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.output.run \
  --frozen-manifest artifacts/models/sample_16/final_71a3fdbe5a54/manifest.json \
  --score-manifest artifacts/predictions/test/sample_16/10c0a5ad290f/manifest.json \
  --normalized-dir artifacts/normalized/sample_16 \
  --output-dir artifacts/output/sample_16
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.output.validate \
  --matching artifacts/output/sample_16/matching_results.tsv \
  --candidate artifacts/output/sample_16/candidate_pairs.tsv \
  --normalized-dir artifacts/normalized/sample_16
```

The second command calls the competition's `utils/validate_submission.py` logic with `--check-ids` against a temporary TSV projection of the sampled normalized IDs. It fails on the validator's otherwise soft matched-subset warning. The development output is **not** the full-test submission.

On the AWS host, run the full train stages in this order from the repository root, replacing the `dev.yaml` examples above with `configs/aws_cpu.yaml` and omitting all sample-modulus overrides:

```bash
CFG=chimera_submission/code/business_entity_resolution/configs/aws_cpu.yaml
MOD=chimera_submission.code.business_entity_resolution.src
.venv/bin/python -m $MOD.normalization.run --config "$CFG" --split train --source 1
.venv/bin/python -m $MOD.normalization.run --config "$CFG" --split train --source 2
.venv/bin/python -m $MOD.normalization.run --config "$CFG" --split train --source 3
.venv/bin/python -m $MOD.blocking.run_exact --config "$CFG" --split train
.venv/bin/python -m $MOD.blocking.run_rare_numeric --config "$CFG" --split train
.venv/bin/python -m $MOD.retrieval.run_exact_address --config "$CFG" --split train
.venv/bin/python -m $MOD.retrieval.run_tfidf --config "$CFG" --split train --threads 8
.venv/bin/python -m $MOD.retrieval.run_union --config "$CFG" --split train --cap 30
.venv/bin/python -m $MOD.evaluation.blocking --sample-modulus 1 --caps 30
```

**Inspect full-universe pair recall, complete-entity recall, candidate count, stage runtime, and peak RSS at this gate.** Rework retrieval if the projected end-to-end budgets or recall are unacceptable. Then continue:

```bash
.venv/bin/python -m $MOD.features.run --config "$CFG" --split train --cap 30
.venv/bin/python -m $MOD.training.run_pairs --config "$CFG" --negative-ratio 5
.venv/bin/python -m $MOD.scoring.run_lightgbm --config "$CFG" --negative-ratio 5 --threads 8
.venv/bin/python -m $MOD.entity_decision.run --config "$CFG"
.venv/bin/python -m $MOD.entity_decision.run_optimize --config "$CFG"
.venv/bin/python -m $MOD.training.run_finalize --config "$CFG"
```

Use the **full** frozen manifest printed by `run_finalize` in the next commands; replace the placeholders with those exact artifact paths. Do not substitute the development manifest or tune thresholds from test scores. Inspect the full validation macro F0.5 and singleton/zero/one/many diagnostics before freezing.

```bash
.venv/bin/python -m $MOD.inference.run --config "$CFG" \
  --frozen-manifest <FULL_FROZEN_MANIFEST>
.venv/bin/python -m $MOD.output.run \
  --frozen-manifest <FULL_FROZEN_MANIFEST> \
  --score-manifest <FULL_TEST_SCORE_MANIFEST> \
  --normalized-dir artifacts/normalized/full \
  --output-dir chimera_submission/output
.venv/bin/python -m $MOD.output.validate \
  --matching chimera_submission/output/matching_results.tsv \
  --candidate chimera_submission/output/candidate_pairs.tsv \
  --test-dir dataset/student_resource/dataset/test
```

The final validator uses the actual full test TSVs with ID checks enabled. The output writer independently enforces that every matched ID belongs to the exact final scored candidate set. The sample result and remaining acceptance gates are in `reports/phase17_submission_development.md`.
