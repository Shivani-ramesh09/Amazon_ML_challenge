# Business Entity Resolution Pipeline

Team: `chimera`

Phase 1 provides a streaming contract and data profile. The full matching pipeline is being built phase by phase; `src/main.py` is not yet runnable.

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

Run the same command with `--source 2` and `--source 3`, and with `--split test`, for all six source files. Omit `--sample-modulus` to use the selected config's default (dev: 1/128 rows, AWS: full). The sample is for engineering checks only; independently sampled target IDs do not support valid candidate-recall claims.

Phase 3 exact blocking, after the three normalized files for a split exist:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.blocking.run_exact --split train --sample-modulus 16
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.blocking.eval_exact --normalized-dir artifacts/normalized/sample_16 --candidate artifacts/blocking/sample_16/train_exact.parquet
```

The second command reports **conditional sample recall** only. Full-universe recall and entity completeness are later evaluation gates.

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

The cap audit expects cached K=5,10,15,20,30 training runs. The final scored candidates are the `part-*.parquet` files in the `final_dir` printed by `run_union`. Sample recall is conditional on target IDs being independently sampled; use Phase 7 for the full blocking report.

Phase 7 evaluates each channel, union, and cap on both pair and entity measures, and profiles missed ground-truth pairs:

```bash
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.evaluation.blocking --sample-modulus 16
.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.evaluation.miss_diagnostics --sample-modulus 16
```

The evaluation JSON is saved in `artifacts/reports/blocking/`. The default candidate cap is now 30 because the measured recall gain over 20 justified the extra candidates on the development sample. Full-universe recall and final entity F0.5 still require the production run.

Phase 8 builds 36 numeric pair features from the exact final candidate set using bounded Arrow gathers and 250,000-pair batches in dev:

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
