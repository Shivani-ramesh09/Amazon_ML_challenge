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
