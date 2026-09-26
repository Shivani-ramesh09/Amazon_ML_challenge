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
