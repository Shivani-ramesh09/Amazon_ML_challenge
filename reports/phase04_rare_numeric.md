# Phase 4 rare-token and numeric/postal blocking — 2026-09-26

Run `.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.blocking.run_rare_numeric --split train --sample-modulus 8`, then `...src.blocking.eval_cheap --normalized-dir artifacts/normalized/sample_8 --exact artifacts/blocking/sample_8/train_exact.parquet --additional artifacts/blocking/sample_8/train_rare_numeric.parquet`. Both channels index only provided normalized target data. Rare name tokens are ordered by target document frequency; tokens with DF >64 never enter the rare index. Composite keys require a name token with DF <=5,000 plus country and either postal or house number; no number-only global join exists. Target postings use compact uint32 arrays. Each query scans at most three rare token keys and two name tokens combined with at most two address values; each composite posting lookup is capped at 128 and counted when oversized. Output caps are 15 rare-token and 10 numeric/postal pairs per S1. The index and candidate artifacts have separate signatures, so a query change reuses the target index.

| Development universe | 1/16 | 1/8 |
| --- | ---: | ---: |
| S1 queries / targets | 138,401 / 646,068 | 276,025 / 1,290,798 |
| Token vocabulary | 177,139 | 285,479 |
| Rare / composite keys | 174,848 / 852,684 | 282,108 / 1,523,848 |
| Largest rare / composite posting | 64 / 103 | 64 / 145 |
| Composite keys above 128 | 0 | 1 |
| Rare / numeric emitted pairs | 559,500 / 196,756 | 1,214,087 / 490,566 |
| Eligible GT pairs in sampled universe | 29,999 | 119,236 |
| Exact union GT hits | 11,296 | 44,442 |
| Rare incremental GT over exact | 6,413 | 23,557 |
| Numeric incremental GT over exact+rare | 4,945 | 20,492 |
| All cheap-channel union GT hits | 22,654 | 88,491 |
| Conditional cheap-union pair recall | 75.52% | 74.22% |
| Index + retrieval wall time | 15.32 s | 34.66 s |
| Peak RSS | 1.24 GB | 2.18 GB |
| Index / pair Parquet artifact | 127.8 / 6.55 MB | 242.2 / 15.38 MB |

The 1/8 run scanned ~2.34M rare postings and ~1.17M composite postings across 276k queries. House-number keys account for 1,160,000 composite posting scans; postal keys account for only 8,188. Of 1,523,848 composite keys, 1,332,723 have just one target. The two channels add many unique true matches beyond exact name, so they meet the Phase 4 usefulness gate. The 1/16-to-1/8 recall decrease is a warning that the full target universe can increase ambiguity and posting truncation. These are **conditional sampled-universe** truth statistics, not full candidate recall. Phase 7 must rerun on the full universe and report entity any-hit/complete recall and cap sweeps before model training.

On test 1/16, the channel processed 107,510 S1 and 622,373 targets in 16.16 s, peaking at 1.21 GB; it emitted 755,931 raw channel pairs. Of 16,124 sampled France S1 rows, 15,230 received at least one rare/numeric candidate. This is unlabeled coverage, not match correctness. The observed postal sparsity means most numeric composite evidence is expected to be house/building numbers; exact posting-type counts are recorded in the artifact manifest.

Naively multiplying the 1/8 time by eight gives ~4.6 minutes for full train. Because token vocabulary, composite keys, posting collisions and Python dictionary overhead do not scale linearly, a 3x time margin is ~13.9 minutes. That is above the provisional 10-minute allowance for this individual stage but small relative to the 3-hour overall training budget; the next full-scale gate must verify it before accepting more retrieval work. Linear RSS projection is ~17.4 GB; a 2x safety margin is ~35 GB, below the practical 50-GB target. If full dictionary growth exceeds this, move composite postings to sorted Arrow arrays or tighten token selection while measuring lost GT pairs.
