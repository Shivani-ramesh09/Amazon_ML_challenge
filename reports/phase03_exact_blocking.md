# Phase 3 exact/core blocking benchmark — 2026-09-26

> Historical development benchmark on independently sampled targets. Entity-quality results are superseded by [the entity-complete dev correction](dev_sample_correction.md); stage throughput remains an engineering observation.

Command: `.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.blocking.run_exact --split train --sample-modulus 8`. Normalized S1 and target rows were independently sampled by ID suffix modulo 8. Evaluation command: `.venv/bin/python -m chimera_submission.code.business_entity_resolution.src.blocking.eval_exact --normalized-dir artifacts/normalized/sample_8 --candidate artifacts/blocking/sample_8/train_exact.parquet`. All raw input comes from the competition files. The index is a keyed collection of compact uint32 target postings plus columnar target metadata; pairs are streamed to Parquet. Oversized blocks (`>64` targets) use country plus postal, house, exact address, and address-token secondary keys. Blocks needing a 30-pair channel cap are ranked by address/number/postal/country evidence with stable target-ID ties. Trimmed raw hits and truth losses are reported.

| Development universe | 1/16 | 1/8 |
| --- | ---: | ---: |
| S1 queries | 138,401 | 276,025 |
| S2+S3 targets | 646,068 | 1,290,798 |
| Clean/core unique keys | 597,488 / 571,952 | 1,152,972 / 1,088,901 |
| Clean/core largest blocks | 71 / 79 | 135 / 156 |
| Clean/core oversized keys | 3 / 4 | 18 / 92 |
| Clean/core oversized queries | 0 / 41 | 16 / 2,481 |
| Clean/core trimmed queries | 28 / 1,675 | 1,084 / 8,913 |
| Clean/core raw hits dropped by refinement/cap | 144 / 16,374 | 9,577 / 287,916 |
| Clean/core emitted channel pairs | 85,321 / 243,642 | 330,193 / 743,649 |
| Unique emitted pairs after dedup | 248,417 | 819,861 |
| Eligible GT pairs in sampled universe | 29,999 | 119,236 |
| Clean/core GT hits before pruning | 6,517 / 11,296 | 25,804 / 44,451 |
| Core-only GT hits | 4,779 | 18,647 |
| Union GT hits after pruning | 11,296 | 44,442 |
| True pairs lost to exact pruning | 0 | 9 |
| Conditional exact pair recall after pruning | 37.65% | 37.27% |
| Index + retrieval wall time | 5.85 s | 12.50 s |
| Peak RSS | 1.20 GB | 2.13 GB |
| Index artifact / channel Parquet | 119.7 MB / 1.77 MB | 236.3 MB / 6.34 MB |

These recall denominators include only truth pairs whose S1 *and* target survived independent sampling. They are not estimates of the final candidate recall in the full target universe. Removing most distractors shrinks common-name blocks and makes cap loss optimistic. The growth from 1/16 to 1/8 already raised core oversized keys from 4 to 92, so full-block distribution needs remeasurement before accepting the full candidate cap. No labels are used in retrieval; truth is read only by the separate audit.

The initial arbitrary first-30 policy on the 1/8 sample lost 200 of 44,451 exact-recoverable truth pairs and emitted 839,542 unique pairs. Ranking bounded block members by available address evidence reduced the loss to 9 pairs while emitting 819,861. The 1/16 sample went from 10 lost pairs to zero. This is a measurable retrieval improvement within the same runtime class. At 1/8, unique candidate count averaged 2.97 per S1 (median 0, P90 9, P95 16, P99 37, maximum 55; 157,121 S1 had none). Phase 4–6 must add complementary channels and then enforce the final cap; Phase 7 must evaluate full any-hit/complete recall, especially for many-match entities.

Test-side 1/16 run indexed 622,373 sampled targets and processed 107,510 S1 queries in 5.90 s, peaking at 1.15 GB. France had 16,124 S1 sample rows; 3,534 received at least one exact/core candidate (20,279 raw channel pairs). No France label or parser branch was dropped. The index and candidate artifacts are cached separately; an unchanged rerun returns a cache hit in under a second, while a changed S1 query file reuses the target index.

At 1/8 scale, a simple 8x projection gives ~100 s and ~1.9 GB index artifact for full train. Python dictionary growth, increasingly common blocks, and Parquet output may be nonlinear. Allowing a 3x runtime margin yields ~5 minutes, inside the combined cheap-blocking allowance; a 2x RAM margin over the 8x sample projection gives ~34 GB, below the 50-GB target but high enough to require an AWS full-index check. The dev target subset is too small to certify full-scale recall. If full blocks or memory exceed these bounds, refine high-frequency keys or move key/posting storage to sorted Arrow arrays before Phase 7, and remeasure lost GT pairs.
