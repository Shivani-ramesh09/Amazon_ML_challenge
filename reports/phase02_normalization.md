# Phase 2 normalization benchmark — 2026-09-26

All six source TSVs use the same streaming Arrow TSV reader and deterministic Python normalizer. Benchmark mode retains IDs whose numeric suffix is divisible by 16; the parser still scans every raw row. This is a resource-safe engineering benchmark, not a valid retrieval-quality sample because independently selected targets omit many S1 truth matches. Normalized raw and derived strings are written to zstd Parquet; no external identity data is used.

| File | Scanned | Written (1/16) | Elapsed | Peak RSS | Parquet bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train S1 | 2,206,821 | 138,401 | 3.09 s | 0.429 GB | 14,222,526 |
| Train S2 | 5,034,616 | 315,168 | 7.34 s | 0.517 GB | 34,536,161 |
| Train S3 | 5,285,603 | 330,900 | 7.53 s | 0.519 GB | 35,809,982 |
| Test S1 | 1,732,544 | 107,510 | 2.50 s | 0.385 GB | 11,172,638 |
| Test S2 | 4,887,273 | 305,451 | 7.65 s | 0.470 GB | 34,450,701 |
| Test S3 | 5,082,316 | 316,922 | 7.95 s | 0.485 GB | 34,832,168 |

Training sample total: 784,469 normalized rows, 17.96 s, 84.6 MB of Parquet. A second S1 measurement at 1/4 sampling (552,299 normalized rows) took 8.91 s at 0.429 GB RSS and wrote 56.1 MB. The two S1 points imply roughly 1.15 s fixed scan overhead plus 14 microseconds per normalized row. Applying this per-row slope to all 12.53M training source records suggests ~3 minutes for full normalization, plus variation from text length, disk writes, and target source composition. A conservative 3x margin is ~9 minutes, still below the provisional 25-minute training allocation. The ~11.7M test records should be similar, below the provisional 20-minute test normalization allocation. Memory is batch-bound rather than proportional to total rows; the measured peak was ~0.52 GB, but full output size is projected near 1.35 GB train and 1.3 GB test, subject to compression scaling. Remeasure full throughput and peak RSS on the AWS host before accepting the production target.

The training sample has 733,126 distinct raw names, 703,157 distinct clean names and 661,660 distinct core names. There are 22,755 clean keys that combine two or more different raw strings; core has 5.9% fewer distinct keys than clean. This is useful variant consolidation but signals potential large core blocks, so Phase 3 must profile key cardinality and refine oversized blocks. 44.68% of sample rows change beyond simple casefold/trim because of punctuation or other normalization.

In the 1/16 test S1 sample, 16,124 France rows passed the open-set path; 57 contain a recognizable five-digit postal code. The low postal coverage is in the provided data, so downstream retrieval must not rely on postal presence. India test sample has only 7 recognized six-digit PINs; numeric/address-token features are likely more broadly useful. These are observed parser outputs, not ground-truth quality measurements. Raw names/addresses and the original country remain available in every Parquet row.

Parquet cache signature includes source SHA-256, sample modulus, and normalizer implementation hashes. Repeating the same S1 command returned `cache_hit: true` without rewriting the artifact. Nine unit/integration tests pass. Candidate recall, entity F0.5 and model effects are intentionally deferred until their corresponding phases.
