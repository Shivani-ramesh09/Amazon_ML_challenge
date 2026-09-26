# Phase 6 — bounded candidate union

> Historical development benchmark on independently sampled targets. Entity-quality results are superseded by [the entity-complete dev correction](dev_sample_correction.md); stage throughput remains an engineering observation.

The final candidate builder streams the three channel Parquets into S1-hash shards, then deduplicates each shard by `(S1 ID, target ID)`. It ORs five provenance flags, retains the best retrieval scores/ranks, applies a deterministic cheap evidence order, and caps the set per S1. The capped Parquet shards are the sole input planned for pair scoring. Raw sharding and cap/ranking artifacts have separate cache keys.

## Measured development benchmark

Train sample modulus 16: 138,401 S1 rows and 646,068 target rows (the sample is independently selected, so all recall below is conditional on both IDs being sampled). Three channels emitted 3,151,562 rows. The union held 2,714,358 distinct pairs, removed 437,204 duplicate channel rows, and had 6,483 S1 rows with no candidate. Raw sharding took 3.97 s, peak RSS 0.305 GB, artifact 27.5 MB; rank/cap 20 took 0.86 s, process peak RSS 0.81 GB, artifact 21.9 MB. The largest processed raw shard was 799,455 rows. Four of 64 shards were occupied because IDs were sampled modulo 16.

With 29,999 GT pairs eligible in the sampled target universe, the revised evidence order gives:

| Cap | Final pairs | Mean / sampled S1 | GT pairs hit | Conditional pair recall |
|---:|---:|---:|---:|---:|
| 5 | 626,180 | 4.52 | 21,916 | 73.06% |
| 10 | 1,202,930 | 8.69 | 22,668 | 75.56% |
| 15 | 1,749,480 | 12.64 | 23,093 | 76.98% |
| 20 | 2,282,700 | 16.49 | 23,309 | 77.70% |
| 30 | 2,658,769 | 19.21 | 23,704 | 79.02% |

The first rank formula put 1,422 numeric/postal-retrieved GT pairs at positions 21–30, so cap 20 recovered only 21,897 GT pairs (72.99%). Raising numeric/postal evidence in the cheap order lifted cap-20 recovery to 23,309 (77.70%) without increasing candidate volume. Cap 30 still recovers 395 more GT pairs than cap 20; Phase 7 will decide the accepted production cap after entity-level completeness and volume analysis. This cheap ordering uses only pair evidence and no truth at inference.

An arithmetic 16× extrapolation gives about 50 million raw rows and 42.5 million final pairs at cap 30 for train, and ~77 seconds for the measured sharding/rank work. The estimate assumes near-linear I/O and balanced 64-way sharding; full-data disk pressure and hash/index costs remain unmeasured. The shard row bound is similar to the sample's four occupied shards. CPU time/RAM are comfortably inside the phase's 10-minute/30-GB AWS envelope, but pair-feature cost at cap 30 must be measured later.

Tests: 21 unit tests pass, including provenance OR, dedup, empty S1, cap, stable tie, cache reuse and cap-only recomputation. The exact commands are in the README. This is a conditional sampled recall result, not a full-universe candidate-recall claim.
