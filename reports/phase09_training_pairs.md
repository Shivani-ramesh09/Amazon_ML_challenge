# Phase 9 — grouped training pairs

Feature rows are labeled only after retrieval, candidate capping, and feature computation. The existing stable hash split assigns entire S1 entities to 85% training or 15% validation. All retrieved positive pairs are retained; validation keeps its full candidate distribution. Training negatives combine high-similarity false pairs (name, address, exact/core, TF-IDF, postal/house evidence) with a deterministic smaller easy tail. A train S1 with no retrieved positive contributes its hardest false pair so the model sees zero-match cases. No truth can rescue a pair excluded by retrieval.

On the modulus-16, cap-30 development run, 2,658,769 feature pairs split into 398,726 untouched validation pairs and 2,260,043 training-entity candidate pairs. Validation has 3,558 positive pairs; training entities have 20,146. Thus all 23,704 GT pairs recovered by Phase 7 remain labeled. Training and validation S1 sets have zero overlap, and the sampled train set has no duplicate pair. The split contains 117,576 train S1 and 20,825 validation S1, including entities with zero candidates.

| Requested per-positive negative quota | Training rows | Positives | Hard negatives | Easy negatives | Realized neg:pos | Runtime | Peak RSS | Artifact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 206,223 | 20,146 | 168,133 | 17,944 | 9.24 | 3.19 s | 1.46 GB | 16.3 MB |
| 10 | 292,267 | 20,146 | 238,703 | 33,418 | 13.51 | 3.18 s | 1.42 GB | 18.9 MB |
| 20 | 432,225 | 20,146 | 360,190 | 51,889 | 20.45 | 3.12 s | 1.38 GB | 22.4 MB |

The realized 5-policy ratio exceeds five because most sampled S1 rows have no *eligible* GT partner in the independently sampled target universe. They contribute one hard negative each. This does not imply those S1 are true singletons. In the full target universe, positive density and therefore the realized ratio will differ; LightGBM selection must use entity-level validation F0.5, not the requested ratio alone. The 5-policy is the initial model baseline because it is the smallest of the tested training sets.

Each run processes one S1-hash feature shard at a time and writes separate Parquet files for sampled training and complete validation. A naive 16× row-work projection is ~51 seconds plus full GT ingestion and larger truth-map memory; full-data throughput and peak RAM still require measurement. The stage is well below the five-minute target on the development workload. Tests cover all-positive retention, hard/easy quota, repeatability, split isolation, malformed policy, and validation exclusion.
