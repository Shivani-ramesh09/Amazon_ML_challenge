# Phase 10 — CPU LightGBM pair scorer

> Historical development benchmark on independently sampled targets. Entity-quality results are superseded by [the entity-complete dev correction](dev_sample_correction.md); stage throughput remains an engineering observation.

The baseline scorer uses 36 fixed float32 features, LightGBM binary objective, 63 leaves, depth 10, learning rate 0.05, 0.8 feature/bagging fractions, 8 CPU threads, a fixed seed, and early stopping on held-out S1 entities. It stores a text model and scores every validation candidate in bounded batches. Training uses the Phase 9 five-negatives-per-positive quota policy; the realized ratio is reported in that phase. Pair metrics are diagnostics, never the model-selection objective.

| Dev universe | Train pairs | Validation pairs | Best tree | Fit | Total | Peak RSS | Pair AP | Pair P@0.5 | Pair R@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Modulus 16 | 206,223 | 398,726 | 293 | 3.64 s | 4.63 s | 0.57 GB | 0.9797 | 0.8789 | 0.9503 |
| Modulus 8 | 604,980 | 859,324 | 472 | 10.68 s | 13.94 s | 0.72 GB | 0.9841 | 0.9047 | 0.9549 |

The modulus-8 model is 3.28 MB and its complete score artifact is 10.23 MB. The highest gain features are address token-set ratio, address word Jaccard, address ratio, number Jaccard, and address length ratio. This confirms the value of strong address features but does not prove that the model predicts correct *sets* for S1 entities. That is evaluated next, including zero-candidate S1 rows and true singletons.

The 8-way training sample has zero overlap between train and validation S1 IDs. Score output has exactly one finite score per held-out candidate and retains pair IDs and true labels for audit. Save/load tests confirm equivalent predictions; all project tests pass. No test labels or validation S1 features enter LightGBM training.

The larger sample took 10.7 s to fit 605k pairs. A rough projection to ~30M full-universe sampled training pairs gives about 9–15 minutes of fitting depending on final boosting round and data density, against the 20-minute stage target. This is an extrapolation: histogram construction and disk/memory pressure may change throughput. A 30M×36 float32 input matrix alone is ~4.3 GB; a ~6M-row validation matrix is ~0.9 GB, before LightGBM bins, copies, gradients, and Polars buffers. The planned <40-GB stage budget is plausible, not verified on AWS. If full profiling violates it, reduce negative rows or boosting complexity based on entity-level F0.5 impact.
