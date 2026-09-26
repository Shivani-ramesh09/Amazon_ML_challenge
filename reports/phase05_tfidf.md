# Phase 5 sparse character TF-IDF retrieval — 2026-09-26

> Historical development benchmark on independently sampled targets. Entity-quality results are superseded by [the entity-complete dev correction](dev_sample_correction.md); stage throughput remains an engineering observation.

The first global sparse top-N probe was too slow: on 646,068 target names, 1,000 queries with unpruned 3–5 `char_wb` n-grams took 2.90 s. The target CSR had 31.1M nonzeros (0.234 GB), so RAM was not the issue; frequent n-grams caused expensive sparse multiplication. A naive full-scale projection was many hours. Pruning document frequencies above 1% of targets cut the same probe to 0.381 s/1k queries; at 0.1% it took 0.072 s/1k with 5.5M target nonzeros. This measured bottleneck motivated the `max_df=0.001` CPU design. All products use `sparse-dot-topn` with CSR float32, top-K=20 and similarity threshold 0.2; no dense pairwise matrix is allocated.

| Development universe | 1/16 train | 1/8 train | 1/16 test |
| --- | ---: | ---: | ---: |
| S1 queries / targets | 138,401 / 646,068 | 276,025 / 1,290,798 | 107,510 / 622,373 |
| Vectorizer vocabulary | 60,000 | 60,000 | frozen train 60,000 |
| Target CSR nonzeros | 5,505,841 | 10,979,865 | 5,349,741 |
| Target/transpose CSR memory | 0.043 / 0.041 GB | 0.087 / 0.082 GB | 0.042 / 0.040 GB |
| Index build/transform | 12.67 s | 25.41 s | 8.83 s |
| Retrieval | 5.35 s | 14.88 s | 7.78 s |
| Peak RSS | 1.03 GB | 1.84 GB | 0.42 GB |
| Zero-feature S1 queries | 35,083 | 68,540 | 24,341 |
| Top-N candidate pairs | 2,066,343 | 4,149,700 | 1,663,112 |
| Transposed CSR artifact | 44.3 MB | 88.1 MB | 43.0 MB |
| Candidate Parquet | 14.9 MB | 29.3 MB | 12.8 MB |

The 1/16 sample has 29,999 eligible GT pairs. Cheap channels recovered 22,654; TF-IDF added 1,082 uniquely, lifting conditional pair recall from 75.52% to 79.12%. At 1/8 scale, 119,236 eligible GT pairs yielded 88,491 cheap hits; TF-IDF added 3,757, lifting conditional recall from 74.22% to 77.37%. These are sampled-universe pair recalls, not full candidate recall or entity F0.5. Phase 6/7 must measure lost GT from dedup/pruning/caps and entity complete recall.

A stricter `max_df=0.0005` saved ~0.9 s of retrieval on the 1/16 sample but uniquely recovered only 935 GT pairs beyond cheap channels, versus 1,082 at 0.001. A conservative `strong_exact` gate skipped 7,350/138,401 queries, saved about 0.2–0.3 s, and lost 57 additional true pairs. Therefore the baseline searches all nonempty/empty-name queries through the same code path; empty and zero-feature rows cost little sparse multiplication. The gate remains implemented and auditable but is not selected. This is an evidence-based exception to the proposed Tier-2 skip rule, especially because a strong exact hit can coexist with additional true matches.

The test index transformed France names with the **frozen train-fitted vectorizer**; it did not fit on test. In the test 1/16 sample, 16,103/16,124 France S1 rows received at least one TF-IDF candidate. That is unlabeled coverage, not accuracy. France made test retrieval slower per query than train, so test projections use its measured workload.

From the 1/8 train benchmark, an 8x query and 8x target projection gives roughly 16 minutes retrieval plus 3–4 minutes index fitting for full train. This assumes sparse multiplication scales with target posting length and query count; high-frequency pruning changes the term distribution as corpus size grows, so a 2x runtime margin (~40 minutes) is the practical gate against the provisional 40-minute TF-IDF allocation. A naive 8x RSS projection from 1.84 GB is ~15 GB; a 2x safety margin is ~30 GB, under the 50-GB target. The actual full AWS run must measure nnz, posting distributions, wall time and RSS before final model training. The versioned index key contains the DF/feature settings; candidate signatures include K, threshold, gate and query hash. Changing model or decision thresholds does not rebuild this index.
