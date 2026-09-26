# Phase 14 — Exact-address retrieval ablation

Phase 13 found 16,151 missing held-out true links, including 1,335 with identical nonempty cleaned addresses and 3,877 with low core-name similarity but high address similarity. We therefore tested a bounded exact-address channel after completing the CPU lexical baseline. The comparison uses the same corrected, truth-complete sample of 138,401 train S1 entities, 1,095,422 sampled and GT-linked target rows, 479,353 GT links, 20,825 held-out S1 entities, and cap 30. Retrieval never uses labels. The test target universe is not represented in this experiment.

| Metric | Baseline | Exact address | Change |
| --- | ---: | ---: | ---: |
| Final candidate pairs | 2,895,797 | 2,904,239 | +8,442 |
| GT pairs recovered | 372,430 | 380,738 | +8,308 |
| Pair candidate recall | 0.776943 | 0.794275 | +0.017332 |
| Entity any-hit recall | 0.944239 | 0.952433 | +0.008195 |
| Entity complete recall | 0.531967 | 0.558097 | +0.026131 |
| Candidate oracle macro F0.5 | 0.891490 | 0.902707 | +0.011217 |
| Optimized held-out macro F0.5 | 0.875444 | **0.885423** | **+0.009979** |
| Optimized micro precision | 0.987455 | 0.985266 | -0.002190 |
| Optimized micro recall | 0.760189 | 0.778656 | +0.018467 |
| False-positive singleton entities | 42 / 1,203 | 53 / 1,203 | +11 |

The treatment retrained LightGBM on 37 features and reoptimized the entity policy rather than applying baseline thresholds to a changed score distribution. Selected treatment thresholds: singleton 0.65, pair 0.65, strong 0.90, gap 0.40, max count 30. Nearby policy scores ranged from 0.885145 to 0.885423, so the gain is not a single-point threshold spike. `reports/ablation.csv` contains baseline and treatment rows.

The exact-address channel generated 52,225 pairs, of which 50,883 were true links in this reduced target universe. It recovered 8,783 GT links exclusive of all other raw channels before cap and 8,308 additional GT links after cap. It indexed 969,248 address keys; maximum block size was eight and no blocks exceeded the configured limit of 64. The index build took 1.864 s and reached 0.856 GB process RSS; the serialized index was 133.4 MB. Retrieval took 0.190 s, reached 1.349 GB process RSS, and wrote 0.65 MB. These RSS figures include process/runtime state and are stage-local measurements, not an additive peak.

Treatment-dependent stages measured 6.032 s for union/final cap, 4.533 s for blocking evaluation, 39.287 s for 2.904M-pair features (1.335 GB peak RSS), 3.270 s for training-pair selection (1.439 GB), 15.937 s for model fit and validation scoring (0.966 GB), and 1.703 s for threshold search. The full sample treatment rerun was under two minutes excluding previously cached normalization and other channels. At full train scale, address-index memory could be several GB rather than 133 MB, and block-size/cardinality distributions can grow with the full distractor universe. The AWS <=50 GB and ~3-hour training/~2-hour inference targets therefore remain unverified until a full run; extrapolating this channel alone is insufficient to certify the pipeline.

The gain targets a measured error class, so the channel is retained. It uses country as an open-set string, requires a meaningful cleaned address, bounds oversized blocks with house/postal evidence, records dropped/trimmed counts, and adds provenance to the final candidates and feature schema. The full-universe run must remeasure country slices, block sizes, candidate recall, macro F0.5, peak RSS, and France behavior. No global address TF-IDF or external data was added. All 39 unit tests passed.
