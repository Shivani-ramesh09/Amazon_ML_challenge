# Phase 16 development inference; production gate open

The frozen-model test path ran on the 1/16 development sample with the same code paths intended for AWS. It normalized the three test sources, built exact, rare/numeric, exact-address, and train-vectorizer char-TF-IDF candidates, unioned/deduplicated/capped at 30, created 37 pair features, and scored every final candidate. The train-fitted vectorizer and final model hashes matched the frozen manifest. No test labels were read, no model or vectorizer was fitted on test, and no threshold was tuned to test scores.

| Stage | Wall seconds |
| --- | ---: |
| Normalize S1/S2/S3 | 18.909 |
| Exact-name blocking | 5.753 |
| Rare/numeric blocking | 16.675 |
| Exact-address blocking | 2.903 |
| Sparse char TF-IDF | 19.168 |
| Candidate union/cap | 4.286 |
| Pair features | 31.254 |
| Frozen-model scoring | 5.233 |
| **End-to-end** | **104.185** |

There were 107,510 sampled test S1 entities, 2,192,254 final candidate pairs, and exactly 2,192,254 scored pairs. The capped candidate set left 4,044 S1 entities with zero candidates; these still require one output row each in Phase 17. Peak child-process RSS was 1.214 GB on this sample. Stage RSS tracking uses the cumulative maximum of child-process peaks, so it is an upper bound for earlier stages after the peak is reached, not a sum. Pair features were the largest measured wall-time component. All 43 unit tests passed.

Scaling 104 seconds by the roughly 16-fold S1 sample factor would suggest ~28 minutes, but this is **not** a production runtime estimate: the full 10M-target universe increases ambiguous block sizes, TF-IDF multiplication work, candidate density, address-index size, and feature count. Full AWS inference over 1,732,544 S1, actual peak RSS, ~2-hour budget, France behavior, and final output coverage are still open acceptance gates. The score manifest is `artifacts/predictions/test/sample_16/44519b86bf36/manifest.json`; it is a development artifact, not a submission.
