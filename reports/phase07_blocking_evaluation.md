# Phase 7 — blocking evaluation

> Historical development benchmark on independently sampled targets. Entity-quality results are superseded by [the entity-complete dev correction](dev_sample_correction.md); stage throughput remains an engineering observation.

The shardwise evaluator reads final candidates exactly as the scorer will see them. It includes zero-candidate S1 rows, compares retrieved pairs to ground truth, and reports pair, any-hit, complete-hit, country, cardinality, and best-case entity F0.5. It runs per S1-hash shard rather than collecting all candidate pairs. A hand fixture checks singleton, multi-match, retrieval miss, and F0.5 arithmetic.

On the deterministic modulus-16 train sample, 138,401 S1 and 646,068 S2/S3 rows yielded 29,999 **eligible** GT pairs. This is a conditional experiment: independently sampled targets omit most true partners. Of selected S1, 7,827 are true singletons, while 111,344 have zero *eligible* GT pairs. The latter number must never be treated as true singleton prevalence. The sampled macro oracle over all S1 is therefore inflated; the eligible-positive oracle and pair recall are the relevant development diagnostics. Full-universe recall is unmeasured.

| Retrieval set | Candidates | GT hits | Conditional pair recall | Any-hit | Complete-hit | Eligible-positive oracle F0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Exact clean | 85,321 | 6,517 | 21.72% | 23.53% | 20.02% | 21.94% |
| Exact core | 243,642 | 11,296 | 37.65% | 40.05% | 35.24% | 37.91% |
| Rare token | 559,500 | 11,725 | 39.08% | 39.60% | 38.66% | 39.19% |
| Numeric/postal | 196,756 | 16,788 | 55.96% | 57.84% | 54.12% | 56.19% |
| TF-IDF name | 2,066,343 | 14,711 | 49.04% | 50.10% | 48.09% | 49.21% |
| Union before cap | 2,714,358 | 23,736 | 79.12% | 80.36% | 77.80% | 79.24% |
| Cap 5 | 626,180 | 21,916 | 73.06% | 74.80% | 71.48% | 73.34% |
| Cap 10 | 1,202,930 | 22,668 | 75.56% | 77.06% | 74.08% | 75.75% |
| Cap 15 | 1,749,480 | 23,093 | 76.98% | 78.38% | 75.55% | 77.13% |
| Cap 20 | 2,282,700 | 23,309 | 77.70% | 79.02% | 76.31% | 77.83% |
| Cap 30 | 2,658,769 | 23,704 | 79.02% | 80.26% | 77.69% | 79.13% |

Cap 30 is the default for subsequent development and AWS profiles: it retains 395 more GT pairs than cap 20 for 376,069 additional candidates (+16.5%), and loses only 32 GT pairs against the uncapped union. The measured scorer throughput in Phase 8 can still force a different cost/quality decision. At cap 30, median/P90/P95/P99/max candidates are 20/30/30/30/30, with 6,483 zero-candidate S1. The pre-cap union has max 65 and P99 37. The approximate full-train cap-30 volume is 42.5 million pairs if sample density scales; full-data block frequencies and pair counts may differ.

The ordered-channel marginal GT gains were 6,517 exact clean, 4,779 exact core, 6,413 rare, 4,945 numeric/postal, and 1,082 TF-IDF. Exclusive GT hits were 552 core, 321 rare, 4,070 numeric/postal, and 1,082 TF-IDF. Exact blocking recorded 3 oversized clean-name keys and 4 oversized core-name keys in this sample, plus 144 and 16,374 raw hits dropped after evidence ranking. These counts are logged in the channel manifests; the final candidate cap is separately accounted for.

Country slice exposes a material weakness: at cap 30, US conditional pair recall is 85.01% on 17,897 eligible pairs, but India is 70.15% on 12,102. Of 6,263 uncapped-union misses, 3,593 are India. In those India misses, 2,411 have core-name RapidFuzz ratio below 50, while 1,988 have address ratio at least 75 and 1,329 have both low name and high address agreement. Representative missed names include Latin-script S1 names paired with Indic-script target names. This supports an **address-evidence retrieval experiment** after the CPU lexical baseline, with unique GT gain and time/RAM measured before adoption. It does not justify blindly running global address KNN. US misses include 1,679 with name ratio at least 75, suggesting a separate fuzzy-channel top-K/threshold recall audit.

Evaluation took 3.36 s and peaked at 0.70 GB RSS on this sample; all channel and cap Parquet inputs total 111.8 MB. A linear row-count projection is under a minute, but full truth loading, hash join sizes, and storage throughput can change that. The code processes one of 64 S1 shards at a time, so peak join size is bounded. The observed retrieval quality is a **baseline ceiling**, not an accepted final score: full-universe blocking evaluation on AWS and Phase 13–14 error-driven improvement remain required before a final model is frozen.
