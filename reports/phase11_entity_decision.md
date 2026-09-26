# Phase 11 — entity-level decision and metric

Validation scores are sorted within each S1, with stable target-ID ties, then cached as one row per held-out entity. Each row includes candidate IDs, scores, channel counts, TF-IDF ranks, top three scores, top/second gap, top-three mean, and top candidate provenance. S1 entities with no candidate receive an explicit empty group. The initial deterministic policy predicts zero when the top score is below the singleton threshold, one for a very strong isolated top score, and multiple when several candidates exceed the pair threshold. It never forces top-1. Thresholds remain tunable in Phase 12.

The official per-entity metric is implemented as `1.25 TP / (1.25 TP + FP + 0.25 FN)`, with empty truth and empty prediction scoring 1.0. Tests reproduce the competition’s two-true/three-predicted 0.714 example, singleton success/failure, partial multiple-match credit, and complete S1 accounting. The prior blocking oracle denominator was independently corrected and committed before this phase; the cap-30 corrected retrieval ceiling is 0.8915 macro F0.5 on the entity-complete development universe.

On the corrected modulus-16 development validation split, the initial policy (`singleton=0.50`, `strong=0.98`, `pair=0.50`, `gap=0.25`, max 30) scores **0.87363 macro entity F0.5**, with micro precision **0.98115** and micro recall **0.76509**. It scores all 20,825 validation S1 entities, including 138 with zero candidates, from 435,361 candidate pairs. Among 1,203 true singletons, 1,134 are correctly left empty and 69 receive false positives: **94.26% singleton accuracy** and **5.74% singleton false-positive rate**. The model predicts 56,118 links, of which 55,060 are true and 1,058 false; 16,905 true links are missed. The positive-entity macro score is 0.86940.

| True cardinality | Pred zero | Pred one | Pred many |
|---|---:|---:|---:|
| Zero | 1,134 | 63 | 6 |
| One | 235 | 880 | 26 |
| Many | 853 | 2,301 | 15,327 |

India macro F0.5 is 0.80412 across 8,269 S1; US is 0.91941 across 12,556 S1. The India gap tracks Phase 7 retrieval misses, especially names in different scripts with similar addresses. The baseline is only 0.01786 below the cap-30 retrieval oracle, so decision thresholds may recover some score, but retrieval improvements may have more leverage. The development target universe includes all truth partners but fewer background distractors than production; these scores are not full-universe estimates.

Grouping took 0.56 s, peaked at 0.35 GB RSS, and wrote 6.26 MB. Metric evaluation took 1.22 s in the same process. A linear projection over ~15× more held-out entities/pairs is well inside the three-minute stage target, but full AWS RSS/runtime must still be measured. All project tests pass before the phase commit.
