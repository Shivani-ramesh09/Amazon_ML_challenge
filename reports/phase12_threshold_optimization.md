# Phase 12 — entity F0.5 threshold optimization

The optimizer reads cached, sorted validation groups and complete truth sets. It pads at most 30 candidate scores per S1 into compact arrays, then evaluates the exact per-entity F0.5 for each policy. The search tested 180 valid coarse policies over singleton, pair, strong-match, gap, and max-match settings, followed by 18 coordinate refinements. It checked nearby thresholds (±0.02 on the four continuous settings) for the best candidates and selected a stable policy within 0.0005 of the highest raw score. No retrieval, feature generation, or model training was repeated; no test scores or labels were used.

| Policy | Singleton | Pair | Strong | Gap | Max matches | Macro F0.5 | Micro P | Micro R | Singleton FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial | 0.50 | 0.50 | 0.98 | 0.25 | 30 | 0.87363 | 0.98115 | 0.76509 | 69 / 1,203 |
| Optimized | 0.70 | 0.70 | 0.90 | 0.40 | 30 | **0.87544** | **0.98746** | 0.76019 | **42 / 1,203** |

The macro gain is 0.00181 on 20,825 held-out S1 entities. Singleton accuracy rises from 94.26% to 96.51%; false-positive singleton rate drops from 5.74% to 3.49%. The optimized policy predicts zero for 2,318 S1, one for 3,149, and many for 15,358. It yields 54,707 TP, 695 FP, and 17,258 FN. The 0.89149 cap-30 retrieval oracle leaves ~0.01605 macro points beyond this policy/model on the development universe.

The selected policy's immediate neighbors have macro F0.5 between 0.87508 and 0.87544, with mean 0.87538. The small variation supports a broad threshold region rather than a one-point spike; strong-match threshold variants 0.85–0.90 tie at the optimum. The selected JSON is a **provisional** development policy, not a test-tuned or production-frozen policy. The reduced background target universe can make both scores and thresholds optimistic relative to full AWS data. Phase 15 must validate and freeze the full-universe configuration before final inference.

The search took 1.76 seconds and peaked at 0.21 GB RSS. A simple ~16× entity projection is ~28 seconds, comfortably inside the five-minute stage target, with larger full-data truth loading and storage still to measure. Hand tests verify exact equivalence to set-based F0.5 for zero/one/many cases and an optimizer case where higher singleton threshold removes false links. The baseline and optimized policies are recorded in `reports/ablation.csv` with candidate recall, entity any/complete recall, precision, recall, runtime and memory.
