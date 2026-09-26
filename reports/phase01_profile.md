# Phase 1 data contract and profile — 2026-09-26

Commands: `python3 -m chimera_submission.code.business_entity_resolution.src.data.profile --split train --config chimera_submission/code/business_entity_resolution/configs/dev.yaml` and same with `--split test`. Source file hashes, per-file times, length bins and the deterministic split manifest are in ignored `artifacts/profile/`. This is a full dataset scan despite using the dev config, because it is a low-memory integrity gate. The matching stages will use the dev sample modulus.

| Split | S1 | S2 | S3 | GT pairs | Wall time | Peak RSS | Report bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 2,206,821 | 5,034,616 | 5,285,603 | 7,638,365 | 50.55 s | 1.489 GB | 8,768 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | — | 35.57 s | 0.962 GB | 8,618 |

All seven TSVs parsed with exact headers and tab delimiter. Source IDs were unique and had correct prefixes; every training S1 has exactly one truth row; all 7,638,365 truth references resolve to a training S2/S3 ID. No malformed rows occurred. Source counts match the problem statement. The scan processed ~1.33 GB train source/truth input and ~1.19 GB test input. It is currently single-threaded, which is acceptable for this <1-minute phase and avoids multiplying Python ID-set memory. Full-scan measurements are stronger than a small-sample linear extrapolation, but AWS filesystem and CPU may differ; allow at least 2x I/O margin and remeasure there. This stage fits its ~10-minute planning allowance and is well below the 8-GB Phase 1 AWS / 4-GB dev RAM gates.

Training S1 countries: US 1,323,633; India 883,188. Test S1 countries: US 663,106; India 809,986; France 259,452. France is 14.98% of test S1 and absent from train, reinforcing the open-set parser/feature design. Training S2 has 168,967 blank addresses; S3 has 175,916. The truth has 123,247 singletons (5.59%). Cardinality 1 is 119,157, 2–3 is 906,053, and 4+ is 1,058,364; maximum observed is 11. Thus candidate recall must be evaluated at the **complete-entity** level, and a top-1 policy would be severely mismatched to the data. The stable hash split yields 1,875,389 training and 331,432 validation S1 entities (~84.98/15.02%).

Source name-length upper bounds from fixed buckets: train S1 P50<=40, P90<=40, P99<=80 characters; exact bucket counts and sampled common names are in the artifact. Name-frequency sample is 1/128 of rows and is only a profiling estimate; Phase 3 must compute exact block distributions. The Phase 1 source IDs and truth targets were validated in compact integer sets; this approach is not a proposal to keep all candidates in Python containers.

Five unit/integration tests pass under both system Python 3.14 and project venv Python 3.12. The pinned CPU environment resolved and installed 14 exact packages; an offline re-audit succeeded. Direct and transitive versions are in `requirements.txt`. Phase 1 produces no candidate set or score, so candidate recall, macro F0.5, model runtime and test-inference runtime remain unmeasured.
