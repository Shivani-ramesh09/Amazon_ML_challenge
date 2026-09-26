# ML Challenge 2026: Business Entity Resolution — chimera

Status: development validation only. The production training and full-test submission have not been run; the tracked `output/` TSVs are header-only placeholders. Do not submit them.

## Methodology

S1 is the reference source. For each S1 entity, the model predicts zero, one, or multiple S2/S3 IDs. The metric is macro entity-level F0.5, including full credit for a correctly predicted singleton. There is no one-to-one matching assumption. Only the supplied train/test fields are used; no external identity data, geocoding, APIs, or enrichment are involved.

The CPU pipeline reads TSVs in batches and preserves raw text. Deterministic NFKC/casefold normalization creates clean/core business names, cleaned addresses, country keys, postal candidates, and house/number tokens. India and US address patterns have tailored parsing; France and unknown countries use a generic fallback. Country is an open-set string, used as relationship evidence rather than a fixed country one-hot vocabulary.

Candidate generation unions exact cleaned/core name, rare-name-token, name-plus-number/postal, sparse character 3–5-gram TF-IDF top-K, and measured exact-cleaned-address channels. Common/oversized blocks are refined or bounded; the final union is deduplicated and capped at 30 candidates per S1. Exact-address retrieval was introduced after error analysis found cross-script name matches with identical addresses. Sparse top-N multiplication avoids dense pairwise matrices. The exact final candidate set is cached and later serialized to `candidate_pairs.tsv`.

Each candidate receives 37 numeric name, address, country, missingness, and retrieval-provenance features. All recovered positive pairs and deterministic hard/easy negative samples train a CPU LightGBM classifier. A stable entity-level S1 split supplies validation candidates; no pairs from a held-out S1 enter training. A cached zero/one/many policy searches singleton, pair, strong-match and score-gap thresholds against exact macro F0.5. The final scorer refits on all labeled training entities for the best held-out boosting-round count. The held-out score belongs to the **pre-refit** model and is not an independent assessment of that refit.

## Development measurements

The truth-complete 1/16 development train sample has 138,401 S1 entities and 479,353 true links; target background density is lower than production, making results optimistic. With all channels and cap 30, 380,738 true links are retrieved (pair candidate recall 0.794275), entity any-hit recall is 0.952433, and entity complete recall is 0.558097. The optimized held-out policy achieves macro F0.5 **0.885423**, micro precision **0.985266**, micro recall **0.778656**, and 53 false-positive predictions among 1,203 true singleton entities. The exact-address ablation added 8,308 recovered true links after cap and raised macro F0.5 by 0.009979 relative to the preceding lexical baseline.

The 1/16 test smoke run normalized 107,510 S1 entities, generated and scored 2,192,254 final candidates in 104.185 seconds, and peaked at 1.214 GB child-process RSS. The output writer generated one row per sampled S1 in both TSVs; the official validator passed with ID checks and zero warnings against the sampled test ID universe. These unlabeled test predictions carry no accuracy claim.

## Remaining production gates

The 8-vCPU/64-GB machine must run the full training pipeline and report actual stage times, peak RSS, candidate recall, macro F0.5, precision/recall, and singleton/zero/one/many performance. Only then should the full frozen scorer run against all 1,732,544 test S1 entities. The two canonical `chimera_submission/output/` TSVs must then be generated from the exact scored candidate set and pass `utils/validate_submission.py` against the complete raw test directory with `--check-ids`. The ~3-hour training, ~2-hour inference, and <50-GB RSS targets remain unverified.

Exact reproduction commands, dependencies, cache layout, and the current development artifact paths are in `code/business_entity_resolution/README.md`. Design and measured phase reports are in the repository's `architecture.md`, `plan.md`, `TODO.md`, and `reports/` directory. No GPU implementation is used.
