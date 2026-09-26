"""Held-out entity error taxonomy with bounded representative examples."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import resource
import time

import polars as pl
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from chimera_submission.code.business_entity_resolution.src.entity_decision.policy import DecisionPolicy


def classify_entity(truth: set[str], candidates: list[str], scores: list[float],
                    predicted: list[str], pair_features: dict[str, dict] | None = None) -> set[str]:
    """Non-exclusive entity tags; caller adds text-level subtypes for misses."""
    features = pair_features or {}
    candidate_set = set(candidates)
    predicted_set = set(predicted)
    tags = set()
    if truth - candidate_set:
        tags.add("retrieval_miss")
    if (truth & candidate_set) - predicted_set:
        tags.add("retrieved_false_negative")
    if predicted_set - truth:
        tags.add("singleton_false_positive" if not truth else "non_singleton_false_positive")
    if truth and predicted and predicted[0] not in truth:
        tags.add("wrong_top_candidate")
    if truth & candidate_set and candidates and candidates[0] not in truth:
        best_true = max(scores[index] for index, target in enumerate(candidates) if target in truth)
        if scores[0] > best_true:
            tags.add("true_candidate_ranked_below_false")
    if len(truth) > 1 and truth - predicted_set:
        tags.add("missing_extra_multi_match")
    if truth and not predicted_set:
        tags.add("false_negative_entity")
    for target in (truth & candidate_set) - predicted_set:
        row = features.get(target)
        if row and row.get("address_ratio", 1.0) < 0.35:
            tags.add("address_conflict")
        if row and (row.get("postal_conflict", 0) or row.get("house_conflict", 0)):
            tags.add("numeric_postal_conflict")
    for target in predicted_set - truth:
        row = features.get(target)
        if row and row.get("name_core_exact", 0) and row.get("address_ratio", 1.0) < 0.5:
            tags.add("common_name_ambiguity")
    return tags


def _normalized_lookup(normalized_dir: Path, source_ids: set[str],
                       target_ids: set[str]) -> tuple[dict[str, dict], dict[str, dict]]:
    fields = ["entity_id", "name_raw", "name_core", "address_clean", "postal_candidate",
              "house_number", "country_key"]
    result = ({}, {})
    for path, wanted, destination in ((normalized_dir / "train_s1.parquet", source_ids, result[0]),
                                      (normalized_dir / "train_s2.parquet", target_ids, result[1]),
                                      (normalized_dir / "train_s3.parquet", target_ids, result[1])):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000, columns=fields):
            frame = pl.from_arrow(batch)
            for row in frame.filter(pl.col("entity_id").is_in(wanted)).to_dicts():
                destination[row["entity_id"]] = row
    return result


def analyze_errors(group_dir: Path, validation_features_dir: Path, normalized_dir: Path,
                   truth: dict[str, set[str]], policy: DecisionPolicy, *, examples_per_tag: int = 5) -> dict:
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    feature_columns = ["source1_entity_id", "candidate_entity_id", "name_core_exact", "address_ratio",
                       "postal_conflict", "house_conflict", "country_different"]
    needed_targets = set()
    for path in sorted(group_dir.glob("part-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000,
                                                       columns=["source1_entity_id", "candidate_ids"]):
            for s1_id, candidate_ids in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                needed_targets.update(truth[s1_id] - set(candidate_ids))
    source_rows, target_rows = _normalized_lookup(normalized_dir, set(truth), needed_targets)
    counts = Counter()
    pair_counts = Counter()
    by_country = defaultdict(Counter)
    by_cardinality = defaultdict(Counter)
    examples = defaultdict(list)
    processed = set()
    for path in sorted(group_dir.glob("part-*.parquet")):
        feature_path = validation_features_dir / path.name
        features: dict[str, dict[str, dict]] = defaultdict(dict)
        if feature_path.is_file():
            feature_frame = pl.read_parquet(feature_path, columns=feature_columns)
            for row in feature_frame.to_dicts():
                features[row["source1_entity_id"]][row["candidate_entity_id"]] = row
        for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000):
            columns = {name: batch.column(batch.schema.get_field_index(name)).to_pylist()
                       for name in ("source1_entity_id", "candidate_ids", "scores", "country")}
            for s1_id, candidate_ids, scores, country in zip(*(columns[name] for name in columns)):
                processed.add(s1_id)
                gt = truth[s1_id]
                prediction = policy.decide(candidate_ids, scores)
                tags = classify_entity(gt, candidate_ids, scores, prediction, features.get(s1_id))
                missing = gt - set(candidate_ids)
                miss_detail = None
                pair_counts["retrieval_missed_gt_pairs"] += len(missing)
                pair_counts["retrieved_unselected_gt_pairs"] += len((gt & set(candidate_ids)) - set(prediction))
                for target in sorted(missing):
                    left = source_rows[s1_id]
                    right = target_rows.get(target)
                    if right is None:
                        raise ValueError(f"GT target missing in normalized sample: {target}")
                    name_ratio = fuzz.ratio(left["name_core"] or "", right["name_core"] or "")
                    address_ratio = fuzz.ratio(left["address_clean"] or "", right["address_clean"] or "")
                    if miss_detail is None:
                        miss_detail = {"s1_name": (left["name_core"] or "")[:100],
                                       "target_name": (right["name_core"] or "")[:100],
                                       "s1_address": (left["address_clean"] or "")[:120],
                                       "target_address": (right["address_clean"] or "")[:120],
                                       "name_ratio": round(name_ratio, 1),
                                       "address_ratio": round(address_ratio, 1)}
                    pair_counts["retrieval_miss_name_below_50"] += name_ratio < 50
                    pair_counts["retrieval_miss_address_at_least_75"] += address_ratio >= 75
                    pair_counts["retrieval_miss_low_name_high_address"] += name_ratio < 50 and address_ratio >= 75
                    pair_counts["retrieval_miss_exact_address"] += bool(
                        left["address_clean"] and left["address_clean"] == right["address_clean"])
                    pair_counts["retrieval_miss_house_same"] += bool(
                        left["house_number"] and left["house_number"] == right["house_number"])
                    if name_ratio < 50 and address_ratio >= 75:
                        tags.add("name_representation_gap")
                    if (left["name_raw"] and not left["name_core"]) or (
                        right["name_raw"] and not right["name_core"]):
                        tags.add("normalization_failure")
                    if left["country_key"] and right["country_key"] and left["country_key"] != right["country_key"]:
                        tags.add("country_issue")
                    if left["postal_candidate"] and right["postal_candidate"] and left["postal_candidate"] != right["postal_candidate"]:
                        tags.add("numeric_postal_conflict")
                card = "zero" if not gt else "one" if len(gt) == 1 else "many"
                counts["entities"] += 1
                by_country[country or "<missing>"]["entities"] += 1
                by_cardinality[card]["entities"] += 1
                for tag in sorted(tags):
                    counts[tag] += 1
                    by_country[country or "<missing>"][tag] += 1
                    by_cardinality[card][tag] += 1
                    if len(examples[tag]) < examples_per_tag:
                        examples[tag].append({"s1": s1_id, "country": country,
                                              "truth_count": len(gt), "candidate_count": len(candidate_ids),
                                              "predicted_count": len(prediction),
                                              "top_score": scores[0] if scores else None,
                                              "missing_gt_ids": sorted(missing)[:3],
                                              "miss_detail": miss_detail})
    if processed != set(truth):
        raise ValueError("group S1 set does not match validation truth")
    for tag in ("retrieval_miss", "retrieved_false_negative", "singleton_false_positive",
                "false_negative_entity", "wrong_top_candidate", "true_candidate_ranked_below_false",
                "missing_extra_multi_match", "normalization_failure", "address_conflict",
                "common_name_ambiguity", "country_issue", "numeric_postal_conflict",
                "name_representation_gap"):
        counts.setdefault(tag, 0)
    return {"policy": policy.as_dict(), "entity_counts": dict(counts), "pair_counts": dict(pair_counts),
            "by_country": {key: dict(value) for key, value in by_country.items()},
            "by_cardinality": {key: dict(value) for key, value in by_cardinality.items()},
            "examples": dict(examples),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
            "actual_threads": 1}
