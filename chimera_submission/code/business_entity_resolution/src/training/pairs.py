"""Entity-grouped labels and deterministic hard/easy negative sampling."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import polars as pl
import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.data.profile import split_bucket


def read_truth(path: Path, selected_s1: set[str] | None = None) -> dict[str, set[str]]:
    truth = {}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        if next(reader) != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("unexpected truth schema")
        for s1_id, raw_targets in reader:
            if selected_s1 is None or s1_id in selected_s1:
                truth[s1_id] = set(raw_targets.split(",")) if raw_targets else set()
    return truth


def label_and_split(frame: pl.DataFrame, truth: dict[str, set[str]],
                    validation_ids: set[str]) -> pl.DataFrame:
    labels = []
    validation = []
    for s1_id, target_id in frame.select(["source1_entity_id", "candidate_entity_id"]).iter_rows():
        if s1_id not in truth:
            raise ValueError(f"candidate S1 {s1_id} is absent from truth")
        labels.append(target_id in truth[s1_id])
        validation.append(s1_id in validation_ids)
    return frame.with_columns(pl.Series("label", labels, dtype=pl.Boolean),
                              pl.Series("is_validation", validation, dtype=pl.Boolean))


def select_train_pairs(frame: pl.DataFrame, *, negative_ratio: int = 5,
                       hard_fraction: float = 0.8, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """Keep all positives; take high-similarity false pairs and a small easy tail."""
    if negative_ratio < 1 or not 0 <= hard_fraction <= 1:
        raise ValueError("invalid negative sampling policy")
    if frame["is_validation"].any():
        raise ValueError("training selection received validation rows")
    positive = frame.filter(pl.col("label"))
    negative = frame.filter(~pl.col("label"))
    positive_counts = frame.group_by("source1_entity_id").agg(
        pl.col("label").sum().cast(pl.Int32).alias("positive_count"))
    negative = negative.join(positive_counts, on="source1_entity_id")
    quota = pl.when(pl.col("positive_count") > 0).then(
        pl.col("positive_count") * negative_ratio).otherwise(1)
    negative = negative.with_columns(quota.alias("negative_quota"))
    negative = negative.with_columns(
        (pl.col("negative_quota") * hard_fraction).ceil().cast(pl.Int32).alias("hard_quota"),
        (
            2.0 * pl.max_horizontal("name_ratio", "name_token_set_ratio", "address_ratio") +
            2.0 * pl.col("name_core_exact") + pl.col("name_clean_exact") +
            pl.col("best_tfidf_score") + 0.25 * pl.col("retrieval_channel_count") +
            0.25 * pl.col("postal_same") + 0.25 * pl.col("house_same")
        ).alias("hardness"),
    )
    negative = negative.sort(["source1_entity_id", "hardness", "candidate_entity_id"],
                             descending=[False, True, False]).with_columns(
        pl.col("candidate_entity_id").cum_count().over("source1_entity_id").alias("hard_rank"))
    hard = negative.filter(pl.col("hard_rank") <= pl.col("hard_quota"))
    remainder = negative.filter(pl.col("hard_rank") > pl.col("hard_quota"))
    remainder = remainder.with_columns(
        pl.col("candidate_entity_id").hash(seed=seed).alias("random_order"))
    remainder = remainder.sort(["source1_entity_id", "random_order", "candidate_entity_id"]).with_columns(
        pl.col("candidate_entity_id").cum_count().over("source1_entity_id").alias("easy_rank"))
    easy = remainder.filter(pl.col("easy_rank") <= pl.col("negative_quota") - pl.col("hard_quota"))
    columns = frame.columns
    selected = pl.concat([positive, hard.select(columns), easy.select(columns)], how="vertical")
    stats = {"positive_pairs": positive.height, "available_negative_pairs": negative.height,
             "hard_negative_pairs": hard.height, "easy_negative_pairs": easy.height,
             "training_pairs": selected.height,
             "actual_negative_to_positive": (hard.height + easy.height) / positive.height if positive.height else None}
    if selected.filter(pl.col("label")).height != positive.height:
        raise AssertionError("positive candidate lost during sampling")
    return selected, stats


def build_training_pairs(features_dir: Path, normalized_s1: Path, truth_path: Path,
                         output_dir: Path, *, validation_percent: int = 15, seed: int = 42,
                         negative_ratio: int = 5, hard_fraction: float = 0.8) -> dict:
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    s1_ids = pq.read_table(normalized_s1, columns=["entity_id"])["entity_id"].to_pylist()
    validation_ids = {s1 for s1 in s1_ids
                      if split_bucket(s1, seed=seed, validation_percent=validation_percent) == "validation"}
    truth = read_truth(truth_path, set(s1_ids))
    if len(truth) != len(s1_ids):
        raise ValueError("selected S1 IDs do not all appear in ground truth")
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "validation").mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Path]] = {}
    for path in features_dir.glob("part-*-batch-*.parquet"):
        grouped.setdefault(path.name.split("-batch-")[0], []).append(path)
    totals = Counter()
    shard_reports = []
    for shard, paths in sorted(grouped.items()):
        frame = pl.read_parquet(sorted(paths))
        labeled = label_and_split(frame, truth, validation_ids)
        validation = labeled.filter(pl.col("is_validation")).drop("is_validation")
        train_all = labeled.filter(~pl.col("is_validation"))
        selected, stats = select_train_pairs(train_all, negative_ratio=negative_ratio,
                                             hard_fraction=hard_fraction, seed=seed)
        train_path = output_dir / "train" / f"{shard}.parquet"
        validation_path = output_dir / "validation" / f"{shard}.parquet"
        selected.drop("is_validation").write_parquet(train_path, compression="zstd")
        validation.write_parquet(validation_path, compression="zstd")
        totals.update({key: value for key, value in stats.items()
                       if key in ("positive_pairs", "available_negative_pairs", "hard_negative_pairs",
                                  "easy_negative_pairs", "training_pairs")})
        totals["validation_pairs"] += validation.height
        totals["validation_positive_pairs"] += validation.filter(pl.col("label")).height
        totals["input_feature_pairs"] += frame.height
        shard_reports.append({"shard": shard, **stats, "validation_pairs": validation.height})
    return {**dict(totals), "selected_s1_rows": len(s1_ids), "validation_s1_rows": len(validation_ids),
            "train_s1_rows": len(s1_ids) - len(validation_ids),
            "actual_negative_to_positive": (totals["hard_negative_pairs"] + totals["easy_negative_pairs"]) /
                                           totals["positive_pairs"] if totals["positive_pairs"] else None,
            "shards": shard_reports,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
            "artifact_bytes": sum(path.stat().st_size for path in output_dir.glob("*/*.parquet")),
            "actual_threads": 1}


def training_signature(feature_manifest: Path, truth_path: Path, *, validation_percent: int,
                       seed: int, negative_ratio: int, hard_fraction: float) -> str:
    feature = json.loads(feature_manifest.read_text())["signature"]
    with truth_path.open("rb") as file:
        truth_hash = hashlib.file_digest(file, "sha256").hexdigest()
    return hashlib.sha256(json.dumps({"features": feature, "truth": truth_hash,
                                      "validation_percent": validation_percent, "seed": seed,
                                      "negative_ratio": negative_ratio, "hard_fraction": hard_fraction,
                                      "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                     sort_keys=True).encode()).hexdigest()
