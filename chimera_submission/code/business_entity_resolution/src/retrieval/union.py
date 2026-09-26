"""Sharded, bounded candidate union and deterministic per-S1 cap."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

RAW_SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()), ("candidate_entity_id", pa.string()),
    ("channel", pa.string()), ("block_size", pa.int32()),
    ("cheap_score", pa.float32()), ("tfidf_score", pa.float32()),
    ("tfidf_rank", pa.int16()),
])
FINAL_COLUMNS = (
    "source1_entity_id", "candidate_entity_id", "retrieved_by_exact_name",
    "retrieved_by_core_name", "retrieved_by_rare_token", "retrieved_by_number_postal",
    "retrieved_by_tfidf", "retrieval_channel_count", "exact_block_size",
    "best_tfidf_score", "tfidf_rank", "best_cheap_score", "preliminary_score",
)
RAW_FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _s1_bucket(s1_id: str, shards: int) -> int:
    return int(s1_id.partition("-")[2]) % shards


def shard_channels(channel_paths: list[Path], output_dir: Path, *, shards: int = 64) -> dict:
    """Scan channel Parquet in batches; never hold all emitted pairs in Python."""
    if shards < 1:
        raise ValueError("shards must be positive")
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[int, pq.ParquetWriter] = {}
    buffers = {bucket: {field.name: [] for field in RAW_SCHEMA} for bucket in range(shards)}
    counts = Counter()

    def flush(bucket: int):
        columns = buffers[bucket]
        if not columns["source1_entity_id"]:
            return
        if bucket not in writers:
            writers[bucket] = pq.ParquetWriter(output_dir / f"part-{bucket:03d}.parquet.tmp", RAW_SCHEMA, compression="zstd")
        writers[bucket].write_table(pa.Table.from_pydict(columns, schema=RAW_SCHEMA), row_group_size=10_000)
        for values in columns.values():
            values.clear()

    try:
        for path in channel_paths:
            parquet = pq.ParquetFile(path)
            columns = parquet.schema_arrow.names
            required = {"source1_entity_id", "candidate_entity_id", "channel"}
            if not required.issubset(columns):
                raise ValueError(f"{path}: missing {sorted(required - set(columns))}")
            for batch in parquet.iter_batches(batch_size=100_000):
                values = {name: batch.column(batch.schema.get_field_index(name)).to_pylist() for name in columns}
                for row in range(batch.num_rows):
                    s1_id = values["source1_entity_id"][row]
                    channel = values["channel"][row]
                    bucket = _s1_bucket(s1_id, shards)
                    output = buffers[bucket]
                    output["source1_entity_id"].append(s1_id)
                    output["candidate_entity_id"].append(values["candidate_entity_id"][row])
                    output["channel"].append(channel)
                    output["block_size"].append(
                        values["exact_block_size"][row] if channel.startswith("exact_") else
                        values["blocking_key_size"][row] if channel in ("rare_token", "numeric_postal") else None
                    )
                    output["cheap_score"].append(values["retrieval_score"][row] if channel in ("rare_token", "numeric_postal") else None)
                    output["tfidf_score"].append(values["tfidf_score"][row] if channel == "tfidf_name" else None)
                    output["tfidf_rank"].append(values["tfidf_rank"][row] if channel == "tfidf_name" else None)
                    counts[f"{channel}_raw_pairs"] += 1
                    counts["raw_pairs"] += 1
                    if len(output["source1_entity_id"]) >= 10_000:
                        flush(bucket)
        for bucket in range(shards):
            flush(bucket)
    except Exception:
        for writer in writers.values():
            writer.close()
        for path in output_dir.glob("*.tmp"):
            path.unlink()
        raise
    for bucket, writer in writers.items():
        writer.close()
        os.replace(output_dir / f"part-{bucket:03d}.parquet.tmp", output_dir / f"part-{bucket:03d}.parquet")
    return {**dict(counts), "shard_files": len(writers), "elapsed_seconds": round(time.perf_counter() - start, 3),
            "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
            "artifact_bytes": sum(path.stat().st_size for path in output_dir.glob("*.parquet")), "actual_threads": 1}


def _dedup_rank(frame: pl.DataFrame, cap: int) -> tuple[pl.DataFrame, int]:
    flags = {
        "retrieved_by_exact_name": "exact_clean", "retrieved_by_core_name": "exact_core",
        "retrieved_by_rare_token": "rare_token", "retrieved_by_number_postal": "numeric_postal",
        "retrieved_by_tfidf": "tfidf_name",
    }
    aggregated = frame.group_by(["source1_entity_id", "candidate_entity_id"]).agg(
        [(pl.col("channel") == channel).any().alias(name) for name, channel in flags.items()] + [
            pl.when(pl.col("channel").str.starts_with("exact_"))
              .then(pl.col("block_size")).otherwise(None).min().alias("exact_block_size"),
            pl.col("tfidf_score").max().alias("best_tfidf_score"),
            pl.col("tfidf_rank").min().alias("tfidf_rank"),
            pl.col("cheap_score").max().alias("best_cheap_score"),
        ]
    )
    count_expr = sum(pl.col(name).cast(pl.Int8) for name in flags)
    aggregated = aggregated.with_columns(count_expr.alias("retrieval_channel_count"))
    # Cheap ordering only. The pair model gets the full provenance and raw scores.
    rank_expr = (
        4.0 * pl.col("retrieved_by_exact_name").cast(pl.Float32) +
        3.0 * pl.col("retrieved_by_core_name").cast(pl.Float32) +
        1.5 * pl.col("retrieved_by_rare_token").cast(pl.Float32) +
        4.0 * pl.col("retrieved_by_number_postal").cast(pl.Float32) +
        4.0 * pl.col("best_tfidf_score").fill_null(0.0) +
        0.1 * pl.col("best_cheap_score").fill_null(0.0) +
        1.0 * (pl.col("retrieval_channel_count").cast(pl.Float32) - 1.0) +
        2.0 / (pl.col("exact_block_size").fill_null(1000).cast(pl.Float32) + 1.0)
    )
    aggregated = aggregated.with_columns(rank_expr.cast(pl.Float32).alias("preliminary_score"))
    before = aggregated.height
    ranked = aggregated.sort(["source1_entity_id", "preliminary_score", "candidate_entity_id"],
                             descending=[False, True, False])
    kept = ranked.group_by("source1_entity_id", maintain_order=True).head(cap)
    return kept.select(FINAL_COLUMNS), before


def merge_shards(raw_dir: Path, output_dir: Path, *, cap: int, query_rows: int) -> dict:
    if cap < 1:
        raise ValueError("cap must be positive")
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_pairs = unique_pairs = final_pairs = entities_with_candidates = 0
    max_raw_shard_rows = max_output_shard_rows = 0
    for path in sorted(raw_dir.glob("part-*.parquet")):
        frame = pl.read_parquet(path)
        raw_pairs += frame.height
        max_raw_shard_rows = max(max_raw_shard_rows, frame.height)
        kept, before = _dedup_rank(frame, cap)
        unique_pairs += before
        final_pairs += kept.height
        max_output_shard_rows = max(max_output_shard_rows, kept.height)
        entities_with_candidates += kept.select(pl.col("source1_entity_id").n_unique()).item()
        output_path = output_dir / path.name
        temporary = output_path.with_suffix(".parquet.tmp")
        kept.write_parquet(temporary, compression="zstd")
        os.replace(temporary, output_path)
        del frame, kept
    return {
        "raw_pairs": raw_pairs, "unique_pairs_before_cap": unique_pairs, "final_pairs": final_pairs,
        "deduplicated_pairs": raw_pairs - unique_pairs, "pruned_by_cap": unique_pairs - final_pairs,
        "entities_with_candidates": entities_with_candidates,
        "zero_candidate_entities": query_rows - entities_with_candidates,
        "max_raw_shard_rows": max_raw_shard_rows, "max_output_shard_rows": max_output_shard_rows,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
        "artifact_bytes": sum(path.stat().st_size for path in output_dir.glob("*.parquet")),
        "actual_threads": 1,
    }


def build_union(channel_paths: list[Path], query_path: Path, raw_root: Path, final_root: Path,
                *, cap: int = 20, shards: int = 64) -> dict:
    """Cache raw channel sharding separately from ranking/cap experiments."""
    raw_signature = hashlib.sha256(json.dumps({
        "channels": [_sha256(path) for path in channel_paths], "raw_format": RAW_FORMAT_VERSION,
        "shards": shards,
    }, sort_keys=True).encode()).hexdigest()
    final_signature = hashlib.sha256(json.dumps({
        "raw": raw_signature, "query": _sha256(query_path), "cap": cap,
        "ranking_code": _sha256(Path(__file__)),
    }, sort_keys=True).encode()).hexdigest()
    raw_dir = raw_root / raw_signature[:12]
    final_dir = final_root / f"k{cap}_{final_signature[:12]}"
    raw_manifest = raw_dir / "manifest.json"
    final_manifest = final_dir / "manifest.json"
    if final_manifest.is_file():
        cached = json.loads(final_manifest.read_text())
        if cached.get("signature") == final_signature:
            return {"cache_hit": True, **cached}
    raw_hit = False
    if raw_manifest.is_file():
        cached = json.loads(raw_manifest.read_text())
        raw_hit = cached.get("signature") == raw_signature
    if raw_hit:
        raw_stats = cached["stats"]
    else:
        raw_stats = shard_channels(channel_paths, raw_dir, shards=shards)
        temporary = raw_manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"signature": raw_signature, "stats": raw_stats}, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, raw_manifest)
    query_rows = pq.ParquetFile(query_path).metadata.num_rows
    final_stats = merge_shards(raw_dir, final_dir, cap=cap, query_rows=query_rows)
    report = {"signature": final_signature, "raw_cache_hit": raw_hit, "cap": cap,
              "raw_dir": str(raw_dir), "final_dir": str(final_dir),
              "raw": raw_stats, "final": final_stats}
    temporary = final_manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, final_manifest)
    return report
